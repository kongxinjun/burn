//! Differential probe: every case runs on `Device::default()` (BURN_DEVICE) and on Flex,
//! and compares the results. Inputs come in several memory layouts with the same logical
//! values, so layout bugs show up too. Each #[test] is one op family; it prints one line
//! per case (`ok` / `FAIL`) and fails with the list of mismatches.
#![allow(clippy::all, unused)]

extern crate alloc;

use burn_tensor::activation as act;
use burn_tensor::module;
use burn_tensor::ops::{
    AvgPoolOptions, ConvOptions, ConvTransposeOptions, MaxPoolOptions, CoordinateTransformMode, GridSampleMode,
    GridSampleOptions, GridSamplePaddingMode, InterpolateMode, InterpolateOptions, PadMode,
};
use burn_tensor::{
    Bool, DType, Device, FloatDType, IndexingUpdateOp, Int, IntDType, Tensor, TensorData, s,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

// ---------------------------------------------------------------- harness

fn dut() -> Device {
    Device::default()
}
fn refd() -> Device {
    Device::flex()
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [-1, 1).
    fn f(&mut self) -> f32 {
        (self.next_u32() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn randv(shape: &[usize], seed: u64) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..numel(shape)).map(|_| r.f()).collect()
}

/// Distinct values (a shuffled arange scaled to [-1, 1)), so argmax/sort have no ties.
fn distinctv(shape: &[usize], seed: u64) -> Vec<f32> {
    let n = numel(shape);
    let mut idx: Vec<usize> = (0..n).collect();
    let mut r = Rng::new(seed);
    for i in (1..n).rev() {
        let j = (r.next_u32() as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx.iter()
        .map(|&i| (i as f32) / (n as f32) * 2.0 - 1.0)
        .collect()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    s
}

/// Host permute: output[o] = input[...] with out shape = shape[perm[i]].
fn host_permute(v: &[f32], shape: &[usize], perm: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p]).collect();
    let in_st = strides(shape);
    let out_st = strides(&out_shape);
    let n = v.len();
    let mut out = vec![0.0; n];
    for o in 0..n {
        let mut rem = o;
        let mut src = 0;
        for i in 0..out_shape.len() {
            let c = rem / out_st[i];
            rem %= out_st[i];
            src += c * in_st[perm[i]];
        }
        out[o] = v[src];
    }
    (out, out_shape)
}

fn inv_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

#[derive(Clone, Copy, Debug)]
enum Layout {
    Contig,
    /// Physical order NHWC-like: last logical dim is not innermost.
    PermLast,
    /// Physical order reversed.
    PermRev,
    /// Last two dims swapped in memory.
    Transposed,
    /// Every second row of a twice-as-large dim 1.
    Strided,
    /// Inner window of a larger tensor (offset + non-unit outer strides).
    Window,
    /// Stored flipped along the last dim.
    Flipped,
}

const LAYOUTS: [Layout; 7] = [
    Layout::Contig,
    Layout::PermLast,
    Layout::PermRev,
    Layout::Transposed,
    Layout::Strided,
    Layout::Window,
    Layout::Flipped,
];

/// A rank-D float tensor with logical values `v` and shape `shape`, laid out as `layout`.
fn make<const D: usize>(v: &[f32], shape: [usize; D], layout: Layout, d: &Device) -> Tensor<D> {
    match layout {
        Layout::Contig => Tensor::from_data(TensorData::new(v.to_vec(), shape), d),
        Layout::PermLast | Layout::PermRev => {
            // physical order `phys`: logical dim phys[i] is the i-th physical dim.
            let phys: Vec<usize> = match layout {
                Layout::PermLast => {
                    let mut p = vec![0];
                    p.extend(2..D);
                    p.push(1);
                    if D == 1 { vec![0] } else { p }
                }
                _ => (0..D).rev().collect(),
            };
            let (pv, ps) = host_permute(v, &shape, &phys);
            let ps: [usize; D] = ps.try_into().unwrap();
            let t: Tensor<D> = Tensor::from_data(TensorData::new(pv, ps), d);
            let inv: [usize; D] = inv_perm(&phys).try_into().unwrap();
            t.permute(inv.map(|x| x as isize))
        }
        Layout::Transposed => {
            if D < 2 {
                return make(v, shape, Layout::Contig, d);
            }
            let mut perm: Vec<usize> = (0..D).collect();
            perm.swap(D - 1, D - 2);
            let (pv, ps) = host_permute(v, &shape, &perm);
            let ps: [usize; D] = ps.try_into().unwrap();
            let t: Tensor<D> = Tensor::from_data(TensorData::new(pv, ps), d);
            t.swap_dims(D - 1, D - 2)
        }
        Layout::Strided => {
            // Double dim 0 (or the only dim), keep even entries.
            let mut big = shape;
            big[0] *= 2;
            let inner: usize = shape[1..].iter().product();
            let mut bv = vec![-7.0f32; numel(&big)];
            for i in 0..shape[0] {
                bv[(2 * i) * inner..(2 * i + 1) * inner]
                    .copy_from_slice(&v[i * inner..(i + 1) * inner]);
            }
            let t: Tensor<D> = Tensor::from_data(TensorData::new(bv, big), d);
            t.slice_dim(0, s![0..big[0];2])
        }
        Layout::Window => {
            let mut big = shape;
            for i in 0..D {
                big[i] += 2;
            }
            let bst = strides(&big);
            let st = strides(&shape);
            let mut bv = vec![9.0f32; numel(&big)];
            for o in 0..v.len() {
                let mut rem = o;
                let mut dst = 0;
                for i in 0..D {
                    let c = rem / st[i];
                    rem %= st[i];
                    dst += (c + 1) * bst[i];
                }
                bv[dst] = v[o];
            }
            let mut t: Tensor<D> = Tensor::from_data(TensorData::new(bv, big), d);
            for i in 0..D {
                t = t.narrow(i, 1, shape[i]);
            }
            t
        }
        Layout::Flipped => {
            let last = shape[D - 1];
            let mut fv = v.to_vec();
            for chunk in fv.chunks_mut(last) {
                chunk.reverse();
            }
            let t: Tensor<D> = Tensor::from_data(TensorData::new(fv, shape), d);
            t.flip([D - 1])
        }
    }
}

fn to_f64(data: &TensorData) -> Vec<f64> {
    data.iter::<f64>().collect()
}

#[derive(Clone, Copy)]
struct Tol {
    atol: f64,
    rtol: f64,
}
const EXACT: Tol = Tol { atol: 0.0, rtol: 0.0 };
const F32: Tol = Tol { atol: 1e-4, rtol: 1e-4 };
const LOOSE: Tol = Tol { atol: 2e-3, rtol: 2e-3 };
const MM: Tol = Tol { atol: 2e-2, rtol: 1e-2 };
const HALF: Tol = Tol { atol: 2e-2, rtol: 2e-2 };

fn compare(a: &TensorData, b: &TensorData, tol: Tol) -> Result<(), String> {
    if a.shape() != b.shape() {
        return Err(format!("shape {:?} vs ref {:?}", a.shape(), b.shape()));
    }
    let av = to_f64(a);
    let bv = to_f64(b);
    let mut bad = 0usize;
    let mut first = None;
    let mut maxerr = 0f64;
    for (i, (&x, &y)) in av.iter().zip(bv.iter()).enumerate() {
        let ok = if x.is_nan() || y.is_nan() {
            x.is_nan() && y.is_nan()
        } else if x.is_infinite() || y.is_infinite() {
            x == y
        } else {
            let e = (x - y).abs();
            if e > maxerr {
                maxerr = e;
            }
            e <= tol.atol + tol.rtol * y.abs()
        };
        if !ok {
            bad += 1;
            if first.is_none() {
                first = Some(i);
            }
        }
    }
    if bad == 0 {
        Ok(())
    } else {
        let i = first.unwrap();
        let lo = i.saturating_sub(2);
        let hi = (i + 4).min(av.len());
        Err(format!(
            "{bad}/{} mismatched (maxerr {maxerr:.3e}); first at {i}: got {:?} ref {:?} (dtype {:?} vs {:?})",
            av.len(),
            &av[lo..hi],
            &bv[lo..hi],
            a.dtype(),
            b.dtype()
        ))
    }
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<String>() {
        s.lines().next().unwrap_or("").chars().take(300).collect()
    } else if let Some(s) = e.downcast_ref::<&str>() {
        s.chars().take(300).collect()
    } else {
        "panic".into()
    }
}

struct Probe {
    group: &'static str,
    fails: Vec<String>,
    total: usize,
}

impl Probe {
    fn new(group: &'static str) -> Self {
        Probe {
            group,
            fails: vec![],
            total: 0,
        }
    }

    /// Runs `f` on the device under test and on Flex, compares the outputs.
    fn case(&mut self, name: impl Into<String>, tol: Tol, f: impl Fn(&Device) -> Vec<TensorData>) {
        let name = name.into();
        self.total += 1;
        let r = catch_unwind(AssertUnwindSafe(|| f(&refd())));
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                println!("SKIP\t{}\t{name}\tref panicked: {}", self.group, panic_msg(e));
                return;
            }
        };
        let o = catch_unwind(AssertUnwindSafe(|| f(&dut())));
        let res = match o {
            Err(e) => Err(format!("device panicked: {}", panic_msg(e))),
            Ok(o) => {
                let mut res = Ok(());
                for (k, (a, b)) in o.iter().zip(r.iter()).enumerate() {
                    if let Err(m) = compare(a, b, tol) {
                        res = Err(format!("output {k}: {m}"));
                        break;
                    }
                }
                res
            }
        };
        match res {
            Ok(()) => println!("ok\t{}\t{name}", self.group),
            Err(m) => {
                println!("FAIL\t{}\t{name}\t{m}", self.group);
                self.fails.push(format!("{name}: {m}"));
            }
        }
    }

    fn finish(self) {
        println!(
            "SUMMARY\t{}\t{} cases, {} failed",
            self.group,
            self.total,
            self.fails.len()
        );
        if !self.fails.is_empty() {
            panic!(
                "{} of {} cases failed:\n{}",
                self.fails.len(),
                self.total,
                self.fails.join("\n")
            );
        }
    }
}

/// Uniform way to read any tensor kind back.
trait IntoData {
    fn data(self) -> TensorData;
}
impl<const D: usize> IntoData for Tensor<D> {
    fn data(self) -> TensorData {
        self.into_data()
    }
}
impl<const D: usize> IntoData for Tensor<D, Int> {
    fn data(self) -> TensorData {
        self.into_data()
    }
}
impl<const D: usize> IntoData for Tensor<D, Bool> {
    fn data(self) -> TensorData {
        self.int().into_data()
    }
}

macro_rules! out {
    ($($t:expr),+ $(,)?) => { vec![$(IntoData::data($t)),+] };
}

const SHAPE: [usize; 4] = [2, 3, 5, 7];

// ---------------------------------------------------------------- reductions

#[test]
fn probe_reduce() {
    let mut p = Probe::new("reduce");
    let v = distinctv(&SHAPE, 1);
    for layout in LAYOUTS {
        for dim in 0..4 {
            let x = |d: &Device| make(&v, SHAPE, layout, d);
            p.case(format!("sum_dim{dim} {layout:?}"), F32, |d| out![x(d).sum_dim(dim)]);
            p.case(format!("mean_dim{dim} {layout:?}"), F32, |d| out![x(d).mean_dim(dim)]);
            p.case(format!("max_dim{dim} {layout:?}"), EXACT, |d| out![x(d).max_dim(dim)]);
            p.case(format!("min_dim{dim} {layout:?}"), EXACT, |d| out![x(d).min_dim(dim)]);
            p.case(format!("argmax{dim} {layout:?}"), EXACT, |d| out![x(d).argmax(dim)]);
            p.case(format!("argmin{dim} {layout:?}"), EXACT, |d| out![x(d).argmin(dim)]);
            p.case(format!("max_dim_with_indices{dim} {layout:?}"), EXACT, |d| {
                let (a, b) = x(d).max_dim_with_indices(dim);
                out![a, b]
            });
            p.case(format!("prod_dim{dim} {layout:?}"), F32, |d| {
                out![x(d).mul_scalar(0.2).add_scalar(1.0).prod_dim(dim)]
            });
            p.case(format!("max_abs_dim{dim} {layout:?}"), EXACT, |d| out![x(d).max_abs_dim(dim)]);
            p.case(format!("any_dim{dim} {layout:?}"), EXACT, |d| {
                out![x(d).greater_elem(0.9).any_dim(dim)]
            });
            p.case(format!("all_dim{dim} {layout:?}"), EXACT, |d| {
                out![x(d).greater_elem(-0.9).all_dim(dim)]
            });
            p.case(format!("softmax{dim} {layout:?}"), F32, |d| out![act::softmax(x(d), dim)]);
            p.case(format!("log_softmax{dim} {layout:?}"), F32, |d| {
                out![act::log_softmax(x(d), dim)]
            });
            p.case(format!("sum_dims[{dim},3] {layout:?}"), F32, |d| {
                out![x(d).sum_dims(&[dim, 3])]
            });
        }
        let x = |d: &Device| make(&v, SHAPE, layout, d);
        p.case(format!("sum {layout:?}"), F32, |d| out![x(d).sum()]);
        p.case(format!("mean {layout:?}"), F32, |d| out![x(d).mean()]);
        p.case(format!("max {layout:?}"), EXACT, |d| out![x(d).max()]);
        p.case(format!("min {layout:?}"), EXACT, |d| out![x(d).min()]);
        p.case(format!("prod {layout:?}"), LOOSE, |d| {
            out![x(d).mul_scalar(0.01).add_scalar(1.0).prod()]
        });
        p.case(format!("any {layout:?}"), EXACT, |d| out![x(d).greater_elem(0.999).any()]);
        p.case(format!("all {layout:?}"), EXACT, |d| out![x(d).greater_elem(-0.999).all()]);
    }
    // Long reductions (multi-plane / multi-cube paths).
    for &(n, m) in &[(1usize, 4097usize), (3, 33000), (7, 1025), (1, 1_000_003), (513, 3)] {
        let v = randv(&[n, m], 2);
        for layout in [Layout::Contig, Layout::Transposed] {
            let x = |d: &Device| make(&v, [n, m], layout, d);
            p.case(format!("long sum_dim1 {n}x{m} {layout:?}"), LOOSE, |d| out![x(d).sum_dim(1)]);
            p.case(format!("long sum_dim0 {n}x{m} {layout:?}"), LOOSE, |d| out![x(d).sum_dim(0)]);
            p.case(format!("long max_dim1 {n}x{m} {layout:?}"), EXACT, |d| out![x(d).max_dim(1)]);
            p.case(format!("long argmax1 {n}x{m} {layout:?}"), EXACT, |d| out![x(d).argmax(1)]);
            p.case(format!("long argmin0 {n}x{m} {layout:?}"), EXACT, |d| out![x(d).argmin(0)]);
            p.case(format!("long mean {n}x{m} {layout:?}"), LOOSE, |d| out![x(d).mean()]);
            p.case(format!("long softmax1 {n}x{m} {layout:?}"), F32, |d| {
                out![act::softmax(x(d), 1)]
            });
        }
    }
    p.finish();
}

#[test]
fn probe_reduce_nan_ties() {
    let mut p = Probe::new("reduce_nan_ties");
    // NaN at various positions in a row of 8.
    for pos in [0usize, 1, 5, 7] {
        let mut v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        v[pos] = f32::NAN;
        let x = |d: &Device| Tensor::<2>::from_data(TensorData::new(v.clone(), [1, 8]), d);
        p.case(format!("max_dim nan@{pos}"), EXACT, |d| out![x(d).max_dim(1)]);
        p.case(format!("min_dim nan@{pos}"), EXACT, |d| out![x(d).min_dim(1)]);
        p.case(format!("argmax nan@{pos}"), EXACT, |d| out![x(d).argmax(1)]);
        p.case(format!("argmin nan@{pos}"), EXACT, |d| out![x(d).argmin(1)]);
        p.case(format!("max nan@{pos}"), EXACT, |d| out![x(d).max()]);
        p.case(format!("min nan@{pos}"), EXACT, |d| out![x(d).min()]);
        p.case(format!("sum nan@{pos}"), EXACT, |d| out![x(d).sum()]);
        p.case(format!("cummax nan@{pos}"), EXACT, |d| out![x(d).cummax(1)]);
        p.case(format!("cummin nan@{pos}"), EXACT, |d| out![x(d).cummin(1)]);
        p.case(format!("max_abs_dim nan@{pos}"), EXACT, |d| out![x(d).max_abs_dim(1)]);
        p.case(format!("topk2 nan@{pos}"), EXACT, |d| out![x(d).topk(2, 1)]);
        p.case(format!("sort nan@{pos}"), EXACT, |d| out![x(d).sort(1)]);
        p.case(format!("max_pool1d nan@{pos}"), EXACT, |d| {
            out![module::max_pool1d(x(d).unsqueeze_dim(0), MaxPoolOptions::new([3]))]
        });
        p.case(format!("max_pool2d nan@{pos}"), EXACT, |d| {
            out![module::max_pool2d(
                x(d).reshape([1, 1, 2, 4]),
                MaxPoolOptions::new([2, 2]).with_stride([1, 1])
            )]
        });
        p.case(format!("max_pool2d_idx nan@{pos}"), EXACT, |d| {
            let (a, i) = module::max_pool2d_with_indices(
                x(d).reshape([1, 1, 2, 4]),
                MaxPoolOptions::new([2, 2]).with_stride([2, 2]),
            );
            out![a, i]
        });
        p.case(format!("max_pair nan@{pos}"), EXACT, |d| {
            out![x(d).max_pair(x(d).flip([1]))]
        });
        p.case(format!("clamp nan@{pos}"), EXACT, |d| out![x(d).clamp(1.0, 5.0)]);
        p.case(format!("softmax nan@{pos}"), EXACT, |d| out![act::softmax(x(d), 1)]);
    }
    // Ties: first index wins (PyTorch semantics, and Burn's flex).
    let v = vec![1.0f32, 3.0, 3.0, 0.0, 3.0, -1.0, 3.0, 2.0];
    for len in [8usize] {
        let x = |d: &Device| Tensor::<2>::from_data(TensorData::new(v.clone(), [1, len]), d);
        p.case("argmax ties", EXACT, |d| out![x(d).argmax(1)]);
        p.case("argmin ties", EXACT, |d| out![x(d).neg().argmin(1)]);
        p.case("max_dim_with_indices ties", EXACT, |d| {
            let (a, b) = x(d).max_dim_with_indices(1);
            out![a, b]
        });
    }
    // Long tie row: all zeros, argmax must be 0.
    for n in [64usize, 1000, 5000, 70000] {
        let x = |d: &Device| Tensor::<2>::zeros([3, n], d);
        p.case(format!("argmax zeros {n}"), EXACT, |d| out![x(d).argmax(1)]);
        p.case(format!("argmin zeros {n}"), EXACT, |d| out![x(d).argmin(1)]);
        let xi = |d: &Device| Tensor::<2, Int>::zeros([3, n], d);
        p.case(format!("int argmax zeros {n}"), EXACT, |d| out![xi(d).argmax(1)]);
    }
    // -0.0 vs 0.0, inf.
    let v = vec![-0.0f32, 0.0, f32::INFINITY, f32::NEG_INFINITY, -3.0, 5.0];
    let x = |d: &Device| Tensor::<1>::from_data(TensorData::new(v.clone(), [6]), d);
    p.case("sum inf-inf", EXACT, |d| out![x(d).sum()]);
    p.case("max inf", EXACT, |d| out![x(d).max()]);
    p.case("argmin inf", EXACT, |d| out![x(d).argmin(0)]);
    p.case("sort inf", EXACT, |d| out![x(d).sort(0)]);
    p.finish();
}

// ---------------------------------------------------------------- scans / sort

#[test]
fn probe_scan_sort() {
    let mut p = Probe::new("scan_sort");
    let v = distinctv(&SHAPE, 3);
    for layout in LAYOUTS {
        for dim in 0..4 {
            let x = |d: &Device| make(&v, SHAPE, layout, d);
            p.case(format!("cumsum{dim} {layout:?}"), F32, |d| out![x(d).cumsum(dim)]);
            p.case(format!("cumprod{dim} {layout:?}"), F32, |d| {
                out![x(d).mul_scalar(0.3).add_scalar(1.0).cumprod(dim)]
            });
            p.case(format!("cummax{dim} {layout:?}"), EXACT, |d| out![x(d).cummax(dim)]);
            p.case(format!("cummin{dim} {layout:?}"), EXACT, |d| out![x(d).cummin(dim)]);
            p.case(format!("sort{dim} {layout:?}"), EXACT, |d| out![x(d).sort(dim)]);
            p.case(format!("sort_desc_idx{dim} {layout:?}"), EXACT, |d| {
                let (a, b) = x(d).sort_descending_with_indices(dim);
                out![a, b]
            });
            p.case(format!("argsort{dim} {layout:?}"), EXACT, |d| out![x(d).argsort(dim)]);
            p.case(format!("topk2_idx{dim} {layout:?}"), EXACT, |d| {
                let (a, b) = x(d).topk_with_indices(2, dim);
                out![a, b]
            });
        }
    }
    for n in [100usize, 1025, 4096, 20000] {
        let v = randv(&[2, n], 4);
        let x = |d: &Device| Tensor::<2>::from_data(TensorData::new(v.clone(), [2, n]), d);
        p.case(format!("long cumsum {n}"), LOOSE, |d| out![x(d).cumsum(1)]);
        p.case(format!("long cummax {n}"), EXACT, |d| out![x(d).cummax(1)]);
        p.case(format!("long sort {n}"), EXACT, |d| out![x(d).sort(1)]);
        p.case(format!("long argsort {n}"), EXACT, |d| out![x(d).argsort(1)]);
        p.case(format!("long topk5 {n}"), EXACT, |d| {
            let (a, b) = x(d).topk_with_indices(5, 1);
            out![a, b]
        });
        p.case(format!("long topk{n}/2"), EXACT, |d| out![x(d).topk(n / 2, 1)]);
    }
    p.finish();
}

// ---------------------------------------------------------------- matmul

#[test]
fn probe_matmul() {
    let mut p = Probe::new("matmul");
    let shapes: &[(usize, usize, usize, usize)] = &[
        (1, 1, 1, 1),
        (2, 3, 5, 7),
        (1, 1, 300, 1),
        (3, 1, 64, 1),
        (3, 1, 70, 33),
        (2, 33, 70, 1),
        (2, 17, 33, 9),
        (1, 128, 128, 128),
        (4, 1, 257, 513),
        (1, 65, 1, 65),
        (2, 100, 3, 100),
        (1, 255, 129, 63),
        (6, 31, 64, 47),
    ];
    for &(b, m, k, n) in shapes {
        let lv = randv(&[b, m, k], 10);
        let rv = randv(&[b, k, n], 11);
        for (ll, rl) in [
            (Layout::Contig, Layout::Contig),
            (Layout::Transposed, Layout::Contig),
            (Layout::Contig, Layout::Transposed),
            (Layout::Transposed, Layout::Transposed),
            (Layout::PermRev, Layout::PermLast),
            (Layout::Window, Layout::Strided),
        ] {
            p.case(format!("mm {b}x{m}x{k}x{n} {ll:?}/{rl:?}"), MM, |d| {
                out![make(&lv, [b, m, k], ll, d).matmul(make(&rv, [b, k, n], rl, d))]
            });
        }
        // Broadcast batch: lhs batch 1.
        let l1 = randv(&[1, m, k], 12);
        p.case(format!("mm bcast-lhs {b}x{m}x{k}x{n}"), MM, |d| {
            out![make(&l1, [1, m, k], Layout::Contig, d).matmul(make(&rv, [b, k, n], Layout::Contig, d))]
        });
        p.case(format!("mm expand-lhs {b}x{m}x{k}x{n}"), MM, |d| {
            out![make(&l1, [1, m, k], Layout::Contig, d)
                .expand([b, m, k])
                .matmul(make(&rv, [b, k, n], Layout::Contig, d))]
        });
        let r1 = randv(&[1, k, n], 13);
        p.case(format!("mm bcast-rhs-T {b}x{m}x{k}x{n}"), MM, |d| {
            out![make(&lv, [b, m, k], Layout::Contig, d).matmul(make(&r1, [1, k, n], Layout::Transposed, d))]
        });
        // 4D broadcast [2,1,m,k] x [1,b,k,n]
        let l4 = randv(&[2, 1, m, k], 14);
        p.case(format!("mm 4d-bcast {b}x{m}x{k}x{n}"), MM, |d| {
            out![make(&l4, [2, 1, m, k], Layout::Contig, d)
                .matmul(make(&rv, [b, k, n], Layout::Contig, d).unsqueeze_dim(0))]
        });
        // f16 / bf16 against f32 flex.
        for dt in [FloatDType::F16, FloatDType::BF16] {
            p.case(format!("mm {dt:?} {b}x{m}x{k}x{n}"), Tol { atol: 5e-2 + 4e-3 * k as f64, rtol: 2e-2 }, |d| {
                let l = make(&lv, [b, m, k], Layout::Contig, d).cast(dt);
                let r = make(&rv, [b, k, n], Layout::Transposed, d).cast(dt);
                if d.as_dispatch() == refd().as_dispatch() {
                    out![l.cast(FloatDType::F32).matmul(r.cast(FloatDType::F32))]
                } else {
                    out![l.matmul(r).cast(FloatDType::F32)]
                }
            });
        }
        // int matmul
        p.case(format!("int mm {b}x{m}x{k}x{n}"), EXACT, |d| {
            let l = make(&lv, [b, m, k], Layout::Contig, d).mul_scalar(5.0).int();
            let r = make(&rv, [b, k, n], Layout::Transposed, d).mul_scalar(5.0).int();
            out![l.matmul(r)]
        });
    }
    p.finish();
}

// ---------------------------------------------------------------- conv

#[test]
fn probe_conv() {
    let mut p = Probe::new("conv");
    // (n, cin, h, w, cout, kh, kw, stride, pad, dil, groups)
    let cases: &[(usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize)] = &[
        (2, 3, 13, 11, 4, 3, 3, 1, 1, 1, 1),
        (2, 3, 13, 11, 4, 3, 3, 2, 0, 1, 1),
        (1, 4, 13, 11, 6, 3, 3, 1, 2, 2, 2),
        (2, 8, 9, 9, 8, 3, 3, 1, 1, 1, 8),
        (2, 8, 9, 9, 16, 3, 3, 2, 1, 1, 8),
        (1, 16, 12, 12, 32, 1, 1, 1, 0, 1, 1),
        (3, 6, 15, 14, 4, 5, 5, 3, 2, 1, 2),
        (1, 5, 10, 17, 7, 3, 1, 1, 1, 1, 1),
        (1, 3, 10, 17, 7, 1, 4, 2, 0, 1, 1),
        (2, 64, 16, 16, 64, 3, 3, 1, 1, 1, 1),
        (1, 32, 20, 20, 48, 3, 3, 2, 1, 1, 4),
        (1, 3, 33, 33, 8, 7, 7, 2, 3, 1, 1),
        (4, 1, 5, 5, 1, 5, 5, 1, 0, 1, 1),
        (1, 12, 8, 8, 12, 3, 3, 1, 3, 3, 12),
    ];
    for &(n, ci, h, w, co, kh, kw, st, pd, dl, g) in cases {
        let xv = randv(&[n, ci, h, w], 20);
        let wv = randv(&[co, ci / g, kh, kw], 21);
        let bv = randv(&[co], 22);
        let name = format!("conv2d n{n} c{ci}->{co} {h}x{w} k{kh}x{kw} s{st} p{pd} d{dl} g{g}");
        let opts = ConvOptions::new([st, st], [pd, pd], [dl, dl], g);
        for (xl, wl) in [
            (Layout::Contig, Layout::Contig),
            (Layout::PermLast, Layout::Contig),
            (Layout::Contig, Layout::PermLast),
            (Layout::Window, Layout::Transposed),
        ] {
            p.case(format!("{name} {xl:?}/{wl:?}"), MM, |d| {
                out![module::conv2d(
                    make(&xv, [n, ci, h, w], xl, d),
                    make(&wv, [co, ci / g, kh, kw], wl, d),
                    Some(Tensor::from_data(TensorData::new(bv.clone(), [co]), d)),
                    opts.clone()
                )]
            });
        }
        p.case(format!("{name} f16"), Tol { atol: 5e-2, rtol: 3e-2 }, |d| {
            let x = make(&xv, [n, ci, h, w], Layout::Contig, d).cast(FloatDType::F16);
            let wt = make(&wv, [co, ci / g, kh, kw], Layout::Contig, d).cast(FloatDType::F16);
            if d.as_dispatch() == refd().as_dispatch() {
                out![module::conv2d(x.cast(FloatDType::F32), wt.cast(FloatDType::F32), None, opts.clone())]
            } else {
                out![module::conv2d(x, wt, None, opts.clone()).cast(FloatDType::F32)]
            }
        });
    }
    // Asymmetric padding.
    let xv = randv(&[1, 3, 10, 9], 23);
    let wv = randv(&[4, 3, 3, 3], 24);
    p.case("conv2d asym pad", MM, |d| {
        out![module::conv2d(
            make(&xv, [1, 3, 10, 9], Layout::Contig, d),
            make(&wv, [4, 3, 3, 3], Layout::Contig, d),
            None,
            ConvOptions::new_with_padding([1, 2], [(0, 2), (1, 0)], [1, 1], 1)
        )]
    });
    // conv_transpose2d: (n, cin, h, w, cout, k, stride, pad, pad_out, dil, groups)
    let tcases: &[(usize, usize, usize, usize, usize, usize, usize, usize, usize, usize, usize)] = &[
        (2, 3, 7, 6, 4, 3, 2, 1, 1, 1, 1),
        (1, 4, 5, 5, 6, 4, 2, 1, 0, 1, 2),
        (1, 3, 6, 7, 2, 3, 1, 0, 0, 2, 1),
        (2, 6, 4, 4, 6, 3, 3, 0, 2, 1, 3),
        (1, 8, 9, 9, 8, 3, 2, 1, 1, 1, 8),
        (1, 32, 8, 8, 16, 4, 2, 1, 0, 1, 1),
        (1, 2, 5, 3, 3, 5, 3, 2, 1, 2, 1),
    ];
    for &(n, ci, h, w, co, k, st, pd, po, dl, g) in tcases {
        let xv = randv(&[n, ci, h, w], 25);
        let wv = randv(&[ci, co / g, k, k], 26);
        let bv = randv(&[co], 27);
        let name = format!("convT2d n{n} c{ci}->{co} {h}x{w} k{k} s{st} p{pd} op{po} d{dl} g{g}");
        let opts = ConvTransposeOptions::new([st, st], [pd, pd], [po, po], [dl, dl], g);
        for xl in [Layout::Contig, Layout::PermLast, Layout::Transposed] {
            p.case(format!("{name} {xl:?}"), MM, |d| {
                out![module::conv_transpose2d(
                    make(&xv, [n, ci, h, w], xl, d),
                    make(&wv, [ci, co / g, k, k], Layout::Contig, d),
                    Some(Tensor::from_data(TensorData::new(bv.clone(), [co]), d)),
                    opts.clone()
                )]
            });
        }
    }
    // conv1d / conv3d / convT1d
    let xv = randv(&[2, 4, 29], 28);
    let wv = randv(&[6, 2, 5], 29);
    for (st, pd, dl) in [(1, 2, 1), (2, 0, 1), (1, 4, 2), (3, 1, 1)] {
        for xl in [Layout::Contig, Layout::PermLast] {
            p.case(format!("conv1d s{st} p{pd} d{dl} g2 {xl:?}"), MM, |d| {
                out![module::conv1d(
                    make(&xv, [2, 4, 29], xl, d),
                    make(&wv, [6, 2, 5], Layout::Contig, d),
                    None,
                    ConvOptions::new([st], [pd], [dl], 2)
                )]
            });
            p.case(format!("convT1d s{st} p{pd} d{dl} {xl:?}"), MM, |d| {
                out![module::conv_transpose1d(
                    make(&xv, [2, 4, 29], xl, d),
                    make(&randv(&[4, 3, 5], 30), [4, 3, 5], Layout::Contig, d),
                    None,
                    ConvTransposeOptions::new([st], [pd], [st.saturating_sub(1).min(dl)], [dl], 1)
                )]
            });
        }
    }
    let xv = randv(&[1, 4, 6, 7, 5], 31);
    let wv = randv(&[4, 2, 3, 3, 3], 32);
    for (st, pd, dl) in [(1, 1, 1), (2, 0, 1), (1, 2, 2)] {
        for xl in [Layout::Contig, Layout::PermLast] {
            p.case(format!("conv3d s{st} p{pd} d{dl} g2 {xl:?}"), MM, |d| {
                out![module::conv3d(
                    make(&xv, [1, 4, 6, 7, 5], xl, d),
                    make(&wv, [4, 2, 3, 3, 3], Layout::Contig, d),
                    None,
                    ConvOptions::new([st; 3], [pd; 3], [dl; 3], 2)
                )]
            });
            p.case(format!("convT3d s{st} p{pd} d{dl} {xl:?}"), MM, |d| {
                out![module::conv_transpose3d(
                    make(&xv, [1, 4, 6, 7, 5], xl, d),
                    make(&randv(&[4, 2, 3, 3, 3], 33), [4, 2, 3, 3, 3], Layout::Contig, d),
                    None,
                    ConvTransposeOptions::new([st; 3], [pd; 3], [0; 3], [dl; 3], 2)
                )]
            });
        }
    }
    p.finish();
}

// ---------------------------------------------------------------- pooling

#[test]
fn probe_pool() {
    let mut p = Probe::new("pool");
    let shape = [2, 3, 13, 11];
    let v = distinctv(&shape, 40);
    // (k, stride, pad, dil, ceil)
    let mcases: &[(usize, usize, usize, usize, bool)] = &[
        (3, 2, 1, 1, false),
        (2, 2, 0, 1, true),
        (3, 1, 1, 2, false),
        (3, 3, 1, 1, true),
        (5, 2, 2, 1, true),
        (2, 1, 0, 3, false),
        (4, 3, 2, 1, true),
    ];
    for layout in [Layout::Contig, Layout::PermLast, Layout::Transposed, Layout::Window] {
        let x = |d: &Device| make(&v, shape, layout, d);
        for &(k, st, pd, dl, ceil) in mcases {
            let o = MaxPoolOptions::new([k, k])
                .with_stride([st, st])
                .with_padding([pd, pd])
                .with_dilation([dl, dl])
                .with_ceil_mode(ceil);
            p.case(format!("maxpool2d k{k} s{st} p{pd} d{dl} c{ceil} {layout:?}"), EXACT, |d| {
                let (a, i) = module::max_pool2d_with_indices(x(d), o.clone());
                out![module::max_pool2d(x(d), o.clone()), a, i]
            });
            if dl == 1 {
                for cip in [true, false] {
                    let o = AvgPoolOptions::new([k, k])
                        .with_stride([st, st])
                        .with_padding([pd, pd])
                        .with_count_include_pad(cip)
                        .with_ceil_mode(ceil);
                    p.case(
                        format!("avgpool2d k{k} s{st} p{pd} c{ceil} cip{cip} {layout:?}"),
                        F32,
                        |d| out![module::avg_pool2d(x(d), o.clone())],
                    );
                }
            }
            let o = AvgPoolOptions::new([k, k])
                .with_stride([st, st])
                .with_padding([pd, pd])
                .with_dilation([dl, dl])
                .with_count_include_pad(false)
                .with_ceil_mode(ceil);
            p.case(format!("avgpool2d dil k{k} s{st} p{pd} d{dl} c{ceil} {layout:?}"), F32, |d| {
                out![module::avg_pool2d(x(d), o.clone())]
            });
        }
        for os in [[1usize, 1], [3, 5], [7, 4], [13, 11], [20, 3], [6, 6]] {
            p.case(format!("adaptive_avg2d {os:?} {layout:?}"), F32, |d| {
                out![module::adaptive_avg_pool2d(x(d), os)]
            });
        }
    }
    // 1d / 3d
    let v1 = distinctv(&[2, 3, 23], 41);
    for layout in [Layout::Contig, Layout::PermLast] {
        let x = |d: &Device| make(&v1, [2, 3, 23], layout, d);
        for &(k, st, pd, dl, ceil) in mcases {
            p.case(format!("maxpool1d k{k} s{st} p{pd} d{dl} c{ceil} {layout:?}"), EXACT, |d| {
                let o = MaxPoolOptions::new([k]).with_stride([st]).with_padding([pd]).with_dilation([dl]).with_ceil_mode(ceil);
                let (a, i) = module::max_pool1d_with_indices(x(d), o.clone());
                out![a, i]
            });
            p.case(format!("avgpool1d k{k} s{st} p{pd} c{ceil} {layout:?}"), F32, |d| {
                let o = AvgPoolOptions::new([k]).with_stride([st]).with_padding([pd]).with_ceil_mode(ceil).with_count_include_pad(false);
                out![module::avg_pool1d(x(d), o)]
            });
        }
        for os in [1usize, 5, 7, 23, 40] {
            p.case(format!("adaptive_avg1d {os} {layout:?}"), F32, |d| {
                out![module::adaptive_avg_pool1d(x(d), os)]
            });
        }
    }
    let v3 = distinctv(&[1, 2, 7, 6, 5], 42);
    for layout in [Layout::Contig, Layout::PermLast] {
        let x = |d: &Device| make(&v3, [1, 2, 7, 6, 5], layout, d);
        for &(k, st, pd, dl, ceil) in &mcases[..4] {
            p.case(format!("maxpool3d k{k} s{st} p{pd} d{dl} c{ceil} {layout:?}"), EXACT, |d| {
                let o = MaxPoolOptions::new([k; 3]).with_stride([st; 3]).with_padding([pd; 3]).with_dilation([dl; 3]).with_ceil_mode(ceil);
                let (a, i) = module::max_pool3d_with_indices(x(d), o.clone());
                out![a, i]
            });
            p.case(format!("avgpool3d k{k} s{st} p{pd} c{ceil} {layout:?}"), F32, |d| {
                let o = AvgPoolOptions::new([k; 3]).with_stride([st; 3]).with_padding([pd; 3]).with_ceil_mode(ceil);
                out![module::avg_pool3d(x(d), o)]
            });
        }
        for os in [[1usize, 1, 1], [3, 4, 2], [7, 6, 5], [9, 2, 3]] {
            p.case(format!("adaptive_avg3d {os:?} {layout:?}"), F32, |d| {
                out![module::adaptive_avg_pool3d(x(d), os)]
            });
        }
    }
    p.finish();
}

// ---------------------------------------------------------------- interpolate / grid sample

#[test]
fn probe_interpolate() {
    let mut p = Probe::new("interpolate");
    let modes = || {
        vec![
            ("nearest", InterpolateMode::Nearest),
            ("nearest_exact", InterpolateMode::NearestExact),
            ("bilinear_hp", InterpolateMode::Bilinear { coordinate_transform: CoordinateTransformMode::HalfPixel }),
            ("bilinear_ac", InterpolateMode::Bilinear { coordinate_transform: CoordinateTransformMode::AlignCorners }),
            ("bicubic_hp", InterpolateMode::Bicubic { coordinate_transform: CoordinateTransformMode::HalfPixel, a: -0.75 }),
            ("bicubic_ac", InterpolateMode::Bicubic { coordinate_transform: CoordinateTransformMode::AlignCorners, a: -0.5 }),
            ("lanczos3_hp", InterpolateMode::Lanczos3 { coordinate_transform: CoordinateTransformMode::HalfPixel }),
            ("lanczos3_ac", InterpolateMode::Lanczos3 { coordinate_transform: CoordinateTransformMode::AlignCorners }),
        ]
    };
    for &shape in &[[2usize, 3, 9, 7], [10, 3, 8, 8], [1, 1, 1, 1], [1, 4, 1, 6], [3, 2, 31, 2]] {
        let v = randv(&shape, 50);
        for layout in [Layout::Contig, Layout::PermLast, Layout::Window] {
            let x = |d: &Device| make(&v, shape, layout, d);
            for (mname, mode) in modes() {
                for size in [[17usize, 23], [5, 3], [1, 1], [9, 7], [2, 13]] {
                    let o = InterpolateOptions::new(mode.clone()).with_output_size(size);
                    p.case(format!("{mname} {shape:?}->{size:?} {layout:?}"), LOOSE, |d| {
                        out![module::interpolate(x(d), o.clone())]
                    });
                }
                for sf in [[1.5f32, 0.7], [2.0, 2.0], [0.5, 3.0]] {
                    let o = InterpolateOptions::new(mode.clone()).with_scale_factor(sf);
                    p.case(format!("{mname} {shape:?}x{sf:?} {layout:?}"), LOOSE, |d| {
                        out![module::interpolate(x(d), o.clone())]
                    });
                }
            }
        }
    }
    // grid_sample
    let xv = randv(&[2, 3, 6, 5], 51);
    let gv: Vec<f32> = randv(&[2, 4, 7, 2], 52).iter().map(|g| g * 1.3).collect();
    for mode in [GridSampleMode::Nearest, GridSampleMode::Bilinear] {
        for pm in [GridSamplePaddingMode::Zeros, GridSamplePaddingMode::Border, GridSamplePaddingMode::Reflection] {
            for ac in [false, true] {
                for layout in [Layout::Contig, Layout::PermLast] {
                    p.case(format!("grid_sample {mode:?} {pm:?} ac{ac} {layout:?}"), LOOSE, |d| {
                        let o = GridSampleOptions::new(mode).with_padding_mode(pm).with_align_corners(ac);
                        out![make(&xv, [2, 3, 6, 5], layout, d).grid_sample_2d(make(&gv, [2, 4, 7, 2], layout, d), o)]
                    });
                }
            }
        }
    }
    p.finish();
}

// ---------------------------------------------------------------- indexing

#[test]
fn probe_indexing() {
    let mut p = Probe::new("indexing");
    let v = distinctv(&SHAPE, 60);
    let mut r = Rng::new(61);
    for layout in LAYOUTS {
        let x = |d: &Device| make(&v, SHAPE, layout, d);
        for dim in 0..4 {
            let n = SHAPE[dim];
            // gather with an index tensor of a different size along dim
            let mut ishape = SHAPE;
            ishape[dim] = n + 2;
            let iv: Vec<f32> = (0..numel(&ishape)).map(|_| (r.next_u32() as usize % n) as f32).collect();
            let idx = |d: &Device, l: Layout| make(&iv, ishape, l, d).int();
            p.case(format!("gather{dim} {layout:?}"), EXACT, |d| out![x(d).gather(dim, idx(d, Layout::Contig))]);
            p.case(format!("gather{dim} {layout:?} idx-perm"), EXACT, |d| out![x(d).gather(dim, idx(d, Layout::PermLast))]);
            // scatter add with duplicates; values = same shape as idx
            let vv = randv(&ishape, 62 + dim as u64);
            for (oname, op) in [("assign1", IndexingUpdateOp::Add), ("add", IndexingUpdateOp::Add)] {
                if oname == "assign1" { continue; }
                p.case(format!("scatter_{oname}{dim} {layout:?}"), F32, |d| {
                    out![x(d).scatter(dim, idx(d, Layout::Contig), make(&vv, ishape, Layout::PermLast, d), op.clone())]
                });
            }
            let sel: Vec<i32> = vec![(n - 1) as i32, 0, (n / 2) as i32, 0, (n - 1) as i32];
            let si = |d: &Device| Tensor::<1, Int>::from_data(TensorData::new(sel.clone(), [5]), d);
            p.case(format!("select{dim} {layout:?}"), EXACT, |d| out![x(d).select(dim, si(d))]);
            let mut sshape = SHAPE;
            sshape[dim] = 5;
            let sv = randv(&sshape, 70 + dim as u64);
            p.case(format!("select_assign_add{dim} {layout:?}"), F32, |d| {
                out![x(d).select_assign(dim, si(d), make(&sv, sshape, Layout::Transposed, d), IndexingUpdateOp::Add)]
            });
            p.case(format!("flip{dim} {layout:?}"), EXACT, |d| out![x(d).flip([dim])]);
            p.case(format!("roll{dim} {layout:?}"), EXACT, |d| out![x(d).roll(&[-2i64], &[dim])]);
            p.case(format!("repeat_dim{dim} {layout:?}"), EXACT, |d| out![x(d).repeat_dim(dim, 3)]);
            p.case(format!("cat{dim} {layout:?}"), EXACT, |d| {
                out![Tensor::cat(vec![x(d), make(&v, SHAPE, Layout::Contig, d), x(d).flip([3])], dim)]
            });
            p.case(format!("narrow{dim} {layout:?}"), EXACT, |d| out![x(d).narrow(dim, 1, n - 1).exp()]);
        }
        p.case(format!("slice step {layout:?}"), EXACT, |d| {
            out![x(d).slice(s![.., 0..3;2, 4..0;-1, 1..7;3]).add_scalar(1.0)]
        });
        p.case(format!("slice_assign {layout:?}"), EXACT, |d| {
            out![x(d).slice_assign(s![.., 1..3, 0..5;2, 2..6], Tensor::ones([2, 2, 3, 4], d))]
        });
        p.case(format!("triu {layout:?}"), EXACT, |d| out![x(d).triu(1)]);
        p.case(format!("tril {layout:?}"), EXACT, |d| out![x(d).tril(-1)]);
        p.case(format!("mask_where {layout:?}"), EXACT, |d| {
            let m = make(&v, SHAPE, Layout::PermRev, d).greater_elem(0.0);
            out![x(d).mask_where(m, make(&v, SHAPE, Layout::Contig, d).neg())]
        });
        p.case(format!("mask_fill bcast {layout:?}"), EXACT, |d| {
            let m = make(&v[..7].to_vec(), [1, 1, 1, 7], Layout::Contig, d).greater_elem(0.0).expand(SHAPE);
            out![x(d).mask_fill(m, -100.0)]
        });
        for pm in [PadMode::Constant(2.5), PadMode::Reflect, PadMode::Edge] {
            p.case(format!("pad {pm:?} {layout:?}"), EXACT, |d| out![x(d).pad([(1, 2), (2, 1)], pm.clone())]);
        }
        p.case(format!("expand+add {layout:?}"), F32, |d| {
            out![x(d).add(make(&v[..21].to_vec(), [3, 1, 7], Layout::Contig, d).expand([2, 3, 5, 7]))]
        });
        p.case(format!("unfold {layout:?}"), EXACT, |d| out![x(d).unfold::<5, _>(3, 3, 2)]);
        p.case(format!("unfold4d {layout:?}"), EXACT, |d| {
            out![module::unfold4d(x(d), [3, 2], burn_tensor::ops::UnfoldOptions::new([2, 1], [1, 0], [1, 2]))]
        });
    }
    // Large scatter-add with heavy duplicates (atomics).
    let n = 100_000;
    let iv: Vec<i32> = (0..n).map(|i| (i * 7 % 13) as i32).collect();
    let vv = randv(&[n], 80);
    p.case("scatter_add dup 100k->13", LOOSE, |d| {
        out![Tensor::<1>::zeros([13], d).scatter(
            0,
            Tensor::<1, Int>::from_data(TensorData::new(iv.clone(), [n]), d),
            Tensor::from_data(TensorData::new(vv.clone(), [n]), d),
            IndexingUpdateOp::Add
        )]
    });
    p.case("select_assign dup 100k->13", LOOSE, |d| {
        out![Tensor::<1>::zeros([13], d).select_assign(
            0,
            Tensor::<1, Int>::from_data(TensorData::new(iv.clone(), [n]), d),
            Tensor::from_data(TensorData::new(vv.clone(), [n]), d),
            IndexingUpdateOp::Add
        )]
    });
    p.finish();
}

// ---------------------------------------------------------------- elementwise math

#[test]
fn probe_elementwise() {
    let mut p = Probe::new("elementwise");
    let special = vec![
        0.0f32, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, 1e-7, -1e-7, 1e-30, 88.0, 89.0, -88.0,
        -104.0, 1e10, -1e10, 3e38, -3e38, 1e20, 15.0, -15.0, 20.0, -20.0, 50.0, -50.0, 100.0,
        -100.0, 0.1, -0.1, 0.999, 1.0, -1.0, 7.0, -7.0, f32::INFINITY, f32::NEG_INFINITY,
        f32::NAN, 1e-40, 65504.0, 70000.0, -0.75, 0.33,
    ];
    let n = special.len();
    let x = |d: &Device| Tensor::<1>::from_data(TensorData::new(special.clone(), [n]), d);
    macro_rules! un {
        ($name:expr, $tol:expr, $f:expr) => {
            p.case($name, $tol, |d| out![$f(x(d))]);
        };
    }
    un!("exp", F32, |t: Tensor<1>| t.exp());
    un!("log", F32, |t: Tensor<1>| t.log());
    un!("log1p", F32, |t: Tensor<1>| t.log1p());
    un!("sqrt", F32, |t: Tensor<1>| t.sqrt());
    un!("tanh", F32, |t: Tensor<1>| t.tanh());
    un!("sinh", F32, |t: Tensor<1>| t.sinh());
    un!("cosh", F32, |t: Tensor<1>| t.cosh());
    un!("sin", LOOSE, |t: Tensor<1>| t.clamp(-1e4, 1e4).sin());
    un!("cos", LOOSE, |t: Tensor<1>| t.clamp(-1e4, 1e4).cos());
    un!("tan", LOOSE, |t: Tensor<1>| t.clamp(-1.5, 1.5).tan());
    un!("asin", F32, |t: Tensor<1>| t.asin());
    un!("acos", F32, |t: Tensor<1>| t.acos());
    un!("atan", F32, |t: Tensor<1>| t.atan());
    un!("asinh", F32, |t: Tensor<1>| t.asinh());
    un!("acosh", F32, |t: Tensor<1>| t.acosh());
    un!("atanh", F32, |t: Tensor<1>| t.atanh());
    un!("erf", F32, |t: Tensor<1>| t.erf());
    un!("recip", F32, |t: Tensor<1>| t.recip());
    un!("round", EXACT, |t: Tensor<1>| t.round());
    un!("floor", EXACT, |t: Tensor<1>| t.floor());
    un!("ceil", EXACT, |t: Tensor<1>| t.ceil());
    un!("trunc", EXACT, |t: Tensor<1>| t.trunc());
    un!("sign", EXACT, |t: Tensor<1>| t.sign());
    un!("abs", EXACT, |t: Tensor<1>| t.abs());
    un!("square", F32, |t: Tensor<1>| t.square());
    un!("powf_scalar 3", F32, |t: Tensor<1>| t.powf_scalar(3.0));
    un!("powf_scalar 2", F32, |t: Tensor<1>| t.powf_scalar(2.0));
    un!("powf_scalar 0.5", F32, |t: Tensor<1>| t.powf_scalar(0.5));
    un!("powf_scalar -1", F32, |t: Tensor<1>| t.powf_scalar(-1.0));
    un!("powf_scalar 0", F32, |t: Tensor<1>| t.powf_scalar(0.0));
    un!("powf_scalar 2.5", F32, |t: Tensor<1>| t.powf_scalar(2.5));
    un!("powi_scalar 3", F32, |t: Tensor<1>| t.powi_scalar(3));
    un!("powi_scalar -2", F32, |t: Tensor<1>| t.powi_scalar(-2));
    un!("remainder_scalar 2", F32, |t: Tensor<1>| t.remainder_scalar(2.0));
    un!("remainder_scalar -2", F32, |t: Tensor<1>| t.remainder_scalar(-2.0));
    un!("fmod_scalar -2", F32, |t: Tensor<1>| t.fmod_scalar(-2.0));
    un!("fmod_scalar 0.3", F32, |t: Tensor<1>| t.fmod_scalar(0.3));
    un!("div_scalar 0", EXACT, |t: Tensor<1>| t.div_scalar(0.0));
    un!("clamp", EXACT, |t: Tensor<1>| t.clamp(-1.0, 2.0));
    un!("round_to 2", F32, |t: Tensor<1>| t.clamp(-1e6, 1e6).round_to(2));
    un!("is_nan", EXACT, |t: Tensor<1>| t.is_nan());
    un!("is_inf", EXACT, |t: Tensor<1>| t.is_inf());
    un!("is_finite", EXACT, |t: Tensor<1>| t.is_finite());
    un!("sigmoid", F32, |t: Tensor<1>| act::sigmoid(t));
    un!("log_sigmoid", F32, |t: Tensor<1>| act::log_sigmoid(t));
    un!("softplus", F32, |t: Tensor<1>| act::softplus(t, 1.0));
    un!("silu", F32, |t: Tensor<1>| act::silu(t));
    un!("gelu", F32, |t: Tensor<1>| act::gelu(t));
    un!("gelu_approximate", F32, |t: Tensor<1>| act::gelu_approximate(t));
    un!("mish", F32, |t: Tensor<1>| act::mish(t));
    un!("elu", F32, |t: Tensor<1>| act::elu(t, 1.0));
    un!("selu", F32, |t: Tensor<1>| act::selu(t));
    un!("celu", F32, |t: Tensor<1>| act::celu(t, 0.5));
    un!("hard_sigmoid", F32, |t: Tensor<1>| act::hard_sigmoid(t, 0.2, 0.5));
    un!("hard_swish", F32, |t: Tensor<1>| act::hard_swish(t));
    un!("softsign", F32, |t: Tensor<1>| act::softsign(t));
    un!("relu6", EXACT, |t: Tensor<1>| act::relu6(t));
    un!("leaky_relu", F32, |t: Tensor<1>| act::leaky_relu(t, 0.01));
    un!("tanhshrink", F32, |t: Tensor<1>| act::tanhshrink(t));
    un!("int()", EXACT, |t: Tensor<1>| t.clamp(-1e9, 1e9).int());
    un!("cast f16", EXACT, |t: Tensor<1>| t.cast(FloatDType::F16).cast(FloatDType::F32));
    un!("cast bf16", EXACT, |t: Tensor<1>| t.cast(FloatDType::BF16).cast(FloatDType::F32));
    // Rounding to f16/bf16 of values between representable numbers.
    let mids: Vec<f32> = (0..200).map(|i| 1.0 + (i as f32) * (1.0 / 4096.0)).chain((0..200).map(|i| 1.0 + (i as f32) * (1.0 / 512.0))).collect();
    let m = mids.len();
    let xm = |d: &Device| Tensor::<1>::from_data(TensorData::new(mids.clone(), [m]), d);
    p.case("cast f16 rne", EXACT, |d| out![xm(d).cast(FloatDType::F16).cast(FloatDType::F32)]);
    p.case("cast bf16 rne", EXACT, |d| out![xm(d).cast(FloatDType::BF16).cast(FloatDType::F32)]);
    // Binary on special pairs.
    let y: Vec<f32> = special.iter().rev().cloned().collect();
    let yt = |d: &Device| Tensor::<1>::from_data(TensorData::new(y.clone(), [n]), d);
    p.case("atan2", F32, |d| out![x(d).atan2(yt(d))]);
    p.case("hypot", F32, |d| out![x(d).hypot(yt(d))]);
    p.case("powf", F32, |d| out![x(d).clamp(-3.0, 3.0).powf(yt(d).clamp(-3.0, 3.0).round())]);
    p.case("powf frac", F32, |d| out![x(d).clamp(-3.0, 3.0).powf(yt(d).clamp(-3.0, 3.0))]);
    p.case("remainder", F32, |d| out![x(d).clamp(-1e6, 1e6).remainder(yt(d).clamp(-1e6, 1e6))]);
    p.case("fmod", F32, |d| out![x(d).clamp(-1e6, 1e6).fmod(yt(d).clamp(-1e6, 1e6))]);
    p.case("div", F32, |d| out![x(d).div(yt(d))]);
    p.case("max_pair", EXACT, |d| out![x(d).max_pair(yt(d))]);
    p.case("min_pair", EXACT, |d| out![x(d).min_pair(yt(d))]);
    p.case("greater", EXACT, |d| out![x(d).greater(yt(d))]);
    p.case("equal", EXACT, |d| out![x(d).equal(yt(d))]);
    p.case("not_equal", EXACT, |d| out![x(d).not_equal(yt(d))]);
    p.case("lower_equal", EXACT, |d| out![x(d).lower_equal(yt(d))]);
    p.case("is_close", EXACT, |d| out![x(d).is_close(yt(d), None, None)]);
    // f16 variants of unaries
    let fv: Vec<f32> = special.iter().map(|v| v.clamp(-60000.0, 60000.0)).collect();
    let xf = |d: &Device| Tensor::<1>::from_data(TensorData::new(fv.clone(), [n]), d).cast(FloatDType::F16);
    for (name, f) in [
        ("exp", (|t: Tensor<1>| t.exp()) as fn(Tensor<1>) -> Tensor<1>),
        ("tanh", |t| t.tanh()),
        ("sigmoid", |t| act::sigmoid(t)),
        ("softplus", |t| act::softplus(t, 1.0)),
        ("log_sigmoid", |t| act::log_sigmoid(t)),
        ("silu", |t| act::silu(t)),
        ("erf", |t| t.erf()),
        ("sqrt", |t| t.sqrt()),
        ("log1p", |t| t.log1p()),
        ("round", |t| t.round()),
        ("recip", |t| t.recip()),
        ("powf_scalar 2", |t| t.powf_scalar(2.0)),
        ("cosh", |t| t.cosh()),
        ("sinh", |t| t.sinh()),
        ("hypot", |t| t.clone().hypot(t)),
        ("atanh", |t| t.atanh()),
        ("acosh", |t| t.acosh()),
        ("asinh", |t| t.asinh()),
        ("mish", |t| act::mish(t)),
        ("elu", |t| act::elu(t, 1.0)),
    ] {
        p.case(format!("f16 {name}"), HALF, |d| out![f(xf(d)).cast(FloatDType::F32)]);
    }
    p.finish();
}

// ---------------------------------------------------------------- int ops

#[test]
fn probe_int() {
    let mut p = Probe::new("int");
    let a: Vec<i32> = vec![-7, 7, -7, 7, 0, 5, -5, 13, -13, 1, -1, 100, -100, i32::MAX, i32::MIN + 1, 6, -6, 9];
    let b: Vec<i32> = vec![2, -2, -2, 2, 3, 5, 3, -4, 4, -1, 1, 7, -7, 2, 3, -3, 4, -9];
    let n = a.len();
    let ta = |d: &Device| Tensor::<1, Int>::from_data(TensorData::new(a.clone(), [n]), d);
    let tb = |d: &Device| Tensor::<1, Int>::from_data(TensorData::new(b.clone(), [n]), d);
    p.case("int div", EXACT, |d| out![ta(d).div(tb(d))]);
    p.case("int remainder", EXACT, |d| out![ta(d).remainder(tb(d))]);
    p.case("int div_scalar -3", EXACT, |d| out![ta(d).div_scalar(-3)]);
    p.case("int remainder_scalar -3", EXACT, |d| out![ta(d).remainder_scalar(-3)]);
    p.case("int remainder_scalar 3", EXACT, |d| out![ta(d).remainder_scalar(3)]);
    p.case("int abs", EXACT, |d| out![ta(d).abs()]);
    p.case("int sign", EXACT, |d| out![ta(d).sign()]);
    p.case("int neg", EXACT, |d| out![ta(d).neg()]);
    p.case("int powi_scalar 2", EXACT, |d| out![ta(d).clamp(-1000, 1000).powi_scalar(2)]);
    p.case("int powi", EXACT, |d| out![ta(d).clamp(-10, 10).powi(tb(d).abs().clamp(0, 5))]);
    p.case("int clamp", EXACT, |d| out![ta(d).clamp(-6, 8)]);
    p.case("int max_pair", EXACT, |d| out![ta(d).max_pair(tb(d))]);
    p.case("int mul wrap", EXACT, |d| out![ta(d).mul(tb(d))]);
    p.case("int add wrap", EXACT, |d| out![ta(d).add(tb(d).mul_scalar(1_000_000_000))]);
    p.case("int bitwise_and", EXACT, |d| out![ta(d).bitwise_and(tb(d))]);
    p.case("int bitwise_xor", EXACT, |d| out![ta(d).bitwise_xor(tb(d))]);
    p.case("int bitwise_not", EXACT, |d| out![ta(d).bitwise_not()]);
    p.case("int shl", EXACT, |d| out![ta(d).bitwise_left_shift(tb(d).abs().clamp(0, 30))]);
    p.case("int shr", EXACT, |d| out![ta(d).bitwise_right_shift(tb(d).abs().clamp(0, 30))]);
    p.case("int float()", EXACT, |d| out![ta(d).float()]);
    p.case("int cast i64", EXACT, |d| out![ta(d).cast(IntDType::I64)]);
    p.case("int cast i8", EXACT, |d| out![ta(d).cast(IntDType::I8)]);
    p.case("int cast u8", EXACT, |d| out![ta(d).cast(IntDType::U8)]);
    p.case("int cast i16", EXACT, |d| out![ta(d).cast(IntDType::I16)]);
    p.case("int to f16", EXACT, |d| out![ta(d).clamp(-60000, 60000).float().cast(FloatDType::F16).cast(FloatDType::F32)]);
    p.case("int sum wrap", EXACT, |d| out![ta(d).sum()]);
    p.case("int mean", EXACT, |d| out![ta(d).clamp(-1000, 1000).mean()]);
    p.case("int prod", EXACT, |d| out![tb(d).prod()]);
    p.case("int cumsum", EXACT, |d| out![ta(d).clamp(-1000, 1000).cumsum(0)]);
    p.case("int cumprod", EXACT, |d| out![tb(d).cumprod(0)]);
    p.case("int sort", EXACT, |d| out![ta(d).sort(0)]);
    p.case("int argsort", EXACT, |d| out![ta(d).argsort(0)]);
    p.case("int topk", EXACT, |d| out![ta(d).topk(3, 0)]);
    p.case("int argmax", EXACT, |d| out![ta(d).argmax(0)]);
    p.case("int argmin", EXACT, |d| out![ta(d).argmin(0)]);
    p.case("int max", EXACT, |d| out![ta(d).max()]);
    p.case("int min", EXACT, |d| out![ta(d).min()]);
    p.case("int cummax", EXACT, |d| out![ta(d).cummax(0)]);
    p.case("int equal", EXACT, |d| out![ta(d).equal(tb(d))]);
    p.case("int greater_elem", EXACT, |d| out![ta(d).greater_elem(5)]);
    p.case("int div 0", EXACT, |d| out![ta(d).div(tb(d).mul_scalar(0)).mul_scalar(0)]);
    // Int on views
    let v = distinctv(&SHAPE, 90);
    for layout in LAYOUTS {
        let x = |d: &Device| make(&v, SHAPE, layout, d).mul_scalar(50.0).int();
        for dim in 0..4 {
            p.case(format!("int sum_dim{dim} {layout:?}"), EXACT, |d| out![x(d).sum_dim(dim)]);
            p.case(format!("int prod_dim{dim} {layout:?}"), EXACT, |d| out![x(d).clamp(-3, 3).prod_dim(dim)]);
            p.case(format!("int max_dim{dim} {layout:?}"), EXACT, |d| out![x(d).max_dim(dim)]);
            p.case(format!("int argmin{dim} {layout:?}"), EXACT, |d| out![x(d).argmin(dim)]);
            p.case(format!("int cumsum{dim} {layout:?}"), EXACT, |d| out![x(d).cumsum(dim)]);
            p.case(format!("int mean_dim{dim} {layout:?}"), EXACT, |d| out![x(d).mean_dim(dim)]);
            p.case(format!("int sort{dim} {layout:?}"), EXACT, |d| out![x(d).sort(dim)]);
            p.case(format!("int remainder{dim} {layout:?}"), EXACT, |d| out![x(d).remainder_scalar(-7)]);
        }
    }
    p.finish();
}

// ---------------------------------------------------------------- half-precision reductions

#[test]
fn probe_half_reduce() {
    let mut p = Probe::new("half_reduce");
    let v = randv(&SHAPE, 100);
    for dt in [FloatDType::F16, FloatDType::BF16] {
        for layout in [Layout::Contig, Layout::PermLast, Layout::Transposed] {
            // Device runs in half precision; reference in f32 on the half-rounded input.
            let x = |d: &Device| make(&v, SHAPE, layout, d).cast(dt);
            let is_ref = |d: &Device| d.as_dispatch() == refd().as_dispatch();
            for dim in 0..4 {
                macro_rules! h {
                    ($name:expr, $tol:expr, $f:expr) => {
                        p.case(format!("{:?} {} dim{dim} {layout:?}", dt, $name), $tol, |d| {
                            let t = x(d);
                            if is_ref(d) {
                                out![$f(t.cast(FloatDType::F32), dim)]
                            } else {
                                out![$f(t, dim).cast(FloatDType::F32)]
                            }
                        });
                    };
                }
                h!("sum_dim", HALF, |t: Tensor<4>, k| t.sum_dim(k));
                h!("mean_dim", HALF, |t: Tensor<4>, k| t.mean_dim(k));
                h!("max_dim", HALF, |t: Tensor<4>, k| t.max_dim(k));
                h!("prod_dim", HALF, |t: Tensor<4>, k| t.add_scalar(1.5).prod_dim(k));
                h!("softmax", HALF, |t: Tensor<4>, k| act::softmax(t, k));
                h!("log_softmax", HALF, |t: Tensor<4>, k| act::log_softmax(t, k));
                h!("cumprod", HALF, |t: Tensor<4>, k| t.add_scalar(1.5).cumprod(k));
                h!("sort", HALF, |t: Tensor<4>, k| t.sort(k));
                p.case(format!("{dt:?} argmax dim{dim} {layout:?}"), EXACT, |d| {
                    let t = x(d);
                    if is_ref(d) { out![t.cast(FloatDType::F32).argmax(dim)] } else { out![t.argmax(dim)] }
                });
            }
            // Long sums: accumulation precision. 4096 ones etc.
            for n in [2048usize, 4096, 10000] {
                p.case(format!("{dt:?} ones sum {n} {layout:?}"), Tol { atol: 0.5, rtol: 1e-3 }, |d| {
                    let t = Tensor::<2>::ones([2, n], d).cast(dt);
                    if is_ref(d) { out![t.cast(FloatDType::F32).sum_dim(1)] } else { out![t.sum_dim(1).cast(FloatDType::F32)] }
                });
                p.case(format!("{dt:?} ones mean {n}"), Tol { atol: 1e-2, rtol: 1e-2 }, |d| {
                    let t = Tensor::<2>::ones([2, n], d).cast(dt);
                    if is_ref(d) { out![t.cast(FloatDType::F32).mean_dim(1)] } else { out![t.mean_dim(1).cast(FloatDType::F32)] }
                });
                p.case(format!("{dt:?} ones sum all {n}"), Tol { atol: 0.5, rtol: 1e-3 }, |d| {
                    let t = Tensor::<2>::ones([2, n], d).cast(dt);
                    if is_ref(d) { out![t.cast(FloatDType::F32).sum()] } else { out![t.sum().cast(FloatDType::F32)] }
                });
                p.case(format!("{dt:?} ones avgpool {n}"), Tol { atol: 1e-2, rtol: 1e-2 }, |d| {
                    let t = Tensor::<3>::ones([1, 1, n], d).cast(dt);
                    let o = AvgPoolOptions::new([n]);
                    if is_ref(d) { out![module::avg_pool1d(t.cast(FloatDType::F32), o)] } else { out![module::avg_pool1d(t, o).cast(FloatDType::F32)] }
                });
                p.case(format!("{dt:?} ones adaptive_avg {n}"), Tol { atol: 1e-2, rtol: 1e-2 }, |d| {
                    let t = Tensor::<3>::ones([1, 1, n], d).cast(dt);
                    if is_ref(d) { out![module::adaptive_avg_pool1d(t.cast(FloatDType::F32), 1)] } else { out![module::adaptive_avg_pool1d(t, 1).cast(FloatDType::F32)] }
                });
                p.case(format!("{dt:?} ones softmax {n}"), Tol { atol: 1e-5, rtol: 2e-2 }, |d| {
                    let t = Tensor::<2>::ones([2, n], d).cast(dt);
                    if is_ref(d) { out![act::softmax(t.cast(FloatDType::F32), 1)] } else { out![act::softmax(t, 1).cast(FloatDType::F32)] }
                });
                p.case(format!("{dt:?} ones matmul k{n}"), Tol { atol: 0.5, rtol: 1e-2 }, |d| {
                    let t = Tensor::<2>::ones([2, n], d).cast(dt);
                    let r = Tensor::<2>::ones([n, 3], d).cast(dt);
                    if is_ref(d) { out![t.cast(FloatDType::F32).matmul(r.cast(FloatDType::F32))] } else { out![t.matmul(r).cast(FloatDType::F32)] }
                });
            }
            // Large values: f16 overflow in intermediate (variance / sum of squares).
            p.case(format!("{dt:?} big softmax {layout:?}"), HALF, |d| {
                let t = make(&v, SHAPE, layout, d).mul_scalar(200.0).cast(dt);
                if is_ref(d) { out![act::softmax(t.cast(FloatDType::F32), 3)] } else { out![act::softmax(t, 3).cast(FloatDType::F32)] }
            });
            p.case(format!("{dt:?} big mean {layout:?}"), Tol { atol: 5.0, rtol: 2e-2 }, |d| {
                let t = make(&v, SHAPE, layout, d).mul_scalar(30000.0).add_scalar(30000.0).cast(dt);
                if is_ref(d) { out![t.cast(FloatDType::F32).mean_dim(3)] } else { out![t.mean_dim(3).cast(FloatDType::F32)] }
            });
        }
    }
    p.finish();
}

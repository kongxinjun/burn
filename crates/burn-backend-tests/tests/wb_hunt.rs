//! Bug-hunting probe (fork-only harness, never upstream): every case runs on
//! `Device::default()` (BURN_DEVICE) and on Flex, and compares the results. Families target code
//! that changed after 03f4ddaed: fused concat, fused NaN predicates, nan_max/nan_min, fusion
//! search at syncs (random programs), quantized matmul. Prints `FAIL` lines and one `SUMMARY`
//! per family; `WB_VERBOSE=1` also prints `ok` lines.
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
            Ok(()) => if verbose() { println!("ok\t{}\t{name}", self.group) },
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
fn verbose() -> bool {
    std::env::var("WB_VERBOSE").is_ok_and(|v| v == "1")
}

fn is_ref(d: &Device) -> bool {
    d.as_dispatch() == refd().as_dispatch()
}

/// Number of cases a family runs, scaled by `WB_SCALE` (default 1).
fn scale(n: usize) -> usize {
    let s: f64 = std::env::var("WB_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    ((n as f64) * s).max(1.0) as usize
}

// ---------------------------------------------------------------- fused cat

/// What runs around the `cat` in the same fusion block.
#[derive(Clone, Copy, Debug)]
enum CatEpi {
    Plain,
    Affine,
    AddOther,
    AddOtherPermuted,
    SwapOut,
    ReduceAfter,
    BiasLast,
    Producers,
    Flatten,
    SameTwice,
    NarrowAfter,
    CatOfCat,
}

const CAT_EPIS: [CatEpi; 12] = [
    CatEpi::Plain,
    CatEpi::Affine,
    CatEpi::AddOther,
    CatEpi::AddOtherPermuted,
    CatEpi::SwapOut,
    CatEpi::ReduceAfter,
    CatEpi::BiasLast,
    CatEpi::Producers,
    CatEpi::Flatten,
    CatEpi::SameTwice,
    CatEpi::NarrowAfter,
    CatEpi::CatOfCat,
];

const CAT_SIZES: &[&[usize]] = &[
    &[1, 1],
    &[3, 5],
    &[4, 4],
    &[8, 8],
    &[2, 3, 7],
    &[0, 3],
    &[5, 0, 2],
    &[1, 2, 3, 4, 5, 6, 7, 8, 9],
    &[4, 1, 4, 1, 4],
    &[1; 17],
    &[2; 12],
];

const CAT_MIXES: &[&[Layout]] = &[
    &[Layout::Contig],
    &[Layout::Contig, Layout::PermRev],
    &[Layout::Transposed, Layout::Strided, Layout::Window],
    &[Layout::Flipped, Layout::PermLast],
];

fn cat_inputs<const D: usize>(
    base: [usize; D],
    dim: usize,
    sizes: &[usize],
    layouts: &[Layout],
    seed: u64,
    d: &Device,
) -> Vec<Tensor<D>> {
    sizes
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut sh = base;
            sh[dim] = s;
            let v = randv(&sh, seed + i as u64);
            let layout = if numel(&sh) == 0 { Layout::Contig } else { layouts[i % layouts.len()] };
            make(&v, sh, layout, d)
        })
        .collect()
}

fn cat_epi<const D: usize>(xs: Vec<Tensor<D>>, dim: usize, epi: CatEpi, seed: u64, d: &Device) -> Vec<TensorData> {
    let other = |sh: [usize; D], layout: Layout| make(&randv(&sh, seed + 999), sh, layout, d);
    match epi {
        CatEpi::Plain => out![Tensor::cat(xs, dim)],
        CatEpi::Affine => out![Tensor::cat(xs, dim).mul_scalar(2.0).add_scalar(1.0)],
        CatEpi::AddOther => {
            let c = Tensor::cat(xs, dim);
            let o = other(c.dims(), Layout::Contig);
            out![c + o]
        }
        CatEpi::AddOtherPermuted => {
            let c = Tensor::cat(xs, dim);
            let sh = c.dims();
            let o = if numel(&sh) == 0 { other(sh, Layout::Contig) } else { other(sh, Layout::PermRev) };
            out![c + o]
        }
        CatEpi::SwapOut => out![Tensor::cat(xs, dim).swap_dims(0, D - 1).mul_scalar(3.0)],
        CatEpi::ReduceAfter => {
            let c = Tensor::cat(xs, dim).exp();
            let r = if D > 1 { (dim + 1) % D } else { 0 };
            out![c.sum_dim(r)]
        }
        CatEpi::BiasLast => {
            let c = Tensor::cat(xs, dim);
            let mut sh = [1; D];
            sh[D - 1] = c.dims()[D - 1];
            let b = other(sh, Layout::Contig);
            out![c + b]
        }
        CatEpi::Producers => {
            let xs = xs.into_iter().map(|x| x.mul_scalar(2.0).tanh()).collect();
            out![Tensor::cat(xs, dim).add_scalar(0.5)]
        }
        CatEpi::Flatten => {
            let c = Tensor::cat(xs, dim);
            let n = c.shape().num_elements();
            out![c.reshape([n]).add_scalar(1.0)]
        }
        CatEpi::SameTwice => {
            let a = xs[0].clone();
            let mut all = vec![a.clone()];
            all.extend(xs);
            all.push(a);
            out![Tensor::cat(all, dim).mul_scalar(2.0)]
        }
        CatEpi::NarrowAfter => {
            let c = Tensor::cat(xs, dim);
            let n = c.dims()[dim];
            if n == 0 {
                out![c]
            } else {
                out![c.narrow(dim, n / 3, n - n / 3).mul_scalar(2.0)]
            }
        }
        CatEpi::CatOfCat => {
            let c = Tensor::cat(xs, dim);
            out![Tensor::cat(vec![c.clone().neg(), c], dim).add_scalar(1.0)]
        }
    }
}

fn cat_family<const D: usize>(p: &mut Probe, bases: &[([usize; D], usize)]) {
    let mut rng = Rng::new(7);
    for &(base, dim) in bases {
        for (si, sizes) in CAT_SIZES.iter().enumerate() {
            for epi in CAT_EPIS {
                let mixes = [0, 1 + (rng.next_u32() as usize) % (CAT_MIXES.len() - 1)];
                for mi in mixes {
                    let mix = CAT_MIXES[mi];
                    let seed = (si * 100 + mi) as u64;
                    p.case(format!("r{D} {base:?} dim{dim} sizes{sizes:?} {epi:?} mix{mi}"), F32, |d| {
                        cat_epi(cat_inputs(base, dim, sizes, mix, seed, d), dim, epi, seed, d)
                    });
                }
            }
            // Other dtypes, a few shapes of the same op.
            let mix = CAT_MIXES[1 + si % (CAT_MIXES.len() - 1)];
            let seed = 5000 + si as u64;
            for dt in [FloatDType::F16, FloatDType::BF16] {
                p.case(format!("r{D} {base:?} dim{dim} sizes{sizes:?} {dt:?} affine"), HALF, |d| {
                    let xs: Vec<Tensor<D>> = cat_inputs(base, dim, sizes, mix, seed, d).into_iter().map(|x| x.cast(dt)).collect();
                    out![Tensor::cat(xs, dim).mul_scalar(2.0).add_scalar(1.0).cast(FloatDType::F32)]
                });
            }
            p.case(format!("r{D} {base:?} dim{dim} sizes{sizes:?} int affine"), EXACT, |d| {
                let xs: Vec<Tensor<D, Int>> = cat_inputs(base, dim, sizes, mix, seed, d).into_iter().map(|x| x.mul_scalar(50.0).int()).collect();
                out![Tensor::cat(xs, dim).mul_scalar(3).add_scalar(1)]
            });
            p.case(format!("r{D} {base:?} dim{dim} sizes{sizes:?} bool not"), EXACT, |d| {
                let xs: Vec<Tensor<D, Bool>> = cat_inputs(base, dim, sizes, mix, seed, d).into_iter().map(|x| x.greater_elem(0.0)).collect();
                out![Tensor::cat(xs, dim).bool_not()]
            });
            p.case(format!("r{D} {base:?} dim{dim} sizes{sizes:?} int sum_dim"), EXACT, |d| {
                let xs: Vec<Tensor<D, Int>> = cat_inputs(base, dim, sizes, mix, seed, d).into_iter().map(|x| x.mul_scalar(50.0).int()).collect();
                out![Tensor::cat(xs, dim).add_scalar(2).sum_dim(D - 1)]
            });
        }
    }
}

#[test]
fn hunt_cat_r1_r2() {
    let mut p = Probe::new("fused_cat_r12");
    cat_family::<1>(&mut p, &[([0], 0)]);
    cat_family::<2>(&mut p, &[([3, 0], 1), ([0, 8], 0), ([4, 0], 1)]);
    p.finish();
}

#[test]
fn hunt_cat_r3() {
    let mut p = Probe::new("fused_cat_r3");
    cat_family::<3>(&mut p, &[([2, 0, 5], 1), ([2, 4, 0], 2), ([0, 3, 8], 0)]);
    p.finish();
}

#[test]
fn hunt_cat_r4() {
    let mut p = Probe::new("fused_cat_r4");
    cat_family::<4>(&mut p, &[([0, 3, 4, 8], 0), ([2, 0, 4, 8], 1), ([2, 3, 0, 8], 2), ([2, 3, 4, 0], 3)]);
    p.finish();
}

// ---------------------------------------------------------------- NaN predicates and nan reductions

/// Values in [-1, 1) with NaN at a `nan_rate` fraction and a few infinities.
fn nanv(shape: &[usize], seed: u64, nan_rate: f32) -> Vec<f32> {
    let mut r = Rng::new(seed);
    (0..numel(shape))
        .map(|_| {
            let u = (r.f() + 1.0) / 2.0;
            if u < nan_rate {
                f32::NAN
            } else if u > 0.97 {
                if r.f() > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY }
            } else {
                r.f()
            }
        })
        .collect()
}

#[test]
fn hunt_nan() {
    let mut p = Probe::new("nan");
    let shape = SHAPE;
    let vs = [nanv(&shape, 11, 0.2), nanv(&shape, 12, 0.85)];
    let w = nanv(&shape, 13, 0.3);
    let bshape = [1, 3, 1, 7];
    let bv = nanv(&bshape, 14, 0.3);
    for (vi, v) in vs.iter().enumerate() {
        for layout in LAYOUTS {
            for dt in [FloatDType::F32, FloatDType::F16, FloatDType::BF16] {
                // On the reference a half input is rounded, then computed in f32.
                let x = |d: &Device| {
                    let t = make(v, shape, layout, d).cast(dt);
                    if is_ref(d) { t.cast(FloatDType::F32) } else { t }
                };
                let y = |d: &Device| {
                    let t = make(&w, shape, Layout::Contig, d).cast(dt);
                    if is_ref(d) { t.cast(FloatDType::F32) } else { t }
                };
                let b = |d: &Device| {
                    let t = make(&bv, bshape, Layout::Contig, d).cast(dt);
                    if is_ref(d) { t.cast(FloatDType::F32) } else { t }
                };
                let f = |t: Tensor<4>| t.cast(FloatDType::F32);
                let tag = format!("v{vi} {dt:?} {layout:?}");
                p.case(format!("is_nan {tag}"), EXACT, |d| out![x(d).is_nan()]);
                p.case(format!("is_finite {tag}"), EXACT, |d| out![x(d).is_finite()]);
                p.case(format!("is_inf {tag}"), EXACT, |d| out![x(d).is_inf()]);
                p.case(format!("nan_or {tag}"), EXACT, |d| out![x(d).is_nan().bool_or(y(d).is_nan())]);
                p.case(format!("nan_or_gt {tag}"), EXACT, |d| out![x(d).is_nan().bool_or(x(d).greater_elem(0.5))]);
                p.case(format!("scaled is_nan {tag}"), EXACT, |d| out![x(d).mul_scalar(2.0).add_scalar(1.0).is_nan()]);
                p.case(format!("exp is_nan {tag}"), EXACT, |d| out![x(d).exp().is_nan()]);
                p.case(format!("inf-inf is_nan {tag}"), EXACT, |d| out![(x(d) - y(d)).is_nan()]);
                p.case(format!("nan count {tag}"), EXACT, |d| out![x(d).is_nan().int().sum_dim(3)]);
                p.case(format!("mask_fill nan {tag}"), EXACT, |d| {
                    let t = x(d);
                    out![f(t.clone().mask_fill(t.is_nan(), 0.0))]
                });
                p.case(format!("max_pair {tag}"), EXACT, |d| out![f(x(d).max_pair(y(d)))]);
                p.case(format!("min_pair {tag}"), EXACT, |d| out![f(x(d).min_pair(y(d)))]);
                p.case(format!("max_pair bcast {tag}"), EXACT, |d| out![f(x(d).max_pair(b(d)))]);
                p.case(format!("min_pair bcast rev {tag}"), EXACT, |d| out![f(b(d).min_pair(x(d)))]);
                p.case(format!("max_pair chain {tag}"), HALF, |d| out![f(x(d).mul_scalar(0.5).max_pair(y(d).add_scalar(0.25)).exp())]);
                p.case(format!("nan_max {tag}"), EXACT, |d| out![x(d).nan_max().cast(FloatDType::F32)]);
                p.case(format!("nan_min {tag}"), EXACT, |d| out![x(d).nan_min().cast(FloatDType::F32)]);
                for dim in 0..4 {
                    p.case(format!("nan_max_dim{dim} {tag}"), EXACT, |d| out![f(x(d).nan_max_dim(dim))]);
                    p.case(format!("nan_min_dim{dim} {tag}"), EXACT, |d| out![f(x(d).nan_min_dim(dim))]);
                    p.case(format!("max_dim{dim} {tag}"), EXACT, |d| out![f(x(d).max_dim(dim))]);
                    p.case(format!("min_dim{dim} {tag}"), EXACT, |d| out![f(x(d).min_dim(dim))]);
                    p.case(format!("nan_max_dim{dim} then add {tag}"), HALF, |d| out![f(x(d).nan_max_dim(dim).add_scalar(1.0).exp())]);
                }
            }
        }
    }
    // Int pairwise extrema (no NaN mask applies).
    for layout in LAYOUTS {
        let x = |d: &Device| make(&vs[0], shape, layout, d).mask_fill(make(&vs[0], shape, layout, d).is_nan(), 0.0).clamp(-1.0, 1.0).mul_scalar(100.0).int();
        let y = |d: &Device| make(&w, shape, Layout::Contig, d).mask_fill(make(&w, shape, Layout::Contig, d).is_nan(), 0.0).clamp(-1.0, 1.0).mul_scalar(100.0).int();
        p.case(format!("int max_pair {layout:?}"), EXACT, |d| out![x(d).max_pair(y(d))]);
        p.case(format!("int min_pair {layout:?}"), EXACT, |d| out![x(d).min_pair(y(d))]);
    }
    p.finish();
}

// ---------------------------------------------------------------- random programs

/// A random program over rank-3 float tensors: element-wise, broadcast, layout, reshape, cat,
/// reduce, matmul, scan and softmax ops, with syncs (`into_data`) at random points so the fusion
/// search sees segments cut by syncs. The structure comes from `seed`, the values from `vseed`.
/// Every op is continuous in its inputs so a rounding difference cannot flip a branch.
struct Fuzz<'a> {
    rng: Rng,
    d: &'a Device,
    pool: Vec<Tensor<3>>,
    inputs: Vec<Tensor<3>>,
    synced: Vec<TensorData>,
    mask: Tensor<3, Bool>,
    log: Vec<String>,
}

impl<'a> Fuzz<'a> {
    fn pick(&mut self) -> usize {
        self.rng.next_u32() as usize % self.pool.len()
    }
    fn coin(&mut self, n: u32) -> u32 {
        self.rng.next_u32() % n
    }
    fn partner_same(&mut self, i: usize) -> Option<usize> {
        let dims = self.pool[i].dims();
        let c: Vec<usize> = (0..self.pool.len()).filter(|&j| j != i && self.pool[j].dims() == dims).collect();
        if c.is_empty() { None } else { Some(c[self.rng.next_u32() as usize % c.len()]) }
    }
    fn push(&mut self, t: Tensor<3>, what: String) {
        if t.shape().num_elements() > 16384 {
            return;
        }
        self.log.push(format!("{what} -> {:?}", t.dims()));
        self.pool.push(t);
        if self.pool.len() > 8 {
            self.pool.remove(0);
        }
    }

    fn step(&mut self) {
        let i = self.pick();
        let a = self.pool[i].clone();
        let dims = a.dims();
        let op = self.coin(20);
        match op {
            0 => {
                let k = self.coin(10);
                let t = match k {
                    0 => a.tanh(),
                    1 => a.tanh().exp(),
                    2 => a.abs(),
                    3 => a.neg(),
                    4 => a.sin(),
                    5 => a.abs().add_scalar(0.1).sqrt(),
                    6 => act::sigmoid(a),
                    7 => act::relu(a),
                    8 => a.clone().mul(a).tanh(),
                    _ => a.erf(),
                };
                self.push(t, format!("unary{k} p{i}"));
            }
            1 => {
                let c = self.rng.f() * 2.0;
                let e = self.rng.f();
                self.push(a.mul_scalar(c).add_scalar(e), format!("affine p{i}"));
            }
            2 | 3 => {
                if let Some(j) = self.partner_same(i) {
                    let b = self.pool[j].clone();
                    let k = self.coin(6);
                    let t = match k {
                        0 => a + b,
                        1 => a - b,
                        2 => (a * b).tanh(),
                        3 => a / b.abs().add_scalar(1.0),
                        4 => a.max_pair(b),
                        _ => a.min_pair(b),
                    };
                    self.push(t, format!("binary{k} p{i} p{j}"));
                }
            }
            4 => {
                let j = self.pick();
                let b = self.pool[j].clone();
                if let Some(r) = (0..3).find(|&r| {
                    let bd = b.dims();
                    (0..3).all(|k| k == r || bd[k] == dims[k] || bd[k] == 1 || dims[k] == 1) && (0..3).all(|k| k == r || bd[k] == dims[k] || bd[k] == 1)
                }) {
                    let k = self.coin(3);
                    let red = match k {
                        0 => b.sum_dim(r).mul_scalar(1.0 / dims[r].max(1) as f32),
                        1 => b.max_dim(r),
                        _ => b.mean_dim(r),
                    };
                    let rd = red.dims();
                    if (0..3).all(|k| rd[k] == dims[k] || rd[k] == 1) {
                        self.push(a + red, format!("bcast_reduce{k} p{i} p{j} dim{r}"));
                    }
                }
            }
            5 => {
                let (x, y) = match self.coin(3) { 0 => (0, 1), 1 => (1, 2), _ => (0, 2) };
                self.push(a.swap_dims(x, y), format!("swap p{i} {x}{y}"));
            }
            6 => {
                let perm = [[1, 2, 0], [2, 0, 1], [2, 1, 0]][self.coin(3) as usize];
                self.push(a.permute(perm), format!("permute p{i} {perm:?}"));
            }
            7 => {
                let k = self.coin(3);
                let [x, y, z] = dims;
                let t = match k {
                    0 => a.reshape([y, x, z]),
                    1 => a.reshape([1, x * y, z]),
                    _ => a.reshape([x, y * z, 1]),
                };
                self.push(t, format!("reshape{k} p{i}"));
            }
            8 => {
                let r = self.coin(3) as usize;
                let n = dims[r];
                if n >= 2 {
                    self.push(a.narrow(r, n / 3, n - n / 3), format!("narrow p{i} dim{r}"));
                }
            }
            9 => {
                let r = self.coin(3) as usize;
                let n = dims[r];
                let t = if let (Some(j), true) = (self.partner_same(i), self.coin(2) == 0) {
                    let b = self.pool[j].clone();
                    Tensor::cat(vec![a, b], r)
                } else if n >= 2 {
                    let part = a.clone().narrow(r, 0, n / 2);
                    Tensor::cat(vec![a, part], r)
                } else {
                    Tensor::cat(vec![a.clone(), a.neg()], r)
                };
                self.push(t, format!("cat p{i} dim{r}"));
            }
            10 => {
                let r = self.coin(3) as usize;
                let k = self.coin(4);
                let t = match k {
                    0 => a.sum_dim(r).mul_scalar(1.0 / dims[r].max(1) as f32),
                    1 => a.mean_dim(r),
                    2 => a.max_dim(r),
                    _ => a.min_dim(r),
                };
                let t = if self.coin(2) == 0 { t.expand(dims) } else { t };
                self.push(t, format!("reduce{k} p{i} dim{r}"));
            }
            11 => {
                let j = self.pick();
                self.synced.push(self.pool[j].clone().into_data());
                self.log.push(format!("sync p{j}"));
            }
            12 => {
                let r = self.coin(3) as usize;
                self.push(a.flip([r]), format!("flip p{i} dim{r}"));
            }
            13 => {
                let [x, y, z] = dims;
                let t = if let (Some(j), true) = (self.partner_same(i), self.coin(2) == 0) {
                    let b = self.pool[j].clone();
                    a.swap_dims(1, 2).matmul(b).mul_scalar(1.0 / y as f32)
                } else {
                    a.clone().matmul(a.swap_dims(1, 2)).mul_scalar(1.0 / z as f32)
                };
                self.push(t, format!("matmul p{i}"));
            }
            14 => {
                let r = self.coin(3) as usize;
                let t = if self.coin(2) == 0 { act::softmax(a, r) } else { act::log_softmax(a, r).mul_scalar(0.25) };
                self.push(t, format!("softmax p{i} dim{r}"));
            }
            15 => {
                let r = self.coin(3) as usize;
                self.push(a.cumsum(r).mul_scalar(1.0 / dims[r].max(1) as f32), format!("cumsum p{i} dim{r}"));
            }
            16 => {
                if dims == self.mask.dims() {
                    let m = self.mask.clone();
                    let t = if self.coin(2) == 0 {
                        a.clone().mask_where(m, a.mul_scalar(-0.5))
                    } else {
                        a.mask_fill(m, 0.25)
                    };
                    self.push(t, format!("mask p{i}"));
                }
            }
            17 => {
                let r = self.coin(3) as usize;
                if dims[r] >= 2 {
                    self.push(a.slice_dim(r, s![..;2]), format!("step_slice p{i} dim{r}"));
                }
            }
            18 => {
                let r = self.coin(3) as usize;
                let n = dims[r];
                let idx: Vec<i64> = (0..n.max(1)).map(|_| (self.rng.next_u32() as usize % n.max(1)) as i64).collect();
                let idx = Tensor::<1, Int>::from_data(TensorData::new(idx, [n.max(1)]), self.d);
                self.push(a.select(r, idx), format!("select p{i} dim{r}"));
            }
            _ => {
                let r = self.coin(3) as usize;
                if dims.iter().product::<usize>() * 2 <= 8192 {
                    self.push(a.repeat_dim(r, 2), format!("repeat p{i} dim{r}"));
                }
            }
        }
    }
}

const FUZZ_DIMS: [usize; 11] = [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 32];

fn fuzz_shape(seed: u64) -> [usize; 3] {
    let mut r = Rng::new(seed ^ 0x5eed);
    loop {
        let s = [0, 1, 2].map(|_| FUZZ_DIMS[r.next_u32() as usize % FUZZ_DIMS.len()]);
        if numel(&s) <= 2048 {
            return s;
        }
    }
}

/// `shape` with each dim above 1 mapped to another size above 1, equal dims to equal sizes.
fn fuzz_shape_like(shape: [usize; 3], seed: u64) -> [usize; 3] {
    let mut r = Rng::new(seed ^ 0xa11);
    let big = [2usize, 3, 4, 5, 7, 8, 9, 16];
    let mut map: Vec<(usize, usize)> = vec![(1, 1)];
    shape.map(|s| {
        if let Some(&(_, t)) = map.iter().find(|(f, _)| *f == s) {
            t
        } else {
            let t = big[r.next_u32() as usize % big.len()];
            map.push((s, t));
            t
        }
    })
}

/// Runs program `seed` on `shape` with values `vseed`. With `grad`, also returns the gradients
/// of the inputs for a loss over the final pool.
fn fuzz_run(seed: u64, steps: usize, shape: [usize; 3], vseed: u64, d: &Device, grad: bool) -> (Vec<TensorData>, Vec<String>) {
    let d = if grad { d.clone().autodiff() } else { d.clone() };
    let mut vr = Rng::new(vseed);
    let mut inputs = vec![];
    for k in 0..3u64 {
        let layout = LAYOUTS[vr.next_u32() as usize % LAYOUTS.len()];
        let mut t = make(&randv(&shape, vseed * 7 + k), shape, layout, &d);
        if grad {
            t = t.require_grad();
        }
        inputs.push(t);
    }
    let mask = make(&randv(&shape, vseed * 7 + 5), shape, Layout::Contig, &d).greater_elem(0.0);
    let mut fz = Fuzz { rng: Rng::new(seed), d: &d, pool: inputs.clone(), inputs: inputs.clone(), synced: vec![], mask, log: vec![] };
    for _ in 0..steps {
        fz.step();
    }
    let mut outs = std::mem::take(&mut fz.synced);
    let n = fz.pool.len();
    let finals: Vec<Tensor<3>> = fz.pool[n.saturating_sub(3)..].to_vec();
    if grad {
        let mut loss: Option<Tensor<1>> = None;
        for (k, t) in finals.iter().enumerate() {
            let term = t.clone().mul_scalar(0.5 + k as f32).tanh().sum();
            loss = Some(match loss { None => term, Some(l) => l + term });
        }
        let g = loss.unwrap().backward();
        for t in &inputs {
            match t.grad(&g) {
                Some(gr) => outs.push(gr.into_data()),
                None => outs.push(TensorData::new(vec![0.0f32; numel(&shape)], shape)),
            }
        }
    } else {
        for t in finals {
            outs.push(t.into_data());
        }
    }
    (outs, fz.log)
}

fn fuzz_family(p: &mut Probe, first: u64, count: usize, grad: bool) {
    for seed in first..first + count as u64 {
        let shape = fuzz_shape(seed);
        let shape2 = fuzz_shape_like(shape, seed);
        let steps = 8 + (seed % 17) as usize;
        let (_, log) = fuzz_run(seed, steps, shape, 1, &refd(), grad);
        p.case(format!("seed{seed} {shape:?}/{shape2:?} steps{steps}{} [{}]", if grad { " grad" } else { "" }, log.join("; ")), LOOSE, |d| {
            // Same program three times: a fresh plan, the cached plan on new values, another shape.
            let mut all = fuzz_run(seed, steps, shape, 1, d, grad).0;
            all.extend(fuzz_run(seed, steps, shape, 2, d, grad).0);
            all.extend(fuzz_run(seed, steps, shape2, 3, d, grad).0);
            all
        });
    }
}

#[test]
fn hunt_fuzz_a() {
    let mut p = Probe::new("fuzz_a");
    fuzz_family(&mut p, 0, scale(150), false);
    p.finish();
}

#[test]
fn hunt_fuzz_b() {
    let mut p = Probe::new("fuzz_b");
    fuzz_family(&mut p, 10_000, scale(150), false);
    p.finish();
}

#[test]
fn hunt_fuzz_grad() {
    let mut p = Probe::new("fuzz_grad");
    fuzz_family(&mut p, 20_000, scale(120), true);
    p.finish();
}

// ---------------------------------------------------------------- quantization

mod quant {
    use super::*;
    use burn_tensor::quantization::{BlockExtent, BlockSize, QuantScheme, QuantStore, QuantValue, ScaleDtype};

    pub fn schemes() -> Vec<(&'static str, QuantScheme, f64)> {
        let q8 = QuantScheme::default().with_value(QuantValue::Q8S).with_store(QuantStore::PackedU32(0));
        vec![
            ("q8s tensor", q8.per_tensor(ScaleDtype::F32), 0.02),
            ("q8s block32", q8.per_block([32], ScaleDtype::F32), 0.02),
            ("q8s block2x16 f16+tensor", q8.per_block([2, 16], ScaleDtype::F16).per_tensor(ScaleDtype::F32), 0.03),
            ("q8s native block16", q8.with_store(QuantStore::Native).per_block([16], ScaleDtype::F32), 0.02),
            (
                "q4s block32",
                QuantScheme::default().with_value(QuantValue::Q4S).with_store(QuantStore::PackedU32(0)).per_block([32], ScaleDtype::F32),
                0.3,
            ),
            ("fp8 per_row", QuantScheme::fp8_per_row(), 0.15),
            (
                "e4m3 native block32",
                QuantScheme::default().with_value(QuantValue::E4M3).with_store(QuantStore::Native).per_block([32], ScaleDtype::F32),
                0.15,
            ),
            (
                "e5m2 block row",
                QuantScheme::default().with_value(QuantValue::E5M2).with_store(QuantStore::PackedU32(0)).per_block(
                    BlockSize::with_rest(&[BlockExtent::WholeAxis], BlockExtent::VALUE),
                    ScaleDtype::F32,
                ),
                0.3,
            ),
            ("nvfp4", QuantScheme::nvfp4(), 0.6),
            ("mxfp4 row factored", QuantScheme::mxfp4_row_factored(), 0.6),
        ]
    }
}

/// Rows of distinct magnitudes, so per-row and per-block scales differ.
fn qvals(shape: &[usize], seed: u64) -> Vec<f32> {
    let last = *shape.last().unwrap();
    randv(shape, seed)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v * (1.0 + ((i / last) % 5) as f32) * (1.0 + 0.5 * ((i % last) / 16) as f32))
        .collect()
}

#[test]
fn hunt_quant() {
    let mut p = Probe::new("quant");
    for (name, scheme, step) in quant::schemes() {
        let qt = Tol { atol: step, rtol: 0.02 };
        let mt = Tol { atol: step * 8.0, rtol: 0.03 };
        for &(k, n) in &[(32usize, 32usize), (64, 64), (96, 48), (64, 96), (128, 16)] {
            let w = qvals(&[k, n], (k * n) as u64);
            let wt = qvals(&[n, k], (k * n + 1) as u64);
            let tag = format!("{name} k{k} n{n}");
            p.case(format!("roundtrip {tag}"), qt, |d| {
                let q = Tensor::<2>::from_data(TensorData::new(w.clone(), [k, n]), d).quantize_dynamic(&scheme);
                out![q.dequantize()]
            });
            p.case(format!("roundtrip transposed input {tag}"), qt, |d| {
                let x = make(&w, [k, n], Layout::Transposed, d);
                out![x.quantize_dynamic(&scheme).dequantize()]
            });
            p.case(format!("q swap_dims {tag}"), qt, |d| {
                let q = Tensor::<2>::from_data(TensorData::new(w.clone(), [k, n]), d).quantize_dynamic(&scheme);
                out![q.swap_dims(0, 1).dequantize()]
            });
            p.case(format!("q narrow rows {tag}"), qt, |d| {
                let q = Tensor::<2>::from_data(TensorData::new(w.clone(), [k, n]), d).quantize_dynamic(&scheme);
                out![q.narrow(0, 2, k - 4).dequantize()]
            });
            p.case(format!("q reshape {tag}"), qt, |d| {
                let q = Tensor::<2>::from_data(TensorData::new(w.clone(), [k, n]), d).quantize_dynamic(&scheme);
                out![q.reshape([2, k / 2, n]).dequantize()]
            });
            p.case(format!("q add {tag}"), qt, |d| {
                let q = Tensor::<2>::from_data(TensorData::new(w.clone(), [k, n]), d).quantize_dynamic(&scheme);
                let o = Tensor::<2>::from_data(TensorData::new(randv(&[k, n], 3), [k, n]), d);
                out![q.dequantize() + o]
            });
            for &m in &[1usize, 3, 17, 48] {
                let lv = randv(&[m, k], (m * 31 + k) as u64);
                for lay in [Layout::Contig, Layout::Transposed] {
                    p.case(format!("matmul m{m} {lay:?} {tag}"), mt, |d| {
                        let q = Tensor::<2>::from_data(TensorData::new(w.clone(), [k, n]), d).quantize_dynamic(&scheme);
                        let l = make(&lv, [m, k], lay, d);
                        out![l.clone().matmul(q.clone()), l.matmul(q.dequantize())]
                    });
                    p.case(format!("matmul rhs^T m{m} {lay:?} {tag}"), mt, |d| {
                        let q = Tensor::<2>::from_data(TensorData::new(wt.clone(), [n, k]), d).quantize_dynamic(&scheme).transpose();
                        let l = make(&lv, [m, k], lay, d);
                        out![l.clone().matmul(q.clone()), l.matmul(q.dequantize())]
                    });
                }
                p.case(format!("matmul+bias+relu m{m} {tag}"), mt, |d| {
                    let q = Tensor::<2>::from_data(TensorData::new(w.clone(), [k, n]), d).quantize_dynamic(&scheme);
                    let l = Tensor::<2>::from_data(TensorData::new(lv.clone(), [m, k]), d);
                    let b = Tensor::<2>::from_data(TensorData::new(randv(&[1, n], 9), [1, n]), d);
                    out![act::relu(l.matmul(q) + b)]
                });
            }
            // Batched rhs, lhs broadcast over the batch.
            let wb = qvals(&[3, k, n], (k * n + 2) as u64);
            p.case(format!("batched matmul {tag}"), mt, |d| {
                let q = Tensor::<3>::from_data(TensorData::new(wb.clone(), [3, k, n]), d).quantize_dynamic(&scheme);
                let l = Tensor::<3>::from_data(TensorData::new(randv(&[1, 5, k], 4), [1, 5, k]), d);
                out![l.clone().matmul(q.clone()), l.matmul(q.dequantize())]
            });
        }
    }
    p.finish();
}

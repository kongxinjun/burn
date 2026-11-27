#![allow(clippy::all, unused)]
extern crate alloc;
use burn_tensor::{Device, Tensor, TensorData, Int, FloatDType};
use burn_tensor::module;
use burn_tensor::ops::AvgPoolOptions;

fn randv(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }).collect()
}

fn maxdiff(a: &TensorData, b: &TensorData) -> f32 {
    let a: Vec<f32> = a.iter::<f32>().collect();
    let b: Vec<f32> = b.iter::<f32>().collect();
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
fn focus_matmul_bcast() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    for &b in &[2usize, 3] {
        for &m in &[2usize, 4, 8, 16, 33] {
            for &k in &[1usize, 4, 8, 16, 64, 70] {
                for &n in &[1usize, 2, 4] {
                    for mode in ["lhs1", "rhs1"] {
                        let (ls, rs) = if mode == "lhs1" { ([1, m, k], [b, k, n]) } else { ([b, m, k], [1, k, n]) };
                        let lv = randv(ls.iter().product(), 1);
                        let rv = randv(rs.iter().product(), 2);
                        let run = |d: &Device| {
                            let l = Tensor::<3>::from_data(TensorData::new(lv.clone(), ls), d);
                            let r = Tensor::<3>::from_data(TensorData::new(rv.clone(), rs), d);
                            l.matmul(r).into_data()
                        };
                        let e = maxdiff(&run(&dev), &run(&flex));
                        if e > 1e-2 { bad.push(format!("{mode} b{b} m{m} k{k} n{n}: maxdiff {e}")); }
                    }
                }
            }
        }
    }
    println!("{}", bad.join("\n"));
    assert!(bad.is_empty(), "{} bad", bad.len());
}

#[test]
fn focus_argmax_nan() {
    let dev = Device::default();
    let mut bad = vec![];
    for n in [3usize, 4, 5, 7, 8, 9, 16, 32, 33, 100, 256, 1000] {
        for pos in [0usize, 1, n / 2, n - 1] {
            let mut v: Vec<f32> = (0..n).map(|i| i as f32).collect();
            v[pos] = f32::NAN;
            let t = Tensor::<2>::from_data(TensorData::new(v.clone(), [1, n]), &dev);
            let am: Vec<i64> = t.clone().argmax(1).into_data().iter::<i64>().collect();
            let an: Vec<i64> = t.clone().argmin(1).into_data().iter::<i64>().collect();
            let mx: Vec<f32> = t.clone().max_dim(1).into_data().iter::<f32>().collect();
            let (_, mi) = t.clone().max_dim_with_indices(1);
            let mi: Vec<i64> = mi.into_data().iter::<i64>().collect();
            let ok = am[0] == pos as i64 && an[0] == pos as i64 && mx[0].is_nan() && mi[0] == pos as i64;
            if !ok { bad.push(format!("n{n} nan@{pos}: argmax {am:?} argmin {an:?} max_dim {mx:?} max_dim_idx {mi:?}")); }
            // column direction
            let t2 = Tensor::<2>::from_data(TensorData::new(v.clone(), [n, 1]), &dev);
            let am2: Vec<i64> = t2.clone().argmax(0).into_data().iter::<i64>().collect();
            if am2[0] != pos as i64 { bad.push(format!("col n{n} nan@{pos}: argmax {am2:?}")); }
        }
    }
    println!("{}", bad.join("\n"));
    assert!(bad.is_empty(), "{} bad", bad.len());
}

#[test]
fn focus_avgpool_half() {
    let dev = Device::default();
    let mut out = vec![];
    for dt in [FloatDType::F16, FloatDType::BF16] {
        for n in [64usize, 256, 300, 1024, 2049, 4096] {
            let t = Tensor::<3>::ones([1, 1, n], &dev).cast(dt);
            let a: Vec<f32> = module::avg_pool1d(t.clone(), AvgPoolOptions::new([n])).cast(FloatDType::F32).into_data().iter::<f32>().collect();
            let side = (n as f64).sqrt() as usize;
            let t2 = Tensor::<4>::ones([1, 1, side, side], &dev).cast(dt);
            let a2: Vec<f32> = module::avg_pool2d(t2, AvgPoolOptions::new([side, side])).cast(FloatDType::F32).into_data().iter::<f32>().collect();
            let a3: Vec<f32> = module::adaptive_avg_pool1d(t, 1).cast(FloatDType::F32).into_data().iter::<f32>().collect();
            out.push(format!("{dt:?} n{n}: avg_pool1d {a:?}  avg_pool2d {side}x{side} {a2:?}  adaptive {a3:?}"));
        }
    }
    println!("{}", out.join("\n"));
}

#[test]
fn focus_mm_one() {
    let dev = Device::default();
    let flex = Device::flex();
    let lv: Vec<f32> = (0..8).map(|i| i as f32).collect(); // [1,4,2]
    let rv: Vec<f32> = (0..4).map(|i| (i + 1) as f32).collect(); // [2,2,1]
    let run = |d: &Device| {
        let l = Tensor::<3>::from_data(TensorData::new(lv.clone(), [1, 4, 2]), d);
        let r = Tensor::<3>::from_data(TensorData::new(rv.clone(), [2, 2, 1]), d);
        l.matmul(r).into_data().iter::<f32>().collect::<Vec<f32>>()
    };
    println!("device {:?}\nflex   {:?}", run(&dev), run(&flex));
}

#[test]
fn focus_mm_shared_rhs() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    for &(b, m, k, n) in &[(8usize, 1usize, 4usize, 2usize), (8, 1, 16, 3), (2, 32, 8, 4), (2, 32, 8, 1), (4, 64, 16, 8), (3, 1, 5, 2), (16, 1, 8, 4), (2, 1, 8, 64)] {
        let lv = randv(b * m * k, 1);
        let rv = randv(k * n, 2);
        let run = |d: &Device| {
            let l = Tensor::<3>::from_data(TensorData::new(lv.clone(), [b, m, k]), d);
            let r = Tensor::<3>::from_data(TensorData::new(rv.clone(), [1, k, n]), d);
            l.matmul(r).into_data()
        };
        let e = maxdiff(&run(&dev), &run(&flex));
        println!("shared-rhs b{b} m{m} k{k} n{n}: maxdiff {e}");
        if e > 1e-2 { bad.push(format!("b{b} m{m} k{k} n{n}")); }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

#[test]
fn focus_mm_shared_rhs_colmajor() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    for &(b, m, k, n) in &[(8usize, 1usize, 4usize, 2usize), (8, 1, 16, 3), (16, 1, 8, 4), (2, 32, 8, 4), (4, 64, 16, 8), (2, 1, 2, 4)] {
        let lv = randv(b * m * k, 1);
        let rv = randv(k * n, 2);
        let run = |d: &Device| {
            let l = Tensor::<3>::from_data(TensorData::new(lv.clone(), [b, m, k]), d);
            let r = Tensor::<3>::from_data(TensorData::new(rv.clone(), [1, n, k]), d).swap_dims(1, 2);
            l.matmul(r).into_data()
        };
        let e = maxdiff(&run(&dev), &run(&flex));
        println!("shared-rhs-col b{b} m{m} k{k} n{n}: maxdiff {e}");
        if e > 1e-2 { bad.push(format!("b{b} m{m} k{k} n{n}")); }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

#[test]
fn focus_mm_unit_row_stride() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    for &(b, k, n) in &[(2usize, 2usize, 4usize), (8, 4, 2), (3, 16, 5), (2, 70, 33)] {
        let lv = randv(b * k, 1);
        let rv = randv(k * n, 2);
        for variant in ["lhs [b,k,1]^T", "rhs col-major", "both"] {
            let run = |d: &Device| {
                let l = Tensor::<3>::from_data(TensorData::new(lv.clone(), [b, k, 1]), d).swap_dims(1, 2);
                let r = if variant == "lhs [b,k,1]^T" {
                    Tensor::<3>::from_data(TensorData::new(rv.clone(), [1, k, n]), d)
                } else {
                    Tensor::<3>::from_data(TensorData::new(rv.clone(), [1, n, k]), d).swap_dims(1, 2)
                };
                let l = if variant == "rhs col-major" { Tensor::<3>::from_data(TensorData::new(lv.clone(), [b, 1, k]), d) } else { l };
                l.matmul(r).into_data()
            };
            let e = maxdiff(&run(&dev), &run(&flex));
            println!("unit-row-stride {variant} b{b} k{k} n{n}: maxdiff {e}");
            if e > 1e-2 { bad.push(format!("{variant} b{b} k{k} n{n}")); }
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

fn cmp_line(name: &str, a: TensorData, b: TensorData, bad: &mut Vec<String>) {
    let e = maxdiff(&a, &b);
    let n = a.iter::<f32>().zip(b.iter::<f32>()).filter(|(x, y)| (x - y).abs() > 1e-4).count();
    println!("{name}: maxdiff {e} ({n}/{} off)", a.num_elements());
    if e > 1e-4 || a.shape() != b.shape() { bad.push(name.to_string()); }
}

#[test]
fn focus_fused_log_softmax() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    for shape in [[2usize, 3, 5, 7], [2, 4, 1, 8], [3, 2, 4, 4], [1, 3, 5, 7]] {
        let v = randv(shape.iter().product(), 7);
        let x = |d: &Device| Tensor::<4>::from_data(TensorData::new(v.clone(), shape), d);
        for dim in 0..4 {
            let tag = format!("{shape:?} dim{dim}");
            cmp_line(&format!("log_softmax {tag}"), burn_tensor::activation::log_softmax(x(&dev), dim).into_data(), burn_tensor::activation::log_softmax(x(&flex), dim).into_data(), &mut bad);
            // decomposed, all in one fusion graph
            let f = |d: &Device| {
                let t = x(d);
                let max = t.clone().detach().max_dim(dim);
                let shifted = t - max;
                let lse = shifted.clone().exp().sum_dim(dim).log();
                (shifted - lse).into_data()
            };
            cmp_line(&format!("decomposed {tag}"), f(&dev), f(&flex), &mut bad);
            // without the max shift
            let g = |d: &Device| { let t = x(d); (t.clone() - t.exp().sum_dim(dim).log()).into_data() };
            cmp_line(&format!("x - log(sum(exp x)) {tag}"), g(&dev), g(&flex), &mut bad);
            let h = |d: &Device| { let t = x(d); (t.clone() - t.sum_dim(dim)).into_data() };
            cmp_line(&format!("x - sum(x) {tag}"), h(&dev), h(&flex), &mut bad);
            let h2 = |d: &Device| { let t = x(d); (t.clone() - t.sum_dim(dim).exp()).into_data() };
            cmp_line(&format!("x - exp(sum(x)) {tag}"), h2(&dev), h2(&flex), &mut bad);
            let h3 = |d: &Device| { let t = x(d); let s = t.clone().add_scalar(1.0); (s.clone() - s.sum_dim(dim)).into_data() };
            cmp_line(&format!("s=x+1; s - sum(s) {tag}"), h3(&dev), h3(&flex), &mut bad);
            let h4 = |d: &Device| { let t = x(d); (t.clone() - t.mean_dim(dim)).powi_scalar(2).mean_dim(dim).into_data() };
            cmp_line(&format!("var {tag}"), h4(&dev), h4(&flex), &mut bad);
        }
    }
    println!("BAD: {bad:?}");
    assert!(bad.is_empty(), "{} bad", bad.len());
}

#[test]
fn focus_fused_two_reduce() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    let shape = [2usize, 3, 5, 7];
    let v = randv(shape.iter().product(), 7);
    let x = |d: &Device| Tensor::<4>::from_data(TensorData::new(v.clone(), shape), d);
    for dim in 0..3 {
        macro_rules! c {
            ($name:expr, $f:expr) => {{
                let f = $f;
                cmp_line(&format!("{} dim{dim}", $name), f(&dev), f(&flex), &mut bad);
            }};
        }
        c!("a: t - max(t)", |d: &Device| { let t = x(d); (t.clone() - t.max_dim(dim)).into_data() });
        c!("b: s=t-max; s - sum(s)", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); (s.clone() - s.sum_dim(dim)).into_data() });
        c!("c: s=t-max; sum(exp s)", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); s.exp().sum_dim(dim).into_data() });
        c!("d: s=t-max; s - sum(exp s)", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); (s.clone() - s.exp().sum_dim(dim)).into_data() });
        c!("e: s=t-sum; s - sum(s)", |d: &Device| { let t = x(d); let s = t.clone() - t.sum_dim(dim); (s.clone() - s.sum_dim(dim)).into_data() });
        c!("f: s=t*2; s - max(s) - sum(s)", |d: &Device| { let t = x(d); let s = t.mul_scalar(2.0); (s.clone() - s.clone().max_dim(dim) - s.sum_dim(dim)).into_data() });
        c!("g: s=t-max; sum(s) only", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); s.sum_dim(dim).into_data() });
        c!("h: s=t-mean; s - mean(s)", |d: &Device| { let t = x(d); let s = t.clone() - t.mean_dim(dim); (s.clone() - s.mean_dim(dim)).into_data() });
        c!("i: s=t-max; (s, sum(s)) materialize s first", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); let _ = s.clone().into_data(); (s.clone() - s.sum_dim(dim)).into_data() });
    }
    println!("BAD: {bad:?}");
    assert!(bad.is_empty(), "{} bad", bad.len());
}

#[test]
fn focus_fused_log_detach() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    let shape = [2usize, 3, 5, 7];
    let v = randv(shape.iter().product(), 7);
    let x = |d: &Device| Tensor::<4>::from_data(TensorData::new(v.clone(), shape), d);
    for dim in 0..4 {
        macro_rules! c {
            ($name:expr, $f:expr) => {{
                let f = $f;
                cmp_line(&format!("{} dim{dim}", $name), f(&dev), f(&flex), &mut bad);
            }};
        }
        c!("j: s=t-max; s - log(sum(exp s))", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); (s.clone() - s.exp().sum_dim(dim).log()).into_data() });
        c!("k: s=t-detach(t).max; s - sum(exp s)", |d: &Device| { let t = x(d); let s = t.clone() - t.detach().max_dim(dim); (s.clone() - s.exp().sum_dim(dim)).into_data() });
        c!("l: s=t-max; s - log(sum(s))", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); (s.clone() - s.sum_dim(dim).abs().log()).into_data() });
        c!("m: s=t-max; s - exp(sum(s))", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); (s.clone() - s.sum_dim(dim).exp()).into_data() });
        c!("n: s=t-max; s - (sum(s)+1)", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); (s.clone() - s.sum_dim(dim).add_scalar(1.0)).into_data() });
        c!("o: s=t+1; s - (sum(s)*2)", |d: &Device| { let t = x(d); let s = t.add_scalar(1.0); (s.clone() - s.sum_dim(dim).mul_scalar(2.0)).into_data() });
        c!("p: t - (sum(t)+1)", |d: &Device| { let t = x(d); (t.clone() - t.sum_dim(dim).add_scalar(1.0)).into_data() });
        c!("q: s=t-max; s - (max(s)+1)", |d: &Device| { let t = x(d); let s = t.clone() - t.max_dim(dim); (s.clone() - s.max_dim(dim).add_scalar(1.0)).into_data() });
        c!("r: s=t-sum; s - (sum(s)+1)", |d: &Device| { let t = x(d); let s = t.clone() - t.sum_dim(dim); (s.clone() - s.sum_dim(dim).add_scalar(1.0)).into_data() });
    }
    println!("BAD: {bad:?}");
    assert!(bad.is_empty(), "{} bad", bad.len());
}

#[test]
fn focus_fused_norms() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    let shape = [4usize, 3, 6, 5];
    let v = randv(shape.iter().product(), 9);
    let x = |d: &Device| Tensor::<4>::from_data(TensorData::new(v.clone(), shape), d);
    let bn = |d: &Device| {
        let o = module::batch_norm_train(x(d), Tensor::ones([3], d), Tensor::zeros([3], d), 1e-5);
        o.output.into_data()
    };
    cmp_line("batch_norm_train [4,3,6,5]", bn(&dev), bn(&flex), &mut bad);
    for dim in 0..4 {
        let manual = |d: &Device| {
            let t = x(d);
            let c = t.clone() - t.mean_dim(dim);
            (c.clone() / c.powi_scalar(2).mean_dim(dim).add_scalar(1e-5).sqrt()).into_data()
        };
        cmp_line(&format!("manual norm dim{dim}"), manual(&dev), manual(&flex), &mut bad);
        let ls = |d: &Device| burn_tensor::activation::log_softmax(x(d).mul_scalar(3.0), dim).into_data();
        cmp_line(&format!("log_softmax(3x) dim{dim}"), ls(&dev), ls(&flex), &mut bad);
    }
    println!("BAD: {bad:?}");
    assert!(bad.is_empty(), "{} bad", bad.len());
}

#[test]
fn focus_grid_sample_layout() {
    use burn_tensor::ops::{GridSampleMode, GridSampleOptions, GridSamplePaddingMode};
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    let (n, c, h, w, gh, gw) = (2usize, 3usize, 6usize, 5usize, 4usize, 7usize);
    let xv = randv(n * c * h * w, 51);
    let gv: Vec<f32> = randv(n * gh * gw * 2, 52).iter().map(|g| g * 0.9).collect();
    for mode in [GridSampleMode::Bilinear] {
        for (xl, gl) in [("contig", "contig"), ("nhwc", "contig"), ("contig", "from [N,2,H,W]"), ("nhwc", "from [N,2,H,W]")] {
            let run = |d: &Device| {
                let x = if xl == "nhwc" {
                    // same logical values, channels-last memory
                    Tensor::<4>::from_data(TensorData::new(xv.clone(), [n, c, h, w]), d).permute([0, 2, 3, 1]);
                    let t = Tensor::<4>::from_data(TensorData::new(xv.clone(), [n, c, h, w]), &Device::flex()).permute([0, 2, 3, 1]).into_data();
                    Tensor::<4>::from_data(t, d).permute([0, 3, 1, 2])
                } else {
                    Tensor::<4>::from_data(TensorData::new(xv.clone(), [n, c, h, w]), d)
                };
                let g = if gl == "contig" {
                    Tensor::<4>::from_data(TensorData::new(gv.clone(), [n, gh, gw, 2]), d)
                } else {
                    // flow-style: stored [N, 2, H, W], viewed as [N, H, W, 2]
                    let t = Tensor::<4>::from_data(TensorData::new(gv.clone(), [n, gh, gw, 2]), &Device::flex()).permute([0, 3, 1, 2]).into_data();
                    Tensor::<4>::from_data(t, d).permute([0, 2, 3, 1])
                };
                let o = GridSampleOptions::new(mode).with_padding_mode(GridSamplePaddingMode::Zeros).with_align_corners(false);
                x.grid_sample_2d(g, o).into_data()
            };
            cmp_line(&format!("grid_sample {mode:?} x={xl} grid={gl}"), run(&dev), run(&flex), &mut bad);
        }
    }
    println!("BAD: {bad:?}");
    assert!(bad.is_empty(), "{} bad", bad.len());
}

#[test]
fn focus_avgpool_half_realistic() {
    let dev = Device::default();
    for dt in [FloatDType::F16, FloatDType::BF16] {
        for k in [3usize, 7, 14, 28] {
            let shape = [1usize, 16, 28, 28];
            let v: Vec<f32> = randv(shape.iter().product(), 11).iter().map(|x| x.abs() * 4.0).collect();
            let x = Tensor::<4>::from_data(TensorData::new(v, shape), &dev).cast(dt);
            let o = AvgPoolOptions::new([k, k]).with_stride([k, k]);
            let got = module::avg_pool2d(x.clone(), o.clone()).cast(FloatDType::F32).into_data();
            let want = module::avg_pool2d(x.cast(FloatDType::F32), o).into_data();
            let g: Vec<f32> = got.iter::<f32>().collect();
            let w: Vec<f32> = want.iter::<f32>().collect();
            let rel = g.iter().zip(w.iter()).map(|(a, b)| ((a - b) / b).abs()).fold(0.0, f32::max);
            println!("{dt:?} avg_pool2d k{k}: max rel err vs f32 accumulation {rel:.4}");
        }
    }
}

#[test]
fn focus_mm_rotation_and_grad() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    // R [1,3,3] applied to a batch of points [b,3,1]
    let r: Vec<f32> = vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let p: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let run = |d: &Device| {
        let rt = Tensor::<3>::from_data(TensorData::new(r.clone(), [1, 3, 3]), d);
        let pt = Tensor::<3>::from_data(TensorData::new(p.clone(), [4, 3, 1]), d);
        rt.matmul(pt).into_data()
    };
    let (a, b) = (run(&dev), run(&flex));
    println!("rotate device {:?}\nrotate flex   {:?}", a.iter::<f32>().collect::<Vec<_>>(), b.iter::<f32>().collect::<Vec<_>>());
    cmp_line("R[1,3,3] @ p[4,3,1]", a, b, &mut bad);
    // gradient through the rhs of a shared-lhs batched mat-vec
    let grad = |d: &Device| {
        let d = d.clone().autodiff();
        let w = Tensor::<3>::from_data(TensorData::new(randv(12, 3), [1, 4, 3]), &d);
        let x = Tensor::<3>::from_data(TensorData::new(randv(6, 4), [2, 3, 1]), &d).require_grad();
        let y = w.matmul(x.clone()).mul_scalar(1.0);
        let g = y.sum().backward();
        x.grad(&g).unwrap().into_data()
    };
    cmp_line("grad x of W[1,4,3] @ x[2,3,1]", grad(&dev), grad(&flex), &mut bad);
    println!("BAD: {bad:?}");
    assert!(bad.is_empty());
}

#[test]
fn focus_mm_bcast_many_batches() {
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    for &(b, m, k) in &[(8usize, 2usize, 4usize), (16, 3, 3), (100, 3, 3), (64, 4, 16), (1000, 3, 3), (4, 3, 3)] {
        let lv = randv(m * k, 1);
        let rv = randv(b * k, 2);
        let run = |d: &Device| {
            let l = Tensor::<3>::from_data(TensorData::new(lv.clone(), [1, m, k]), d);
            let r = Tensor::<3>::from_data(TensorData::new(rv.clone(), [b, k, 1]), d);
            l.matmul(r).into_data()
        };
        cmp_line(&format!("[1,{m},{k}] @ [{b},{k},1]"), run(&dev), run(&flex), &mut bad);
    }
    println!("BAD: {bad:?}");
    assert!(bad.is_empty());
}

#[test]
fn focus_conv_asym_pad() {
    use burn_tensor::ops::{ConvOptions, PadMode};
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    // (cin, cout, h, w, k, stride, pads [(top,bottom),(left,right)])
    let cases: &[(usize, usize, usize, usize, [usize; 2], [usize; 2], [(usize, usize); 2])] = &[
        (3, 4, 10, 9, [3, 3], [1, 2], [(0, 2), (1, 0)]),
        (3, 4, 10, 9, [3, 3], [1, 1], [(0, 2), (1, 0)]),
        (3, 4, 10, 9, [3, 3], [1, 1], [(1, 0), (0, 1)]),
        (3, 4, 10, 9, [3, 3], [2, 2], [(0, 1), (0, 1)]),
        (8, 8, 16, 16, [3, 3], [1, 1], [(0, 2), (2, 0)]),
        (16, 32, 12, 12, [2, 2], [1, 1], [(0, 1), (0, 1)]),
        (3, 4, 10, 9, [1, 1], [1, 1], [(1, 0), (0, 0)]),
    ];
    for &(ci, co, h, w, k, st, pads) in cases {
        let xv = randv(ci * h * w, 23);
        let wv = randv(co * ci * k[0] * k[1], 24);
        let direct = |d: &Device| {
            module::conv2d(
                Tensor::<4>::from_data(TensorData::new(xv.clone(), [1, ci, h, w]), d),
                Tensor::<4>::from_data(TensorData::new(wv.clone(), [co, ci, k[0], k[1]]), d),
                None,
                ConvOptions::new_with_padding(st, pads, [1, 1], 1),
            )
            .into_data()
        };
        // independent reference on the same device: pad explicitly, then conv without padding
        let padded = |d: &Device| {
            let x = Tensor::<4>::from_data(TensorData::new(xv.clone(), [1, ci, h, w]), d)
                .pad([pads[0], pads[1]], PadMode::Constant(0.0));
            module::conv2d(
                x,
                Tensor::<4>::from_data(TensorData::new(wv.clone(), [co, ci, k[0], k[1]]), d),
                None,
                ConvOptions::new(st, [0, 0], [1, 1], 1),
            )
            .into_data()
        };
        let tag = format!("c{ci}->{co} {h}x{w} k{k:?} s{st:?} pad{pads:?}");
        let (a, b, c) = (direct(&dev), padded(&dev), direct(&flex));
        let e1 = maxdiff(&a, &b);
        let e2 = maxdiff(&a, &c);
        let e3 = maxdiff(&b, &c);
        println!("conv asym {tag}: direct-vs-padded {e1:.3e}, direct-vs-flex {e2:.3e}, padded-vs-flex {e3:.3e}");
        if e1 > 2e-2 || e2 > 2e-2 { bad.push(tag); }
    }
    println!("BAD: {bad:?}");
    assert!(bad.is_empty());
}

#[test]
fn focus_conv_asym_sweep() {
    use burn_tensor::ops::{ConvOptions, PadMode};
    let dev = Device::default();
    let flex = Device::flex();
    let mut bad = vec![];
    for &(ci, co) in &[(8usize, 8usize), (16, 16), (4, 8)] {
        for &hw in &[16usize, 20, 32] {
            for &n in &[1usize, 2] {
                for pads in [[(0usize, 2usize), (2usize, 0usize)], [(0, 1), (0, 1)], [(2, 0), (0, 2)]] {
                    let xv = randv(n * ci * hw * hw, 31);
                    let wv = randv(co * ci * 9, 32);
                    let run = |d: &Device| {
                        module::conv2d(
                            Tensor::<4>::from_data(TensorData::new(xv.clone(), [n, ci, hw, hw]), d),
                            Tensor::<4>::from_data(TensorData::new(wv.clone(), [co, ci, 3, 3]), d),
                            None,
                            ConvOptions::new_with_padding([1, 1], pads, [1, 1], 1),
                        )
                        .into_data()
                    };
                    let e = maxdiff(&run(&dev), &run(&flex));
                    let tag = format!("n{n} c{ci}->{co} {hw}x{hw} pad{pads:?}");
                    println!("CONVCASE {tag}: maxdiff {e:.3e}");
                    if e > 5e-2 { bad.push(tag); }
                }
            }
        }
    }
    println!("BAD: {bad:?}");
    assert!(bad.is_empty());
}

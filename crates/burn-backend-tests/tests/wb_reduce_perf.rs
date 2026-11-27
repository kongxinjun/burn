#![allow(clippy::all, unused)]
//! Times fused reduce graphs whose read block materializes an intermediate (the case the reduce
//! fuser's in-place setting affects). Prints `ok<TAB>perf<TAB><case> <ms>` lines.
use burn_tensor::{Device, Tensor, TensorData};
use std::time::Instant;

fn randv(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }).collect()
}

fn time<F: Fn() -> Tensor<4>>(dev: &Device, name: &str, f: F) {
    for _ in 0..10 { let _ = f(); }
    dev.sync().unwrap();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..20 { let _ = f(); }
        dev.sync().unwrap();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0 / 20.0);
    }
    println!("ok\tperf\t{name} {best:.3} ms");
}

#[test]
fn reduce_perf() {
    let dev = Device::default();
    for shape in [[16usize, 64, 64, 64], [8, 21, 256, 256]] {
        let v = randv(shape.iter().product(), 3);
        let x = Tensor::<4>::from_data(TensorData::new(v, shape), &dev);
        for dim in [1usize, 2] {
            time(&dev, &format!("log_softmax {shape:?} dim{dim}"), || {
                burn_tensor::activation::log_softmax(x.clone().mul_scalar(1.0), dim)
            });
            time(&dev, &format!("norm {shape:?} dim{dim}"), || {
                let t = x.clone().mul_scalar(1.0);
                let c = t.clone() - t.mean_dim(dim);
                c.clone() / c.powi_scalar(2).mean_dim(dim).add_scalar(1e-5).sqrt()
            });
            time(&dev, &format!("s-2sum(s) {shape:?} dim{dim}"), || {
                let s = x.clone().add_scalar(1.0);
                s.clone() - s.sum_dim(dim).mul_scalar(2.0)
            });
        }
    }
}

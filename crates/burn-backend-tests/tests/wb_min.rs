//! Minimal repros for findings of the hunt (fork-only harness). Each test prints
//! `ok|FAIL<TAB>min<TAB>name<TAB>detail` and only uses API that also exists on older commits.
#![allow(clippy::all, unused)]

use burn_tensor::{DType, Device, FloatDType, Tensor, TensorData};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn report(name: &str, f: impl FnOnce() -> Result<(), String>) {
    let r = catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|e| {
        let m = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        Err(format!("panicked: {}", m.lines().next().unwrap_or("").chars().take(300).collect::<String>()))
    });
    match r {
        Ok(()) => println!("ok\tmin\t{name}"),
        Err(m) => println!("FAIL\tmin\t{name}\t{m}"),
    }
}

fn close(got: &TensorData, want: &[f32]) -> Result<(), String> {
    let g: Vec<f32> = got.iter::<f32>().collect();
    let bad = g.iter().zip(want).any(|(a, b)| (a - b).abs() > 1e-3 * (1.0 + b.abs()));
    if g.len() != want.len() || bad { Err(format!("got {g:?} want {want:?}")) } else { Ok(()) }
}

#[test]
fn min_flex32_matmul() {
    let d = Device::default();
    println!("device {d:?} supports Flex32: {}", d.supports_dtype(DType::Flex32));
    report("flex32 matmul 2x2", || {
        let a = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &d).cast(FloatDType::Flex32);
        let c = a.clone().matmul(a);
        println!("out dtype {:?}", c.dtype());
        close(&c.cast(FloatDType::F32).into_data(), &[7.0, 10.0, 15.0, 22.0])
    });
    report("flex32 matmul [8,8,7]x[8,7,5]", || {
        let a = Tensor::<3>::ones([8, 8, 7], &d).cast(FloatDType::Flex32);
        let b = Tensor::<3>::ones([8, 7, 5], &d).cast(FloatDType::Flex32);
        close(&a.matmul(b).cast(FloatDType::F32).into_data(), &[7.0; 8 * 8 * 5])
    });
    report("f32 matmul 2x2 (control)", || {
        let a = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &d);
        close(&a.clone().matmul(a).into_data(), &[7.0, 10.0, 15.0, 22.0])
    });
}

fn rt(shape: [usize; 3], seed: u64, d: &Device) -> Tensor<3> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let n: usize = shape.iter().product();
    let v: Vec<f32> = (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    Tensor::from_data(TensorData::new(v, shape), d)
}

/// Full transcription of fuzz seed 13 (wb_hunt), to be reduced.
#[test]
fn min_fuzz13_full() {
    let d = Device::default();
    report("fuzz13 full", || {
        let s = [16, 16, 5];
        let (t1, t2) = (rt(s, 1, &d), rt(s, 2, &d));
        let t3 = t2.clone().max_pair(t1.clone());
        let t4 = t3.clone().select(2, Tensor::<1, burn_tensor::Int>::from_data([1, 3, 2, 2, 0], &d));
        let t5 = t1.clone().swap_dims(0, 2);
        let _ = t4.clone().into_data();
        let t6 = t5.slice_dim(2, burn_tensor::s![..;2]);
        let t7 = burn_tensor::activation::log_softmax(t3, 0).mul_scalar(0.25);
        let t8 = t4.sum_dim(1).mul_scalar(1.0 / 16.0).expand(s);
        let t9 = t2.swap_dims(1, 2).matmul(t1).mul_scalar(1.0 / 16.0);
        let t10 = t6.clone().flip([0]);
        let t11 = burn_tensor::activation::softmax(t7, 0);
        let t12 = t9.flip([2]);
        let _t13 = t6.tanh();
        let _t14 = t10.clone().permute([2, 0, 1]);
        let _t15 = t11.select(2, Tensor::<1, burn_tensor::Int>::from_data([2, 1, 0, 2, 1], &d));
        let _t16 = t8.slice_dim(2, burn_tensor::s![..;2]);
        let t17 = t10.clone().select(0, Tensor::<1, burn_tensor::Int>::from_data([0, 2, 3, 0, 2], &d));
        let t18 = burn_tensor::activation::softmax(t10, 1);
        let t19 = Tensor::cat(vec![t12.clone(), t12.narrow(1, 0, 2)], 1);
        let _ = (t17.into_data(), t18.into_data(), t19.into_data());
        Ok(())
    });
}

#[test]
fn min_fuzz13_mask() {
    let Ok(m) = std::env::var("WB_MASK") else { return };
    let m = u32::from_str_radix(&m, 16).unwrap();
    let on = |b: u32| m & (1 << b) != 0;
    let d = Device::default();
    report(&format!("fuzz13 mask {m:x}"), || {
        let s = [16, 16, 5];
        let (t1, t2) = (rt(s, 1, &d), rt(s, 2, &d));
        let t3 = if on(8) { t2.clone().max_pair(t1.clone()) } else { t2.clone() };
        let t4 = if on(9) { t3.clone().select(2, Tensor::<1, burn_tensor::Int>::from_data([1, 3, 2, 2, 0], &d)) } else { t3.clone() };
        let t5 = t1.clone().swap_dims(0, 2);
        if on(0) { let _ = t4.clone().into_data(); }
        let t6 = if on(10) { t5.slice_dim(2, burn_tensor::s![..;2]) } else { t5.narrow(2, 0, 8) };
        let t7 = burn_tensor::activation::log_softmax(t3, 0).mul_scalar(0.25);
        let t8 = t4.sum_dim(1).mul_scalar(1.0 / 16.0).expand(s);
        let t9 = t2.swap_dims(1, 2).matmul(t1).mul_scalar(1.0 / 16.0);
        let t10 = if on(11) { t6.clone().flip([0]) } else { t6.clone() };
        let t11 = burn_tensor::activation::softmax(t7, 0);
        let t12 = if on(12) { t9.flip([2]) } else { t9 };
        if on(1) { let _t13 = t6.tanh(); }
        if on(2) { let _t14 = t10.clone().permute([2, 0, 1]); }
        if on(3) { let _t15 = t11.select(2, Tensor::<1, burn_tensor::Int>::from_data([2, 1, 0, 2, 1], &d)); }
        if on(4) { let _t16 = t8.slice_dim(2, burn_tensor::s![..;2]); }
        let t17 = t10.clone().select(0, Tensor::<1, burn_tensor::Int>::from_data([0, 2, 3, 0, 2], &d));
        let t18 = burn_tensor::activation::softmax(t10, 1);
        let t19 = Tensor::cat(vec![t12.clone(), t12.narrow(1, 0, 2)], 1);
        if on(5) { let _ = t17.into_data(); }
        if on(6) { let _ = t18.into_data(); }
        if on(7) { let _ = t19.into_data(); }
        Ok(())
    });
}

#[test]
fn min_fuzz13_mask2() {
    let Ok(m) = std::env::var("WB_MASK2") else { return };
    let m = u32::from_str_radix(&m, 16).unwrap();
    let on = |b: u32| m & (1 << b) != 0;
    let d = Device::default();
    report(&format!("fuzz13 mask2 {m:x}"), || {
        let s = [16, 16, 5];
        let (t1, t2) = (rt(s, 1, &d), rt(s, 2, &d));
        let t6 = t1.clone().swap_dims(0, 2).narrow(2, 0, 8);
        if on(0) { let t7 = burn_tensor::activation::log_softmax(t2.clone(), 0).mul_scalar(0.25); if on(1) { let _t11 = burn_tensor::activation::softmax(t7, 0); } }
        if on(2) { let _t8 = t2.clone().sum_dim(1).mul_scalar(1.0 / 16.0).expand(s); }
        if on(3) {
            let t9 = t2.clone().swap_dims(1, 2).matmul(t1.clone()).mul_scalar(1.0 / 16.0);
            if on(4) { let _t19 = Tensor::cat(vec![t9.clone(), t9.narrow(1, 0, 2)], 1); }
        }
        let t17 = t6.clone().select(0, Tensor::<1, burn_tensor::Int>::from_data([0, 2, 3, 0, 2], &d));
        if on(5) { let _t18 = burn_tensor::activation::softmax(t6, 1); }
        let _ = t17.into_data();
        Ok(())
    });
}

/// Fusion panics at burn-cubecl-fusion/src/engine/trace/fuser.rs:123 (unwrap on None).
#[test]
fn min_fuse_select_then_softmax() {
    let d = Device::default();
    let idx = |d: &Device| Tensor::<1, burn_tensor::Int>::from_data([0, 2], d);
    report("select(view) + softmax(view) [16,16,5]", || {
        let x = rt([16, 16, 5], 1, &d).swap_dims(0, 2).narrow(2, 0, 8);
        let y = x.clone().select(0, idx(&d));
        let _z = burn_tensor::activation::softmax(x, 1);
        let _ = y.into_data();
        Ok(())
    });
    report("select + softmax, plain input [3,4]", || {
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], &d);
        let y = x.clone().select(0, idx(&d));
        let _z = burn_tensor::activation::softmax(x, 1);
        close(&y.into_data(), &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0])
    });
    report("select + exp().sum_dim, plain input [3,4]", || {
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], &d);
        let y = x.clone().select(0, idx(&d));
        let _z = x.exp().sum_dim(1);
        close(&y.into_data(), &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0])
    });
    report("select + sum_dim, plain input [3,4]", || {
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], &d);
        let y = x.clone().select(0, idx(&d));
        let _z = x.sum_dim(1);
        close(&y.into_data(), &[1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0])
    });
}

#[test]
fn min_v() {
    let Ok(v) = std::env::var("WB_V") else { return };
    let d = Device::default();
    report(&format!("variant {v}"), || {
        let s = [16, 16, 5];
        let t1 = rt(s, 1, &d);
        let _t2 = if v.contains('b') { Some(rt(s, 2, &d)) } else { None };
        let n = if v.contains('i') { vec![0i64, 2, 3, 0, 2] } else { vec![0i64, 2] };
        let base = if v.contains('k') { t1.clone() } else { t1 };
        let t6 = if v.contains('s') { base.swap_dims(0, 2) } else { base.reshape([5, 16, 16]) };
        let t6 = if v.contains('n') { t6.narrow(2, 0, 8) } else { t6 };
        let len = n.len();
        let t17 = t6.clone().select(0, Tensor::<1, burn_tensor::Int>::from_data(TensorData::new(n, [len]), &d));
        let _t18 = if v.contains('x') { t6.exp().sum_dim(1) } else { burn_tensor::activation::softmax(t6, 1) };
        let _ = t17.into_data();
        Ok(())
    });
}

/// `select` whose output has its input's shape, beside a reduce-based op on the same input.
#[test]
fn min_select_same_shape() {
    let v = std::env::var("WB_SEL").unwrap_or_else(|_| "sum".into());
    let d = Device::default();
    report(&format!("select same shape {v}"), || {
        let x = Tensor::<2>::from_data([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], &d);
        let idx = Tensor::<1, burn_tensor::Int>::from_data([2, 0, 1], &d);
        let y = x.clone().select(0, idx);
        let _z = match v.as_str() {
            "softmax" => burn_tensor::activation::softmax(x, 1),
            "sum" => x.sum_dim(1),
            "expsum" => x.exp().sum_dim(1),
            "max" => x.max_dim(1),
            "sub_max" => x.clone() - x.max_dim(1),
            "gather" => {
                let i = Tensor::<2, burn_tensor::Int>::from_data([[3, 2, 1, 0], [0, 1, 2, 3], [1, 1, 1, 1]], &d);
                let g = x.clone().gather(1, i);
                let _ = x.sum_dim(1);
                return close(&g.into_data(), &[4.0, 3.0, 2.0, 1.0, 5.0, 6.0, 7.0, 8.0, 10.0, 10.0, 10.0, 10.0]);
            }
            _ => x.exp(),
        };
        close(&y.into_data(), &[9.0, 10.0, 11.0, 12.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    });
}

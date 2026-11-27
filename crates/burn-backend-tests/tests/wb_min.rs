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

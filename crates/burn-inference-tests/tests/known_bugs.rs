//! Confirmed upstream bugs, reduced to their smallest reproduction. Each test fails while its bug
//! remains, so CI runs them in a step allowed to fail: a test that starts passing marks a fix.

use burn::tensor::{Device, Distribution, Tensor, Tolerance};
use burn_inference_tests::harness::{ACCELERATED, TestCase, device, run};

/// Found by lab-fuzz; PyTorch agrees with Flex. Wrong on Metal, wgpu and CubeCL CPU, with and
/// without fusion: a reduction over the third axis of this permutation reads the wrong elements.
fn cubecl_sum_of_permuted_tensor() {
    let flex = Device::flex();
    flex.seed(7);
    let data = Tensor::<4>::random([3, 4, 4, 5], Distribution::Default, &flex).into_data();
    let sum = |device: &Device| {
        Tensor::<4>::from_data(data.clone(), device)
            .permute([1, 3, 0, 2])
            .sum_dim(2)
            .into_data()
    };

    sum(&device()).assert_approx_eq::<f32>(&sum(&flex), Tolerance::rel_abs(1e-4, 1e-4));
}

fn main() {
    let mut tests: Vec<TestCase> = Vec::new();
    if ACCELERATED {
        tests.push((
            "cubecl_sum_of_permuted_tensor",
            cubecl_sum_of_permuted_tensor,
        ));
    }

    run(tests);
}

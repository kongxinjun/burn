//! End-to-end inference on complete models. See the README for what each test guards against.

#![allow(clippy::single_range_in_vec_init)]

use std::thread;

use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::transformer::{TransformerDecoderConfig, TransformerDecoderInput};
use burn::tensor::{Bool, Device, Distribution, Int, Tensor, TensorData, Tolerance};
use burn_inference_tests::harness::{ACCELERATED, TestCase, device, run};
use burn_inference_tests::{Array, Reference, TinyResNet, TinyTransformer};

fn float_tensor<const D: usize>(array: &Array<f32>, device: &Device) -> Tensor<D> {
    Tensor::from_data(
        TensorData::new(array.data.clone(), array.shape.clone()),
        device,
    )
}

fn int_tensor(array: &Array<i64>, device: &Device) -> Tensor<2, Int> {
    Tensor::from_data(
        TensorData::new(array.data.clone(), array.shape.clone()),
        device,
    )
}

fn bool_tensor(array: &Array<bool>, device: &Device) -> Tensor<2, Bool> {
    Tensor::from_data(
        TensorData::new(array.data.clone(), array.shape.clone()),
        device,
    )
}

fn expected(array: &Array<f32>) -> TensorData {
    TensorData::new(array.data.clone(), array.shape.clone())
}

#[track_caller]
fn assert_close(actual: TensorData, expected: &TensorData) {
    actual.assert_approx_eq::<f32>(expected, Tolerance::rel_abs(2e-3, 2e-4));
}

/// Reproducible images, created on the reference device so every backend sees the same input.
fn images(batch_size: usize, height: usize, width: usize, seed: u64) -> TensorData {
    let flex = Device::flex();
    flex.seed(seed);
    Tensor::<4>::random([batch_size, 3, height, width], Distribution::Default, &flex).into_data()
}

/// Reproducible token ids within the model's vocabulary.
fn tokens(batch_size: usize, seq_length: usize) -> TensorData {
    let data = (0..batch_size * seq_length)
        .map(|i| ((i * 7 + 3) % 31 + 1) as i64)
        .collect::<Vec<_>>();
    TensorData::new(data, [batch_size, seq_length])
}

fn resnet_matches_pytorch() {
    let device = device();
    let reference = Reference::load().resnet;
    let model = TinyResNet::pretrained(&device);

    let output = model.forward(float_tensor(&reference.input, &device));

    assert_close(output.into_data(), &expected(&reference.output));
}

fn transformer_matches_pytorch() {
    let device = device();
    let reference = Reference::load().transformer;
    let model = TinyTransformer::pretrained(&device);

    let output = model.forward(int_tensor(&reference.tokens, &device), None);

    assert_close(output.into_data(), &expected(&reference.output));
}

fn padded_transformer_matches_pytorch() {
    let device = device();
    let reference = Reference::load().transformer;
    let model = TinyTransformer::pretrained(&device);

    let output = model.forward(
        int_tensor(&reference.padded_tokens, &device),
        Some(bool_tensor(&reference.mask_pad, &device)),
    );

    assert_close(output.into_data(), &expected(&reference.padded_output));
}

fn padding_does_not_change_real_tokens() {
    let device = device();
    let model = TinyTransformer::pretrained(&device);
    let [real, padded] = [5, 12];
    let short = tokens(1, real);

    let mut ids = short.clone().try_to_vec::<i64>().unwrap();
    ids.resize(padded, 0);
    let mask = (0..padded).map(|i| i >= real).collect::<Vec<_>>();

    let alone = model.forward(Tensor::from_data(short, &device), None);
    let batched = model.forward(
        Tensor::from_data(TensorData::new(ids, [1, padded]), &device),
        Some(Tensor::from_data(
            TensorData::new(mask, [1, padded]),
            &device,
        )),
    );

    assert_close(
        batched.slice([0..1, 0..real]).into_data(),
        &alone.into_data(),
    );
}

fn batching_does_not_change_outputs() {
    let device = device();
    let model = TinyResNet::pretrained(&device);

    for batch_size in [2, 7, 16] {
        let batch = Tensor::<4>::from_data(images(batch_size, 16, 16, 7), &device);
        let output = model.forward(batch.clone());

        for i in 0..batch_size {
            let single = model.forward(batch.clone().slice([i..i + 1]));
            assert_close(
                output.clone().slice([i..i + 1]).into_data(),
                &single.into_data(),
            );
        }
    }
}

fn accelerated_device_matches_flex() {
    let device = device();
    let flex = Device::flex();
    let [resnet, resnet_flex] = [&device, &flex].map(TinyResNet::pretrained);
    let [transformer, transformer_flex] = [&device, &flex].map(TinyTransformer::pretrained);

    for (batch_size, height, width) in [(1, 16, 16), (4, 32, 24), (9, 20, 48)] {
        let input = images(batch_size, height, width, 11);
        let output = resnet.forward(Tensor::from_data(input.clone(), &device));
        let reference = resnet_flex.forward(Tensor::from_data(input, &flex));
        assert_close(output.into_data(), &reference.into_data());
    }

    for (batch_size, seq_length) in [(1, 3), (4, 8), (6, 12)] {
        let input = tokens(batch_size, seq_length);
        let output = transformer.forward(Tensor::from_data(input.clone(), &device), None);
        let reference = transformer_flex.forward(Tensor::from_data(input, &flex), None);
        assert_close(output.into_data(), &reference.into_data());
    }
}

fn repeated_inference_is_stable() {
    let device = device();
    let model = TinyResNet::pretrained(&device);
    let shapes = [(1, 16, 16), (4, 32, 24), (2, 16, 16), (8, 20, 48)];
    let inputs = shapes.map(|(batch_size, height, width)| images(batch_size, height, width, 13));
    let first = inputs
        .clone()
        .map(|input| model.forward(Tensor::from_data(input, &device)).into_data());

    // Enough executions for fusion to replace its first plans with cached and autotuned ones.
    for _ in 0..50 {
        for (input, first) in inputs.iter().zip(&first) {
            let output = model.forward(Tensor::from_data(input.clone(), &device));
            assert_close(output.into_data(), first);
        }
    }
}

fn concurrent_inference_matches_sequential() {
    let device = device();
    let model = TinyTransformer::pretrained(&device);
    let inputs = (1..=8)
        .map(|batch_size| tokens(batch_size, 4 + batch_size))
        .collect::<Vec<_>>();
    let sequential = inputs
        .iter()
        .map(|input| {
            model
                .forward(Tensor::from_data(input.clone(), &device), None)
                .into_data()
        })
        .collect::<Vec<_>>();

    let workers = inputs
        .into_iter()
        .zip(sequential)
        .map(|(input, expected)| {
            let (model, device) = (model.clone(), device.clone());
            thread::spawn(move || {
                for _ in 0..16 {
                    let output = model.forward(Tensor::from_data(input.clone(), &device), None);
                    assert_close(output.into_data(), &expected);
                }
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        worker.join().expect("The worker should not panic");
    }
}

fn cached_decoding_matches_full_decoding() {
    let device = device();
    let [batch_size, seq_length, d_model] = [2, 12, 16];

    for norm_first in [false, true] {
        device.seed(42);
        let decoder = TransformerDecoderConfig::new(d_model, 32, 2, 2)
            .with_dropout(0.0)
            .with_norm_first(norm_first)
            .init(&device);
        let flex = Device::flex();
        flex.seed(17);
        let shape = [batch_size, seq_length, d_model];
        let target = Tensor::<3>::random(shape, Distribution::Default, &flex).to_device(&device);
        let memory = Tensor::<3>::random(shape, Distribution::Default, &flex).to_device(&device);

        let mask = generate_autoregressive_mask(batch_size, seq_length, &device);
        let full = decoder.forward(
            TransformerDecoderInput::new(target.clone(), memory.clone()).target_mask_attn(mask),
        );

        let mut cache = decoder.new_autoregressive_cache();
        let steps = (1..=seq_length)
            .map(|length| {
                let mask = generate_autoregressive_mask(batch_size, length, &device);
                let input = TransformerDecoderInput::new(
                    target.clone().slice([0..batch_size, 0..length]),
                    memory.clone(),
                )
                .target_mask_attn(mask);
                decoder
                    .forward_autoregressive_inference(input, &mut cache)
                    .slice([0..batch_size, length - 1..length])
            })
            .collect::<Vec<_>>();

        Tensor::cat(steps, 1)
            .into_data()
            .assert_approx_eq::<f32>(&full.into_data(), Tolerance::rel_abs(5e-3, 1e-4));
    }
}

fn main() {
    let mut tests: Vec<TestCase> = vec![
        ("resnet_matches_pytorch", resnet_matches_pytorch),
        ("transformer_matches_pytorch", transformer_matches_pytorch),
        (
            "padded_transformer_matches_pytorch",
            padded_transformer_matches_pytorch,
        ),
        (
            "padding_does_not_change_real_tokens",
            padding_does_not_change_real_tokens,
        ),
        (
            "batching_does_not_change_outputs",
            batching_does_not_change_outputs,
        ),
        ("repeated_inference_is_stable", repeated_inference_is_stable),
        (
            "concurrent_inference_matches_sequential",
            concurrent_inference_matches_sequential,
        ),
        (
            "cached_decoding_matches_full_decoding",
            cached_decoding_matches_full_decoding,
        ),
    ];
    if ACCELERATED {
        tests.push((
            "accelerated_device_matches_flex",
            accelerated_device_matches_flex,
        ));
    }
    run(tests);
}

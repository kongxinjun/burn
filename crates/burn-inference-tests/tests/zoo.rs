//! Pretrained models checked against the outputs of the frameworks they were exported from.

#![allow(clippy::single_range_in_vec_init)]

use std::thread;

use burn::tensor::{Bool, Device, Int, Tensor, TensorData, Tolerance};
use burn_inference_tests::harness::{ACCELERATED, TestCase, device, run};
use burn_inference_tests::zoo::{DistilGpt2, MiniLm, MobileNetV2, Reference, ResNet50};

/// PyTorch and Burn round differently, and errors accumulate across dozens of layers.
fn tolerance() -> Tolerance<f32> {
    Tolerance::rel_abs(5e-3, 5e-3)
}

#[track_caller]
fn assert_close(actual: TensorData, expected: &TensorData) {
    actual.assert_approx_eq::<f32>(expected, tolerance());
}

fn mask_pad(attention_mask: TensorData, device: &Device) -> Tensor<2, Bool> {
    Tensor::<2, Int>::from_data(attention_mask, device).equal_elem(0)
}

fn classifier_matches_reference(name: &str, forward: impl Fn(Tensor<4>) -> Tensor<2>) {
    let reference = Reference::load(name);
    let input = Tensor::from_data(reference.get("input"), &device());

    assert_close(forward(input).into_data(), &reference.get("logits"));
}

fn classifier_ignores_batching(name: &str, forward: impl Fn(Tensor<4>) -> Tensor<2>) {
    let input = Tensor::<4>::from_data(Reference::load(name).get("input"), &device());
    let [batch_size, ..] = input.dims();
    let batched = forward(input.clone());

    for i in 0..batch_size {
        assert_close(
            batched.clone().slice([i..i + 1]).into_data(),
            &forward(input.clone().slice([i..i + 1])).into_data(),
        );
    }
}

fn resnet50_matches_torchvision() {
    let model = ResNet50::pretrained(&device());
    classifier_matches_reference("resnet50", |x| model.forward(x));
}

fn resnet50_ignores_batching() {
    let model = ResNet50::pretrained(&device());
    classifier_ignores_batching("resnet50", |x| model.forward(x));
}

fn mobilenet_v2_matches_torchvision() {
    let model = MobileNetV2::pretrained(&device());
    classifier_matches_reference("mobilenet_v2", |x| model.forward(x));
}

fn mobilenet_v2_ignores_batching() {
    let model = MobileNetV2::pretrained(&device());
    classifier_ignores_batching("mobilenet_v2", |x| model.forward(x));
}

fn minilm_matches_transformers() {
    let device = device();
    let reference = Reference::load("minilm");
    let model = MiniLm::pretrained(&device);
    let mask = mask_pad(reference.get("attention_mask"), &device);

    let hidden = model.forward(
        Tensor::from_data(reference.get("input_ids"), &device),
        Tensor::from_data(reference.get("token_type_ids"), &device),
        mask.clone(),
    );

    assert_close(hidden.clone().into_data(), &reference.get("hidden"));
    assert_close(
        model.embed(hidden, mask).into_data(),
        &reference.get("embedding"),
    );
}

fn minilm_ignores_padding() {
    let device = device();
    let reference = Reference::load("minilm");
    let model = MiniLm::pretrained(&device);
    let input_ids = Tensor::<2, Int>::from_data(reference.get("input_ids"), &device);
    let token_type_ids = Tensor::<2, Int>::from_data(reference.get("token_type_ids"), &device);
    let mask = mask_pad(reference.get("attention_mask"), &device);
    let padded = model.embed(
        model.forward(input_ids.clone(), token_type_ids.clone(), mask.clone()),
        mask.clone(),
    );

    let lengths = mask
        .clone()
        .bool_not()
        .int()
        .sum_dim(1)
        .into_data()
        .convert::<i64>()
        .try_into_vec::<i64>()
        .unwrap();
    for (i, length) in lengths.into_iter().enumerate() {
        let length = length as usize;
        let row = [i..i + 1, 0..length];
        let mask = mask.clone().slice(row.clone());
        let alone = model.forward(
            input_ids.clone().slice(row.clone()),
            token_type_ids.clone().slice(row),
            mask.clone(),
        );
        assert_close(
            padded.clone().slice([i..i + 1]).into_data(),
            &model.embed(alone, mask).into_data(),
        );
    }
}

fn prompt_and_generated() -> (Vec<i64>, Vec<i64>) {
    let reference = Reference::load("distilgpt2");
    let tokens = |name| reference.get(name).try_into_vec::<i64>().unwrap();

    (tokens("prompt"), tokens("generated"))
}

fn distilgpt2_matches_transformers() {
    let device = device();
    let reference = Reference::load("distilgpt2");
    let model = DistilGpt2::pretrained(&device);

    let logits = model.forward(Tensor::from_data(reference.get("prompt"), &device));

    assert_close(logits.into_data(), &reference.get("logits"));
}

fn distilgpt2_generates_like_transformers() {
    let device = device();
    let model = DistilGpt2::pretrained(&device);
    let (prompt, generated) = prompt_and_generated();

    let tokens = model.generate_full(&prompt, generated.len() - prompt.len(), &device);

    assert_eq!(tokens, generated);
}

fn distilgpt2_cached_generation_matches_transformers() {
    let device = device();
    let model = DistilGpt2::pretrained(&device);
    let (prompt, generated) = prompt_and_generated();

    let tokens = model.generate_cached(&prompt, generated.len() - prompt.len(), &device);

    assert_eq!(tokens, generated);
}

fn concurrent_requests_match_reference() {
    let device = device();
    let reference = Reference::load("mobilenet_v2");
    let model = MobileNetV2::pretrained(&device);
    let input = reference.get("input");
    let expected = reference.get("logits");

    let workers = (0..4)
        .map(|_| {
            let (model, device, input, expected) = (
                model.clone(),
                device.clone(),
                input.clone(),
                expected.clone(),
            );
            thread::spawn(move || {
                for _ in 0..8 {
                    let output = model.forward(Tensor::from_data(input.clone(), &device));
                    assert_close(output.into_data(), &expected);
                }
            })
        })
        .collect::<Vec<_>>();

    for worker in workers {
        worker.join().expect("The worker should not panic");
    }
}

fn accelerated_device_matches_flex() {
    let device = device();
    let flex = Device::flex();
    let input = Reference::load("resnet50").get("input");

    let resnet = ResNet50::pretrained(&device).forward(Tensor::from_data(input.clone(), &device));
    let reference = ResNet50::pretrained(&flex).forward(Tensor::from_data(input.clone(), &flex));
    assert_close(resnet.into_data(), &reference.into_data());

    let mobilenet =
        MobileNetV2::pretrained(&device).forward(Tensor::from_data(input.clone(), &device));
    let reference = MobileNetV2::pretrained(&flex).forward(Tensor::from_data(input, &flex));
    assert_close(mobilenet.into_data(), &reference.into_data());
}

fn main() {
    let mut tests: Vec<TestCase> = vec![
        ("resnet50_matches_torchvision", resnet50_matches_torchvision),
        ("resnet50_ignores_batching", resnet50_ignores_batching),
        (
            "mobilenet_v2_matches_torchvision",
            mobilenet_v2_matches_torchvision,
        ),
        (
            "mobilenet_v2_ignores_batching",
            mobilenet_v2_ignores_batching,
        ),
        ("minilm_matches_transformers", minilm_matches_transformers),
        ("minilm_ignores_padding", minilm_ignores_padding),
        (
            "distilgpt2_matches_transformers",
            distilgpt2_matches_transformers,
        ),
        (
            "distilgpt2_generates_like_transformers",
            distilgpt2_generates_like_transformers,
        ),
        (
            "distilgpt2_cached_generation_matches_transformers",
            distilgpt2_cached_generation_matches_transformers,
        ),
        (
            "concurrent_requests_match_reference",
            concurrent_requests_match_reference,
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

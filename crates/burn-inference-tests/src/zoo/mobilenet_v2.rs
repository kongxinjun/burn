use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    tensor::{Device, Tensor},
};

use super::load_weights;

/// Expansion, output channels, repeats and first stride of every stage.
const STAGES: [[usize; 4]; 7] = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
];

fn conv(
    channels: [usize; 2],
    kernel: usize,
    stride: usize,
    groups: usize,
    device: &Device,
) -> Conv2d {
    let padding = (kernel - 1) / 2;
    Conv2dConfig::new(channels, [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(
            padding, padding, padding, padding,
        ))
        .with_groups(groups)
        .with_bias(false)
        .init(device)
}

fn relu6(x: Tensor<4>) -> Tensor<4> {
    x.clamp(0.0, 6.0)
}

#[derive(Module, Debug)]
struct ConvNorm {
    conv: Conv2d,
    norm: BatchNorm,
}

impl ConvNorm {
    fn new(
        channels: [usize; 2],
        kernel: usize,
        stride: usize,
        groups: usize,
        device: &Device,
    ) -> Self {
        Self {
            conv: conv(channels, kernel, stride, groups, device),
            norm: BatchNormConfig::new(channels[1]).init(device),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        relu6(self.norm.forward(self.conv.forward(x)))
    }
}

#[derive(Module, Debug)]
struct InvertedResidual {
    expand: Option<ConvNorm>,
    depthwise: ConvNorm,
    project: Conv2d,
    project_norm: BatchNorm,
    residual: bool,
}

impl InvertedResidual {
    fn new(channels: [usize; 2], stride: usize, expansion: usize, device: &Device) -> Self {
        let [channels_in, channels_out] = channels;
        let hidden = channels_in * expansion;

        Self {
            expand: (expansion != 1).then(|| ConvNorm::new([channels_in, hidden], 1, 1, 1, device)),
            depthwise: ConvNorm::new([hidden, hidden], 3, stride, hidden, device),
            project: conv([hidden, channels_out], 1, 1, 1, device),
            project_norm: BatchNormConfig::new(channels_out).init(device),
            residual: stride == 1 && channels_in == channels_out,
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let x = match &self.expand {
            Some(expand) => expand.forward(input.clone()),
            None => input.clone(),
        };
        let x = self.depthwise.forward(x);
        let x = self.project_norm.forward(self.project.forward(x));

        if self.residual { input + x } else { x }
    }
}

/// MobileNetV2 with torchvision's `IMAGENET1K_V2` weights.
#[derive(Module, Debug)]
pub struct MobileNetV2 {
    first: ConvNorm,
    blocks: Vec<InvertedResidual>,
    last: ConvNorm,
    pool: AdaptiveAvgPool2d,
    classifier: Linear,
}

impl MobileNetV2 {
    /// Creates the model with its pretrained weights.
    pub fn pretrained(device: &Device) -> Self {
        let mut channels_in = 32;
        let mut blocks = Vec::new();
        for [expansion, channels_out, repeats, stride] in STAGES {
            for index in 0..repeats {
                let stride = if index == 0 { stride } else { 1 };
                blocks.push(InvertedResidual::new(
                    [channels_in, channels_out],
                    stride,
                    expansion,
                    device,
                ));
                channels_in = channels_out;
            }
        }

        let mut model = Self {
            first: ConvNorm::new([3, 32], 3, 2, 1, device),
            blocks,
            last: ConvNorm::new([320, 1280], 1, 1, 1, device),
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            classifier: LinearConfig::new(1280, 1000).init(device),
        };
        load_weights(&mut model, "mobilenet_v2");
        model
    }

    /// Computes the class logits of a batch of images.
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let mut x = self.first.forward(images);
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = self.last.forward(x);

        self.classifier.forward(self.pool.forward(x).flatten(1, 3))
    }
}

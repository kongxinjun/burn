use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
    },
    tensor::{Device, Tensor},
};

use super::load_weights;

const EXPANSION: usize = 4;

fn conv(channels: [usize; 2], kernel: usize, stride: usize, device: &Device) -> Conv2d {
    let padding = kernel / 2;
    Conv2dConfig::new(channels, [kernel, kernel])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(
            padding, padding, padding, padding,
        ))
        .with_bias(false)
        .init(device)
}

#[derive(Module, Debug)]
struct Downsample {
    conv: Conv2d,
    bn: BatchNorm,
}

#[derive(Module, Debug)]
struct Bottleneck {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    conv3: Conv2d,
    bn3: BatchNorm,
    downsample: Option<Downsample>,
    relu: Relu,
}

impl Bottleneck {
    fn new(channels_in: usize, width: usize, stride: usize, device: &Device) -> Self {
        let channels_out = width * EXPANSION;
        let downsample = (stride != 1 || channels_in != channels_out).then(|| Downsample {
            conv: conv([channels_in, channels_out], 1, stride, device),
            bn: BatchNormConfig::new(channels_out).init(device),
        });

        Self {
            conv1: conv([channels_in, width], 1, 1, device),
            bn1: BatchNormConfig::new(width).init(device),
            conv2: conv([width, width], 3, stride, device),
            bn2: BatchNormConfig::new(width).init(device),
            conv3: conv([width, channels_out], 1, 1, device),
            bn3: BatchNormConfig::new(channels_out).init(device),
            downsample,
            relu: Relu::new(),
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let shortcut = match &self.downsample {
            Some(downsample) => downsample
                .bn
                .forward(downsample.conv.forward(input.clone())),
            None => input.clone(),
        };
        let x = self
            .relu
            .forward(self.bn1.forward(self.conv1.forward(input)));
        let x = self.relu.forward(self.bn2.forward(self.conv2.forward(x)));
        let x = self.bn3.forward(self.conv3.forward(x));

        self.relu.forward(x + shortcut)
    }
}

fn layer(
    channels_in: usize,
    width: usize,
    blocks: usize,
    stride: usize,
    device: &Device,
) -> Vec<Bottleneck> {
    let channels_out = width * EXPANSION;
    (0..blocks)
        .map(|index| match index {
            0 => Bottleneck::new(channels_in, width, stride, device),
            _ => Bottleneck::new(channels_out, width, 1, device),
        })
        .collect()
}

/// ResNet-50 with torchvision's `IMAGENET1K_V2` weights.
#[derive(Module, Debug)]
pub struct ResNet50 {
    conv1: Conv2d,
    bn1: BatchNorm,
    maxpool: MaxPool2d,
    layer1: Vec<Bottleneck>,
    layer2: Vec<Bottleneck>,
    layer3: Vec<Bottleneck>,
    layer4: Vec<Bottleneck>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear,
    relu: Relu,
}

impl ResNet50 {
    /// Creates the model with its pretrained weights.
    pub fn pretrained(device: &Device) -> Self {
        let mut model = Self {
            conv1: conv([3, 64], 7, 2, device),
            bn1: BatchNormConfig::new(64).init(device),
            maxpool: MaxPool2dConfig::new([3, 3])
                .with_strides([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(),
            layer1: layer(64, 64, 3, 1, device),
            layer2: layer(256, 128, 4, 2, device),
            layer3: layer(512, 256, 6, 2, device),
            layer4: layer(1024, 512, 3, 2, device),
            avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            fc: LinearConfig::new(2048, 1000).init(device),
            relu: Relu::new(),
        };
        load_weights(&mut model, "resnet50");
        model
    }

    /// Computes the class logits of a batch of images.
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let x = self
            .relu
            .forward(self.bn1.forward(self.conv1.forward(images)));
        let mut x = self.maxpool.forward(x);
        for layer in [&self.layer1, &self.layer2, &self.layer3, &self.layer4] {
            for block in layer {
                x = block.forward(x);
            }
        }

        self.fc.forward(self.avgpool.forward(x).flatten(1, 3))
    }
}

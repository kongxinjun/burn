//! Models and fixtures shared by the end-to-end inference tests.

use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
        PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{Bool, Device, Int, Tensor},
};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use serde::Deserialize;

const RESNET_WEIGHTS: &[u8] = include_bytes!("../fixtures/resnet.safetensors");
const TRANSFORMER_WEIGHTS: &[u8] = include_bytes!("../fixtures/transformer.safetensors");
const REFERENCE: &str = include_str!("../fixtures/reference.json");

/// A tensor stored in the reference file.
#[derive(Clone, Debug, Deserialize)]
pub struct Array<E> {
    /// The dimensions of the tensor.
    pub shape: Vec<usize>,
    /// The elements of the tensor, in row-major order.
    pub data: Vec<E>,
}

/// The expected outputs of [`TinyResNet`].
#[derive(Debug, Deserialize)]
pub struct ResNetReference {
    /// The images given to the model.
    pub input: Array<f32>,
    /// The logits computed by PyTorch.
    pub output: Array<f32>,
}

/// The expected outputs of [`TinyTransformer`].
#[derive(Debug, Deserialize)]
pub struct TransformerReference {
    /// The token ids of the unpadded batch.
    pub tokens: Array<i64>,
    /// The logits PyTorch computes for `tokens`.
    pub output: Array<f32>,
    /// The token ids of the padded batch.
    pub padded_tokens: Array<i64>,
    /// The positions of `padded_tokens` that are padding.
    pub mask_pad: Array<bool>,
    /// The logits PyTorch computes for `padded_tokens`.
    pub padded_output: Array<f32>,
}

/// The outputs computed by `fixtures/generate.py`.
#[derive(Debug, Deserialize)]
pub struct Reference {
    /// The residual network's reference.
    pub resnet: ResNetReference,
    /// The transformer's reference.
    pub transformer: TransformerReference,
}

impl Reference {
    /// Parses the reference outputs bundled with the tests.
    pub fn load() -> Self {
        serde_json::from_str(REFERENCE).expect("The reference file should be valid")
    }
}

/// The shortcut of a residual block whose output shape differs from its input.
#[derive(Module, Debug)]
pub struct Downsample {
    conv: Conv2d,
    norm: BatchNorm,
}

/// A residual block of two convolutions.
#[derive(Module, Debug)]
pub struct Block {
    conv1: Conv2d,
    norm1: BatchNorm,
    conv2: Conv2d,
    norm2: BatchNorm,
    downsample: Option<Downsample>,
    relu: Relu,
}

impl Block {
    fn new(channels: [usize; 2], stride: usize, device: &Device) -> Self {
        let [channels_in, channels_out] = channels;
        let conv = |channels, stride| {
            Conv2dConfig::new(channels, [3, 3])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device)
        };
        let downsample = (stride != 1 || channels_in != channels_out).then(|| Downsample {
            conv: Conv2dConfig::new(channels, [1, 1])
                .with_stride([stride, stride])
                .with_bias(false)
                .init(device),
            norm: BatchNormConfig::new(channels_out).init(device),
        });

        Self {
            conv1: conv(channels, stride),
            norm1: BatchNormConfig::new(channels_out).init(device),
            conv2: conv([channels_out, channels_out], 1),
            norm2: BatchNormConfig::new(channels_out).init(device),
            downsample,
            relu: Relu::new(),
        }
    }

    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let shortcut = match &self.downsample {
            Some(downsample) => downsample
                .norm
                .forward(downsample.conv.forward(input.clone())),
            None => input.clone(),
        };
        let x = self
            .relu
            .forward(self.norm1.forward(self.conv1.forward(input)));
        let x = self.norm2.forward(self.conv2.forward(x));

        self.relu.forward(x + shortcut)
    }
}

/// A small residual image classifier.
#[derive(Module, Debug)]
pub struct TinyResNet {
    stem: Conv2d,
    stem_norm: BatchNorm,
    blocks: Vec<Block>,
    pool: AdaptiveAvgPool2d,
    head: Linear,
    relu: Relu,
}

impl TinyResNet {
    /// Creates the model with the weights exported from PyTorch.
    pub fn pretrained(device: &Device) -> Self {
        let mut model = Self {
            stem: Conv2dConfig::new([3, 8], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_bias(false)
                .init(device),
            stem_norm: BatchNormConfig::new(8).init(device),
            blocks: vec![
                Block::new([8, 8], 1, device),
                Block::new([8, 16], 2, device),
            ],
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            head: LinearConfig::new(16, 5).init(device),
            relu: Relu::new(),
        };
        load_weights(&mut model, RESNET_WEIGHTS);
        model
    }

    /// Computes the logits of a batch of images.
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let mut x = self
            .relu
            .forward(self.stem_norm.forward(self.stem.forward(images)));
        for block in &self.blocks {
            x = block.forward(x);
        }

        self.head.forward(self.pool.forward(x).flatten(1, 3))
    }
}

/// A small pre-norm transformer producing logits for every token.
#[derive(Module, Debug)]
pub struct TinyTransformer {
    embedding: Embedding,
    position: Embedding,
    encoder: TransformerEncoder,
    head: Linear,
}

impl TinyTransformer {
    /// Creates the model with the weights exported from PyTorch.
    pub fn pretrained(device: &Device) -> Self {
        let mut model = Self {
            embedding: EmbeddingConfig::new(32, 16).init(device),
            position: EmbeddingConfig::new(12, 16).init(device),
            encoder: TransformerEncoderConfig::new(16, 32, 2, 2)
                .with_dropout(0.0)
                .with_norm_first(true)
                .init(device),
            head: LinearConfig::new(16, 8).init(device),
        };
        load_weights(&mut model, TRANSFORMER_WEIGHTS);
        model
    }

    /// Computes the logits of every token, ignoring the positions set in `mask_pad`.
    pub fn forward(&self, tokens: Tensor<2, Int>, mask_pad: Option<Tensor<2, Bool>>) -> Tensor<3> {
        let [batch_size, seq_length] = tokens.dims();
        let positions = Tensor::<1, Int>::arange(0..seq_length as i64, &tokens.device())
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);
        let x = self.embedding.forward(tokens) + self.position.forward(positions);

        let mut input = TransformerEncoderInput::new(x);
        if let Some(mask_pad) = mask_pad {
            input = input.mask_pad(mask_pad);
        }

        self.head.forward(self.encoder.forward(input))
    }
}

/// Loads PyTorch weights, requiring them to cover exactly the parameters of `model`.
fn load_weights<M: Module>(model: &mut M, weights: &[u8]) {
    let mut store = SafetensorsStore::from_bytes(Some(weights.to_vec()))
        .with_from_adapter(PyTorchToBurnAdapter);
    let result = model
        .load_from(&mut store)
        .expect("The weights should load");

    assert!(result.is_success(), "{result:?}");
    assert!(result.missing.is_empty(), "{result:?}");
    assert!(result.unused.is_empty(), "{result:?}");
}

use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{Bool, Device, Int, Tensor},
};

use super::load_weights;

const D_MODEL: usize = 384;
const EPSILON: f64 = 1e-12;

/// The `all-MiniLM-L6-v2` sentence encoder.
#[derive(Module, Debug)]
pub struct MiniLm {
    word: Embedding,
    position: Embedding,
    token_type: Embedding,
    norm: LayerNorm,
    encoder: TransformerEncoder,
}

impl MiniLm {
    /// Creates the model with its pretrained weights.
    pub fn pretrained(device: &Device) -> Self {
        let mut model = Self {
            word: EmbeddingConfig::new(30522, D_MODEL).init(device),
            position: EmbeddingConfig::new(512, D_MODEL).init(device),
            token_type: EmbeddingConfig::new(2, D_MODEL).init(device),
            norm: LayerNormConfig::new(D_MODEL)
                .with_epsilon(EPSILON)
                .init(device),
            encoder: TransformerEncoderConfig::new(D_MODEL, 1536, 12, 6)
                .with_dropout(0.0)
                .with_layer_norm_eps(EPSILON)
                .init(device),
        };
        load_weights(&mut model, "minilm");
        model
    }

    /// Computes the hidden state of every token, ignoring the positions set in `mask_pad`.
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        token_type_ids: Tensor<2, Int>,
        mask_pad: Tensor<2, Bool>,
    ) -> Tensor<3> {
        let [batch_size, seq_length] = input_ids.dims();
        let positions = Tensor::<1, Int>::arange(0..seq_length as i64, &input_ids.device())
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);
        let x = self.word.forward(input_ids)
            + self.position.forward(positions)
            + self.token_type.forward(token_type_ids);

        self.encoder
            .forward(TransformerEncoderInput::new(self.norm.forward(x)).mask_pad(mask_pad))
    }

    /// Computes normalized sentence embeddings by averaging the tokens that are not padding.
    pub fn embed(&self, hidden: Tensor<3>, mask_pad: Tensor<2, Bool>) -> Tensor<2> {
        let keep = mask_pad.bool_not().float().unsqueeze_dim::<3>(2);
        let mean = (hidden * keep.clone()).sum_dim(1) / keep.sum_dim(1).clamp_min(1e-9);
        let mean = mean.squeeze_dim::<2>(1);
        let norm = mean
            .clone()
            .powi_scalar(2)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1e-12);

        mean / norm
    }
}

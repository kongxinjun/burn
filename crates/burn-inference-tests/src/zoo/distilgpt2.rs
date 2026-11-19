use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        activation::ActivationConfig,
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{Device, Int, Tensor, TensorData},
};

use super::load_weights;

const VOCABULARY: usize = 50257;
const D_MODEL: usize = 768;

/// The `distilgpt2` language model.
#[derive(Module, Debug)]
pub struct DistilGpt2 {
    wte: Embedding,
    wpe: Embedding,
    encoder: TransformerEncoder,
    ln_f: LayerNorm,
    head: Linear,
}

impl DistilGpt2 {
    /// Creates the model with its pretrained weights.
    pub fn pretrained(device: &Device) -> Self {
        let mut model = Self {
            wte: EmbeddingConfig::new(VOCABULARY, D_MODEL).init(device),
            wpe: EmbeddingConfig::new(1024, D_MODEL).init(device),
            encoder: TransformerEncoderConfig::new(D_MODEL, 3072, 12, 6)
                .with_dropout(0.0)
                .with_norm_first(true)
                .with_activation(ActivationConfig::GeluApproximate)
                .init(device),
            ln_f: LayerNormConfig::new(D_MODEL).init(device),
            head: LinearConfig::new(D_MODEL, VOCABULARY)
                .with_bias(false)
                .init(device),
        };
        load_weights(&mut model, "distilgpt2");
        model
    }

    fn embed(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let [batch_size, seq_length] = tokens.dims();
        let positions = Tensor::<1, Int>::arange(0..seq_length as i64, &tokens.device())
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        self.wte.forward(tokens) + self.wpe.forward(positions)
    }

    fn input(&self, tokens: Tensor<2, Int>) -> TransformerEncoderInput {
        let [batch_size, seq_length] = tokens.dims();
        let mask = generate_autoregressive_mask(batch_size, seq_length, &tokens.device());

        TransformerEncoderInput::new(self.embed(tokens)).mask_attn(mask)
    }

    /// Computes the logits of every position.
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let x = self.encoder.forward(self.input(tokens));

        self.head.forward(self.ln_f.forward(x))
    }

    /// Greedily extends `prompt`, recomputing the whole sequence for every token.
    pub fn generate_full(&self, prompt: &[i64], new_tokens: usize, device: &Device) -> Vec<i64> {
        self.generate(prompt, new_tokens, device, |tokens| {
            let [_, seq_length] = tokens.dims();
            self.forward(tokens)
                .slice([0..1, seq_length - 1..seq_length])
        })
    }

    /// Greedily extends `prompt`, reusing the computations of the previous tokens.
    pub fn generate_cached(&self, prompt: &[i64], new_tokens: usize, device: &Device) -> Vec<i64> {
        let cache = core::cell::RefCell::new(self.encoder.new_autoregressive_cache());
        self.generate(prompt, new_tokens, device, |tokens| {
            let [_, seq_length] = tokens.dims();
            let x = self
                .encoder
                .forward_autoregressive_inference(self.input(tokens), &mut cache.borrow_mut())
                .slice([0..1, seq_length - 1..seq_length]);

            self.head.forward(self.ln_f.forward(x))
        })
    }

    fn generate(
        &self,
        prompt: &[i64],
        new_tokens: usize,
        device: &Device,
        next_logits: impl Fn(Tensor<2, Int>) -> Tensor<3>,
    ) -> Vec<i64> {
        let mut tokens = prompt.to_vec();
        for _ in 0..new_tokens {
            let data = TensorData::new(tokens.clone(), [1, tokens.len()]);
            let next = next_logits(Tensor::from_data(data, device))
                .argmax(2)
                .into_data()
                .convert::<i64>()
                .try_into_vec::<i64>()
                .expect("Tokens should be integers");
            tokens.push(next[0]);
        }

        tokens
    }
}

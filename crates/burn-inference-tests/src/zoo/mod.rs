//! Pretrained models exported by `zoo/generate.py`.

mod distilgpt2;
mod minilm;
mod mobilenet_v2;
mod resnet50;

pub use distilgpt2::DistilGpt2;
pub use minilm::MiniLm;
pub use mobilenet_v2::MobileNetV2;
pub use resnet50::ResNet50;

use std::path::PathBuf;

use burn::module::Module;
use burn::tensor::TensorData;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use safetensors::{Dtype, SafeTensors};

/// The directory holding the exported models: `BURN_LAB_MODELS`, or `target/lab-models`.
pub fn models_dir() -> PathBuf {
    std::env::var_os("BURN_LAB_MODELS").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/lab-models"),
        PathBuf::from,
    )
}

/// Loads a model's weights, requiring them to cover exactly its parameters.
fn load_weights<M: Module>(model: &mut M, name: &str) {
    let path = models_dir().join(format!("{name}.safetensors"));
    assert!(
        path.exists(),
        "{} is missing. Run `uv run zoo/generate.py <dir>` and set BURN_LAB_MODELS to <dir>",
        path.display()
    );
    let mut store = SafetensorsStore::from_file(path).with_from_adapter(PyTorchToBurnAdapter);
    let result = model
        .load_from(&mut store)
        .expect("The weights should load");

    assert!(result.is_success(), "{result:?}");
    assert!(result.missing.is_empty(), "{result:?}");
    assert!(result.unused.is_empty(), "{result:?}");
}

/// The inputs and outputs PyTorch computed for a model.
pub struct Reference {
    bytes: Vec<u8>,
}

impl Reference {
    /// Reads the reference of the model called `name`.
    pub fn load(name: &str) -> Self {
        let path = models_dir().join(format!("{name}.reference.safetensors"));
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()));
        Self { bytes }
    }

    /// Reads the tensor called `name`, in the element type it was exported with.
    pub fn get(&self, name: &str) -> TensorData {
        let tensors = SafeTensors::deserialize(&self.bytes).expect("The reference should parse");
        let tensor = tensors.tensor(name).expect("The tensor should exist");
        let shape = tensor.shape().to_vec();
        match tensor.dtype() {
            Dtype::F32 => {
                let data = tensor.data().chunks_exact(4).map(|bytes| {
                    f32::from_le_bytes(bytes.try_into().expect("Chunks hold four bytes"))
                });
                TensorData::new(data.collect::<Vec<_>>(), shape)
            }
            Dtype::I64 => {
                let data = tensor.data().chunks_exact(8).map(|bytes| {
                    i64::from_le_bytes(bytes.try_into().expect("Chunks hold eight bytes"))
                });
                TensorData::new(data.collect::<Vec<_>>(), shape)
            }
            dtype => panic!("Unsupported reference dtype {dtype:?}"),
        }
    }
}

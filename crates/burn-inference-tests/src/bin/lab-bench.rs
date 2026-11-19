//! Measures the latency and memory of pretrained model inference on the device under test.
//!
//! ```sh
//! cargo run --release --bin lab-bench --features metal,fusion -- --scenario resnet50/b1
//! ```
//!
//! Every request reads its output back to the host, so the timings include execution, not only
//! dispatch. The JSON output follows github-action-benchmark's `customSmallerIsBetter` format.

use std::time::{Duration, Instant};

use burn::tensor::{Bool, Device, Int, Tensor};
use burn_inference_tests::harness::{backend_name, device, exit};
use burn_inference_tests::zoo::{DistilGpt2, MiniLm, MobileNetV2, Reference, ResNet50};

/// A request against a loaded model.
type Request = Box<dyn FnMut()>;

const SCENARIOS: [&str; 7] = [
    "resnet50/b1",
    "resnet50/b8",
    "mobilenet_v2/b1",
    "mobilenet_v2/b16",
    "minilm/b4",
    "distilgpt2/prefill",
    "distilgpt2/decode16",
];

/// Repeats the reference batch until it holds `batch_size` images.
fn images(name: &str, batch_size: usize, device: &Device) -> Tensor<4> {
    let reference = Tensor::<4>::from_data(Reference::load(name).get("input"), device);
    let [available, ..] = reference.dims();
    let repeats = batch_size.div_ceil(available);

    reference.repeat_dim(0, repeats).narrow(0, 0, batch_size)
}

/// Loads the model of `scenario` and returns one request on it.
fn load(scenario: &str, device: &Device) -> Request {
    let device = device.clone();
    match scenario {
        "resnet50/b1" | "resnet50/b8" => {
            let batch_size = if scenario.ends_with("b1") { 1 } else { 8 };
            let model = ResNet50::pretrained(&device);
            let input = images("resnet50", batch_size, &device);
            Box::new(move || drop(model.forward(input.clone()).into_data()))
        }
        "mobilenet_v2/b1" | "mobilenet_v2/b16" => {
            let batch_size = if scenario.ends_with("b1") { 1 } else { 16 };
            let model = MobileNetV2::pretrained(&device);
            let input = images("mobilenet_v2", batch_size, &device);
            Box::new(move || drop(model.forward(input.clone()).into_data()))
        }
        "minilm/b4" => {
            let reference = Reference::load("minilm");
            let model = MiniLm::pretrained(&device);
            let ids = Tensor::<2, Int>::from_data(reference.get("input_ids"), &device);
            let types = Tensor::<2, Int>::from_data(reference.get("token_type_ids"), &device);
            let mask: Tensor<2, Bool> =
                Tensor::<2, Int>::from_data(reference.get("attention_mask"), &device).equal_elem(0);
            Box::new(move || {
                let hidden = model.forward(ids.clone(), types.clone(), mask.clone());
                drop(model.embed(hidden, mask.clone()).into_data());
            })
        }
        "distilgpt2/prefill" => {
            let model = DistilGpt2::pretrained(&device);
            let prompt =
                Tensor::<2, Int>::from_data(Reference::load("distilgpt2").get("prompt"), &device);
            Box::new(move || drop(model.forward(prompt.clone()).into_data()))
        }
        "distilgpt2/decode16" => {
            let model = DistilGpt2::pretrained(&device);
            let prompt = Reference::load("distilgpt2")
                .get("prompt")
                .try_into_vec::<i64>()
                .expect("Tokens should be integers");
            Box::new(move || drop(model.generate_cached(&prompt, 16, &device)))
        }
        _ => panic!("Unknown scenario {scenario}, expected one of {SCENARIOS:?}"),
    }
}

/// The peak resident memory of the process, in mebibytes.
fn peak_memory_mib() -> f64 {
    // SAFETY: `getrusage` writes into the zeroed struct it is given.
    let usage = unsafe {
        let mut usage = core::mem::zeroed::<libc::rusage>();
        libc::getrusage(libc::RUSAGE_SELF, &mut usage);
        usage
    };
    // Linux reports kibibytes, macOS bytes.
    let bytes = if cfg!(target_os = "macos") {
        usage.ru_maxrss as f64
    } else {
        usage.ru_maxrss as f64 * 1024.0
    };
    bytes / (1024.0 * 1024.0)
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

struct Options {
    scenarios: Vec<String>,
    warmup: usize,
    iterations: usize,
    json: Option<String>,
    trace: bool,
}

impl Options {
    fn parse() -> Self {
        let mut options = Self {
            scenarios: SCENARIOS.iter().map(ToString::to_string).collect(),
            warmup: 3,
            iterations: 20,
            json: None,
            trace: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .unwrap_or_else(|| panic!("{arg} expects a value"))
            };
            match arg.as_str() {
                "--scenario" => options.scenarios = value().split(',').map(String::from).collect(),
                "--warmup" => options.warmup = value().parse().expect("--warmup is a count"),
                "--iterations" => {
                    options.iterations = value().parse().expect("--iterations is a count")
                }
                "--json" => options.json = Some(value()),
                "--trace" => options.trace = true,
                "--list" => {
                    SCENARIOS.iter().for_each(|scenario| println!("{scenario}"));
                    exit(0);
                }
                _ => panic!("Unknown argument {arg}"),
            }
        }
        options
    }
}

fn main() {
    let options = Options::parse();
    let device = device();
    let backend = backend_name();
    let mut entries = Vec::new();

    println!(
        "{:<22} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "scenario", "load", "first", "p50", "p90", "max", "peak MiB"
    );
    for scenario in &options.scenarios {
        let start = Instant::now();
        let mut request = load(scenario, &device);
        device.sync().expect("Loading should finish");
        let load = milliseconds(start.elapsed());

        let start = Instant::now();
        request();
        let first = milliseconds(start.elapsed());
        for _ in 0..options.warmup {
            request();
        }

        let mut latencies = (0..options.iterations)
            .map(|_| {
                let start = Instant::now();
                request();
                milliseconds(start.elapsed())
            })
            .collect::<Vec<_>>();
        if options.trace {
            println!("{scenario} latencies (ms): {latencies:.2?}");
        }
        latencies.sort_by(f64::total_cmp);
        let memory = peak_memory_mib();

        if latencies.is_empty() {
            println!("{scenario:<22} {load:>9.1} {first:>9.2}");
            continue;
        }
        let [p50, p90, max] = [0.5, 0.9, 1.0].map(|quantile| percentile(&latencies, quantile));
        println!(
            "{scenario:<22} {load:>9.1} {first:>9.2} {p50:>9.2} {p90:>9.2} {max:>9.2} {memory:>9.0}"
        );

        let name = |metric: &str| format!("{backend} {scenario} {metric}");
        entries.extend([
            serde_json::json!({ "name": name("p50"), "unit": "ms", "value": p50 }),
            serde_json::json!({ "name": name("p90"), "unit": "ms", "value": p90 }),
            serde_json::json!({ "name": name("first request"), "unit": "ms", "value": first }),
            serde_json::json!({ "name": name("load"), "unit": "ms", "value": load }),
            serde_json::json!({ "name": name("peak memory"), "unit": "MiB", "value": memory }),
        ]);
    }

    if let Some(path) = options.json {
        let json = serde_json::to_string_pretty(&entries).expect("Entries should serialize");
        std::fs::write(&path, json)
            .unwrap_or_else(|error| panic!("{path} should be writable: {error}"));
    }
    exit(0);
}

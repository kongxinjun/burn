//! Differential fuzzing of inference programs: random chains of tensor and module operations run
//! on the device under test and compared with the Flex reference.
//!
//! ```sh
//! cargo run --release --bin lab-fuzz --features metal,fusion -- --seconds 60
//! cargo run --release --bin lab-fuzz --features metal,fusion -- --seed 1234 --programs 1
//! ```
//!
//! Each program runs in f32, f16 or bf16, several times on the device under test: lazily,
//! synchronizing once so fusion sees the whole chain; eagerly, synchronizing after every operation;
//! on another batch size, when no operation depends on it; and lazily again, once the device has
//! cached kernels, tuning results and memory for it. A mismatch is reduced to the shortest failing
//! prefix and written to `--findings` as JSON.

mod program;

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use burn::tensor::{Device, FloatDType, Tensor};
use burn_inference_tests::harness::{ACCELERATED, backend_name, device, exit};

use program::{DTYPES, Input, Mode, Op, Program, Reference, dtype_name, execute};

/// A run of a program on the device under test that disagrees with the reference.
struct Failure {
    mode: Mode,
    input: Input,
    reference: Reference,
    /// The device's outputs, unless it panicked.
    outputs: Option<Vec<f32>>,
    detail: String,
    panicked: bool,
}

enum Outcome {
    Passed,
    Invalid,
    Failed(Box<Failure>),
}

/// A hash of `message` without its numbers, which vary between runs.
fn message_hash(message: &str) -> u32 {
    let first_lines = message.lines().take(12).collect::<String>();
    first_lines
        .bytes()
        .filter(|byte| !byte.is_ascii_digit())
        .fold(0x811c_9dc5u32, |hash, byte| {
            (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
        })
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(ToString::to_string))
        .unwrap_or_else(|| "unknown panic".to_string())
}

fn check(program: &Program, device: &Device) -> Outcome {
    let Some(reference) = Reference::compute(program, &program.input) else {
        return Outcome::Invalid;
    };
    let resized = program
        .resized
        .as_ref()
        .and_then(|input| Some((input, Reference::compute(program, input)?)));

    let mut runs = vec![
        (Mode::Lazy, &program.input, &reference),
        (Mode::Eager, &program.input, &reference),
    ];
    runs.extend(
        resized
            .as_ref()
            .map(|(input, reference)| (Mode::Resized, *input, reference)),
    );
    runs.push((Mode::Again, &program.input, &reference));

    for (mode, input, reference) in runs {
        let failure = |outputs, detail, panicked| {
            Outcome::Failed(Box::new(Failure {
                mode,
                input: input.clone(),
                reference: reference.clone(),
                outputs,
                detail,
                panicked,
            }))
        };
        let outputs = match catch_unwind(AssertUnwindSafe(|| {
            execute(input, &program.ops, device, program.dtype, mode)
        })) {
            Ok(outputs) => outputs,
            Err(payload) => return failure(None, panic_message(payload.as_ref()), true),
        };
        if outputs.len() != reference.expected.len() {
            let detail = format!(
                "{} elements, expected {}",
                outputs.len(),
                reference.expected.len()
            );
            return failure(Some(outputs), detail, true);
        }
        if let Some((index, actual, expected)) = reference.violation(&outputs) {
            let detail = format!("element {index}: {actual} instead of {expected}");
            return failure(Some(outputs), detail, false);
        }
    }
    Outcome::Passed
}

struct Options {
    seed: u64,
    programs: Option<u64>,
    duration: Duration,
    max_ops: usize,
    dtypes: Vec<FloatDType>,
    findings: PathBuf,
}

impl Options {
    fn parse() -> Self {
        let mut options = Self {
            seed: 0,
            programs: None,
            duration: Duration::from_secs(60),
            max_ops: 12,
            dtypes: DTYPES.map(|(_, dtype)| dtype).to_vec(),
            findings: PathBuf::from("findings"),
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .unwrap_or_else(|| panic!("{arg} expects a value"))
            };
            match arg.as_str() {
                "--seed" => options.seed = value().parse().expect("--seed is an integer"),
                "--programs" => {
                    options.programs = Some(value().parse().expect("--programs is a count"))
                }
                "--seconds" => {
                    options.duration =
                        Duration::from_secs(value().parse().expect("--seconds is a count"))
                }
                "--max-ops" => options.max_ops = value().parse().expect("--max-ops is a count"),
                "--dtypes" => {
                    options.dtypes = value()
                        .split(',')
                        .map(|name| {
                            DTYPES
                                .iter()
                                .find(|(candidate, _)| *candidate == name)
                                .map(|(_, dtype)| *dtype)
                                .unwrap_or_else(|| panic!("--dtypes takes f32, f16 and bf16"))
                        })
                        .collect()
                }
                "--findings" => options.findings = PathBuf::from(value()),
                _ => panic!("Unknown argument {arg}"),
            }
        }
        options
    }
}

fn report(
    options: &Options,
    seed: u64,
    program: &Program,
    device: &Device,
    failure: Failure,
) -> String {
    // The shortest prefix that still fails is the reproduction.
    let (program, failure) = (1..=program.ops.len())
        .find_map(|length| {
            let prefix = program.prefix(length);
            match check(&prefix, device) {
                Outcome::Failed(failure) => Some((prefix, *failure)),
                Outcome::Passed | Outcome::Invalid => None,
            }
        })
        .unwrap_or_else(|| (program.clone(), failure));
    let culprit = program.ops.last().expect("Programs have operations");
    let backend = backend_name();
    let dtype = dtype_name(program.dtype);
    let mode = failure.mode;
    let kind = if failure.panicked {
        "panic"
    } else {
        "mismatch"
    };
    // Panics share their cause across operations, so they are told apart by message.
    let subject = match failure.panicked {
        true => format!("{:08x}", message_hash(&failure.detail)),
        false => culprit.kind(),
    };
    // Only failures specific to a lower precision name it, so a bug found in every dtype gets one
    // issue, and f32 findings keep the fingerprints they had before other dtypes were fuzzed.
    let in_f32 = Program {
        dtype: FloatDType::F32,
        ..program.clone()
    };
    let precision = match program.dtype == FloatDType::F32
        || matches!(check(&in_f32, device), Outcome::Failed(_))
    {
        true => String::new(),
        false => format!("-{dtype}"),
    };
    let fingerprint = format!("{backend}{precision}-{mode:?}-{kind}-{subject}")
        .to_lowercase()
        .replace('+', "-");
    let features =
        std::env::var("LAB_FEATURES").unwrap_or_else(|_| "<backend features>".to_string());

    let finding = serde_json::json!({
        "fingerprint": fingerprint,
        "backend": backend,
        "dtype": dtype,
        "burn_sha": std::env::var("LAB_BURN_SHA").unwrap_or_else(|_| "local".to_string()),
        "mode": format!("{mode:?}"),
        "failure": kind,
        "detail": failure.detail,
        "seed": seed,
        "input_shape": failure.input.shape,
        "culprit": culprit.describe(),
        "program": program.ops.iter().map(Op::describe).collect::<Vec<_>>(),
        "reproduce": format!("cargo run --release --bin lab-fuzz --features {features} -- --seed {seed} --programs 1 --max-ops {} --dtypes {dtype}", options.max_ops),
    });
    std::fs::create_dir_all(&options.findings).expect("The findings directory should be writable");
    let judged = serde_json::json!({
        "dtype": dtype,
        "slack": failure.reference.slack,
        "input_shape": failure.input.shape,
        "input": failure.input.data,
        "ops": program.ops,
        "flex": failure.reference.expected,
        "device": failure.outputs,
    });
    std::fs::write(
        options.findings.join(format!("{fingerprint}.program.json")),
        serde_json::to_string(&judged).expect("Programs serialize"),
    )
    .expect("The program should be writable");
    let path = options.findings.join(format!("{fingerprint}.json"));
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&finding).expect("Findings serialize"),
    )
    .expect("The finding should be writable");
    println!("FINDING {fingerprint} (seed {seed}, {})", failure.detail);
    fingerprint
}

fn main() {
    let mut options = Options::parse();
    if !ACCELERATED {
        eprintln!("Fuzzing compares another backend with Flex: enable a backend feature");
        exit(2);
    }
    let device = device();
    let probe = catch_unwind(AssertUnwindSafe(|| {
        Tensor::<1>::ones([4], &device).into_data()
    }));
    if let Err(payload) = probe {
        eprintln!(
            "The device is unavailable: {}",
            panic_message(payload.as_ref())
        );
        exit(3);
    }
    options.dtypes.retain(|&dtype| {
        let supported = device.supports_dtype(dtype);
        if !supported {
            println!(
                "Skipping {}: the device does not support it",
                dtype_name(dtype)
            );
        }
        supported
    });
    if options.dtypes.is_empty() {
        eprintln!("The device supports none of the requested dtypes");
        exit(3);
    }

    let start = Instant::now();
    let mut fingerprints = std::collections::BTreeSet::new();
    // Passed, failed and rejected programs per dtype.
    let mut counts = BTreeMap::<&str, [u64; 3]>::new();
    let mut resized = 0u64;

    // Panics are expected while reducing a failure; reports carry the messages.
    std::panic::set_hook(Box::new(|_| {}));
    for seed in options.seed.. {
        let done = match options.programs {
            Some(programs) => seed - options.seed >= programs,
            None => start.elapsed() >= options.duration,
        };
        if done {
            break;
        }
        let program = Program::generate(seed, options.max_ops, &options.dtypes);
        let count = counts.entry(dtype_name(program.dtype)).or_default();
        resized += u64::from(program.resized.is_some());
        match check(&program, &device) {
            Outcome::Passed => count[0] += 1,
            Outcome::Invalid => count[2] += 1,
            Outcome::Failed(failure) => {
                count[1] += 1;
                fingerprints.insert(report(&options, seed, &program, &device, *failure));
            }
        }
    }
    let _ = std::panic::take_hook();

    let [passed, failed, invalid] = counts.values().fold([0; 3], |total, count| {
        [0, 1, 2].map(|i| total[i] + count[i])
    });
    println!(
        "{} programs in {:.0?}: {passed} passed, {failed} failed, {invalid} rejected by Flex; {} distinct findings; {resized} also ran on another batch size",
        passed + failed + invalid,
        start.elapsed(),
        fingerprints.len()
    );
    for (dtype, [passed, failed, invalid]) in &counts {
        println!("  {dtype}: {passed} passed, {failed} failed, {invalid} rejected");
    }
    exit(i32::from(failed > 0));
}

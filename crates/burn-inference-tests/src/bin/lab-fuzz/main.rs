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
mod session;

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use burn::tensor::{Device, FloatDType, Tensor};
use burn_inference_tests::harness::{ACCELERATED, backend_name, device, exit, isolate_environment};

use program::{
    DTYPES, Disagreement, Input, Mode, Op, Program, Reference, dtype_name, judge, judged_program,
    message_hash, panic_message,
};
use session::{Case, Recorder, Replay, Verdict};

/// A run of a program on the device under test that disagrees with the reference.
struct Failure {
    mode: Mode,
    input: Input,
    reference: Reference,
    disagreement: Disagreement,
}

enum Outcome {
    Passed,
    Invalid,
    Failed(Box<Failure>),
}

/// Runs `program` in every mode, recording each run, until one disagrees with Flex.
fn check(program: &Program, device: &Device, recorder: &mut Recorder) -> Outcome {
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
        recorder.record(program, mode);
        if let Err(disagreement) = judge(program, input, reference, device, mode) {
            return Outcome::Failed(Box::new(Failure {
                mode,
                input: input.clone(),
                reference: reference.clone(),
                disagreement,
            }));
        }
    }
    Outcome::Passed
}

struct Options {
    /// CubeCL's environment for the run, by default a new one in the findings directory.
    environment: Option<PathBuf>,
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
            environment: None,
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
                "--environment" => options.environment = Some(PathBuf::from(value())),
                _ => panic!("Unknown argument {arg}"),
            }
        }
        options
    }
}

/// Writes a finding for `failure` of `program`, whose session so far is `case`.
fn report(
    options: &Options,
    seed: u64,
    program: &Program,
    device: &Device,
    recorder: &mut Recorder,
    failure: Failure,
    case: Case,
) -> String {
    // The shortest prefix that still fails is the reproduction, with the session up to its failure.
    let (program, failure, case) = (1..=program.ops.len())
        .find_map(|length| {
            let prefix = program.prefix(length);
            match check(&prefix, device, recorder) {
                Outcome::Failed(failure) => {
                    let case = recorder.case(&prefix, options.max_ops, &options.dtypes);
                    Some((prefix, *failure, case))
                }
                Outcome::Passed | Outcome::Invalid => None,
            }
        })
        .unwrap_or_else(|| (program.clone(), failure, case));
    let culprit = program.ops.last().expect("Programs have operations");
    let backend = backend_name();
    let dtype = dtype_name(program.dtype);
    let mode = failure.mode;
    let kind = if failure.disagreement.panicked {
        "panic"
    } else {
        "mismatch"
    };
    // Panics share their cause across operations, so they are told apart by message.
    let subject = match failure.disagreement.panicked {
        true => format!("{:08x}", message_hash(&failure.disagreement.detail)),
        false => culprit.kind(),
    };
    // Only failures specific to a lower precision name it, so a bug found in every dtype gets one
    // issue, and f32 findings keep the fingerprints they had before other dtypes were fuzzed.
    let in_f32 = Program {
        dtype: FloatDType::F32,
        ..program.clone()
    };
    let precision = match program.dtype == FloatDType::F32
        || matches!(check(&in_f32, device, recorder), Outcome::Failed(_))
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
        "detail": failure.disagreement.detail,
        "seed": seed,
        "input_shape": failure.input.shape,
        "culprit": culprit.describe(),
        "program": program.ops.iter().map(Op::describe).collect::<Vec<_>>(),
        "reproduce": format!("cargo run --release --bin lab-fuzz --features {features} -- --seed {seed} --programs 1 --max-ops {} --dtypes {dtype}", options.max_ops),
    });
    std::fs::create_dir_all(&options.findings).expect("The findings directory should be writable");
    let judged = judged_program(
        &program,
        &failure.input,
        &failure.reference,
        failure.disagreement.outputs.as_deref(),
    );
    std::fs::write(
        options.findings.join(format!("{fingerprint}.program.json")),
        serde_json::to_string(&judged).expect("Programs serialize"),
    )
    .expect("The program should be writable");
    std::fs::write(
        options.findings.join(format!("{fingerprint}.case.json")),
        serde_json::to_string(&case).expect("Cases serialize"),
    )
    .expect("The case should be writable");
    let path = options.findings.join(format!("{fingerprint}.json"));
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&finding).expect("Findings serialize"),
    )
    .expect("The finding should be writable");
    println!(
        "FINDING {fingerprint} (seed {seed}, {})",
        failure.disagreement.detail
    );
    fingerprint
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("replay") => replay_command(&args[1..]),
        Some("reduce") => reduce_command(&args[1..]),
        _ => fuzz(Options::parse()),
    }
}

/// The device under test, once it has run an operation.
fn open_device() -> Device {
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
    device
}

/// A new directory for CubeCL's environment, holding a copy of `source` when given: replays write
/// tuning results, which must not reach the next replay.
fn scratch_environment(source: Option<&Path>) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("The clock is after the epoch")
        .as_nanos();
    let scratch =
        std::env::temp_dir().join(format!("lab-environment-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("The scratch environment should be writable");
    if let Some(source) = source {
        let entries = std::fs::read_dir(source)
            .unwrap_or_else(|error| panic!("{} should be readable: {error}", source.display()));
        for entry in entries {
            let path = entry.expect("Environment entries are readable").path();
            // The shared-memory index is rebuilt from the log when the database opens.
            if path.is_file() && !path.to_string_lossy().ends_with("-shm") {
                let name = path.file_name().expect("Files have names");
                std::fs::copy(&path, scratch.join(name)).expect("The environment should copy");
            }
        }
    }
    scratch
}

/// The environment to replay `case` in: `requested`, unless it is `none`, or else the case's own.
fn replay_environment(case: &Case, case_path: &Path, requested: Option<&str>) -> Option<PathBuf> {
    match requested {
        Some("none") => None,
        Some(path) => Some(PathBuf::from(path)),
        None => case.environment.as_ref().map(|environment| {
            case_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(environment)
        }),
    }
}

fn read_case(path: &Path) -> Case {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()));
    serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("{} should hold a case: {error}", path.display()))
}

fn write_json(path: &Path, value: &impl serde::Serialize) {
    let json = serde_json::to_string(value).expect("Values serialize");
    std::fs::write(path, json)
        .unwrap_or_else(|error| panic!("{} should be writable: {error}", path.display()));
}

/// `replay <case> [--program <path>] [--environment <directory>|none]`: replays a case in this
/// process, starting from a copy of its CubeCL environment, and prints what its last run did as
/// `REPLAY <json>`. `--program` writes that run for `ci/oracle.py`; `--environment` replaces the
/// case's environment, or starts from an empty one. Exits with 0 when the run passes, 1 when it
/// fails and 2 when Flex rejects it.
fn replay_command(args: &[String]) -> ! {
    let mut args = args.iter();
    let case_path = PathBuf::from(args.next().expect("replay expects a case"));
    let case = read_case(&case_path);
    let (mut program_path, mut requested) = (None, None);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("{arg} expects a value"))
        };
        match arg.as_str() {
            "--program" => program_path = Some(value()),
            "--environment" => requested = Some(value().as_str()),
            _ => panic!("Unknown argument {arg}"),
        }
    }
    let environment = replay_environment(&case, &case_path, requested);
    let scratch = scratch_environment(environment.as_deref());
    isolate_environment(&scratch);
    let device = open_device();
    std::panic::set_hook(Box::new(|_| {}));
    let (replay, judged) = session::replay(&case, &device);
    if let (Some(path), Some(judged)) = (program_path, judged) {
        write_json(Path::new(path), &judged);
    }
    let _ = std::fs::remove_dir_all(&scratch);
    println!(
        "REPLAY {}",
        serde_json::to_string(&replay).expect("Replays serialize")
    );
    exit(match replay.verdict {
        Verdict::Passed => 0,
        Verdict::Failed | Verdict::Crashed => 1,
        Verdict::Invalid => 2,
    })
}

/// Replays `case` in a new process, so it starts from a fresh device.
fn replay_elsewhere(case: &Case) -> Replay {
    let path = std::env::temp_dir().join(format!("lab-fuzz-{}.case.json", std::process::id()));
    write_json(&path, case);
    let executable = std::env::current_exe().expect("The fuzzer knows its executable");
    let output = Command::new(executable)
        .arg("replay")
        .arg(&path)
        .output()
        .expect("The fuzzer should start a replay");
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("REPLAY "))
        .and_then(|json| serde_json::from_str(json).ok())
        .unwrap_or_else(|| {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail = stderr.lines().rev().take(5).collect::<Vec<_>>();
            Replay {
                verdict: Verdict::Crashed,
                panicked: true,
                detail: format!("{}: {}", output.status, tail.join(" / ")),
            }
        })
}

/// `reduce <case> --out <path> [--tries <n>] [--environment <directory>|none]`: shrinks a case to
/// the runs, operations and values its failure needs, replaying each candidate in a new process up
/// to `--tries` times. Every replay starts from a copy of the same environment.
fn reduce_command(args: &[String]) -> ! {
    let mut args = args.iter();
    let case_path = PathBuf::from(args.next().expect("reduce expects a case"));
    let mut case = read_case(&case_path);
    let (mut out, mut tries, mut requested) = (None, 2, None);
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("{arg} expects a value"))
        };
        match arg.as_str() {
            "--out" => out = Some(PathBuf::from(value())),
            "--tries" => tries = value().parse().expect("--tries is a count"),
            "--environment" => requested = Some(value().as_str()),
            _ => panic!("Unknown argument {arg}"),
        }
    }
    let out = out.expect("reduce expects --out");
    // Candidates are written elsewhere, so they name the environment by its absolute path.
    case.environment = replay_environment(&case, &case_path, requested).map(|environment| {
        std::path::absolute(&environment).expect("The environment has an absolute path")
    });

    let expected = replay_elsewhere(&case);
    if !matches!(expected.verdict, Verdict::Failed | Verdict::Crashed) {
        println!("The case does not reproduce: {expected:?}");
        std::process::exit(3);
    }
    let start = Instant::now();
    let mut replays = 0;
    let reduced = session::reduce(case.clone(), |candidate| {
        (0..tries).any(|_| {
            replays += 1;
            replay_elsewhere(candidate).fails_like(&expected)
        })
    });
    write_json(&out, &reduced);
    let programs = reduced.programs();
    let target = &programs[reduced.runs.last().expect("Cases have a run").program];
    println!(
        "Reduced in {replays} replays and {:.0?}: {} runs to {}, {} operations, {} ({})",
        start.elapsed(),
        case.runs.len(),
        reduced.runs.len(),
        target.ops.len(),
        target
            .ops
            .iter()
            .map(Op::describe)
            .collect::<Vec<_>>()
            .join(" -> "),
        expected.detail
    );
    std::process::exit(0)
}

fn fuzz(mut options: Options) -> ! {
    // A run starts from an empty environment unless told otherwise, so its findings hold every
    // tuning result they depend on.
    let environment = options.environment.clone().unwrap_or_else(|| {
        let environment = options
            .findings
            .join(format!("environment-{}", options.seed));
        let _ = std::fs::remove_dir_all(&environment);
        environment
    });
    std::fs::create_dir_all(&environment).expect("The environment should be writable");
    isolate_environment(&environment);
    let device = open_device();
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
    // Cases sit in the findings directory and name the environment relative to it.
    let environment = std::path::absolute(&environment).expect("Paths can be made absolute");
    let relative = std::path::absolute(&options.findings)
        .ok()
        .and_then(|findings| environment.strip_prefix(findings).ok())
        .map_or_else(|| environment.clone(), Path::to_path_buf);
    let mut recorder = Recorder::new(Some(relative));

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
        match check(&program, &device, &mut recorder) {
            Outcome::Passed => count[0] += 1,
            Outcome::Invalid => count[2] += 1,
            Outcome::Failed(failure) => {
                count[1] += 1;
                let case = recorder.case(&program, options.max_ops, &options.dtypes);
                fingerprints.insert(report(
                    &options,
                    seed,
                    &program,
                    &device,
                    &mut recorder,
                    *failure,
                    case,
                ));
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

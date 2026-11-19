//! Sessions: every run a fuzzing process made on the device under test, in order.
//!
//! A failure can depend on what ran before it in the process: kernels compiled and tuning results
//! cached for other shapes, or memory reused from earlier tensors. Such a failure only reproduces
//! with its session, so each finding carries one as a `Case`, and `reduce` shrinks it to the runs,
//! operations and values the failure needs.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use burn::tensor::{Device, FloatDType};
use burn_inference_tests::harness::backend_name;
use serde::{Deserialize, Serialize};

use crate::program::{
    Input, Mode, Origin, Program, Reference, dtype_by_name, dtype_name, execute, judge,
    judged_program, message_hash, parse_dtype,
};

/// One run of a program on the device under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Run {
    /// The index of the program in `Case::programs`.
    pub(crate) program: usize,
    /// Whether it runs on the input with another batch size.
    pub(crate) resized: bool,
    /// Whether it synchronizes after every operation.
    pub(crate) eager: bool,
}

impl Run {
    fn mode(&self) -> Mode {
        match (self.eager, self.resized) {
            (true, _) => Mode::Eager,
            (false, true) => Mode::Resized,
            (false, false) => Mode::Lazy,
        }
    }

    fn input<'a>(&self, program: &'a Program) -> &'a Input {
        match self.resized {
            true => program
                .resized
                .as_ref()
                .expect("Resized runs are of programs with another batch size"),
            false => &program.input,
        }
    }
}

/// A program of a case: by seed while a session is recorded, in full once it is reduced.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Source {
    /// `Program::generate` with `seed`, cut to `length` operations and run in `dtype`.
    Seed {
        seed: u64,
        length: usize,
        #[serde(with = "dtype_by_name")]
        dtype: FloatDType,
    },
    Program(Program),
}

/// A failure with the session it happened in: the last run disagrees with Flex.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Case {
    pub(crate) backend: String,
    /// The options programs were generated with, to regenerate them from their seeds.
    pub(crate) max_ops: usize,
    pub(crate) dtypes: Vec<String>,
    pub(crate) programs: Vec<Source>,
    pub(crate) runs: Vec<Run>,
}

impl Case {
    /// Every program of the case, regenerated from its seed when needed.
    pub(crate) fn programs(&self) -> Vec<Program> {
        let dtypes = self
            .dtypes
            .iter()
            .map(|name| parse_dtype(name).expect("Cases name dtypes of DTYPES"))
            .collect::<Vec<_>>();
        self.programs
            .iter()
            .map(|source| match source {
                Source::Seed {
                    seed,
                    length,
                    dtype,
                } => {
                    let mut program =
                        Program::generate(*seed, self.max_ops, &dtypes).prefix(*length);
                    program.dtype = *dtype;
                    program
                }
                Source::Program(program) => program.clone(),
            })
            .collect()
    }

    fn last(&self) -> Run {
        *self.runs.last().expect("Cases end with the failing run")
    }

    /// The case with only the runs of `history` before its last one.
    fn with_history(&self, history: &[Run]) -> Self {
        let mut case = self.clone();
        case.runs = history.iter().copied().chain([self.last()]).collect();
        case
    }

    /// The case with its programs in full, without those no run uses.
    fn compacted(&self) -> Self {
        let programs = self.programs();
        let mut indices = HashMap::new();
        let mut kept = Vec::new();
        let runs = self
            .runs
            .iter()
            .map(|run| {
                let program = *indices.entry(run.program).or_insert_with(|| {
                    kept.push(Source::Program(programs[run.program].clone()));
                    kept.len() - 1
                });
                Run { program, ..*run }
            })
            .collect();
        Self {
            programs: kept,
            runs,
            ..self.clone()
        }
    }

    fn with_program(&self, index: usize, program: Program) -> Self {
        let mut case = self.clone();
        case.programs[index] = Source::Program(program);
        case
    }
}

/// Records the runs of a fuzzing process, by seed.
#[derive(Default)]
pub(crate) struct Recorder {
    programs: Vec<Source>,
    known: HashMap<(Origin, &'static str), usize>,
    runs: Vec<Run>,
}

impl Recorder {
    pub(crate) fn record(&mut self, program: &Program, mode: Mode) {
        let programs = &mut self.programs;
        let index = match program.origin {
            Some(origin) => *self
                .known
                .entry((origin, dtype_name(program.dtype)))
                .or_insert_with(|| {
                    programs.push(Source::Seed {
                        seed: origin.seed,
                        length: origin.length,
                        dtype: program.dtype,
                    });
                    programs.len() - 1
                }),
            None => {
                programs.push(Source::Program(program.clone()));
                programs.len() - 1
            }
        };
        self.runs.push(Run {
            program: index,
            resized: mode == Mode::Resized,
            eager: mode == Mode::Eager,
        });
    }

    /// The session so far as a case, ending with the run of `program` just recorded.
    pub(crate) fn case(&self, program: &Program, max_ops: usize, dtypes: &[FloatDType]) -> Case {
        let mut programs = self.programs.clone();
        let last = self.runs.last().expect("A failure follows a run");
        programs[last.program] = Source::Program(program.clone());
        Case {
            backend: backend_name(),
            max_ops,
            dtypes: dtypes.iter().map(|d| dtype_name(*d).to_string()).collect(),
            programs,
            runs: self.runs.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Verdict {
    Passed,
    Failed,
    /// Flex rejects the program.
    Invalid,
    /// The replaying process died without reporting.
    Crashed,
}

/// What the last run of a replayed case did.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Replay {
    pub(crate) verdict: Verdict,
    pub(crate) panicked: bool,
    pub(crate) detail: String,
}

impl Replay {
    /// Whether `self` fails the way `expected` does: a panic with the same message, or a mismatch.
    pub(crate) fn fails_like(&self, expected: &Replay) -> bool {
        match (self.verdict, expected.verdict) {
            (Verdict::Failed, Verdict::Failed) => {
                self.panicked == expected.panicked
                    && (!self.panicked
                        || message_hash(&self.detail) == message_hash(&expected.detail))
            }
            (Verdict::Crashed, Verdict::Crashed) => true,
            _ => false,
        }
    }
}

/// Replays `case` on `device`, and compares its last run with Flex. Also returns that run in the
/// format `ci/oracle.py` judges, when Flex accepts the program.
pub(crate) fn replay(case: &Case, device: &Device) -> (Replay, Option<serde_json::Value>) {
    let programs = case.programs();
    let (last, history) = case.runs.split_last().expect("Cases have a run");
    let target = &programs[last.program];
    let input = last.input(target);
    let Some(reference) = Reference::compute(target, input) else {
        let replay = Replay {
            verdict: Verdict::Invalid,
            panicked: false,
            detail: "Flex rejects the program".to_string(),
        };
        return (replay, None);
    };

    for run in history {
        let program = &programs[run.program];
        // Earlier runs only set up the device; their outputs and panics do not matter.
        let _ = catch_unwind(AssertUnwindSafe(|| {
            execute(
                run.input(program),
                &program.ops,
                device,
                program.dtype,
                run.mode(),
            )
        }));
    }

    let (replay, outputs) = match judge(target, input, &reference, device, last.mode()) {
        Ok(outputs) => {
            let replay = Replay {
                verdict: Verdict::Passed,
                panicked: false,
                detail: String::new(),
            };
            (replay, Some(outputs))
        }
        Err(disagreement) => {
            let replay = Replay {
                verdict: Verdict::Failed,
                panicked: disagreement.panicked,
                detail: disagreement.detail,
            };
            (replay, disagreement.outputs)
        }
    };
    let judged = judged_program(target, input, &reference, outputs.as_deref());
    (replay, Some(judged))
}

/// Shrinks `case` while `fails` holds for it: the runs before the last one, then the failing
/// program's operations, then its values.
pub(crate) fn reduce(case: Case, mut fails: impl FnMut(&Case) -> bool) -> Case {
    let history = &case.runs[..case.runs.len() - 1];
    let needed = if fails(&case.with_history(&[])) {
        Vec::new()
    } else {
        // State usually comes from recent runs: find the shortest failing suffix, then shrink it.
        let mut length = 1;
        let suffix = loop {
            let suffix = &history[history.len().saturating_sub(length)..];
            if length >= history.len() || fails(&case.with_history(suffix)) {
                break suffix;
            }
            length *= 2;
        };
        minimize(suffix.to_vec(), |runs| fails(&case.with_history(runs)))
    };
    let mut case = case.with_history(&needed).compacted();

    // Operations of the failing program, which earlier runs of it share.
    let target = case.last().program;
    let mut program = case.programs()[target].clone();
    loop {
        let mut removed = false;
        for index in (0..program.ops.len()).rev() {
            if program.ops.len() == 1 {
                break;
            }
            let mut candidate = program.clone();
            candidate.ops.remove(index);
            if candidate.is_valid() && fails(&case.with_program(target, candidate.clone())) {
                program = candidate;
                removed = true;
            }
        }
        if !removed {
            break;
        }
    }
    case = case.with_program(target, program);

    // Values: patterns for every program at once, or else one program at a time.
    let programs = case.programs();
    let mut patterned = case.clone();
    for (index, program) in programs.iter().enumerate() {
        patterned = patterned.with_program(index, program.patterned());
    }
    if fails(&patterned) {
        return patterned;
    }
    for (index, program) in programs.iter().enumerate().rev() {
        let candidate = case.with_program(index, program.patterned());
        if fails(&candidate) {
            case = candidate;
        }
    }
    case
}

/// Delta debugging: a smallest subsequence of `items` for which `fails` holds, removing chunks
/// while that keeps it failing.
fn minimize<T: Clone>(mut items: Vec<T>, mut fails: impl FnMut(&[T]) -> bool) -> Vec<T> {
    let mut parts = 2;
    while items.len() >= 2 {
        let chunk = items.len().div_ceil(parts);
        let complement = (0..items.len()).step_by(chunk).find_map(|start| {
            let end = (start + chunk).min(items.len());
            let complement = [&items[..start], &items[end..]].concat();
            fails(&complement).then_some(complement)
        });
        match complement {
            Some(complement) => {
                items = complement;
                parts = (parts - 1).max(2);
            }
            None if parts >= items.len() => break,
            None => parts = (parts * 2).min(items.len()),
        }
    }
    items
}

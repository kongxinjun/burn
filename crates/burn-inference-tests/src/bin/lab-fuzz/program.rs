//! Random programs: their operations, how they are generated and how they run.

use burn::tensor::activation::{gelu, log_softmax, relu, sigmoid, silu, softmax, tanh};
use burn::tensor::module::{
    CoordinateTransformMode, InterpolateMode, InterpolateOptions, adaptive_avg_pool2d, avg_pool2d,
    conv2d, interpolate, max_pool2d,
};
use burn::tensor::ops::{AvgPoolOptions, ConvOptions, MaxPoolOptions};
use burn::tensor::{DType, Device, FloatDType, Int, Tensor, TensorData, bf16, f16};
use serde::Serialize;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub(crate) const MAX_ELEMENTS: usize = 1 << 16;
pub(crate) const MAX_DIM: usize = 96;

/// The float dtypes programs run in, by name.
pub(crate) const DTYPES: [(&str, FloatDType); 3] = [
    ("f32", FloatDType::F32),
    ("f16", FloatDType::F16),
    ("bf16", FloatDType::BF16),
];

pub(crate) fn dtype_name(dtype: FloatDType) -> &'static str {
    DTYPES
        .iter()
        .find(|(_, candidate)| *candidate == dtype)
        .map(|(name, _)| *name)
        .expect("Programs run in a dtype of DTYPES")
}

/// The nearest value of `dtype` to `value`.
pub(crate) fn round_to(dtype: FloatDType, value: f32) -> f32 {
    match dtype {
        FloatDType::F16 => f16::from_f32(value).to_f32(),
        FloatDType::BF16 => bf16::from_f32(value).to_f32(),
        _ => value,
    }
}

/// A xorshift generator: programs must be reproducible from their seed alone.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    pub(crate) fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }

    pub(crate) fn range(&mut self, low: usize, high: usize) -> usize {
        low + self.below(high - low + 1)
    }

    pub(crate) fn chance(&mut self, probability: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= probability
    }

    pub(crate) fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub(crate) fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.unit() * scale).collect()
    }

    pub(crate) fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) enum Unary {
    Relu,
    Abs,
    Neg,
    Sigmoid,
    Tanh,
    Gelu,
    Silu,
    Sin,
    Exp,
    Log1pAbs,
    SqrtAbs,
    Square,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) enum Binary {
    Add,
    Sub,
    Mul,
    Max,
    Min,
    SafeDiv,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) enum Reduce {
    Sum,
    Mean,
    Max,
    Min,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) enum Op {
    Unary(Unary),
    Scalar {
        op: Binary,
        value: f32,
    },
    Broadcast {
        op: Binary,
        shape: [usize; 4],
        data: Vec<f32>,
    },
    Conv {
        channels_out: usize,
        kernel: [usize; 2],
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    },
    MaxPool {
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        ceil: bool,
    },
    AvgPool {
        kernel: usize,
        stride: usize,
        padding: usize,
        count_include_pad: bool,
        ceil: bool,
    },
    AdaptiveAvgPool {
        size: [usize; 2],
    },
    Interpolate {
        mode: usize,
        size: [usize; 2],
    },
    Permute([usize; 4]),
    Flip(usize),
    Narrow {
        dim: usize,
        start: usize,
        length: usize,
    },
    CatWithItself {
        dim: usize,
        other: Unary,
    },
    Repeat {
        dim: usize,
        times: usize,
    },
    Reduce {
        op: Reduce,
        dim: usize,
    },
    Softmax {
        dim: usize,
        log: bool,
    },
    Matmul {
        columns: usize,
        shape: [usize; 4],
        data: Vec<f32>,
    },
    Normalize {
        dim: usize,
    },
    MaskFill {
        threshold: f32,
        value: f32,
    },
    Clamp {
        min: f32,
        max: f32,
    },
    Select {
        dim: usize,
        indices: Vec<i64>,
    },
    Gather {
        dim: usize,
        shape: [usize; 4],
        indices: Vec<i64>,
    },
    Reshape([usize; 4]),
    Argmax {
        dim: usize,
    },
}

impl Op {
    pub(crate) fn kind(&self) -> String {
        let debug = format!("{self:?}");
        let end = debug.find([' ', '(', '{']).unwrap_or(debug.len());
        match self {
            Op::Unary(unary) => format!("Unary{unary:?}"),
            Op::Scalar { op, .. } | Op::Broadcast { op, .. } => format!("{}{op:?}", &debug[..end]),
            Op::Reduce { op, .. } => format!("Reduce{op:?}"),
            _ => debug[..end].to_string(),
        }
    }

    /// A description without the constant data, for reports.
    pub(crate) fn describe(&self) -> String {
        match self {
            Op::Broadcast { op, shape, .. } => {
                format!("Broadcast {{ op: {op:?}, shape: {shape:?} }}")
            }
            Op::Conv {
                channels_out,
                kernel,
                stride,
                padding,
                dilation,
                groups,
                bias,
                ..
            } => format!(
                "Conv {{ channels_out: {channels_out}, kernel: {kernel:?}, stride: {stride}, padding: {padding}, dilation: {dilation}, groups: {groups}, bias: {} }}",
                bias.is_some()
            ),
            Op::Matmul { columns, shape, .. } => {
                format!("Matmul {{ columns: {columns}, rhs: {shape:?} }}")
            }
            Op::Select { dim, indices } => {
                format!("Select {{ dim: {dim}, count: {} }}", indices.len())
            }
            Op::Gather { dim, shape, .. } => format!("Gather {{ dim: {dim}, shape: {shape:?} }}"),
            op => format!("{op:?}"),
        }
    }

    /// Replaces every float constant `v` with `f(v)`.
    pub(crate) fn map_constants(&mut self, f: impl Fn(f32) -> f32) {
        let map = |value: &mut f32| *value = f(*value);
        match self {
            Op::Scalar { value, .. } => map(value),
            Op::Broadcast { data, .. } | Op::Matmul { data, .. } => data.iter_mut().for_each(map),
            Op::Conv { weight, bias, .. } => weight
                .iter_mut()
                .chain(bias.iter_mut().flatten())
                .for_each(map),
            Op::MaskFill { threshold, value } => [threshold, value].into_iter().for_each(map),
            Op::Clamp { min, max } => [min, max].into_iter().for_each(map),
            _ => {}
        }
    }

    /// Whether the operation keeps the batch on axis 0 and holds no constant along it.
    pub(crate) fn batch_independent(&self) -> bool {
        match self {
            Op::Broadcast { shape, .. } | Op::Matmul { shape, .. } => shape[0] == 1,
            Op::Permute(axes) => axes[0] == 0,
            Op::Narrow { dim, .. } | Op::Select { dim, .. } => *dim != 0,
            Op::Gather { .. } | Op::Reshape(_) => false,
            _ => true,
        }
    }
}

pub(crate) fn numel(shape: &[usize; 4]) -> usize {
    shape.iter().product()
}

pub(crate) fn within_limits(shape: &[usize; 4]) -> bool {
    shape.iter().all(|dim| (1..=MAX_DIM * 4).contains(dim)) && numel(shape) <= MAX_ELEMENTS
}

pub(crate) fn pooled(
    size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    ceil: bool,
) -> Option<usize> {
    let span = dilation * (kernel - 1) + 1;
    let padded = size + 2 * padding;
    if padded < span {
        return None;
    }
    let steps = padded - span;
    Some(
        if ceil {
            steps.div_ceil(stride)
        } else {
            steps / stride
        } + 1,
    )
}

pub(crate) const UNARIES: [Unary; 12] = [
    Unary::Relu,
    Unary::Abs,
    Unary::Neg,
    Unary::Sigmoid,
    Unary::Tanh,
    Unary::Gelu,
    Unary::Silu,
    Unary::Sin,
    Unary::Exp,
    Unary::Log1pAbs,
    Unary::SqrtAbs,
    Unary::Square,
];
pub(crate) const BINARIES: [Binary; 6] = [
    Binary::Add,
    Binary::Sub,
    Binary::Mul,
    Binary::Max,
    Binary::Min,
    Binary::SafeDiv,
];

/// Draws an operation valid for `shape` in `dtype`, with the shape it produces.
pub(crate) fn generate_op(
    rng: &mut Rng,
    shape: [usize; 4],
    dtype: FloatDType,
) -> Option<(Op, [usize; 4])> {
    let [n, c, h, w] = shape;
    // Rounding flips comparisons and argmax, so lower precisions leave them out.
    let exact = dtype == FloatDType::F32;
    let op = match rng.below(23) {
        0 | 1 => Op::Unary(rng.pick(&UNARIES)),
        2 => Op::Scalar {
            op: rng.pick(&BINARIES),
            value: rng.unit() * 2.0,
        },
        3 | 4 => {
            let shape = shape.map(|dim| if rng.chance(0.5) { 1 } else { dim });
            Op::Broadcast {
                op: rng.pick(&BINARIES),
                shape,
                data: rng.values(numel(&shape), 1.0),
            }
        }
        5 | 6 => {
            let groups = rng.pick(&[1, 1, 1, c]).max(1);
            let channels_out = if groups == c {
                c * rng.range(1, 2)
            } else {
                rng.range(1, 24)
            };
            if channels_out % groups != 0 {
                return None;
            }
            let kernel = [rng.pick(&[1, 3, 3, 5]), rng.pick(&[1, 3, 3, 5])];
            let (stride, padding, dilation) = (rng.range(1, 2), rng.range(0, 2), rng.range(1, 2));
            let fan_in = (c / groups) * kernel[0] * kernel[1];
            let weight = rng.values(channels_out * fan_in, 1.0 / (fan_in as f32).sqrt());
            let bias = rng.chance(0.5).then(|| rng.values(channels_out, 0.5));
            Op::Conv {
                channels_out,
                kernel,
                stride,
                padding,
                dilation,
                groups,
                weight,
                bias,
            }
        }
        7 => {
            let kernel = rng.range(2, 3);
            Op::MaxPool {
                kernel,
                stride: rng.range(1, 2),
                padding: rng.range(0, kernel / 2),
                dilation: rng.range(1, 2),
                ceil: rng.chance(0.3),
            }
        }
        8 => {
            let kernel = rng.range(2, 3);
            Op::AvgPool {
                kernel,
                stride: rng.range(1, 2),
                padding: rng.range(0, kernel / 2),
                count_include_pad: rng.chance(0.5),
                ceil: rng.chance(0.3),
            }
        }
        9 => Op::AdaptiveAvgPool {
            size: [rng.range(1, h.max(1) + 2), rng.range(1, w.max(1) + 2)],
        },
        10 => Op::Interpolate {
            mode: rng.below(4),
            size: [rng.range(1, h * 2), rng.range(1, w * 2)],
        },
        11 => {
            let mut axes = [0, 1, 2, 3];
            for i in (1..4).rev() {
                axes.swap(i, rng.below(i + 1));
            }
            Op::Permute(axes)
        }
        12 => Op::Flip(rng.below(4)),
        13 => {
            let dim = rng.below(4);
            if shape[dim] < 2 {
                return None;
            }
            let length = rng.range(1, shape[dim] - 1);
            Op::Narrow {
                dim,
                start: rng.below(shape[dim] - length + 1),
                length,
            }
        }
        14 => Op::CatWithItself {
            dim: rng.below(4),
            other: rng.pick(&UNARIES),
        },
        15 => Op::Repeat {
            dim: rng.below(4),
            times: rng.range(2, 3),
        },
        16 => Op::Reduce {
            op: rng.pick(&[Reduce::Sum, Reduce::Mean, Reduce::Max, Reduce::Min]),
            dim: rng.below(4),
        },
        17 => Op::Softmax {
            dim: rng.below(4),
            log: rng.chance(0.3),
        },
        18 => {
            let columns = rng.range(1, 32);
            let shape = [
                if rng.chance(0.5) { 1 } else { n },
                if rng.chance(0.5) { 1 } else { c },
                w,
                columns,
            ];
            Op::Matmul {
                columns,
                shape,
                data: rng.values(numel(&shape), 1.0 / (w as f32).sqrt()),
            }
        }
        19 => Op::Normalize { dim: rng.below(4) },
        20 => match rng.chance(0.5) {
            true if !exact => return None,
            true => Op::MaskFill {
                threshold: rng.unit(),
                value: rng.unit() * 3.0,
            },
            false => Op::Clamp {
                min: -rng.unit().abs(),
                max: rng.unit().abs() + 0.1,
            },
        },
        21 => {
            let dim = rng.below(4);
            match rng.chance(0.5) {
                true => {
                    let count = rng.range(1, shape[dim] + 2);
                    Op::Select {
                        dim,
                        indices: (0..count).map(|_| rng.below(shape[dim]) as i64).collect(),
                    }
                }
                false => {
                    let mut gathered = shape;
                    gathered[dim] = rng.range(1, shape[dim] + 2);
                    let indices = (0..numel(&gathered))
                        .map(|_| rng.below(shape[dim]) as i64)
                        .collect();
                    Op::Gather {
                        dim,
                        shape: gathered,
                        indices,
                    }
                }
            }
        }
        _ => match rng.chance(0.7) {
            true => {
                let total = numel(&shape);
                let mut dims = [1; 4];
                let mut rest = total;
                for dim in dims.iter_mut().take(3) {
                    let divisors = (1..=rest.min(MAX_DIM))
                        .filter(|&d| rest.is_multiple_of(d))
                        .collect::<Vec<_>>();
                    *dim = rng.pick(&divisors);
                    rest /= *dim;
                }
                dims[3] = rest;
                Op::Reshape(dims)
            }
            false if !exact => return None,
            false => Op::Argmax { dim: rng.below(4) },
        },
    };

    let output = output_shape(&op, shape)?;
    within_limits(&output).then_some((op, output))
}

pub(crate) fn output_shape(op: &Op, shape: [usize; 4]) -> Option<[usize; 4]> {
    let [n, c, h, w] = shape;
    Some(match op {
        Op::Conv {
            channels_out,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            ..
        } => {
            if c % groups != 0 {
                return None;
            }
            [
                n,
                *channels_out,
                pooled(h, kernel[0], *stride, *padding, *dilation, false)?,
                pooled(w, kernel[1], *stride, *padding, *dilation, false)?,
            ]
        }
        Op::MaxPool {
            kernel,
            stride,
            padding,
            dilation,
            ceil,
        } => [
            n,
            c,
            pooled(h, *kernel, *stride, *padding, *dilation, *ceil)?,
            pooled(w, *kernel, *stride, *padding, *dilation, *ceil)?,
        ],
        Op::AvgPool {
            kernel,
            stride,
            padding,
            ceil,
            ..
        } => [
            n,
            c,
            pooled(h, *kernel, *stride, *padding, 1, *ceil)?,
            pooled(w, *kernel, *stride, *padding, 1, *ceil)?,
        ],
        Op::AdaptiveAvgPool { size } | Op::Interpolate { size, .. } => [n, c, size[0], size[1]],
        Op::Permute(axes) => axes.map(|axis| shape[axis]),
        Op::Narrow { dim, length, .. } => {
            let mut output = shape;
            output[*dim] = *length;
            output
        }
        Op::CatWithItself { dim, .. } | Op::Repeat { dim, times: 2 } => {
            let mut output = shape;
            output[*dim] *= 2;
            output
        }
        Op::Repeat { dim, times } => {
            let mut output = shape;
            output[*dim] *= times;
            output
        }
        Op::Reduce { dim, .. } | Op::Argmax { dim } => {
            let mut output = shape;
            output[*dim] = 1;
            output
        }
        Op::Matmul { columns, .. } => [n, c, h, *columns],
        Op::Select { dim, indices } => {
            let mut output = shape;
            output[*dim] = indices.len();
            output
        }
        Op::Gather {
            shape: gathered, ..
        } => *gathered,
        Op::Reshape(dims) => *dims,
        _ => shape,
    })
}

#[derive(Clone, Debug)]
pub(crate) struct Input {
    pub(crate) shape: [usize; 4],
    pub(crate) data: Vec<f32>,
}

/// Random operations applied to one input, in one float dtype.
#[derive(Clone, Debug)]
pub(crate) struct Program {
    pub(crate) dtype: FloatDType,
    pub(crate) input: Input,
    pub(crate) ops: Vec<Op>,
    /// An input with another batch size, when every operation is independent of it.
    pub(crate) resized: Option<Input>,
}

impl Program {
    pub(crate) fn generate(seed: u64, max_ops: usize, dtypes: &[FloatDType]) -> Self {
        let mut rng = Rng::new(seed);
        // The dtype and the other batch size come from a stream of their own, so f32 programs are
        // those of earlier versions of the fuzzer.
        let mut variant = Rng::new(seed ^ 0x5EED_D7E5);
        let dtype = variant.pick(dtypes);
        let shape = [
            rng.range(1, 3),
            rng.range(1, 8),
            rng.range(1, 24),
            rng.range(1, 24),
        ];
        let data = rng.values(numel(&shape), 1.0);
        let length = rng.range(2, max_ops);
        let mut ops = Vec::new();
        let mut current = shape;
        let mut attempts = 0;
        while ops.len() < length && attempts < length * 20 {
            attempts += 1;
            if let Some((op, output)) = generate_op(&mut rng, current, dtype) {
                ops.push(op);
                current = output;
            }
        }

        let mut program = Self {
            dtype,
            input: Input { shape, data },
            ops,
            resized: None,
        };
        program.resized = program.resize(&mut variant);
        // The reference and the device start from the same values, so only the computation rounds.
        let round = |value| round_to(dtype, value);
        for value in program.inputs_mut().flat_map(|input| input.data.iter_mut()) {
            *value = round(*value);
        }
        for op in &mut program.ops {
            op.map_constants(round);
        }
        program
    }

    pub(crate) fn resize(&self, rng: &mut Rng) -> Option<Input> {
        if !self.ops.iter().all(Op::batch_independent) {
            return None;
        }
        let mut shape = self.input.shape;
        let sizes = (1..=5).filter(|&n| n != shape[0]).collect::<Vec<_>>();
        shape[0] = rng.pick(&sizes);
        let mut current = shape;
        for op in &self.ops {
            current = output_shape(op, current)?;
            if !within_limits(&current) {
                return None;
            }
        }
        Some(Input {
            shape,
            data: rng.values(numel(&shape), 1.0),
        })
    }

    pub(crate) fn inputs_mut(&mut self) -> impl Iterator<Item = &mut Input> {
        std::iter::once(&mut self.input).chain(self.resized.as_mut())
    }

    /// The program reduced to its first `length` operations.
    pub(crate) fn prefix(&self, length: usize) -> Self {
        let mut prefix = self.clone();
        prefix.ops.truncate(length);
        prefix
    }
}

pub(crate) fn float<const D: usize>(
    data: &[f32],
    shape: [usize; D],
    device: &Device,
    dtype: DType,
) -> Tensor<D> {
    Tensor::from_data(TensorData::new(data.to_vec(), shape), (device, dtype))
}

pub(crate) fn unary(op: Unary, x: Tensor<4>) -> Tensor<4> {
    match op {
        Unary::Relu => relu(x),
        Unary::Abs => x.abs(),
        Unary::Neg => x.neg(),
        Unary::Sigmoid => sigmoid(x),
        Unary::Tanh => tanh(x),
        Unary::Gelu => gelu(x),
        Unary::Silu => silu(x),
        Unary::Sin => x.sin(),
        Unary::Exp => x.clamp(-4.0, 4.0).exp(),
        Unary::Log1pAbs => (x.abs() + 1.0).log(),
        Unary::SqrtAbs => (x.abs() + 1e-3).sqrt(),
        Unary::Square => x.powi_scalar(2) * 0.5,
    }
}

pub(crate) fn binary(op: Binary, x: Tensor<4>, y: Tensor<4>) -> Tensor<4> {
    match op {
        Binary::Add => x + y,
        Binary::Sub => x - y,
        Binary::Mul => x * y,
        Binary::Max => x.max_pair(y),
        Binary::Min => x.min_pair(y),
        Binary::SafeDiv => x / (y.abs() + 1.0),
    }
}

pub(crate) fn apply(op: &Op, x: Tensor<4>, device: &Device) -> Tensor<4> {
    let shape = x.dims();
    let dtype = x.dtype();
    match op {
        Op::Unary(op) => unary(*op, x),
        Op::Scalar { op, value } => {
            let y = Tensor::<4>::full([1, 1, 1, 1], *value, (device, dtype));
            binary(*op, x, y)
        }
        Op::Broadcast { op, shape, data } => binary(*op, x, float(data, *shape, device, dtype)),
        Op::Conv {
            channels_out,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            weight,
            bias,
        } => {
            let weight = float(
                weight,
                [*channels_out, shape[1] / groups, kernel[0], kernel[1]],
                device,
                dtype,
            );
            let bias = bias
                .as_ref()
                .map(|bias| float(bias, [*channels_out], device, dtype));
            let options = ConvOptions::new([*stride; 2], [*padding; 2], [*dilation; 2], *groups);
            conv2d(x, weight, bias, options)
        }
        Op::MaxPool {
            kernel,
            stride,
            padding,
            dilation,
            ceil,
        } => max_pool2d(
            x,
            MaxPoolOptions::new([*kernel; 2])
                .with_stride([*stride; 2])
                .with_padding([*padding; 2])
                .with_dilation([*dilation; 2])
                .with_ceil_mode(*ceil),
        ),
        Op::AvgPool {
            kernel,
            stride,
            padding,
            count_include_pad,
            ceil,
        } => {
            let mut options = AvgPoolOptions::new([*kernel; 2])
                .with_stride([*stride; 2])
                .with_padding([*padding; 2]);
            options.count_include_pad = *count_include_pad;
            options.ceil_mode = *ceil;
            avg_pool2d(x, options)
        }
        Op::AdaptiveAvgPool { size } => adaptive_avg_pool2d(x, *size),
        Op::Interpolate { mode, size } => {
            let mode = match mode {
                0 => InterpolateMode::Nearest,
                1 => InterpolateMode::Bilinear {
                    coordinate_transform: CoordinateTransformMode::HalfPixel,
                },
                2 => InterpolateMode::Bilinear {
                    coordinate_transform: CoordinateTransformMode::AlignCorners,
                },
                _ => InterpolateMode::Bicubic {
                    coordinate_transform: CoordinateTransformMode::HalfPixel,
                    a: -0.75,
                },
            };
            interpolate(x, InterpolateOptions::new(mode).with_output_size(*size))
        }
        Op::Permute(axes) => x.permute(*axes),
        Op::Flip(axis) => x.flip([*axis]),
        Op::Narrow { dim, start, length } => x.narrow(*dim, *start, *length),
        Op::CatWithItself { dim, other } => Tensor::cat(vec![x.clone(), unary(*other, x)], *dim),
        Op::Repeat { dim, times } => x.repeat_dim(*dim, *times),
        Op::Reduce { op, dim } => match op {
            Reduce::Sum => x.sum_dim(*dim),
            Reduce::Mean => x.mean_dim(*dim),
            Reduce::Max => x.max_dim(*dim),
            Reduce::Min => x.min_dim(*dim),
        },
        Op::Softmax { dim, log } => match log {
            true => log_softmax(x, *dim),
            false => softmax(x, *dim),
        },
        Op::Matmul { shape, data, .. } => x.matmul(float(data, *shape, device, dtype)),
        Op::Normalize { dim } => {
            let mean = x.clone().mean_dim(*dim);
            let centered = x - mean;
            let variance = centered.clone().powi_scalar(2).mean_dim(*dim);
            centered / (variance + 1e-5).sqrt()
        }
        Op::MaskFill { threshold, value } => {
            let mask = x.clone().greater_elem(*threshold);
            x.mask_fill(mask, *value)
        }
        Op::Clamp { min, max } => x.clamp(*min, *max),
        Op::Select { dim, indices } => {
            let indices = Tensor::<1, Int>::from_data(
                TensorData::new(indices.clone(), [indices.len()]),
                device,
            );
            x.select(*dim, indices)
        }
        Op::Gather {
            dim,
            shape,
            indices,
        } => {
            let indices =
                Tensor::<4, Int>::from_data(TensorData::new(indices.clone(), *shape), device);
            x.gather(*dim, indices)
        }
        Op::Reshape(dims) => x.reshape(*dims),
        Op::Argmax { dim } => x.argmax(*dim).float() * 0.1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    /// Synchronizes once, so fusion sees the whole chain.
    Lazy,
    /// Synchronizes after every operation.
    Eager,
    /// Lazily, on an input with another batch size.
    Resized,
    /// Lazily again, with kernels, tuning results and memory cached by the earlier runs.
    Again,
}

pub(crate) fn execute(
    input: &Input,
    ops: &[Op],
    device: &Device,
    dtype: FloatDType,
    mode: Mode,
) -> Vec<f32> {
    let mut x = float(&input.data, input.shape, device, dtype.into());
    for op in ops {
        x = apply(op, x, device);
        if mode == Mode::Eager {
            device
                .sync()
                .expect("The device should execute the operation");
        }
    }
    x.into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .expect("The output should be f32")
}

/// The largest finite magnitude of `values`, at least 1: errors are measured against it.
pub(crate) fn scale(values: &[f32]) -> f32 {
    values
        .iter()
        .filter(|v| v.is_finite())
        .fold(1.0f32, |m, v| m.max(v.abs()))
}

/// What a correct device outputs for one input of a program.
#[derive(Clone)]
pub(crate) struct Reference {
    pub(crate) expected: Vec<f32>,
    /// The error the program's dtype allows on top of the f32 tolerance.
    pub(crate) slack: f32,
}

impl Reference {
    /// Flex computes the expected outputs in f32. For a lower precision it also runs the program in
    /// that precision: how far this lands from f32 measures how much the program amplifies
    /// rounding, and so how much error the device may add. Programs that overflow or lose most of
    /// their precision are rejected.
    pub(crate) fn compute(program: &Program, input: &Input) -> Option<Self> {
        let flex = Device::flex();
        let run = |dtype| {
            catch_unwind(AssertUnwindSafe(|| {
                execute(input, &program.ops, &flex, dtype, Mode::Lazy)
            }))
            .ok()
        };
        let expected = run(FloatDType::F32)?;
        if program.dtype == FloatDType::F32 {
            return Some(Self {
                expected,
                slack: 0.0,
            });
        }

        let rounded = run(program.dtype)?;
        let scale = scale(&expected);
        let mut deviation = 0.0f32;
        for (&r, &e) in rounded.iter().zip(&expected) {
            if r.is_finite() != e.is_finite() {
                return None;
            }
            if e.is_finite() {
                deviation = deviation.max((r - e).abs());
            }
        }
        if deviation > 0.25 * scale {
            return None;
        }
        let epsilon = program.dtype.finfo().epsilon as f32;
        Some(Self {
            expected,
            slack: 4.0 * deviation + 8.0 * epsilon * scale,
        })
    }

    /// The element of `actual` furthest from its expected value, among those outside the allowed
    /// error.
    pub(crate) fn violation(&self, actual: &[f32]) -> Option<(usize, f32, f32)> {
        let scale = scale(&self.expected);
        actual
            .iter()
            .zip(&self.expected)
            .enumerate()
            .filter_map(|(index, (&a, &e))| {
                let bad = match (a.is_nan(), e.is_nan()) {
                    (true, true) => false,
                    (true, false) | (false, true) => true,
                    _ if a.is_infinite() || e.is_infinite() => a != e,
                    _ => (a - e).abs() > self.slack + 2e-3 * scale + 2e-3 * e.abs(),
                };
                bad.then_some((index, a, e))
            })
            .max_by(|x, y| (x.1 - x.2).abs().total_cmp(&(y.1 - y.2).abs()))
    }
}

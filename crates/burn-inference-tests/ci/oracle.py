# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy", "torch>=2.4,<3"]
# ///
"""Recomputes lab-fuzz programs with PyTorch to tell which backend is wrong.

Usage: oracle.py <program.json>...   Prints one JSON verdict per program.
"""

import json
import math
import sys

import torch
from torch.nn import functional as F

UNARY = {
    "Relu": torch.relu,
    "Abs": torch.abs,
    "Neg": torch.neg,
    "Sigmoid": torch.sigmoid,
    "Tanh": torch.tanh,
    "Gelu": F.gelu,
    "Silu": F.silu,
    "Sin": torch.sin,
    "Exp": lambda x: x.clamp(-4, 4).exp(),
    "Log1pAbs": lambda x: (x.abs() + 1).log(),
    "SqrtAbs": lambda x: (x.abs() + 1e-3).sqrt(),
    "Square": lambda x: x.pow(2) * 0.5,
}

BINARY = {
    "Add": torch.add,
    "Sub": torch.sub,
    "Mul": torch.mul,
    "Max": torch.maximum,
    "Min": torch.minimum,
    "SafeDiv": lambda x, y: x / (y.abs() + 1),
}

REDUCE = {
    "Sum": lambda x, d: x.sum(d, keepdim=True),
    "Mean": lambda x, d: x.mean(d, keepdim=True),
    "Max": lambda x, d: x.amax(d, keepdim=True),
    "Min": lambda x, d: x.amin(d, keepdim=True),
}


def tensor(data, shape):
    return torch.tensor(data, dtype=torch.float32).reshape(shape)


def interpolate(x, mode, size):
    if mode == 0:
        return F.interpolate(x, size=size, mode="nearest")
    if mode in (1, 2):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=mode == 2)
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False)


def apply(op, x):
    (kind, args), = op.items() if isinstance(op, dict) else ((op, None),)
    if kind == "Unary":
        return UNARY[args](x)
    if kind == "Scalar":
        return BINARY[args["op"]](x, torch.tensor(args["value"], dtype=torch.float32))
    if kind == "Broadcast":
        return BINARY[args["op"]](x, tensor(args["data"], args["shape"]))
    if kind == "Conv":
        channels_in = x.shape[1]
        weight = tensor(
            args["weight"],
            [args["channels_out"], channels_in // args["groups"], *args["kernel"]],
        )
        bias = None if args["bias"] is None else tensor(args["bias"], [args["channels_out"]])
        return F.conv2d(
            x, weight, bias, args["stride"], args["padding"], args["dilation"], args["groups"]
        )
    if kind == "MaxPool":
        return F.max_pool2d(
            x, args["kernel"], args["stride"], args["padding"], args["dilation"], args["ceil"]
        )
    if kind == "AvgPool":
        return F.avg_pool2d(
            x, args["kernel"], args["stride"], args["padding"], args["ceil"], args["count_include_pad"]
        )
    if kind == "AdaptiveAvgPool":
        return F.adaptive_avg_pool2d(x, args["size"])
    if kind == "Interpolate":
        return interpolate(x, args["mode"], args["size"])
    if kind == "Permute":
        return x.permute(args)
    if kind == "Flip":
        return x.flip(args)
    if kind == "Narrow":
        return x.narrow(args["dim"], args["start"], args["length"])
    if kind == "CatWithItself":
        return torch.cat([x, UNARY[args["other"]](x)], args["dim"])
    if kind == "Repeat":
        repeats = [1, 1, 1, 1]
        repeats[args["dim"]] = args["times"]
        return x.repeat(repeats)
    if kind == "Reduce":
        return REDUCE[args["op"]](x, args["dim"])
    if kind == "Softmax":
        return (F.log_softmax if args["log"] else F.softmax)(x, args["dim"])
    if kind == "Matmul":
        return x @ tensor(args["data"], args["shape"])
    if kind == "Normalize":
        centered = x - x.mean(args["dim"], keepdim=True)
        return centered / (centered.pow(2).mean(args["dim"], keepdim=True) + 1e-5).sqrt()
    if kind == "MaskFill":
        return x.masked_fill(x > args["threshold"], args["value"])
    if kind == "Clamp":
        return x.clamp(args["min"], args["max"])
    if kind == "Select":
        return x.index_select(args["dim"], torch.tensor(args["indices"]))
    if kind == "Gather":
        return x.gather(args["dim"], torch.tensor(args["indices"]).reshape(args["shape"]))
    if kind == "Reshape":
        return x.reshape(args)
    if kind == "Argmax":
        return x.argmax(args["dim"], keepdim=True).float() * 0.1
    raise ValueError(f"Unknown operation {kind}")


def agrees(actual, expected):
    """The fuzzer's comparison: NaN matches NaN, errors scale with the largest finite value."""
    if actual is None or len(actual) != len(expected):
        return False
    finite = [abs(v) for v in expected if math.isfinite(v)]
    scale = max([1.0, *finite])
    for a, e in zip(actual, expected):
        a = float("nan") if a is None else a
        if math.isnan(a) or math.isnan(e):
            if math.isnan(a) != math.isnan(e):
                return False
        elif math.isinf(a) or math.isinf(e):
            if a != e:
                return False
        elif abs(a - e) > 2e-3 * scale + 2e-3 * abs(e):
            return False
    return True


def verdict(path):
    program = json.load(open(path))
    x = tensor(program["input"], program["input_shape"])
    try:
        with torch.no_grad():
            for op in program["ops"]:
                x = apply(op, x)
    except Exception as error:  # PyTorch rejects some programs Burn accepts.
        return {"program": path, "verdict": "pytorch-rejected", "detail": str(error)[:300]}

    expected = x.flatten().tolist()
    flex, device = agrees(program["flex"], expected), agrees(program["device"], expected)
    name = {
        (True, False): "device-wrong",
        (False, True): "flex-wrong",
        (False, False): "both-wrong",
        (True, True): "both-agree",
    }[flex, device]
    return {"program": path, "verdict": name}


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(json.dumps(verdict(path)), flush=True)

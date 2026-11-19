# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy",
#   "safetensors",
#   "torch>=2.4,<3",
#   "torchvision>=0.19,<1",
#   "transformers>=4.45,<5",
# ]
# ///
"""Exports pretrained weights under Burn's module paths, with outputs computed by PyTorch."""

import re
import sys
from pathlib import Path

import torch
import torchvision
from safetensors.torch import save_file
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel

MINILM = ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
DISTILGPT2 = ("distilbert/distilgpt2", "2290a62682d06624634c1f46a6ad5be0f47f38aa")

SENTENCES = [
    "Inference engines must agree with the framework that trained the model.",
    "Padding should never change a sentence.",
    "Short text.",
    "A batch mixes long and short sequences, so attention masks do real work here.",
]
PROMPT = "The quick brown fox jumps over the lazy dog because"
GENERATED_TOKENS = 24


def images(batch_size):
    """Smooth, reproducible images normalized like ImageNet inputs."""
    generator = torch.Generator().manual_seed(1234)
    x = F.interpolate(
        torch.rand(batch_size, 3, 14, 14, generator=generator),
        size=224,
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    return (x - mean) / std


def save(out, name, weights, reference):
    save_file({k: v.detach().clone().contiguous() for k, v in weights.items()}, out / f"{name}.safetensors")
    save_file({k: v.detach().clone().contiguous() for k, v in reference.items()}, out / f"{name}.reference.safetensors")
    print(f"{name}: {sum(v.numel() for v in weights.values()):,} parameters")


def export_resnet50(out):
    model = torchvision.models.resnet50(weights="IMAGENET1K_V2").eval()
    weights = {}
    for key, value in model.state_dict().items():
        if key.endswith("num_batches_tracked"):
            continue
        key = key.replace("downsample.0.", "downsample.conv.").replace("downsample.1.", "downsample.bn.")
        weights[key] = value
    x = images(4)
    save(out, "resnet50", weights, {"input": x, "logits": model(x)})


def mobilenet_key(key):
    if key.startswith("classifier.1."):
        return key.replace("classifier.1.", "classifier.")
    match = re.fullmatch(r"features\.(\d+)\.(.+)", key)
    index, rest = int(match[1]), match[2]
    if index in (0, 18):
        name = "first" if index == 0 else "last"
        return f"{name}." + rest.replace("0.", "conv.", 1).replace("1.", "norm.", 1)

    block = f"blocks.{index - 1}"
    has_expansion = index > 1
    parts = [
        ("conv.0.0.", "expand.conv." if has_expansion else "depthwise.conv."),
        ("conv.0.1.", "expand.norm." if has_expansion else "depthwise.norm."),
    ]
    parts += (
        [
            ("conv.1.0.", "depthwise.conv."),
            ("conv.1.1.", "depthwise.norm."),
            ("conv.2.", "project."),
            ("conv.3.", "project_norm."),
        ]
        if has_expansion
        else [("conv.1.", "project."), ("conv.2.", "project_norm.")]
    )
    for old, new in parts:
        if rest.startswith(old):
            return f"{block}.{new}{rest[len(old):]}"
    raise KeyError(key)


def export_mobilenet_v2(out):
    model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2").eval()
    weights = {
        mobilenet_key(key): value
        for key, value in model.state_dict().items()
        if not key.endswith("num_batches_tracked")
    }
    x = images(4)
    save(out, "mobilenet_v2", weights, {"input": x, "logits": model(x)})


def minilm_key(key):
    replacements = [
        ("embeddings.word_embeddings.", "word."),
        ("embeddings.position_embeddings.", "position."),
        ("embeddings.token_type_embeddings.", "token_type."),
        ("embeddings.LayerNorm.", "norm."),
    ]
    for old, new in replacements:
        if key.startswith(old):
            return new + key[len(old):]
    layer = re.fullmatch(r"encoder\.layer\.(\d+)\.(.+)", key)
    rest = layer[2]
    for old, new in [
        ("attention.self.query.", "mha.query."),
        ("attention.self.key.", "mha.key."),
        ("attention.self.value.", "mha.value."),
        ("attention.output.dense.", "mha.output."),
        ("attention.output.LayerNorm.", "norm_1."),
        ("intermediate.dense.", "pwff.linear_inner."),
        ("output.dense.", "pwff.linear_outer."),
        ("output.LayerNorm.", "norm_2."),
    ]:
        if rest.startswith(old):
            return f"encoder.layers.{layer[1]}.{new}{rest[len(old):]}"
    raise KeyError(key)


def export_minilm(out):
    name, revision = MINILM
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModel.from_pretrained(name, revision=revision, attn_implementation="eager").eval()
    weights = {
        minilm_key(key): value
        for key, value in model.state_dict().items()
        if not key.startswith("pooler.") and not key.endswith("position_ids")
    }
    batch = tokenizer(SENTENCES, padding=True, return_tensors="pt")
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    embedding = F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9), dim=1)
    save(
        out,
        "minilm",
        weights,
        {
            "input_ids": batch["input_ids"],
            "token_type_ids": batch["token_type_ids"],
            "attention_mask": batch["attention_mask"],
            "hidden": hidden,
            "embedding": embedding,
        },
    )


def gpt2_weights(model):
    weights = {}
    for key, value in model.state_dict().items():
        if key.endswith((".attn.bias", ".attn.masked_bias")):
            continue
        if key == "lm_head.weight":
            weights["head.weight"] = value
            continue
        key = key.removeprefix("transformer.")
        layer = re.fullmatch(r"h\.(\d+)\.(.+)", key)
        if layer is None:
            weights[key] = value
            continue

        path, rest = f"encoder.layers.{layer[1]}", layer[2]
        kind = rest.rsplit(".", 1)[1]
        # GPT-2's Conv1D stores weights as (input, output); `nn.Linear` stores (output, input).
        linear = value.t() if value.dim() == 2 else value
        if rest.startswith("attn.c_attn."):
            axis = 0
            for head, part in zip(("query", "key", "value"), linear.chunk(3, axis)):
                weights[f"{path}.mha.{head}.{kind}"] = part
        elif rest.startswith("attn.c_proj."):
            weights[f"{path}.mha.output.{kind}"] = linear
        elif rest.startswith("mlp.c_fc."):
            weights[f"{path}.pwff.linear_inner.{kind}"] = linear
        elif rest.startswith("mlp.c_proj."):
            weights[f"{path}.pwff.linear_outer.{kind}"] = linear
        elif rest.startswith("ln_1."):
            weights[f"{path}.norm_2.{kind}"] = value
        elif rest.startswith("ln_2."):
            weights[f"{path}.norm_1.{kind}"] = value
        else:
            raise KeyError(key)
    return weights


def export_distilgpt2(out):
    name, revision = DISTILGPT2
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = GPT2LMHeadModel.from_pretrained(name, revision=revision, attn_implementation="eager").eval()
    encoded = tokenizer(PROMPT, return_tensors="pt")
    prompt = encoded["input_ids"]
    generated = model.generate(
        prompt,
        attention_mask=encoded["attention_mask"],
        max_new_tokens=GENERATED_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    save(
        out,
        "distilgpt2",
        gpt2_weights(model),
        {"prompt": prompt, "logits": model(prompt).logits, "generated": generated},
    )


@torch.no_grad()
def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    export_resnet50(out)
    export_mobilenet_v2(out)
    export_minilm(out)
    export_distilgpt2(out)


if __name__ == "__main__":
    main()

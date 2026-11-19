# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "safetensors", "torch>=2.4,<3"]
# ///
"""Generates the PyTorch weights and reference outputs used by the inference tests."""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).parent


class Downsample(nn.Module):
    def __init__(self, channels_in, channels_out, stride):
        super().__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, 1, stride=stride, bias=False)
        self.norm = nn.BatchNorm2d(channels_out)

    def forward(self, x):
        return self.norm(self.conv(x))


class Block(nn.Module):
    def __init__(self, channels_in, channels_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels_out)
        self.downsample = (
            Downsample(channels_in, channels_out, stride)
            if stride != 1 or channels_in != channels_out
            else None
        )

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        y = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(y)) + shortcut)


class TinyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.stem_norm = nn.BatchNorm2d(8)
        self.blocks = nn.ModuleList([Block(8, 8, 1), Block(8, 16, 2)])
        self.head = nn.Linear(16, 5)

    def forward(self, x):
        x = F.relu(self.stem_norm(self.stem(x)))
        for block in self.blocks:
            x = block(x)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def split(self, x):
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask_pad):
        batch, seq, d_model = x.shape
        query, key, value = self.split(self.query(x)), self.split(self.key(x)), self.split(self.value(x))
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask_pad is not None:
            scores = scores.masked_fill(mask_pad[:, None, None, :], -math.inf)
        context = (scores.softmax(-1) @ value).transpose(1, 2).reshape(batch, seq, d_model)
        return self.output(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear_inner = nn.Linear(d_model, d_ff)
        self.linear_outer = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_outer(F.gelu(self.linear_inner(x)))


class EncoderLayer(nn.Module):
    """Pre-norm layer: `norm_2` guards the attention and `norm_1` the feed-forward."""

    def __init__(self, d_model, d_ff, n_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.pwff = FeedForward(d_model, d_ff)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, mask_pad):
        x = x + self.mha(self.norm_2(x), mask_pad)
        return x + self.pwff(self.norm_1(x))


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, n_layers):
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(d_model, d_ff, n_heads) for _ in range(n_layers))

    def forward(self, x, mask_pad):
        for layer in self.layers:
            x = layer(x, mask_pad)
        return x


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.position = nn.Embedding(12, 16)
        self.encoder = Encoder(16, 32, 2, 2)
        self.head = nn.Linear(16, 8)

    def forward(self, tokens, mask_pad=None):
        positions = torch.arange(tokens.shape[1]).expand(tokens.shape)
        return self.head(self.encoder(self.embedding(tokens) + self.position(positions), mask_pad))


def array(tensor):
    return {"shape": list(tensor.shape), "data": tensor.flatten().tolist()}


def save_weights(model, name):
    state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.endswith("num_batches_tracked")
    }
    save_file(state, HERE / name)


@torch.no_grad()
def main():
    torch.manual_seed(0)

    resnet = TinyResNet().train()
    # Gives the batch norms running statistics that differ from their initial values.
    for _ in range(8):
        resnet(torch.randn(8, 3, 16, 16) * 2.0 + 0.5)
    resnet.eval()
    image = torch.randn(2, 3, 16, 16)

    transformer = TinyTransformer().eval()
    tokens = torch.randint(1, 32, (3, 8))
    mask_pad = torch.tensor([[i >= length for i in range(8)] for length in (8, 5, 3)])
    padded_tokens = tokens.masked_fill(mask_pad, 0)

    save_weights(resnet, "resnet.safetensors")
    save_weights(transformer, "transformer.safetensors")
    reference = {
        "resnet": {"input": array(image), "output": array(resnet(image))},
        "transformer": {
            "tokens": array(tokens),
            "output": array(transformer(tokens)),
            "padded_tokens": array(padded_tokens),
            "mask_pad": array(mask_pad),
            "padded_output": array(transformer(padded_tokens, mask_pad)),
        },
    }
    (HERE / "reference.json").write_text(json.dumps(reference) + "\n")


if __name__ == "__main__":
    main()

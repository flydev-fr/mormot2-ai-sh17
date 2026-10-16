"""LandrixAI - tests/data/infer-ref/tiny-qwen3-hd256: tiny-qwen3 with a head
dimension of 256 (backlog B9).

The Vulkan attention (tools/vk/attention.comp) handles the dimensions t + 128
of a head in a second accumulator (acc1), which Qwen3-0.6B (head_dim 128) and
tiny-qwen3 (32) never reach. This model reaches it: the test compares the
device against the CPU backend of the engine itself, so no reference outputs
are needed - and no torch: numpy writes the safetensors file.

Embeddings, norms and the MLP come from tiny-qwen3; the attention of both
layers is new and seeded: 2 heads of 256 over 1 KV head, BF16 weights.

usage: python make_tiny_headdim.py   (from tools/infer-ref; numpy only)
"""
import hashlib
import json
import os
import platform
import struct
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "tests", "data", "infer-ref")
SRC = os.path.join(DATA, "tiny-qwen3")
DST = os.path.join(DATA, "tiny-qwen3-hd256")
HEADS = 2
KV_HEADS = 1
HEAD_DIM = 256
SEED = 256


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, v in header.items():
        if name == "__metadata__":
            continue
        assert v["dtype"] == "BF16", name
        a, b = v["data_offsets"]
        out[name] = (v["shape"], data[a:b])
    return out


def bf16(values):
    """float32 -> BF16 bytes, rounded to nearest even"""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return bits.astype("<u2").tobytes()


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        shape, raw = tensors[name]
        header[name] = {"dtype": "BF16", "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for b in blobs:
            f.write(b)


def main():
    with open(os.path.join(SRC, "config.json")) as f:
        cfg = json.load(f)
    hidden = cfg["hidden_size"]
    tensors = read_safetensors(os.path.join(SRC, "model.safetensors"))
    rng = np.random.default_rng(SEED)
    for layer in range(cfg["num_hidden_layers"]):
        p = "model.layers.%d.self_attn." % layer
        shapes = {
            "q_proj.weight": [HEADS * HEAD_DIM, hidden],
            "k_proj.weight": [KV_HEADS * HEAD_DIM, hidden],
            "v_proj.weight": [KV_HEADS * HEAD_DIM, hidden],
            "o_proj.weight": [hidden, HEADS * HEAD_DIM],
        }
        for name, shape in shapes.items():
            scale = 1.0 / np.sqrt(shape[1])
            tensors[p + name] = (shape, bf16(rng.normal(0.0, scale, shape)))
        for name in ("q_norm.weight", "k_norm.weight"):
            tensors[p + name] = ([HEAD_DIM], bf16(1.0 + rng.normal(0.0, 0.1, HEAD_DIM)))
    cfg["num_attention_heads"] = HEADS
    cfg["num_key_value_heads"] = KV_HEADS
    cfg["head_dim"] = HEAD_DIM
    os.makedirs(DST, exist_ok=True)
    write_safetensors(os.path.join(DST, "model.safetensors"), tensors)
    with open(os.path.join(DST, "config.json"), "w", newline="\n") as f:
        json.dump(cfg, f, indent=1)
        f.write("\n")
    # the marker FindInferRefDir looks for; no reference outputs (no blob)
    with open(os.path.join(DST, "model.safetensors"), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "dataset": "tiny-qwen3-hd256",
        "note": "tiny-qwen3 with 2 heads of 256 over 1 KV head (seeded numpy "
                "attention weights, BF16): the device against the CPU backend, "
                "no reference outputs",
        "safetensors_sha256": sha,
        "environment": {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "platform": platform.system() + " " + platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "seed": SEED,
            "generator": "tools/infer-ref/make_tiny_headdim.py",
        },
    }
    with open(os.path.join(DST, "manifest.json"), "w", newline="\n") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print("written:", DST)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""An independent reference implementation of the llama forward pass.

This exists to cross-check ``GgufRunner``.  It shares no code with the engine:
it parses the GGUF container itself and evaluates the graph in NumPy, written
straight from the architecture rather than transcribed from the C++.  A sign
error in RoPE, a wrong grouped-query head mapping, a transposed projection or a
misplaced residual all show up as a logit mismatch here, where the engine's own
tests -- which compare the engine against itself -- would not notice.

    python3 tests/reference_llama.py model.gguf --tokens 1,260,261 [--compare oracle.txt]

Only the tensor types the fixture uses (F32 and F16) are implemented; the block
quantisations have their own dedicated tests.
"""

from __future__ import annotations

import argparse
import struct
import sys
from typing import Any

import numpy as np

GGUF_MAGIC = 0x46554747

# ggml_type -> (numpy dtype, elements per block, bytes per block)
PLAIN_TYPES = {0: (np.float32, 1, 4), 1: (np.float16, 1, 2), 30: ("bf16", 1, 2)}

(
    UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY,
    UINT64, INT64, FLOAT64,
) = range(13)

SCALAR = {
    UINT8: ("<B", 1), INT8: ("<b", 1), UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4), FLOAT32: ("<f", 4), BOOL: ("<B", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}


class Reader:
    def __init__(self, blob: bytes) -> None:
        self.b = blob
        self.i = 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise ValueError("truncated GGUF")
        out = self.b[self.i : self.i + n]
        self.i += n
        return out

    def scalar(self, kind: int) -> Any:
        fmt, size = SCALAR[kind]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.take(8))[0]
        return self.take(n).decode("utf-8", "replace")

    def value(self, kind: int) -> Any:
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            inner = struct.unpack("<I", self.take(4))[0]
            n = struct.unpack("<Q", self.take(8))[0]
            return [self.value(inner) for _ in range(n)]
        return self.scalar(kind)


def load_gguf(path: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with open(path, "rb") as fh:
        blob = fh.read()
    r = Reader(blob)
    magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", r.take(24))
    if magic != GGUF_MAGIC:
        raise ValueError(f"{path} is not a GGUF file")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")

    kv: dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        kind = struct.unpack("<I", r.take(4))[0]
        kv[key] = r.value(kind)

    infos = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = struct.unpack("<I", r.take(4))[0]
        ne = [struct.unpack("<Q", r.take(8))[0] for _ in range(n_dims)]
        ttype = struct.unpack("<I", r.take(4))[0]
        offset = struct.unpack("<Q", r.take(8))[0]
        infos.append((name, ne, ttype, offset))

    alignment = int(kv.get("general.alignment", 32))
    data_start = (r.i + alignment - 1) // alignment * alignment

    tensors: dict[str, np.ndarray] = {}
    for name, ne, ttype, offset in infos:
        if ttype not in PLAIN_TYPES:
            raise ValueError(
                f"{name}: this reference only reads F32/F16/BF16, not ggml type {ttype}"
            )
        dtype, _, item = PLAIN_TYPES[ttype]
        count = 1
        for d in ne:
            count *= d
        raw = blob[data_start + offset : data_start + offset + count * item]
        if dtype == "bf16":
            u16 = np.frombuffer(raw, dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        # ggml lists ne fastest-axis first; a weight is [out_features, in_features].
        tensors[name] = arr.reshape(tuple(reversed(ne)))
    return kv, tensors


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps) * w


def rope(vec: np.ndarray, n_heads: int, head_dim: int, rot: int, pos: int, base: float) -> np.ndarray:
    out = vec.reshape(n_heads, head_dim).copy()
    for i in range(0, rot, 2):
        theta = pos * (base ** (-i / rot))
        c, s = np.cos(theta), np.sin(theta)
        x0 = out[:, i].copy()
        x1 = out[:, i + 1].copy()
        out[:, i] = x0 * c - x1 * s
        out[:, i + 1] = x0 * s + x1 * c
    return out.reshape(-1)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def forward(kv: dict[str, Any], t: dict[str, np.ndarray], tokens: list[int]) -> np.ndarray:
    arch = kv.get("general.architecture", "llama")

    def meta(suffix: str, default: Any = None) -> Any:
        return kv.get(f"{arch}.{suffix}", default)

    n_layers = int(meta("block_count"))
    n_heads = int(meta("attention.head_count"))
    n_kv_heads = int(meta("attention.head_count_kv", n_heads))
    n_embd = int(meta("embedding_length"))
    head_dim = int(meta("attention.key_length", 0)) or n_embd // n_heads
    rot = int(meta("rope.dimension_count", 0)) or head_dim
    base = float(meta("rope.freq_base", 10000.0))
    eps = float(meta("attention.layer_norm_rms_epsilon", 1e-5))
    gqa = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)

    embd = t["token_embd.weight"]
    n_pos = len(tokens)
    # [layer][pos] -> the key/value vectors written at that position.
    k_cache = [[None] * n_pos for _ in range(n_layers)]
    v_cache = [[None] * n_pos for _ in range(n_layers)]

    x_final = None
    for pos, tok in enumerate(tokens):
        x = embd[tok].astype(np.float32).copy()
        for layer in range(n_layers):
            p = f"blk.{layer}."
            xb = rms_norm(x, t[p + "attn_norm.weight"], eps)

            q = t[p + "attn_q.weight"] @ xb
            k = t[p + "attn_k.weight"] @ xb
            v = t[p + "attn_v.weight"] @ xb
            for name, vec in ((p + "attn_q.bias", q), (p + "attn_k.bias", k), (p + "attn_v.bias", v)):
                if name in t:
                    vec += t[name]

            q = rope(q, n_heads, head_dim, rot, pos, base)
            k = rope(k, n_kv_heads, head_dim, rot, pos, base)
            k_cache[layer][pos] = k.reshape(n_kv_heads, head_dim)
            v_cache[layer][pos] = v.reshape(n_kv_heads, head_dim)

            qh = q.reshape(n_heads, head_dim)
            attn = np.zeros((n_heads, head_dim), dtype=np.float32)
            for h in range(n_heads):
                kvh = h // gqa
                scores = np.array(
                    [float(qh[h] @ k_cache[layer][p2][kvh]) * scale for p2 in range(pos + 1)],
                    dtype=np.float32,
                )
                w = softmax(scores)
                for p2 in range(pos + 1):
                    attn[h] += w[p2] * v_cache[layer][p2][kvh]

            o = t[p + "attn_output.weight"] @ attn.reshape(-1)
            if p + "attn_output.bias" in t:
                o += t[p + "attn_output.bias"]
            x = x + o

            xb = rms_norm(x, t[p + "ffn_norm.weight"], eps)
            gate = t[p + "ffn_gate.weight"] @ xb
            up = t[p + "ffn_up.weight"] @ xb
            act = gate / (1.0 + np.exp(-gate)) * up
            x = x + t[p + "ffn_down.weight"] @ act
        x_final = x

    normed = rms_norm(x_final, t["output_norm.weight"], eps)
    head = t.get("output.weight", embd)  # tied embeddings when there is no head
    return head @ normed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--tokens", required=True, help="comma-separated token ids")
    ap.add_argument("--compare", help="file of logits, one per line, to compare against")
    ap.add_argument("--tolerance", type=float, default=1e-3)
    args = ap.parse_args()

    tokens = [int(x) for x in args.tokens.split(",") if x.strip()]
    kv, tensors = load_gguf(args.model)
    logits = forward(kv, tensors, tokens)

    if not args.compare:
        for value in logits:
            print(f"{value:.9g}")
        return 0

    with open(args.compare) as fh:
        other = np.array([float(line) for line in fh if line.strip()], dtype=np.float32)
    if other.shape != logits.shape:
        print(f"FAIL: {other.shape} logits from the engine, {logits.shape} from the reference")
        return 1

    diff = np.abs(other - logits)
    scale = max(1.0, float(np.max(np.abs(logits))))
    worst = int(np.argmax(diff))
    rel = float(np.max(diff)) / scale
    print(
        f"reference vs engine: max abs diff {np.max(diff):.3e} "
        f"(relative {rel:.3e}) at index {worst}; "
        f"argmax engine={int(np.argmax(other))} reference={int(np.argmax(logits))}"
    )
    if rel > args.tolerance:
        print("FAIL: the engine's forward pass disagrees with the reference")
        return 1
    if int(np.argmax(other)) != int(np.argmax(logits)):
        print("FAIL: the two implementations would sample different tokens")
        return 1
    print("OK: the engine matches the independent reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Oracle

**Oracle** runs one large language model across several machines. Each node
holds a contiguous slice of the model's layers in its own RAM; activations move
between them over a Thunderbolt bridge (or plain ethernet) as packed tensors.
The master exposes an OpenAI-compatible API, so anything that speaks to OpenAI
speaks to Oracle.

This exists because a 70B model at 4-bit needs ~40 GB of weights, which does not
fit on one laptop but does fit across three. The 2018-era discrete GPUs in such
machines have 2–4 GB of VRAM — nowhere near enough to matter — so Oracle treats
the cluster's **RAM** as the resource being pooled and runs the maths on CPU.

```
  client (curl / OpenAI SDK / any chat UI)
       │  POST /v1/chat/completions          (SSE or buffered)
       ▼
  ┌─────────────────────────── MASTER · 10.10.0.1:8000 ───────────────────────────┐
  │ API key check → rate limit → queue (priority) → concurrency slot              │
  │ tokenise · embed · layers [0, a) · KV cache stays local                       │
  └───────────────────────────────────┬───────────────────────────────────────────┘
                                      │ f16 activations, packed ORCL frame
                                      ▼
                        WORKER 1 · layers [a, b) · KV local
                                      │
                                      ▼
                        WORKER 2 · layers [b, L) + output head
                                      │ f32 logits
                                      ▼
                    MASTER samples the next token and repeats
        control plane: signed handshake on TCP, heartbeats on UDP :9100
```

---

## Quick start

```bash
scripts/build.sh                       # configure, build, run the test suite
scripts/fetch_model.sh                 # ~1.1 GB TinyLlama Q4_K_M into models/
MODEL=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf SINGLE=1 NO_AUTH=1 \
  scripts/run_master.sh
```

Then, in another shell:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain RoPE in one sentence."}],
       "max_tokens":128}'
```

Open <http://127.0.0.1:8000/> for the dashboard.

To check the whole stack without downloading anything — it builds a synthetic
model, runs a real two-node mesh over TCP, and exercises auth, streaming,
concurrency, worker death and recovery:

```bash
scripts/e2e_test.sh
```

---

## What is in the box

| Area | What it does |
|---|---|
| **Model** | GGUF reader, model recognition, dequantisers, llama-family transformer, tokeniser |
| **Transport** | Packed tensor frames over TCP with `writev`, signed registration, reconnect, shared-memory local hop |
| **Security** | API keys, rate limiting, size and concurrency caps, worker HMAC auth, model integrity, audit log |
| **Orchestration** | Priority queue, request ids, concurrency, resource-aware worker scoring, timeouts |
| **API & UI** | `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/cluster`, `/metrics`, dashboard |
| **Deployment** | Static binaries, Dockerfile, Compose mesh, config files, startup scripts |

---

## Models

Oracle reads **GGUF** — llama.cpp's format — directly. Weights stay memory-mapped
and quantised; a row is expanded to f32 inside the mat-vec and thrown away, so a
4-bit 8B model stays around 4.5 GB resident rather than the ~32 GB an up-front
f32 expansion would need.

Inspect a file before deploying it:

```bash
./build/oracle-model-info models/model.gguf --split 32,32,32
```

```
model         Llama-3.1-8B-Instruct
architecture  llama
quantisation  Q4_K_M (4.83 bits/weight)
parameters    8.03B (8030261248)
layers        32
heads         32 q, 8 kv, head_dim 128
memory
  weights            4.58 GiB
  kv per token       128.0 KiB (f16, all layers)
  recommended RAM    5.9 GiB
runnable by Oracle's gguf runner
```

**Quantisations executed:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K. Every one is checked against hand-built
blocks in `tests/test_quant.cpp`.

**Architectures executed:** the llama family graph — RMSNorm, RoPE,
grouped-query attention with a KV cache, SwiGLU — which covers `llama`,
`mistral`, `qwen2` (including its QKV biases), `minicpm`, `deepseek` and `olmo`.
Any other GGUF is still *recognised* (`oracle-model-info` reports everything
about it) but `inspect_gguf` marks it `supported_for_inference: false` with the
reason, rather than running it and producing nonsense. The IQ-series
quantisations are sized and reported but not yet dequantised.

**Tokeniser** comes from the file's own vocabulary: SentencePiece merges with
`<0xNN>` byte fallback, byte-level BPE with the embedded merge table, literal
matching of control tokens, and chat templates for the chatml / llama2 /
llama3 / mistral / gemma families.

---

## Running a mesh

Three machines on a Thunderbolt bridge (see
[`scripts/tb3_bridge.md`](scripts/tb3_bridge.md) for the network setup). Every
node needs **the same GGUF at the same path** — layer weights are read locally,
not streamed, which is what keeps the link carrying only activations.

Edit `configs/cluster.toml` so the `[[nodes]]` hosts match your machines, then:

```bash
# once, on any machine — this secret authenticates nodes to each other
export ORACLE_CLUSTER_SECRET=$(openssl rand -hex 32)

# on each worker (secret must match)
NODE_ID=1 MODEL=/models/model.gguf scripts/run_worker.sh
NODE_ID=2 MODEL=/models/model.gguf scripts/run_worker.sh

# on the master, last: it dials the workers
export ORACLE_API_KEYS="demo:$(openssl rand -hex 24)"
MODEL=/models/model.gguf scripts/run_master.sh
```

Layers are split by node order. Oracle assigns contiguous ranges automatically;
set `layer_start` / `layer_end` per node to override when the machines have
different amounts of RAM.

### Measuring the link first

```bash
# on the far node
./build/oracle-engine-bench-net --listen --port 9200
# on this node
./build/oracle-engine-bench-net --host 10.10.0.2 --port 9200 --hidden 4096
```

It sweeps decode-sized through bulk-sized payloads and reports round-trip
latency (p50/p99/min), one-way throughput and the transfer time for a single
activation frame. A Thunderbolt 3 bridge should clear 8 Gb/s bulk with the
decode hop well under 5 ms.

---

## Docker

```bash
docker build -t oracle:latest .          # the build runs the test suite
docker run --rm -p 8000:8000 \
  -v "$PWD/models:/models:ro" \
  oracle:latest \
  oracle-engine-master --config /etc/oracle/single.toml \
                       --model /models/model.gguf --no-auth
```

A three-container mesh on one host:

```bash
export ORACLE_CLUSTER_SECRET=$(openssl rand -hex 32)
export ORACLE_API_KEYS="demo:$(openssl rand -hex 24)"
MODEL=/absolute/path/model.gguf docker compose up
```

---

## API

`Authorization: Bearer <key>` on everything except `/health`, `/v1/models`,
`/metrics` and the dashboard (configurable).

| Endpoint | Notes |
|---|---|
| `POST /v1/chat/completions` | Streaming and buffered. `stop`, `temperature`, `top_p`, `top_k`, `seed`, `priority`. |
| `POST /v1/completions` | Legacy text completion. |
| `GET /v1/models` | The loaded model plus its geometry and quantisation. |
| `GET /health` | `ok` / `degraded` / `down`, and the live worker count. |
| `GET /cluster` | Pipeline, per-node CPU/RAM/GPU, links, scheduler, security. Admin keys see the audit trail. |
| `GET /metrics` | Prometheus text format. |
| `GET /v1/security` | Admin only: full security state. |
| `GET /` | Dashboard. |

Responses carry an `X-Oracle-Request-Id` header, and buffered completions
include an `oracle` block with `prefill_ms`, `decode_ms` and
`tokens_per_second`.

Because the API is OpenAI-shaped, the official SDKs work unchanged:

```python
from openai import OpenAI
client = OpenAI(base_url="http://10.10.0.1:8000/v1", api_key="<your key>")
print(client.chat.completions.create(
    model="oracle", messages=[{"role": "user", "content": "hello"}]
).choices[0].message.content)
```

---

## Security

Oracle assumes the mesh sits on a network you do not fully trust.

- **API keys** are stored as SHA-256 digests, never in the clear, and compared in
  constant time. Generate one with `oracle-engine-master --generate-key`;
  install it in `configs/api_keys` or `$ORACLE_API_KEYS`.
- **Rate limiting** is a per-key token bucket. A source that keeps getting
  rejected is parked for `ban_seconds`, so a spray costs one map lookup rather
  than a signature check per packet.
- **Request limits**: body size, prompt length, message count, `max_tokens`, and
  both global and per-key concurrency caps.
- **Workers authenticate** with HMAC-SHA256 over a fresh nonce using
  `$ORACLE_CLUSTER_SECRET`. A node that cannot sign, or that claims an id not in
  the config, is refused before it is registered.
- **Frames declare their size up front** and are rejected before allocation when
  they exceed the limit.
- **Model integrity**: set `model_manifest` and `verify_model_integrity` to pin
  each model file's SHA-256. An unknown file is recorded and flagged; a changed
  one refuses to load.
- **Audit log**: every admission decision, handshake and integrity check is
  written as JSON and surfaced on the dashboard.

Starting with `require_api_key = true` and no keys configured is a startup
error, not a silent open door.

---

## Reliability

Every node sends a heartbeat every `heartbeat_interval_ms` carrying its live CPU
load, free RAM, in-flight work and queue depth. After `heartbeat_misses` silent
intervals a node is marked dead, dropped from the connection table, and — the
part that matters — **removed from scheduling**: a request whose pipeline stage
has no healthy worker is refused with `no healthy worker owns layers [a, b)`
rather than dispatched into a black hole.

Heartbeats travel over UDP and activations over TCP, so a node can be answering
one while the other is down — exactly what a restarted worker looks like. A node
counts as healthy only when **both** are up, and the master's reconnect loop is
driven by the link state rather than the heartbeat: any configured peer without
an open activation stream is re-dialled and re-handshaked, with backoff. A worker
that loses its master goes back to waiting for a registration instead of exiting,
so either side can restart independently.

`scripts/e2e_test.sh` kills a worker mid-run and asserts all of this: the
cluster degrades, requests are refused with a specific reason, the worker
rejoins on its own, and inference resumes.

---

## Configuration

| File | Purpose |
|---|---|
| `configs/cluster.toml` | Three-node mesh; the documented reference |
| `configs/single.toml` | One machine, whole model |
| `configs/compose.toml` | Used by `docker-compose.yml` |
| `configs/api_keys.example` | API key file format |

Sections: `[cluster]` (ports, heartbeats, resident sequences), `[model]` (path
and geometry), `[server]` (concurrency, queue depth, timeouts, threads),
`[security]` (everything above), `[[nodes]]` (id, role, host, RAM budget,
optional explicit layer range).

---

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Requires CMake ≥ 3.20 and a C++20 compiler. No external dependencies:
`cpp-httplib` is vendored, and SHA-256, HMAC, JSON and the BLAS kernels are all
in-tree. On macOS the compute layer uses Accelerate and a Metal GEMM is
available; elsewhere it uses a portable threaded CPU backend. Options:
`-DORACLE_BUILD_PYTHON=ON`, `-DORACLE_BUILD_TESTS=OFF`,
`-DORACLE_NATIVE_ARCH=ON`.

### Tests

| Test | Covers |
|---|---|
| `protocol`, `ring_buffer`, `transport_tcp` | Wire format, shared-memory ring, TCP round trip |
| `quant` | Every dequantiser against hand-built blocks; f16/bf16 conversions |
| `gguf` | Recognition, single-node inference, **two-way split producing identical logits**, decode, KV isolation |
| `tokenizer` | SPM merges, byte fallback, control tokens, chat templates |
| `security` | Key handling, rate limits, bans, concurrency, validation, HMAC, integrity |
| `scheduler` | Priority, queue shedding, timeouts, worker scoring, dead-stage refusal |
| `shard_plan`, `pipeline_tiny` | Layer assignment against RAM budgets; end-to-end tiny pipeline |

Tests use a `CHECK` macro rather than `assert`, so they keep checking in the
default Release build.

---

## Layout

```
include/oracle/{types,compute,model,tb3,shard,orch,runner,security,util}
src/{compute,model,tb3-transport,mem-shard-manager,orchestrator-core,metal-cpu-runner,security}
apps/       oracle-engine-master · oracle-engine-worker · oracle-engine-bench-net · oracle-model-info
configs/    cluster.toml · single.toml · compose.toml · api_keys.example
scripts/    build.sh · run_master.sh · run_worker.sh · fetch_model.sh · e2e_test.sh · phase*.sh
tests/      unit and integration tests, plus a synthetic GGUF fixture
```

### Wire format

Packed `TensorHeader` — 76 bytes, magic **ORCL** — followed by the raw payload,
sent as one `writev`. No protobuf or JSON on the hot path. The header carries
dtype, shape, sequence id, token id, flags and a CRC32 of the payload, which is
what lets the master demultiplex concurrent requests by sequence instead of
assuming the next frame is its own. Two Oracle processes on one machine can use
a POSIX shared-memory SPSC ring instead of the socket.

---

## Known limits

- **CPU only outside macOS.** The Metal path implements a GEMM, not the full
  transformer; there is no CUDA backend. Throughput is bounded by memory
  bandwidth.
- **Prefill and decode are one token at a time per sequence.** There is no
  continuous batching; concurrent requests interleave across pipeline stages
  rather than sharing a batched forward pass.
- **IQ-series quantisations** are recognised and sized but not dequantised.
- **MoE, Mamba and encoder-decoder architectures** are not implemented.
- **The pipeline is linear.** There is no tensor parallelism within a layer, and
  no replica failover: if a stage's worker dies, requests are refused until it
  returns.

## License

See [LICENSE](LICENSE).

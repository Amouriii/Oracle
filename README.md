# Oracle

**Oracle** is the standalone **C++20** Thunderbolt-3 LLM engine: packed tensor transport, RAM/VRAM sharding, pipeline orchestration, and pluggable **Accelerate / Metal GEMM + llama.cpp** runners. Oracle-AI (or curl) talks OpenAI HTTP to the master on **:8000**.

70B Q4 is a **RAM pipeline** across machines, not a single-GPU model. 2018 dGPUs (~2–4 GB) are a small hot cache.

```
  iPhone / curl
       |  POST /v1/chat/completions  (SSE)
       v
  MASTER 10.10.0.1 :8000
  |  tokenize, seq_id=k
  |  embed + layers [0, L/3)
  |  KV_local stays on master
  |  send F16 activations  ----------------+
  v                                        |
  WORKER1 10.10.0.2                        |
  |  layers [L/3, 2L/3)                    |
  |  KV_local on W1                        |
  |  send activations  --------------------+-->
  v                                        |
  WORKER2 10.10.0.3                        |
  |  layers [2L/3, L) + lm_head            |
  |  send logits (F32, vocab)  ------------+--> MASTER
  |
  MASTER samples next token, repeats until EOS
  control plane: heartbeat UDP 10.10.0.{1,2,3}:9100
```

## Build

```bash
cmake -S . -B build -DORACLE_BUILD_PYTHON=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Python extension (optional):

```bash
cmake -S . -B build -DORACLE_BUILD_PYTHON=ON
cmake --build build -j
```

Binaries: `build/oracle-engine-master`, `build/oracle-engine-worker`, `build/oracle-engine-bench-net`.

## Run

Single node (phase 4 smoke):

```bash
./build/oracle-engine-master --config configs/cluster.toml --single --runner accelerate
curl -s http://127.0.0.1:8000/v1/models
```

Three-node TB3 mesh: see [`scripts/tb3_bridge.md`](scripts/tb3_bridge.md). Workers:

```bash
./build/oracle-engine-worker --config configs/cluster.toml --id 1 --runner llamacpp
./build/oracle-engine-worker --config configs/cluster.toml --id 2 --runner llamacpp
```

## Phases

| Phase | Script | Gate |
|-------|--------|------|
| 1 Network | `scripts/phase1_net_bench.sh` | >8 Gbps TCP, decode RTT << 5 ms |
| 2 Alloc | `scripts/phase2_alloc_test.sh` | 70B-Q4 layer table fits RAM budgets |
| 3 Tensors | `scripts/phase3_tensor_pass.sh` | ring + GEMM checksum |
| 4 E2E | `scripts/phase4_e2e_infer.sh` | `/v1/models` + short chat |

## Wire format

Packed `TensorHeader` (76 bytes), magic **ORCL**, then raw payload. No protobuf/JSON on the hot path. `writev`/`readv` send header + body. Local hop: POSIX shm SPSC ring.

## Runners

- **Accelerate**: `cblas_sgemm` (identity weights for tests / tiny models).
- **Metal**: `gemm.metal` sgemm for phase-3 tensors. Full 70B transformer kernels are **out of v1**.
- **llama.cpp**: GGUF metadata reader + optional `posix_spawn` RPC; in-process ggml if built with `-DORACLE_HAS_LLAMA_CPP=ON` and the library linked. v1 still owns the activation mesh between **our** workers.

## Layout

`include/oracle/{types,tb3,shard,orch,runner}` · `src/{tb3-transport,mem-shard-manager,orchestrator-core,metal-cpu-runner}` · `apps/` · `configs/cluster.toml`

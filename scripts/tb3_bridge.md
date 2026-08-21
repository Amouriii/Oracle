# Thunderbolt 3 bridge notes (macOS Intel 2018)

Workers clone **only this repo**. Oracle-AI remains an HTTP client aimed at `http://<master>:8000`.

## Addressing

Use Thunderbolt Bridge IPv4, not Wi-Fi:

| Role   | Example IP | Ports                          |
|--------|------------|--------------------------------|
| Master | 10.10.0.1  | TCP 8000 HTTP, TCP 9200 tensors, UDP 9100 heartbeat |
| Worker | 10.10.0.2  | TCP 9200, UDP 9100             |
| Worker | 10.10.0.3  | TCP 9200, UDP 9100             |

System Settings → Network → Thunderbolt Bridge → configure IPv4 manually.

## Bring-up

1. **Wi-Fi off on workers** so the mesh does not hairpin through WLAN.
2. Cable: TB3/USB-C between the Macs (daisy-chain or hub). Confirm `bridge0` / Thunderbolt Bridge is up.
3. Jumbo frames (optional, only if stable):

   ```bash
   sudo ifconfig bridge0 mtu 9000
   ```

   If ping or TCP stalls, revert to 1500. `cluster.toml` `mtu = 9000` is advisory for benches.
4. `TCP_NODELAY` is on in `TB3SocketTransport`. This is **not** NIC-level zero-copy; macOS TCP still copies. The engine avoids extra serialization copies and uses SPSC rings for the local hop.
5. Heartbeats: UDP `10.10.0.{1,2,3}:9100`, carrying each node's live CPU load,
   free RAM and in-flight work. Missing `heartbeat_misses` intervals marks the
   node dead and removes it from scheduling; the master re-dials and
   re-handshakes it with backoff, so a worker that comes back rejoins on its
   own. There is no automatic re-shard: requests are refused with
   `no healthy worker owns layers [a, b)` until the stage is back.
6. Authentication: every node signs its handshake with
   `$ORACLE_CLUSTER_SECRET` (HMAC-SHA256 over a fresh nonce). Export the same
   value on all three machines before starting anything.

## Gate (phase 1)

```bash
# on 10.10.0.2
./build/oracle-engine-bench-net --listen --port 9200
# on 10.10.0.1
scripts/phase1_net_bench.sh 10.10.0.2
```

- `iperf3` + `oracle-engine-bench-net`: **> 8 Gbps** TCP on the bridge.
- Decode activation ping-pong (one hidden state in f16): **RTT well under 5 ms**.

The benchmark sweeps decode-sized through bulk-sized payloads and prints
p50/p99/min round-trip latency alongside one-way throughput, so a link that is
fast in bulk but jittery per-frame is visible rather than averaged away.

Prefill and CPU GEMM dominate; TB3 40 Gbps is not the 70B limiter. 2018 dGPUs (~2–4 GB) are a hot cache only. Weights + KV live in **pooled system RAM** with pipeline-parallel layers.

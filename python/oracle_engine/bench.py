from __future__ import annotations

import argparse
import subprocess
import sys


def net_gate(gbps: float, rtt_ms: float) -> bool:
    return gbps > 8.0 and rtt_ms < 5.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="10.10.0.2")
    args = p.parse_args()
    print(f"run scripts/phase1_net_bench.sh against {args.host}", file=sys.stderr)
    raise SystemExit(subprocess.call(["bash", "scripts/phase1_net_bench.sh", args.host]))


if __name__ == "__main__":
    main()

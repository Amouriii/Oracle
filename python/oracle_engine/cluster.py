from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Node:
    id: int
    role: str
    host: str
    ram_budget_gb: float
    vram_budget_gb: float
    layer_start: int = 0
    layer_end: int = 0


def load_nodes(path: str | Path) -> dict[str, Any]:
    data = tomllib.loads(Path(path).read_text())
    nodes = [Node(**{k: n[k] for k in Node.__dataclass_fields__ if k in n}) for n in data.get("nodes", [])]
    return {"cluster": data.get("cluster", {}), "model": data.get("model", {}), "nodes": nodes}


def print_table(cfg: dict[str, Any]) -> None:
    print(f"cluster={cfg['cluster'].get('name')} nodes={len(cfg['nodes'])}")
    for n in cfg["nodes"]:
        print(f"  id={n.id} role={n.role:6} host={n.host:16} ram={n.ram_budget_gb}G vram={n.vram_budget_gb}G")


def main() -> None:
    p = argparse.ArgumentParser(description="Print TB3 cluster table")
    p.add_argument("--config", default="configs/cluster.toml")
    args = p.parse_args()
    print_table(load_nodes(args.config))


if __name__ == "__main__":
    main()

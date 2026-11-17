"""Traffic profiles.

TKCC's key fidelity point: measure on REAL traffic, not random tensors (random data
activates 13-38% more MoE experts than real traffic). Record a sample of production
requests as JSONL: {"prompt_ids": [...], "output_len": 700}  or {"prompt": "...", "output_len": 700}.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Request:
    prompt_ids: list[int]
    output_len: int


def load_traffic(path: str | Path, limit: int | None = None, tokenizer=None) -> list[Request]:
    reqs: list[Request] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = row.get("prompt_ids")
        if ids is None:
            if tokenizer is None:
                raise ValueError("text prompts need a tokenizer")
            ids = tokenizer.encode(row["prompt"])
        reqs.append(Request(prompt_ids=ids, output_len=int(row["output_len"])))
        if limit and len(reqs) >= limit:
            break
    return reqs


def synth_traffic(path: str | Path, n: int = 32, isl: int = 1024, osl: int = 256,
                  shared_prefix: int = 256, seed: int = 0) -> None:
    """Synthetic placeholder only. Replace with recorded traffic for real runs."""
    rng = random.Random(seed)
    prefix = [rng.randrange(1000, 30000) for _ in range(shared_prefix)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for _ in range(n):
            body = [rng.randrange(1000, 30000) for _ in range(max(isl - shared_prefix, 1))]
            f.write(json.dumps({"prompt_ids": prefix + body,
                                "output_len": max(1, int(rng.gauss(osl, osl * 0.2)))}) + "\n")

#!/usr/bin/env python3
"""Compare one tensor across two local sharded safetensors checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_tensor(root: Path, name: str) -> tuple[Path, str]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        if name in weight_map:
            return root / weight_map[name], name
        matches = sorted(key for key in weight_map if key.endswith(name))
        if len(matches) == 1:
            return root / weight_map[matches[0]], matches[0]
        raise SystemExit(f"Tensor not found or ambiguous in {index_path}: {name}; matches={matches[:10]}")

    single = root / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"No safetensors checkpoint found under {root}")
    from safetensors import safe_open

    with safe_open(single, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if name in keys:
        return single, name
    matches = sorted(key for key in keys if key.endswith(name))
    if len(matches) == 1:
        return single, matches[0]
    raise SystemExit(f"Tensor not found or ambiguous in {single}: {name}; matches={matches[:10]}")


def load_tensor(root: Path, name: str):
    from safetensors import safe_open

    shard, resolved_name = resolve_tensor(root, name)
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(resolved_name)
    return tensor, shard, resolved_name


def stats(tensor) -> dict:
    import numpy as np
    import torch

    values = tensor.detach().float().abs().reshape(-1)
    finite = bool(torch.isfinite(values).all().item())
    if not finite:
        finite_values = values[torch.isfinite(values)]
    else:
        finite_values = values
    if finite_values.numel() == 0:
        return {
            "shape": list(tensor.shape),
            "numel": int(tensor.numel()),
            "finite": False,
        }
    # torch.quantile rejects tensors above an internal element-count limit.
    # NumPy uses a partition-based implementation and handles 3B projection
    # matrices without sampling, so the reported quantiles remain exact.
    quantiles = np.quantile(
        finite_values.numpy(),
        [0.5, 0.99, 0.999],
        method="linear",
    )
    return {
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "finite": finite,
        "abs_max": float(finite_values.max().item()),
        "abs_mean": float(finite_values.mean().item()),
        "abs_p50": float(quantiles[0]),
        "abs_p99": float(quantiles[1]),
        "abs_p999": float(quantiles[2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch

    left, left_shard, left_name = load_tensor(args.left.resolve(), args.tensor)
    right, right_shard, right_name = load_tensor(args.right.resolve(), args.tensor)
    if left.shape != right.shape:
        raise SystemExit(f"Shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")

    left_float = left.float()
    right_float = right.float()
    delta = right_float - left_float
    changed = delta.ne(0)
    changed_count = int(changed.sum().item())
    ratios = torch.empty(0, dtype=torch.float32)
    valid_ratio = changed & left_float.ne(0)
    if valid_ratio.any():
        ratios = (right_float[valid_ratio].abs() / left_float[valid_ratio].abs()).float()

    result = {
        "left": {
            "root": str(args.left.resolve()),
            "shard": str(left_shard),
            "tensor": left_name,
            "stats": stats(left),
        },
        "right": {
            "root": str(args.right.resolve()),
            "shard": str(right_shard),
            "tensor": right_name,
            "stats": stats(right),
        },
        "difference": {
            "stats": stats(delta),
            "changed_count": changed_count,
            "changed_fraction": changed_count / left.numel(),
            "unchanged_count": int(left.numel()) - changed_count,
            "finite": bool(torch.isfinite(delta).all().item()),
        },
        "changed_abs_ratio": None,
    }
    if ratios.numel():
        import numpy as np

        ratio_quantiles = np.quantile(
            ratios.numpy(),
            [0.0, 0.5, 0.99, 1.0],
            method="linear",
        )
        result["changed_abs_ratio"] = {
            "count": int(ratios.numel()),
            "min": float(ratio_quantiles[0]),
            "median": float(ratio_quantiles[1]),
            "p99": float(ratio_quantiles[2]),
            "max": float(ratio_quantiles[3]),
        }

    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

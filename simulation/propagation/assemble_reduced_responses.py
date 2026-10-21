#!/usr/bin/env python3
"""Assemble the propagation capacity responses from its generating studies."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
MATERIALS = ("rotated_stripes", "double_inclusion", "smooth_checker")
RANKS = (0, 4, 8, 16, 32, 64, 96, 128)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Check the assembled responses against the generating studies.")
    args = parser.parse_args()
    paths = [SOURCE / "studies/patch_2d/receiver_and_transfer_responses.npz",
             SOURCE / "studies/patch_2d/capacity_followup_responses.npz"]
    with np.load(paths[0]) as primary, np.load(paths[1]) as capacity:
        arrays = {"time": primary["time"]}
        mapping = {"time": {"file": paths[0].relative_to(SOURCE).as_posix(), "key": "time"}}
        for material in MATERIALS:
            key = f"{material}_full32"
            source_key = f"{material}_n32_full_y"
            arrays[key] = primary[source_key]
            mapping[key] = {"file": paths[0].relative_to(SOURCE).as_posix(), "key": source_key}
            for m in RANKS:
                source_key = f"{material}_n32_r{m}_y" if m <= 32 else f"{material}_fixed_interface_modes_r{m}_y"
                source, path = (primary, paths[0]) if m <= 32 else (capacity, paths[1])
                key = f"{material}_r{m}"
                arrays[key] = source[source_key]
                mapping[key] = {"file": path.relative_to(SOURCE).as_posix(), "key": source_key}
    output = ROOT / "results/reduced_responses.npz"
    maximum = 0.0
    if args.verify:
        with np.load(output) as stored:
            if set(stored.files) != set(arrays):
                raise RuntimeError("Capacity response keys differ from the generating studies.")
            for name, expected in arrays.items():
                np.testing.assert_allclose(stored[name], expected, rtol=1e-12, atol=1e-13)
                maximum = max(maximum, float(np.max(np.abs(stored[name] - expected))))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name("reduced_responses.partial.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(output)
    result = {"passed": True, "mode": "verify" if args.verify else "assemble", "arrays": len(arrays),
              "maximum_absolute_difference": maximum if args.verify else None, "array_sources": mapping,
              "source_sha256": {p.relative_to(SOURCE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    report = ROOT / "results/capacity_trace_linkage.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"array_sources", "source_sha256"}}, indent=2))


if __name__ == "__main__":
    main()

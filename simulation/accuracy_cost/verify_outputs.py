#!/usr/bin/env python3
"""Check completeness, raw-array shapes, and recorded computational hashes."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
cfg = json.loads((HERE / "config.json").read_text())
digest = hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest()
env = json.loads((OUT / "environment.json").read_text())
code_match = {name: hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest() == expected
              for name, expected in env["code_sha256"].items()}
assert all(code_match.values())
expected = len(cfg["materials"]) * len(cfg["grids"]) * (1 + 2 * len(cfg["modes_per_patch"]))
records = sorted(OUT.glob("*_timing.json"))
assert len(records) == expected
assert len(list(OUT.glob("*.npz"))) == expected
arrays = []
for path in records:
    record = json.loads(path.read_text())
    assert record["config_sha256"] == digest
    assert record["cpu_affinity"] == env["cpu_affinity"]
    assert len(record["repetitions"]) == cfg["repetitions"]
    assert all(np.isfinite(rep[key]) and rep[key] >= 0 for rep in record["repetitions"] for key in cfg["timing_stages"])
    npz_path = path.with_name(path.name.replace("_timing.json", ".npz"))
    n, r = record["grid"], record["dimension"]
    with np.load(npz_path) as data:
        assert data["receiver_pressure"].shape == (cfg["samples"], 4, 3)
        assert data["time"].shape == (cfg["samples"],)
        assert data["mass"].shape == (n * n,)
        assert np.min(data["mass"]) > 0
        if record["method"] == "full_sparse_exponential":
            assert data["mass_weighted_pressure"].shape == (cfg["samples"], n * n, 3)
        else:
            assert data["orthonormal_basis"].shape == (n * n, r)
            assert data["reduced_eigenvectors"].shape == (r, r)
            assert data["reduced_eigenvalues"].shape == (r,)
            assert data["initial_modal_coefficients"].shape == (r, 3)
        assert all(np.all(np.isfinite(data[key])) for key in data.files)
        arrays.append({"file": npz_path.name, "bytes": npz_path.stat().st_size,
                       "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                       "arrays": {key: list(data[key].shape) for key in data.files}})
result = {"status": "pass", "setting_count": expected, "timing_repetition_count": expected * cfg["repetitions"],
          "raw_npz_count": len(arrays), "config_sha256": digest, "computational_code_hashes_match": code_match,
          "actual_cpu_affinity": env["cpu_affinity"], "raw_arrays": arrays}
(OUT / "data_integrity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "raw_arrays"}, indent=2))

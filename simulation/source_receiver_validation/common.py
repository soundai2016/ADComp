"""Shared configuration and serialization for the second port layout."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from unequal_speed_interface import kernel as acoustic
from unequal_speed_interface.io import atomic_write, save_arrays

CFG = json.loads((HERE / "config.json").read_text())
acoustic.CFG.clear()
acoustic.CFG.update(CFG)
OUT = HERE / "results"


def config_digest():
    return hashlib.sha256((HERE / "config.json").read_bytes()).hexdigest()


def dump(path, value):
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def csv_write(path, rows):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, buffer.getvalue().encode())


def residue(source, receiver):
    return np.einsum("rj,js->jrs", receiver, source)


def response(times, omega, residues):
    return np.einsum("tj,jrs->trs", np.cos(times[:, None] * omega[None]), residues, optimize=True)


def relative(actual, reference):
    return float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))


def relative_gaps(omega):
    return np.minimum(np.r_[np.inf, np.diff(omega)], np.r_[np.diff(omega), np.inf]) / np.maximum(omega, 1e-100)


def channel_side(receiver, source):
    source_left = CFG["source_centres"][source][0] < CFG["interface_x"]
    receiver_left = CFG["receiver_centres"][receiver][0] < CFG["interface_x"]
    return "same_side" if source_left == receiver_left else "opposite_sides"

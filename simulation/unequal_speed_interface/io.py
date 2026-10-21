"""Complete in-memory encoding followed by atomic artifact replacement."""
from io import BytesIO
import os
from pathlib import Path
import zipfile
import numpy as np


def atomic_write(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name+f".{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size != len(payload):
            raise IOError("Incomplete temporary artifact write")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_arrays(path, **arrays):
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        if set(archive.namelist()) != {key+".npy" for key in arrays}:
            raise IOError("Incomplete NPZ encoding")
    atomic_write(path, payload)

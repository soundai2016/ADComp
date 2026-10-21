"""Record numerical runs, environments, stages, and metrics."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

os.environ.setdefault("LOGURU_AUTOINIT", "False")
from loguru import logger

SCHEMA_VERSION = 1
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND",
    "MKL_NUM_THREADS", "MKL_DYNAMIC", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
    "NPY_DISABLE_CPU_FEATURES",
)
EXCLUDED_DIRECTORIES = {
    ".git", ".venv", "venv", "__pycache__", "results", "logs", "cache", "checks", "data", "models",
}
EPHEMERAL_KEYS = {
    "run_id", "parent_run_id", "run_name", "log_dir", "output_dir", "workdir", "models_dir",
    "source_root", "input_paths", "command", "cwd", "timestamp", "started_at_utc",
    "finished_at_utc", "environment", "hardware", "platform", "cpu_affinity",
}


class RunInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Received {signal.Signals(signum).name}")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_run_id(label=""):
    if label and (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", label) is None or ".." in label):
        raise ValueError("Run label must use letters, digits, underscores, dots or hyphens, without '..'.")
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "-".join(part for part in (prefix, label, uuid.uuid4().hex[:8]) if part)


def _native(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    if hasattr(value, "item"):
        return _native(value.item())
    raise TypeError(f"Unsupported record value: {type(value).__name__}")


def _fingerprint(value):
    payload = json.dumps(_native(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _numerical_configuration(value):
    if isinstance(value, dict):
        return {key: _numerical_configuration(item) for key, item in value.items()
                if key.lower() not in EPHEMERAL_KEYS and not key.lower().endswith("cpu_affinity")}
    if isinstance(value, (list, tuple)):
        return [_numerical_configuration(item) for item in value]
    return _native(value)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(_native(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


atomic_json = _write_json


def _sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _report_signature(path):
    if not path.is_file():
        return None
    return path.stat().st_mtime_ns, _sha256(path)


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _check_report(value, specification):
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    for key, expected in specification.get("equals", {}).items():
        actual = value.get(key)
        matches = actual is expected if isinstance(expected, bool) else actual == expected
        if not matches:
            raise ValueError(f"{key} must be {expected!r}; received {actual!r}")
    for key in specification.get("all_true", ()):
        checks = value.get(key)
        if not isinstance(checks, dict) or not checks or any(item is not True for item in checks.values()):
            raise ValueError(f"{key} must contain nonempty, successful checks")
    for key, limit in specification.get("below", {}).items():
        actual = value.get(key)
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(actual) or not 0 <= actual < limit):
            raise ValueError(f"{key} must be finite and between 0 and {limit}; received {actual!r}")


def _read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def _probe(command, env, timeout=8):
    try:
        result = subprocess.run(command, env=env, text=True, encoding="utf-8", errors="replace",
                                capture_output=True, timeout=timeout, check=False)
        return {"status": "ok" if result.returncode == 0 else "failed",
                "exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "reason": type(error).__name__}


def _environment(env):
    cpu_text = _read_text("/proc/cpuinfo") or ""
    model = re.search(r"^(?:model name|Hardware)\s*:\s*(.+)$", cpu_text, re.M)
    memory_text = _read_text("/proc/meminfo") or ""
    memory = re.search(r"^MemTotal:\s*(\d+)\s+kB", memory_text, re.M)
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    packages = sorted({(dist.metadata.get("Name", "unknown"), dist.version)
                       for dist in importlib.metadata.distributions()})
    backend_code = (
        "import json,numpy,scipy.linalg; from threadpoolctl import threadpool_info; "
        "print(json.dumps({'numpy':numpy.__version__,'thread_pools':threadpool_info()}))"
    )
    backend = _probe([sys.executable, "-c", backend_code], env)
    if backend["status"] == "ok":
        try:
            backend = {"status": "ok", **json.loads(backend["stdout"])}
        except json.JSONDecodeError:
            backend = {"status": "unavailable", "reason": "Invalid probe output"}
    gpu_command = shutil.which("nvidia-smi", path=env.get("PATH"))
    accelerators = {"nvidia": {"query_available": gpu_command is not None},
                    "ascend": {"npu_smi_available": shutil.which("npu-smi", path=env.get("PATH")) is not None}}
    if gpu_command:
        result = _probe([gpu_command, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], env, 3)
        accelerators["nvidia"].update(status=result["status"])
        if result["status"] == "ok":
            devices = []
            for line in result["stdout"].splitlines():
                parts = [part.strip() for part in line.rsplit(",", 2)]
                if len(parts) == 3:
                    devices.append(dict(name=parts[0], driver=parts[1], memory_mib=parts[2]))
            accelerators["nvidia"]["devices"] = devices
    return {
        "captured_at_utc": utc_now(),
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(),
                   "executable": sys.executable, "compiler": platform.python_compiler()},
        "system": {"os": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "cpu": {"model": model.group(1) if model else platform.processor(),
                "logical_count": os.cpu_count(), "affinity": affinity},
        "memory": {"physical_bytes": int(memory.group(1)) * 1024 if memory else None},
        "cgroup": {name: _read_text(path) for name, path in (
            ("cpu_max", "/sys/fs/cgroup/cpu.max"),
            ("cpuset_cpus_effective", "/sys/fs/cgroup/cpuset.cpus.effective"),
            ("memory_max", "/sys/fs/cgroup/memory.max"),
            ("cpu_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            ("cpu_period_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
            ("memory_limit_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        )},
        "packages": {name: version for name, version in packages},
        "thread_environment": {key: env[key] for key in THREAD_VARIABLES if key in env},
        "numerical_backend": backend,
        "backend_probe_scope": "Fresh Python process with the supplied stage environment; individual kernels may apply additional thread controls.",
        "accelerator_inventory": accelerators,
    }


def _environment_fingerprint(snapshot):
    stable = json.loads(json.dumps(snapshot))
    stable.pop("captured_at_utc", None)
    stable.get("python", {}).pop("executable", None)
    backend = stable.get("numerical_backend", {})
    for pool in backend.get("thread_pools", []):
        pool.pop("filepath", None)
    return _fingerprint(stable)


class RunLog:
    """Record a run's inputs, configuration, environment, stages, and metrics."""

    def __init__(self, log_dir, *, mode, command, source_root, input_paths=None,
                 configuration=None, run_id=None, parent_run_id=None, environment=None):
        self.log_dir = Path(log_dir).resolve()
        self.source_root = Path(source_root).resolve()
        self.input_paths = [Path(path).resolve() for path in (input_paths or [])]
        self.configuration = _native(configuration or {})
        self.environment = dict(os.environ if environment is None else environment)
        self.run_id = run_id or new_run_id()
        self._handlers = []
        self._clock = None
        self._previous_sigterm = None
        self.record = dict(schema_version=SCHEMA_VERSION, run_id=self.run_id, parent_run_id=parent_run_id,
                           mode=mode, command=list(map(str, command)), status="created", stages=[], metrics=[],
                           verification_reports=[])
        self.logger = logger.bind(run_id=self.run_id, parent_run_id=parent_run_id)

    def _manifest(self):
        source, configurations, inputs = {}, {}, {}
        generated_paths = [self.log_dir]
        for key in ("output_dir", "workdir", "log_dir", "models_dir"):
            if isinstance(self.configuration.get(key), str):
                generated_paths.append(Path(self.configuration[key]).resolve())
        generated_paths = [path for path in generated_paths
                           if path != self.source_root and path not in self.source_root.parents]

        def generated(path):
            path = path.resolve()
            return any(path == directory or directory in path.parents for directory in generated_paths)

        for directory, directories, filenames in os.walk(self.source_root, followlinks=False):
            directory = Path(directory)
            directories[:] = sorted(name for name in directories
                                    if name not in EXCLUDED_DIRECTORIES and not generated(directory / name))
            for filename in sorted(filenames):
                path = directory / filename
                if not path.is_file() or generated(path):
                    continue
                if path.suffix not in {".py", ".json", ".toml"} and path.name != "requirements.txt":
                    continue
                name = path.relative_to(self.source_root).as_posix()
                source[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
                if path.suffix == ".json":
                    configurations[name] = json.loads(path.read_text())
        for supplied in self.input_paths:
            paths = sorted(path for path in supplied.rglob("*") if path.is_file()) if supplied.is_dir() else [supplied]
            for path in paths:
                try:
                    name = path.relative_to(self.source_root).as_posix()
                except ValueError:
                    name = (Path(supplied.name) / path.relative_to(supplied)).as_posix() if supplied.is_dir() else path.name
                if name in inputs and inputs[name]["sha256"] != _sha256(path):
                    raise ValueError(f"Ambiguous input name: {name}")
                inputs[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        code = {name: item["sha256"] for name, item in source.items() if not name.endswith(".json")}
        numerical = {"mode": self.record["mode"], "configuration": _numerical_configuration(self.configuration),
                     "configs": _numerical_configuration(configurations), "code": code,
                     "inputs": {name: item["sha256"] for name, item in inputs.items()}}
        manifest = dict(schema_version=SCHEMA_VERSION, source=source, inputs=inputs,
                        configuration=self.configuration, numerical_configs=configurations,
                        source_fingerprint=_fingerprint(source), configuration_fingerprint=_fingerprint(numerical))
        _write_json(self.log_dir / "manifest.json", manifest)
        self.record.update(source_fingerprint=manifest["source_fingerprint"],
                           configuration_fingerprint=manifest["configuration_fingerprint"])

    def __enter__(self):
        if self.log_dir.exists() and any(self.log_dir.iterdir()):
            raise FileExistsError(f"Run log directory is not empty: {self.log_dir}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        belongs = lambda record: record["extra"].get("run_id") == self.run_id
        console = lambda record: belongs(record) and record["extra"].get("console", True)
        format_ = "{time:YYYY-MM-DDTHH:mm:ss.SSS!UTC}Z | {level: <7} | {message}"
        common = dict(format=format_, diagnose=False, backtrace=False, enqueue=True, catch=False)
        self._handlers.append(logger.add(sys.stderr, level="INFO", filter=console, **common))
        self._handlers.append(logger.add(self.log_dir / "run.log", level="DEBUG", filter=belongs, encoding="utf-8", **common))
        self._handlers.append(logger.add(self.log_dir / "events.jsonl", level="DEBUG", filter=belongs,
                                         serialize=True, encoding="utf-8", **common))
        self._clock = time.perf_counter()
        if threading.current_thread() is threading.main_thread():
            self._previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._on_sigterm)
        self.record.update(status="running", started_at_utc=utc_now())
        self._save()
        try:
            self._manifest()
            snapshot = _environment(self.environment)
            snapshot.update(schema_version=SCHEMA_VERSION, run_id=self.run_id)
            self.record["environment_fingerprint"] = _environment_fingerprint(
                {key: value for key, value in snapshot.items() if key != "run_id"})
            snapshot["environment_fingerprint"] = self.record["environment_fingerprint"]
            _write_json(self.log_dir / "environment.json", snapshot)
            self._save()
            self.event("run_started", mode=self.record["mode"], source_fingerprint=self.record["source_fingerprint"],
                       configuration_fingerprint=self.record["configuration_fingerprint"],
                       environment_fingerprint=self.record["environment_fingerprint"])
            self.logger.info("Run {} | {} | Python {} | CPU {}", self.run_id, self.record["mode"],
                             snapshot["python"]["version"], snapshot["cpu"]["model"])
            return self
        except BaseException as error:
            self._finish("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error)
            raise

    def _save(self):
        _write_json(self.log_dir / "summary.json", self.record)


    @staticmethod
    def _on_sigterm(signum, frame):
        raise RunInterrupted(signum)

    def event(self, name, **fields):
        self.logger.bind(event=name, **_native(fields)).info("{}", name)

    def metrics(self, name, values, *, step=None, units=None, context=None):
        values = _native(values)
        if not isinstance(values, dict) or not values:
            raise ValueError("Metrics must contain named scalar values.")
        for key, value in values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Metric {key} must be a finite number.")
        if step is not None and (not isinstance(step, int) or step < 0):
            raise ValueError("Metric step must be a nonnegative integer.")
        item = dict(schema_version=SCHEMA_VERSION, run_id=self.run_id, timestamp_utc=utc_now(),
                    name=name, step=step, values=values, units=_native(units or {}), context=_native(context or {}))
        with (self.log_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        self.record["metrics"].append(item)
        self._save()
        self.logger.bind(event="metrics", metric=item).info("Metrics {}: {}", name, values)
        return item

    def _verify_reports(self, stage, specifications, previous):
        for index, (specification, before) in enumerate(zip(specifications, previous), 1):
            path = specification["path"]
            report = dict(stage_id=stage["stage_id"], stage=stage["name"], report=str(path), status="pending",
                          checks={key: value for key, value in specification.items() if key != "path"})
            self.record["verification_reports"].append(report)
            stage.setdefault("verification_reports", []).append(report)
            try:
                if not path.is_file():
                    raise ValueError("Required report was not produced")
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                signature = path.stat().st_mtime_ns, digest
                snapshot = self.log_dir / "verification" / f"{stage['stage_id']:02d}_{index:02d}_{path.name}"
                snapshot.parent.mkdir(exist_ok=True)
                snapshot.write_bytes(raw)
                report.update(snapshot=snapshot.relative_to(self.log_dir).as_posix(), sha256=digest)
                if signature == before:
                    raise ValueError("Required report was not refreshed by this stage")
                value = json.loads(raw, parse_constant=_reject_constant)
                _check_report(value, specification)
                report["status"] = "passed"
                self.logger.bind(event="verification_report", **report).info("Verified {}", path.name)
            except Exception as error:
                report.update(status="failed", error=str(error))
                self.logger.bind(event="verification_report", **report).error("Verification failed: {}", path.name)
                raise RuntimeError(f"Verification report {path}: {error}") from error
            finally:
                self._save()

    @staticmethod
    def _stop(process):
        if process.poll() is not None:
            return
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()

    def run_stage(self, name, command, *, cwd, env=None, log_name=None, console=True, verification_reports=None):
        command = list(map(str, command))
        stage_id = len(self.record["stages"]) + 1
        environment = dict(self.environment if env is None else env)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "stage"
        filename = log_name or f"{stage_id:02d}_{slug}.log"
        if Path(filename).name != filename:
            raise ValueError("Stage log_name must be a filename.")
        path = self.log_dir / "stages" / filename
        path.parent.mkdir(exist_ok=True)
        stage = dict(stage_id=stage_id, name=name, command=command, cwd=str(Path(cwd).resolve()),
                     log=path.relative_to(self.log_dir).as_posix(), started_at_utc=utc_now(), status="running",
                     timing_scope="subprocess_wall",
                     timing_includes=["process_startup", "computation", "output_capture", "logging"],
                     thread_environment={key: environment[key] for key in THREAD_VARIABLES if key in environment})
        self.record["stages"].append(stage)
        self._save()
        self.logger.bind(event="stage_started", stage_id=stage_id, stage_name=name).info("[{}] {}", stage_id, name)
        started = time.perf_counter()
        messages = queue.Queue()
        process = None
        threads = []
        process_elapsed = None
        specifications = []
        for specification in verification_reports or ():
            reported = Path(specification["path"])
            if not reported.is_absolute():
                reported = Path(cwd) / reported
            specifications.append(dict(specification, path=reported.resolve()))

        def read(stream, channel):
            try:
                for line in stream:
                    messages.put((channel, line))
            finally:
                stream.close()
                messages.put((channel, None))

        with path.open("w", encoding="utf-8") as transcript:
            def output(channel, line):
                if line is not None:
                    transcript.write(f"[{channel}] {line}")
                    if not line.endswith("\n"):
                        transcript.write("\n")
                    transcript.flush()
                    self.logger.bind(event="stage_output", stage_id=stage_id, stage_name=name,
                                     stream=channel, console=console).info("{}", line.rstrip("\r\n"))

            try:
                previous = [_report_signature(specification["path"]) for specification in specifications]
                process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                           bufsize=1, start_new_session=os.name == "posix")
                stage["pid"] = process.pid
                for channel, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                    thread = threading.Thread(target=read, args=(stream, channel), daemon=True)
                    thread.start()
                    threads.append(thread)
                completed_streams = 0
                while completed_streams < 2:
                    try:
                        channel, line = messages.get(timeout=.2)
                    except queue.Empty:
                        continue
                    if line is None:
                        completed_streams += 1
                    else:
                        output(channel, line)
                exit_code = process.wait()
                process_elapsed = time.perf_counter() - started
                stage.update(exit_code=exit_code, status="completed" if exit_code == 0 else "failed")
                if exit_code:
                    raise subprocess.CalledProcessError(exit_code, command)
                if specifications:
                    checked_at = time.perf_counter()
                    try:
                        self._verify_reports(stage, specifications, previous)
                    finally:
                        stage["verification_elapsed_seconds"] = time.perf_counter() - checked_at
            except BaseException as error:
                if process is not None:
                    self._stop(process)
                stage.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                             exit_code=process.returncode if process is not None else None,
                             error_type=type(error).__name__, error=str(error))
                self.logger.bind(event="stage_failed", stage_id=stage_id, stage_name=name,
                                 exit_code=stage["exit_code"]).opt(exception=error).error("{}: {}", name, stage["status"])
                raise
            finally:
                for thread in threads:
                    thread.join(timeout=5)
                while True:
                    try:
                        channel, line = messages.get_nowait()
                    except queue.Empty:
                        break
                    output(channel, line)
                stage.update(finished_at_utc=utc_now(),
                             elapsed_seconds=process_elapsed if process_elapsed is not None else time.perf_counter() - started)
                self._save()
                self.logger.bind(event="stage_finished", stage_id=stage_id, stage_name=name,
                                 status=stage["status"], exit_code=stage.get("exit_code"),
                                 elapsed_seconds=stage["elapsed_seconds"]).info(
                    "[{}] {} | {} | {:.3f} s", stage_id, name, stage["status"], stage["elapsed_seconds"])
        return stage

    def _finish(self, status, error=None):
        self.record.update(status=status, finished_at_utc=utc_now(),
                           elapsed_seconds=time.perf_counter() - self._clock if self._clock is not None else 0.)
        if error is not None:
            self.record.update(error_type=type(error).__name__, error=str(error))
            if isinstance(error, RunInterrupted):
                self.record["termination_signal"] = error.signum
            self.logger.bind(event="run_failed", status=status).opt(exception=error).error("Run {}: {}", self.run_id, status)
        else:
            self.logger.bind(event="run_finished", status=status).info("Run {}: {}", self.run_id, status)
        self._save()
        logger.complete()
        for handler in self._handlers:
            logger.remove(handler)
        self._handlers.clear()
        if self._previous_sigterm is not None:
            signal.signal(signal.SIGTERM, self._previous_sigterm)
            self._previous_sigterm = None

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        else:
            status = "completed" if all(stage["status"] == "completed" for stage in self.record["stages"]) else "failed"
        self._finish(status, error)
        return False

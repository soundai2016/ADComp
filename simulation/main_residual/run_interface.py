#!/usr/bin/env python3
"""Evaluate residual bounds for the aligned impedance interface."""
import importlib.util
from run import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", default=[32, 64, 96, 128])
    parser.add_argument("--cpus", type=int, nargs="+", default=CFG["cpu_affinity"])
    args = parser.parse_args()
    if hasattr(os, "sched_getaffinity"):
        available = set(os.sched_getaffinity(0))
        os.sched_setaffinity(0, set(args.cpus) & available or set(sorted(available)[:2]))
    case_dir = HERE.parent / "interface_case"
    spec = importlib.util.spec_from_file_location("interface_acoustic_kernel", case_dir / "kernel.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    name = "aligned_impedance_step"
    orders = [32, 64, 128]
    rows, guards, models, references = [], [], [], []
    start = time.perf_counter()
    for n in args.grids:
        for method in CFG["methods"]:
            for m in orders:
                path = case_dir / "results" / f"{name}_n{n}_{method}_m{m}.npz"
                if not path.exists():
                    raise FileNotFoundError(path)
    for n in args.grids:
        print(f"full state {name} n={n}", flush=True)
        X, Y, rho, bulk, mass, C, J = kernel.build_sparse(n)
        B, R = kernel.source_receiver(n, mass)
        full, record = full_endpoints(n, name, C, J, B)
        references.append(record)
        for method in CFG["methods"]:
            for m in orders:
                path = case_dir / "results" / f"{name}_n{n}_{method}_m{m}.npz"
                rr, gg, audit = evaluate_model(path, n, name, method, m, C, B, full)
                rows.extend(rr); guards.extend(gg); models.append(audit)
                print(f"  {method} m={m}: worst T=8 error {max(r['actual_relative_state_error'] for r in rr if r['time']==8):.5g}", flush=True)
    summary, selections = summarize(rows)
    summary.update({"wall_seconds": time.perf_counter() - start,
                    "main_residual_config_sha256": file_hash(HERE / "config.json"),
                    "interface_config_sha256": file_hash(case_dir / "config.json")})
    csv_write(OUT / "interface_bounds.csv", rows)
    csv_write(OUT / "interface_selection.csv", selections)
    dump(OUT / "interface_summary.json", summary)
    dump(OUT / "interface_floating_point_guards.json", guards)
    dump(OUT / "interface_model_integrity.json", models)
    dump(OUT / "interface_full_reference_checks.json", references)
    print(json.dumps({k: v for k, v in summary.items() if k not in ["groups", "selection_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

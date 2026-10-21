#!/usr/bin/env python3
"""Summarize residual bounds and acceptance for the two reduced spaces."""
import csv
import numpy as np
from run import OUT, CFG, summarize, csv_write, dump


def read_rows(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    integer = {"grid", "modes_per_patch", "dimension", "source"}
    strings = {"material", "method"}
    for row in rows:
        for key, value in row.items():
            if key in strings:
                continue
            if value in {"True", "False"}:
                row[key] = value == "True"
            else:
                row[key] = int(value) if key in integer else float(value)
    return rows


def main():
    rows = read_rows(OUT / "bounds.csv")
    extra = OUT / "interface_bounds.csv"
    if extra.exists():
        rows += read_rows(extra)
    summary, selections = summarize(rows)
    dump(OUT / "combined_summary.json", summary)
    csv_write(OUT / "combined_selection.csv", selections)
    csv_write(OUT / "combined_bounds.csv", rows)
    selected = [row for row in rows if row["time"] == 8]
    table_rows = []
    for family, members in [("Smooth media", CFG["materials"]),
                            ("Impedance step", ["aligned_impedance_step"])]:
        for n in sorted({row["grid"] for row in selected if row["material"] in members}):
            for method in ("craig_bampton", "rational_krylov"):
                group = [row for row in selections if row["material"] in members and row["grid"] == n
                         and row["method"] == method and row["time"] == 8]
                if not group:
                    continue
                record = {"family": family, "grid": n, "method": method}
                for threshold, label in [(.05, "five"), (.1, "ten")]:
                    at_threshold = [row for row in group if row["threshold"] == threshold]
                    record[label + "_actual"] = sum(row["any_actual_model_meets_threshold"] for row in at_threshold)
                    record[label + "_bound"] = sum(row["accepted"] for row in at_threshold)
                    record["queries"] = len(at_threshold)
                matching = [row for row in selected if row["material"] in members and row["grid"] == n
                            and row["method"] == method]
                record["median_effectivity"] = float(np.median([row["bound_effectivity"] for row in matching]))
                table_rows.append(record)
    csv_write(OUT / "acceptance_table.csv", table_rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export numerical tables from results and model parameters."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

HERE = Path(__file__).resolve().parent
MATERIALS = ("rotated_stripes", "double_inclusion", "smooth_checker")
NAMES = dict(zip(MATERIALS, ("Rotated stripes", "Double inclusion", "Smooth checker")))
TIMES = (0.5, 2.0, 8.0)
DEFAULT = "mac0.90_gap0.0050"
CASES = {
    "Source": (128, 32, 2.0, 1, 4, (0, 0), (1, 0)),
    "Receiver": (64, 32, 8.0, 3, 4, (0, 0), (0, 1)),
}
DATA_FILES = {
    "cost": "cost_selections.csv",
    "residual": "residual_acceptance.csv",
    "paths": "residue_paths.csv",
    "grids": "grid_comparisons.csv",
    "ports": "ports.csv",
    "media": "material_coefficients.csv",
}
TABLES = {
    "counts": ("I", "main.tex", "tab:counts"),
    "selection": ("II", "main.tex", "tab:selection"),
    "media": ("S1", "supplementary.tex", "tab:media"),
    "ports": ("S2", "supplementary.tex", "tab:ports"),
    "source_thresholds": ("S3", "supplementary.tex", "tab:supp_relocation_thresholds"),
    "receiver_thresholds": ("S4", "supplementary.tex", "tab:supp_relocation_thresholds_receiver"),
    "residue_paths": ("S5", "supplementary.tex", "tab:supp_relocation_paths"),
    "modal_budgets": ("S6", "supplementary.tex", "tab:supp_q_decomposition"),
    "sweep_counts": ("S7", "supplementary.tex", "tab:sweep_complete"),
    "sweep_stability": ("S8", "supplementary.tex", "tab:sweep_stable"),
    "grid_differences": ("S9", "supplementary.tex", "tab:grid_differences"),
    "residual_acceptance": ("S10", "supplementary.tex", "tab:bound"),
    "cost_selections": ("S11", "supplementary.tex", "tab:cost_full"),
}
BASIS = {
    "s": "np.sin(4 * np.pi * (0.8 * X + 0.6 * Y))",
    "a": "a", "b": "b", "z": "z",
    "cos_2pi_x": "np.cos(2 * np.pi * X)",
    "cos_2pi_y": "np.cos(2 * np.pi * Y)",
    "sin_2pi_x": "np.sin(2 * np.pi * X)",
    "sin_2pi_y": "np.sin(2 * np.pi * Y)",
}
TEX_BASIS = {
    "s": "s", "a": "a", "b": "b", "z": "z",
    "cos_2pi_x": r"\cos(2\pi x)", "cos_2pi_y": r"\cos(2\pi y)",
    "sin_2pi_x": r"\sin(2\pi x)", "sin_2pi_y": r"\sin(2\pi y)",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(rows):
    if len(rows) != 1:
        raise ValueError(f"Expected one matching row, found {len(rows)}")
    return rows[0]


def tex_row(*cells):
    return " & ".join(map(str, cells)) + r"\\"


def material_parameters(path):
    """Read the coefficients of the analytic fields directly from the solver."""
    basis = {ast.dump(ast.parse(expr, mode="eval").body): key for key, expr in BASIS.items()}

    def terms(node, sign=1):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            yield from terms(node.left, sign)
            yield from terms(node.right, sign if isinstance(node.op, ast.Add) else -sign)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            yield basis[ast.dump(node.right)], sign * float(ast.literal_eval(node.left))
        else:
            raise ValueError("Unrecognized analytic material expression")

    function = one([node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "material"])
    rows = []
    for branch in ast.walk(function):
        if not isinstance(branch, ast.If) or not isinstance(branch.test, ast.Compare):
            continue
        test = branch.test
        if not (isinstance(test.left, ast.Name) and test.left.id == "name"
                and isinstance(test.comparators[0], ast.Constant)):
            continue
        material = test.comparators[0].value
        if material not in MATERIALS:
            continue
        for statement in branch.body:
            if not isinstance(statement, ast.Assign) or not isinstance(statement.targets[0], ast.Name):
                continue
            field = {"rho": "log_density", "bulk": "log_bulk_modulus"}.get(statement.targets[0].id)
            if field is None:
                continue
            for order, (name, coefficient) in enumerate(terms(statement.value.args[0])):
                rows.append(dict(material=material, field=field, order=order,
                                 basis=name, coefficient=coefficient))
    if len(rows) != 14:
        raise ValueError("Expected 14 analytic material coefficients")
    return rows


def port_parameters(root):
    layouts = [json.loads((root / module / "config.json").read_text())
               for module in ("unequal_speed_interface", "source_receiver_validation")]
    rows = []
    for prefix, field in (("S", "source_centres"), ("R", "receiver_centres")):
        for index, (before, after) in enumerate(zip(layouts[0][field], layouts[1][field]), start=1):
            rows.append(dict(port=f"{prefix}{index}", layout0_x=before[0], layout0_y=before[1],
                             layout1_x=after[0], layout1_y=after[1]))
    return rows


def simulation_inputs(root):
    paths = {
        "cost": "grid_cost_diagnostics/results/richardson_selections.csv",
        "residual": "main_residual/results/acceptance_table.csv",
        "paths": "frequency_diagnostics/results/threshold_contrasts.csv",
        "grids": "propagation/results/grid_comparisons.csv",
    }
    data = {name: read_csv(root / path) for name, path in paths.items()}
    data["ports"] = port_parameters(root)
    data["media"] = material_parameters(root / "acoustic_core.py")
    return data


def bundled_inputs(root):
    expected = json.loads((root / "inputs.json").read_text())["sha256"]
    for name in DATA_FILES.values():
        if sha256(root / name) != expected[name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    return {key: read_csv(root / name) for key, name in DATA_FILES.items()}


def count_table(rows, metric):
    output, tex = [], []
    for end in TIMES:
        row = dict(time_end=end)
        for group in ("channel", "aggregate"):
            for factor in ("source", "receiver"):
                match = one([r for r in rows if float(r["time_end"]) == end
                             and r["changed_positions"] == factor and r["group"] == group])
                row[f"{factor}_{group}"] = int(match[metric])
                row[f"{group}_comparisons"] = int(match["comparisons"])
        output.append(row)
        tex.append(tex_row(f"{end:g}", *[row[k] for k in
                       ("source_channel", "receiver_channel", "source_aggregate", "receiver_aggregate")]))
    total = {key: "all" if key == "time_end" else sum(row[key] for row in output) for key in output[0]}
    output.append(total)
    tex.append(tex_row("All windows" if metric == "sign_reversals" else "Total",
        *[f"{total[f'{factor}_{group}']}/{total[f'{group}_comparisons']}"
          for group in ("channel", "aggregate") for factor in ("source", "receiver")]))
    return output, tex


def sweep_table(rows):
    output, tex = [], []
    settings = sorted({(float(r["mac_threshold"]), float(r["relative_gap"])) for r in rows})
    for mac, gap in settings:
        row = dict(mac_threshold=mac, relative_gap=gap)
        for factor in ("source", "receiver"):
            for end in TIMES:
                match = one([r for r in rows if float(r["mac_threshold"]) == mac and float(r["relative_gap"]) == gap
                             and r["changed_positions"] == factor and float(r["time_end"]) == end and r["group"] == "channel"])
                row[f"{factor}_T{end:g}"] = int(match["sign_reversals"])
        output.append(row)
        tex.append(tex_row(f"{mac:.2f}", f"{gap:.4f}", *list(row.values())[2:]))
    return output, tex


def case_rows(budgets, case):
    n, m, end, source, receiver, before, after = CASES[case]
    matches = [r for r in budgets if int(r["grid"]) == n and int(r["modes_per_patch"]) == m
               and float(r["time_end"]) == end and r["group"] == "channel"
               and int(r["source"]) == source and int(r["receiver"]) == receiver]
    return {position: [r for r in matches if (int(r["source_layout"]), int(r["receiver_layout"])) == layout]
            for position, layout in (("before", before), ("after", after))}


def threshold_table(budgets, case):
    selected = case_rows(budgets, case)
    output, tex = [], []
    for before in selected["before"]:
        after = one([r for r in selected["after"] if r["selection_id"] == before["selection_id"]])
        row = {key: before[key] for key in ("selection_id", "mac_threshold", "relative_gap")}
        row["matched_count"] = int(before["matched_count"])
        for position, entry in (("before", before), ("after", after)):
            row[f"{position}_change_pp"] = float(entry["change_percentage_points"])
            row[f"{position}_omitted_response_percent"] = 100 * float(entry["matched_reference_omission"])
        output.append(row)
        cells = ([r"\multicolumn{2}{c}{Intersection}"] if before["selection_id"] == "intersection"
                 else [f"{float(row['mac_threshold']):.2f}", f"{float(row['relative_gap']):.4f}"])
        tex.append(tex_row(*cells, row["matched_count"], f"${row['before_change_pp']:+.4f}$",
            f"${row['after_change_pp']:+.4f}$", f"{row['before_omitted_response_percent']:.2f}",
            f"{row['after_omitted_response_percent']:.2f}"))
    return output, tex


def path_table(rows):
    output, tex = [], []
    for case in CASES:
        for theta in (0.0, 1.0, 0.5):
            selected = [r for r in rows if r["case_id"] == case.lower() + "_relocation" and float(r["theta"]) == theta]
            thresholds = [r for r in selected if r["selection"] == "threshold"]
            common = one([r for r in selected if r["selection"] == "intersection"])
            if len(thresholds) != 9:
                raise ValueError("Each residue path requires nine thresholds")
            row = dict(contrast=case, theta=theta)
            for position in ("before", "after"):
                values = [float(r[f"{position}_change_percentage_points"]) for r in thresholds]
                row[f"{position}_minimum_pp"], row[f"{position}_maximum_pp"] = min(values), max(values)
                row[f"{position}_common_pp"] = float(common[f"{position}_change_percentage_points"])
            output.append(row)
            label = {0.0: "0", 1.0: "1", 0.5: "1/2"}[theta]
            tex.append(tex_row(case, rf"$\theta={label}$",
                *[f"$[{row[f'{p}_minimum_pp']:+.4f}, {row[f'{p}_maximum_pp']:+.4f}]$" for p in ("before", "after")],
                f"${row['before_common_pp']:+.4f}$", f"${row['after_common_pp']:+.4f}$"))
    return output, tex


def budget_table(budgets):
    keys = ("same_mode_term", "cross_mode_term", "source_cross_term", "receiver_cross_term", "unmatched_cross_term", "delta_squared_error")
    output, tex = [], []
    for case in CASES:
        for position, rows in case_rows(budgets, case).items():
            match = one([r for r in rows if r["selection_id"] == DEFAULT])
            row = dict(contrast=case, position=position)
            row.update({key: float(match[key]) for key in keys})
            output.append(row)
            tex.append(tex_row(case, position.title(), *[f"${row[key]:+.6f}$" for key in keys]))
    return output, tex


def selection_table(rows):
    selected = sorted([r for r in rows if int(r["grid"]) == 64 and float(r["time_end"]) == 8],
                      key=lambda r: (int(r["source_layout"]), int(r["receiver_layout"])))
    output, tex = [], []
    for item in selected:
        row = dict(layout=item["source_layout"] + item["receiver_layout"],
                   m64_error_percent=float(item["m64_continuous_error_percent"]),
                   m128_error_percent=float(item["m128_continuous_error_percent"]),
                   selection_loss_pp=float(item["selection_loss_percentage_points"]))
        output.append(row)
        tex.append(tex_row(row["layout"], *[f"{value:.3f}" for value in list(row.values())[1:]]))
    return output, tex


def grid_table(rows):
    rows = [r for r in rows if r["comparison"] == "adjacent"]
    pairs = sorted({(int(r["coarse_grid"]), int(r["fine_grid"])) for r in rows})
    output, tex = [], []
    for coarse, fine in pairs:
        selected = [r for r in rows if int(r["coarse_grid"]) == coarse and int(r["fine_grid"]) == fine]
        row = dict(coarse_grid=coarse, fine_grid=fine)
        for end in (2, 8):
            at_time = [r for r in selected if float(r["time_end"]) == end]
            if len(at_time) != 3:
                raise ValueError("Each adjacent-grid comparison requires three media")
            row[f"mean_T{end}_percent"] = 100 * mean(float(r["relative_difference_fine_denominator"]) for r in at_time)
        row["worst_T8_percent"] = 100 * max(float(r["worst_source_relative_difference"]) for r in selected if float(r["time_end"]) == 8)
        output.append(row)
        tex.append(tex_row(rf"${coarse}\to{fine}$", *[f"{value:.3f}" for value in list(row.values())[2:]]))
    return output, tex


def residual_table(rows):
    output, tex = [], []
    keys = ("queries", "five_actual", "five_bound", "ten_actual", "ten_bound")
    for item in rows:
        row = dict(family=item["family"], grid=int(item["grid"]), method=item["method"])
        row.update({key: int(item[key]) for key in keys})
        output.append(row)
        method = {"craig_bampton": "CB", "rational_krylov": "RK"}[item["method"]]
        tex.append(tex_row(row["family"], rf"${row['grid']}^2$/\,{method}", *[row[key] for key in keys]))
    return output, tex


def cost_table(rows):
    output, tex = [], []
    for comparator, tolerances in (("grid512", (0.05, 0.1)), ("richardson512", (0.05,))):
        for material in MATERIALS:
            for tolerance in tolerances:
                item = one([r for r in rows if r["material"] == material and r["comparator"] == comparator
                            and float(r["tolerance"]) == tolerance])
                row = dict(material=material, comparator=comparator, tolerance_percent=100 * tolerance,
                    full_grid=int(item["full_grid"]), full_error_percent=100 * float(item["full_error"]),
                    full_seconds=float(item["full_median_seconds"]), reduced_grid=int(item["reduced_grid"]),
                    reduced_dimension=int(item["reduced_dimension"]), reduced_error_percent=100 * float(item["reduced_error"]),
                    reduced_seconds=float(item["reduced_median_seconds"]), full_over_reduced=float(item["median_full_over_reduced"]))
                output.append(row)
                tex.append(tex_row(NAMES[material], f"{100 * tolerance:g}", row["full_grid"],
                    f"{row['full_error_percent']:.3f}", f"{row['full_seconds']:.3f}",
                    f"({row['reduced_grid']},{row['reduced_dimension']})", f"{row['reduced_error_percent']:.3f}",
                    f"{row['reduced_seconds']:.3f}", f"{row['full_over_reduced']:.2f}"))
    return output, tex


def media_table(rows):
    tex = []
    for material in MATERIALS:
        fields = []
        for field in ("log_density", "log_bulk_modulus"):
            terms = sorted([r for r in rows if r["material"] == material and r["field"] == field], key=lambda r: int(r["order"]))
            expression = "".join(f"{float(r['coefficient']):+.2f}" + TEX_BASIS[r["basis"]] for r in terms).lstrip("+")
            fields.append(f"${expression}$")
        tex.append(tex_row(NAMES[material], *fields))
    return rows, tex


def ports_table(rows):
    tex = []
    for row in rows:
        tex.append(tex_row(row["port"], *[f"$({float(row[f'layout{i}_x']):.2f},{float(row[f'layout{i}_y']):.2f})$" for i in (0, 1)]))
    return rows, tex


def tex_block(name, rows):
    body = list(rows)
    if name == "cost_selections":
        body.insert(6, r"\midrule \multicolumn{9}{l}{Quadratic-extrapolation comparator, $\widehat Y_{512}$}\\")
    return "\n".join(body) + "\n"


def manuscript_source(root, filename, included, stack=()):
    """Expand literal TeX inputs within the source directory."""
    path = (root / filename).resolve()
    if not path.suffix:
        path = path.with_suffix(".tex")
    if not path.is_relative_to(root):
        raise ValueError(f"LaTeX input leaves the manuscript directory: {filename}")
    if path in stack:
        raise ValueError(f"Cyclic LaTeX input: {path.relative_to(root)}")
    if not path.is_file():
        raise ValueError(f"Missing LaTeX input: {path.relative_to(root)}")
    included.add(path)
    return re.sub(r"\\input(?:table)?\s*\{([^}]+)\}",
                  lambda match: manuscript_source(root, match[1], included, (*stack, path)),
                  path.read_text())


def table_body(block, label):
    tabular = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", block, re.S)
    if len(tabular) != 1:
        raise ValueError(f"Expected one tabular environment in {label}")
    text = tabular[0]
    if r"\midrule" in text and r"\bottomrule" in text:
        start = text.index(r"\midrule") + len(r"\midrule")
        end = text.rindex(r"\bottomrule")
    else:
        rules = list(re.finditer(r"\\hline\b", text))
        if len(rules) < 3:
            raise ValueError(f"Cannot locate the header and final rules in {label}")
        start, end = rules[1].end(), rules[-1].start()
    return text[start:end].strip() + "\n"


def verify_manuscript(root, tables):
    root = root.resolve()
    filenames = {entry[1] for entry in TABLES.values()}
    included = set()
    sources = {filename: manuscript_source(root, filename, included)
               for filename in filenames}
    legacy_includes = included - {root / filename for filename in filenames}
    blocks = {filename: re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", text, re.S)
              for filename, text in sources.items()}
    for filename, entries in blocks.items():
        expected_count = sum(entry[1] == filename for entry in TABLES.values())
        if len(entries) != expected_count:
            raise ValueError(f"Expected {expected_count} tables in {filename}")
    verified = {}
    for name, (_, tex) in tables.items():
        number, filename, label = TABLES[name]
        expected = tex_block(name, tex)
        fragment = (root / "tables" / f"{name}.tex").resolve()
        if fragment in included and fragment.read_bytes() != expected.encode():
            raise ValueError(f"Table {number} differs from its generated TeX file: {fragment.name}")
        block = one([text for text in blocks[filename] if rf"\label{{{label}}}" in text])
        body = table_body(block, label)
        if sum("&" in line for line in body.splitlines()) != len(tex):
            raise ValueError(f"Table {number} must contain {len(tex)} data rows.")
        if legacy_includes:
            body = re.sub(r"\s+", "", body.replace(r"\midrule", ""))
            expected = re.sub(r"\s+", "", expected.replace(r"\midrule", ""))
        if body != expected:
            raise ValueError(f"Table {number} differs from the generated rows or their order.")
        verified[name] = len(tex)
    return verified


def export(data_dir, analysis_root, output_dir, simulation_root=None, manuscript_dir=None):
    inputs = simulation_inputs(simulation_root) if simulation_root is not None else bundled_inputs(data_dir)
    budgets = read_csv(analysis_root / "all_budgets.csv")
    thresholds = read_csv(analysis_root / "threshold_counts.csv")
    stable = read_csv(analysis_root / "robustness_counts.csv")
    selection = read_csv(analysis_root / "selection_regret.csv")
    tables = {
        "counts": count_table([r for r in thresholds if r["selection_id"] == DEFAULT], "sign_reversals"),
        "selection": selection_table(selection),
        "media": media_table(inputs["media"]),
        "ports": ports_table(inputs["ports"]),
        "source_thresholds": threshold_table(budgets, "Source"),
        "receiver_thresholds": threshold_table(budgets, "Receiver"),
        "residue_paths": path_table(inputs["paths"]),
        "modal_budgets": budget_table(budgets),
        "sweep_counts": sweep_table(thresholds),
        "sweep_stability": count_table(stable, "all_nine_reversals"),
        "grid_differences": grid_table(inputs["grids"]),
        "residual_acceptance": residual_table(inputs["residual"]),
        "cost_selections": cost_table(inputs["cost"]),
    }
    checks = verify_manuscript(manuscript_dir, tables) if manuscript_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (rows, tex) in tables.items():
        write_csv(output_dir / f"{name}.csv", rows)
        (output_dir / f"{name}.tex").write_text(tex_block(name, tex))
    record = dict(table_count=len(tables), input_source="simulation" if simulation_root else "bundled",
                  csv_rows={name: len(rows) for name, (rows, _) in tables.items()})
    if checks is not None:
        record["manuscript_rows_verified"] = checks
    (output_dir / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Exported {len(tables)} tables as CSV files and TeX row blocks.")
    if checks is not None:
        print(f"Verified all {sum(checks.values())} displayed rows against the manuscript.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data" / "tables")
    parser.add_argument("--analysis-root", type=Path, default=HERE.parent / "results" / "analysis")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "results" / "tables")
    parser.add_argument("--simulation-root", type=Path, help="Read table inputs from complete simulation results")
    parser.add_argument("--manuscript-dir", type=Path, help="Check each displayed row against these TeX sources")
    args = parser.parse_args()
    export(args.data_dir, args.analysis_root, args.output_dir, args.simulation_root, args.manuscript_dir)


if __name__ == "__main__":
    main()

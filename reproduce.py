#!/usr/bin/env python3
"""Reproduce acoustic simulations, analyses, tables, and figures."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def analysis_parameters():
    names = {'GRIDS', 'RANKS', 'WINDOWS', 'MACS', 'GAPS', 'DEFAULT_ID', 'SIGN_TOLERANCE'}
    values = {}
    for node in ast.parse((HERE / 'analysis' / 'core.py').read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = ast.literal_eval(node.value)
    return values


def random_initialization():
    seeds = {}
    for path in sorted((HERE / 'simulation').rglob('*.py')):
        declarations = []
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('default_rng', 'seed'):
                    declarations.append(ast.unparse(node))
        if declarations:
            seeds[path.relative_to(HERE).as_posix()] = declarations
    return seeds


def record_metrics(run, output_dir, mode):
    analysis = output_dir / 'analysis'
    summary = json.loads((analysis / 'summary.json').read_text())
    verification = summary['verification']
    run.metrics('modal_verification', {
        'budget_rows': verification['budget_rows'],
        'position_comparisons': verification['unique_contrasts'],
        'maximum_budget_absolute_defect': verification['maximum_budget_absolute_defect'],
        'maximum_decomposition_relative_defect': verification['maximum_decomposition_relative_defect'],
    }, units={'budget_rows': 'count', 'position_comparisons': 'count',
              'maximum_budget_absolute_defect': 'squared_relative_error',
              'maximum_decomposition_relative_defect': 'relative_error'})
    for position, values in summary['direction_totals'].items():
        run.metrics('position_reversals', values,
                    units={name: 'count' for name in values}, context={'position': position})
    source_span = summary['source_subspace']
    for selection in ('default', 'all_selections'):
        values = source_span[selection]
        run.metrics('source_subspace_signs', values,
                    units={name: 'count' for name in values}, context={'selection': selection})
    checks = source_span['verification']
    run.metrics('source_subspace_verification', checks,
                units={name: ('squared_relative_error' if name.endswith('defect') else
                              'ratio' if name.endswith('condition') else 'count') for name in checks})
    selection = json.loads((analysis / 'selection_regret_summary.json').read_text())
    run.metrics('dimension_selection', {
        'comparison_cases': selection['comparison_cases'],
        'different_choice_cases': selection['different_choice_cases'],
        'mean_loss_pp': selection['mean_selection_loss_pp_all_cases'],
        'maximum_loss_pp': selection['maximum_selection_loss_case']['selection_loss_percentage_points'],
    }, units={'comparison_cases': 'count', 'different_choice_cases': 'count',
              'mean_loss_pp': 'percentage_points', 'maximum_loss_pp': 'percentage_points'})
    with (output_dir / 'tables' / 'cost_selections.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            context = {name: value for name, value in row.items()
                       if name not in ('full_seconds', 'reduced_seconds', 'full_over_reduced')}
            context.update(timing_origin='new_measurement' if mode == 'full' else 'bundled_measurement',
                           timing_scope='solver_setup_and_receiver_output')
            run.metrics('receiver_cost', {
                'full_seconds': float(row['full_seconds']),
                'reduced_seconds': float(row['reduced_seconds']),
                'full_over_reduced': float(row['full_over_reduced']),
            }, units={'full_seconds': 's', 'reduced_seconds': 's', 'full_over_reduced': 'ratio'}, context=context)
    figures = json.loads((output_dir / 'figures' / 'figures.json').read_text())
    tables = json.loads((output_dir / 'tables' / 'verification.json').read_text())
    run.metrics('outputs', {'figures': len(figures['figures']), 'tables': tables['table_count']},
                units={'figures': 'count', 'tables': 'count'})
    run.event('numerical_outputs', analysis='analysis', figures='figures', tables='tables')


def commands(mode, output_dir, workdir):
    python = sys.executable
    steps = []
    analysis_data = HERE / 'data' / 'analysis'
    figure_data = output_dir / 'data' / 'figures'
    analysis_output = output_dir / 'analysis'
    simulation_arguments = []
    if mode in ('full', 'smoke'):
        command = [python, HERE / 'simulation' / 'run.py', '--workdir', workdir]
        if mode == 'smoke':
            command.append('--smoke')
        steps.append(command)
        if mode == 'smoke':
            return steps
        analysis_data = output_dir / 'data' / 'analysis'
        simulation_arguments = ['--simulation-root', workdir]
        steps.append([python, HERE / 'analysis' / 'extract_inputs.py',
                      *simulation_arguments, '--output-dir', analysis_data])
    for script in ('run.py', 'selection_audit.py'):
        steps.append([python, HERE / 'analysis' / script, '--data-dir', analysis_data,
                      '--output-dir', analysis_output, *simulation_arguments])
    steps.append([python, HERE / 'figures' / 'prepare.py',
                  '--analysis-root', analysis_output, '--output-dir', figure_data,
                  *simulation_arguments])
    steps.append([python, HERE / 'figures' / 'render.py',
                  '--data-dir', figure_data, '--output-dir', output_dir / 'figures'])
    steps.append([python, HERE / 'figures' / 'tables.py',
                  '--analysis-root', analysis_output, '--output-dir', output_dir / 'tables',
                  *simulation_arguments])
    return steps


def verification_reports(stage, output_dir, mode):
    reports = {
        'analysis/run.py': (output_dir / 'analysis' / 'verification.json', {'passed': True}),
        'analysis/selection_audit.py': (output_dir / 'analysis' / 'selection_verification.json', {'passed': True}),
        'figures/tables.py': (output_dir / 'tables' / 'verification.json',
                              {'table_count': 13, 'input_source': 'simulation' if mode == 'full' else 'bundled'}),
    }
    if stage not in reports:
        return []
    path, checks = reports[stage]
    return [dict(path=path, equals=checks)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('data', 'full', 'smoke'), default='data',
                        help='data: compact-input replay; full: all simulations; smoke: small-system checks')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'results')
    parser.add_argument('--workdir', type=Path,
                        help='Base for fresh model runs; defaults to sourcecode/models')
    parser.add_argument('--log-dir', type=Path, help='Base directory for separate run records; defaults to OUTPUT_DIR/logs')
    parser.add_argument('--run-name', default='acoustic', help='Short label for this run')
    parser.add_argument('--plan', action='store_true', help='List commands without computing or writing files')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    output_dir = args.output_dir.expanduser().resolve()
    models_dir = (args.workdir or HERE / 'models').expanduser().resolve()
    log_base = (args.log_dir or output_dir / 'logs').expanduser().resolve()
    if output_dir == HERE:
        parser.error('Choose an output directory separate from the sourcecode root.')
    if models_dir == HERE:
        parser.error('Choose a model base separate from the sourcecode root.')
    for directory in (output_dir, models_dir, log_base):
        for protected in (HERE / 'data', HERE / 'analysis', HERE / 'figures', HERE / 'simulation'):
            if directory == protected or protected in directory.parents:
                parser.error('Choose output directories outside the bundled data and program directories.')
    for directory in (output_dir, log_base):
        if directory == models_dir or models_dir in directory.parents or directory in models_dir.parents:
            parser.error('Keep the model base separate from output and log directories; neither may contain the other.')
    workdir = None
    if args.mode != 'data':
        from simulation.run import select_workdir, validate_models_dir
        try:
            validate_models_dir(models_dir)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if args.plan:
        if args.mode != 'data':
            workdir = select_workdir(models_dir)
        steps = commands(args.mode, output_dir, workdir)
        for command in steps:
            print(shlex.join(map(str, command)), flush=True)
        if args.mode in ('full', 'smoke'):
            plan_environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                                    ACOUSTIC_RUN_ID=workdir.name, ACOUSTIC_WORKDIR=str(workdir),
                                    ACOUSTIC_REQUESTED_WORKDIR=str(models_dir))
            subprocess.run([*map(str, steps[0]), '--plan'], check=True, env=plan_environment)
        return
    if sys.version_info < (3, 11):
        parser.error('Python 3.11 or later is required.')
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    from runlog import RunLog, new_run_id

    try:
        run_id = new_run_id(args.run_name)
        if args.mode != 'data':
            workdir = select_workdir(models_dir, run_id)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    log_dir = log_base / run_id
    steps = commands(args.mode, output_dir, workdir)
    configuration = dict(mode=args.mode, analysis=analysis_parameters(),
                         random_initialization=random_initialization(),
                         output_dir=str(output_dir), models_dir=str(models_dir),
                         workdir=str(workdir) if workdir is not None else None, log_dir=str(log_base),
                         run_name=args.run_name)
    input_paths = [HERE / 'data' / name for name in ('analysis', 'figures', 'tables')] if args.mode == 'data' else []
    with RunLog(log_dir, mode=args.mode, command=[sys.executable, *sys.argv],
                source_root=HERE, input_paths=input_paths,
                configuration=configuration, run_id=run_id, environment=environment) as run:
        environment.update(ACOUSTIC_RUN_ID=run_id, ACOUSTIC_LOG_DIR=str(log_dir))
        if args.mode != 'data':
            environment.update(ACOUSTIC_WORKDIR=str(workdir), ACOUSTIC_REQUESTED_WORKDIR=str(models_dir))
            run.event('workdir_selected', requested_workdir=str(models_dir), models_dir=str(models_dir), workdir=str(workdir))
            run.logger.info('Model run: {}', workdir)
        for command in steps:
            name = Path(command[1]).relative_to(HERE).as_posix()
            entry = run.run_stage(name, command, cwd=HERE, env=environment,
                                  verification_reports=verification_reports(name, output_dir, args.mode))
            if name == 'analysis/run.py':
                reported = output_dir / 'analysis' / 'environment.json'
                snapshot = run.log_dir / 'environments' / f"{entry['stage_id']:02d}_analysis.json"
                snapshot.parent.mkdir(exist_ok=True)
                shutil.copyfile(reported, snapshot)
                run.event('kernel_environment', stage=name, report=str(reported),
                          snapshot=snapshot.relative_to(run.log_dir).as_posix(),
                          sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest())
        if args.mode != 'smoke':
            record_metrics(run, output_dir, args.mode)
        run.event('reproduction_completed', output_dir=str(output_dir),
                  workdir=str(workdir) if args.mode != 'data' else None)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode >= 0 else 128 - error.returncode)
    except KeyboardInterrupt:
        raise SystemExit(130)

#!/usr/bin/env python3
"""Assemble figure inputs from simulations or compact data and fresh modal counts."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
MATERIALS = ('rotated_stripes','double_inclusion','smooth_checker')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--simulation-root',type=Path,help='Read complete simulation outputs instead of compact inputs')
    ap.add_argument('--data-dir',type=Path,default=HERE.parent/'data'/'figures')
    ap.add_argument('--analysis-root',type=Path,default=HERE.parent/'results'/'analysis')
    ap.add_argument('--output-dir',type=Path,default=HERE.parent/'results'/'data'/'figures')
    args=ap.parse_args()
    if args.simulation_root is None:
        refresh_counts(args.data_dir, args.analysis_root, args.output_dir)
        return
    src=args.simulation_root.resolve()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    inputs={}; tables={}; arrays={}
    def register(name):
        p=src/name
        inputs[str(p.relative_to(src))]={'sha256':digest(p),'bytes':p.stat().st_size}
        return p
    def records(name):
        with register(name).open(newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            for k,v in row.items():
                try:row[k]=float(v)
                except (TypeError,ValueError):pass
        return rows
    def js(name):return json.loads(register(name).read_text())
    sys.path.insert(0,str(src))
    from acoustic_core import material, CONFIG
    register('acoustic_core.py')
    for m in MATERIALS:
        _,_,rho,bulk=material(128,m)
        arrays[m+'_speed']=np.sqrt(bulk/rho)
        arrays[m+'_impedance']=np.sqrt(bulk*rho)
    tables['original_ports']={k:CONFIG[k] for k in ('source_centres','receiver_centres')}
    tables['relocated_ports']={k:v for k,v in js('source_receiver_validation/config.json').items() if k in ('source_centres','receiver_centres')}
    tables['interface_materials'] = {
        key: {field: value for field, value in js(name).items()
              if field in ('interface_x', 'density', 'bulk_modulus')}
        for key, name in (
            ('equal_speed', 'interface_case/config.json'),
            ('unequal_speed', 'unequal_speed_interface/config.json'),
        )
    }
    for key,name in (
        ('interface_equal','interface_case/results/summary.csv'),
        ('interface_unequal','unequal_speed_interface/results/summary.csv'),
        ('interface_cost','total_error_cost/results/interface_errors.csv'),
        ('waveforms','frequency_diagnostics/results/baseline_waveforms.csv'),
        ('budgets','frequency_diagnostics/results/baseline_budgets.csv'),
        ('relocation_thresholds','frequency_diagnostics/results/threshold_contrasts.csv'),
        ('grid_comparisons','propagation/results/grid_comparisons.csv'),
        ('asymptotic','propagation/results/asymptotic_comparison.csv'),
        ('modal_summary','modal_diagnostics/results/summary.csv'),
        ('reference_evaluation','modal_diagnostics/results/reference_evaluation.csv'),
        ('cost','grid_cost_diagnostics/results/receiver_errors.csv'),
    ): tables[key]=records(name)
    tables['mode_selection']={'selected_modes':js('modal_diagnostics/results/illustrated_mode_selection.json')['selected_modes']}
    with np.load(register('modal_diagnostics/results/illustrated_pressure_modes.npz'),allow_pickle=False) as z:
        for key in z.files:arrays[key]=z[key].copy()
    for m in MATERIALS:
        with np.load(register(f'propagation/results/{m}_n512.npz'),allow_pickle=False) as z:
            t=z['time'].copy();fine=z['receiver_pressure'].copy()
        with np.load(register(f'propagation/results/{m}_n384.npz'),allow_pickle=False) as z:coarse=z['receiver_pressure'].copy()
        window=np.hanning(len(t))[:,None,None]
        f=np.fft.rfftfreq(len(t),d=float(t[1]-t[0]))
        xf=np.fft.rfft(fine*window,axis=0);xd=np.fft.rfft((coarse-fine)*window,axis=0)
        w=np.full(len(f),2.);w[0]=1.
        if len(t)%2==0:w[-1]=1.
        ef=w*np.sum(abs(xf)**2,axis=(1,2));ed=w*np.sum(abs(xd)**2,axis=(1,2))
        arrays[m+'_spectrum_frequency']=f
        arrays[m+'_spectrum_signal']=ef/ef.sum()
        arrays[m+'_spectrum_difference']=ed/ef.sum()
    tables['reduction_comparison']=js('propagation/results/reduction_comparison_summary.json')['mean_over_materials']
    tables['observation_windows']=[r for r in records('observation_diagnostics/results/channel_errors.csv') if r['reference_grid']==512 and r['method']=='craig_bampton' and r['modes_per_patch']==32]
    tables['modal_sensitivity']=[];tables['modal_finite_time']=[]
    for m in MATERIALS:
        tables['modal_sensitivity'] += [r for r in records(f'modal_robustness/results/{m}_sensitivity.csv') if r['selection']=='own' and r['mac_threshold']==.9 and r['relative_gap']==.005]
        tables['modal_finite_time'] += [r for r in records(f'modal_robustness/results/{m}_finite_time.csv') if r['fine_grid']==128 and r['method']=='craig_bampton' and r['m']==32 and (r['group']=='all' or (m=='smooth_checker' and r['group']=='channel' and r['source']==2 and r['receiver']==2 and r['end']==8 and r['window_kind']=='cumulative'))]
    tables['layered']=js('studies/stage1_1d/refinement_summary.json')
    tables['residual']=[r for r in records('main_residual/results/combined_bounds.csv') if r['time']==8 and r['grid'] in (32,64)]
    tables['residual_selection']=[r for r in records('main_residual/results/combined_selection.csv') if r['time']==8]
    tables['extra_residual']=[r for r in records('studies/stage2_2d/energy_certificate_results.csv') if r['time']==8 and r['grid']==32]
    for name in ('threshold_counts.csv','robustness_counts.csv'):
        p=args.analysis_root/name
        if not p.is_file():
            raise FileNotFoundError(f'Missing modal analysis: {p}')
        if p.is_file():
            with p.open(newline='') as f:rows=list(csv.DictReader(f))
            for row in rows:
                for k,v in row.items():
                    try:row[k]=float(v)
                    except (TypeError,ValueError):pass
            tables[name.removesuffix('.csv')]=rows
            inputs['analysis/'+name]={'sha256':digest(p),'bytes':p.stat().st_size}
    (out/'plot_tables.json').write_text(json.dumps(tables,separators=(',',':'),allow_nan=True)+'\n')
    np.savez_compressed(out/'plot_arrays.npz',**arrays)
    provenance = {
        'inputs': inputs,
        'transformations': [
            'Analytic material fields at 128 x 128 cell centers.',
            'Two-layer material parameters copied from the interface simulation configurations.',
            'Physical pressure modes selected by the stated modal rule.',
            'Hann-window spectra with one-sided Parseval weights, normalized by fine-reference spectral power.',
            'CSV rows selected for the plotted models without smoothing.'
        ],
        'outputs': {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                    for p in (out/'plot_tables.json', out/'plot_arrays.npz')}
    }
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'input_count':len(inputs),'tables':len(tables),'array_count':len(arrays),'bytes':sum(p.stat().st_size for p in out.iterdir()),'has_robustness_counts':'robustness_counts' in tables}))

def refresh_counts(data_dir, analysis_root, output_dir):
    metadata = json.loads((data_dir/'provenance.json').read_text())
    for name in ('plot_tables.json', 'plot_arrays.npz'):
        if digest(data_dir/name) != metadata['outputs'][name]['sha256']:
            raise ValueError(f'Figure input hash differs: {name}')
    tables = json.loads((data_dir/'plot_tables.json').read_text())
    with np.load(data_dir/'plot_arrays.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    for name in ('threshold_counts.csv', 'robustness_counts.csv'):
        path = analysis_root/name
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key, value in row.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
        tables[name.removesuffix('.csv')] = rows
        metadata['inputs']['analysis/'+name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'plot_tables.json').write_text(json.dumps(tables, separators=(',', ':'), allow_nan=True)+'\n')
    np.savez_compressed(output_dir/'plot_arrays.npz', **arrays)
    metadata['outputs'] = {path.name: {'sha256': digest(path), 'bytes': path.stat().st_size}
        for path in (output_dir/'plot_tables.json', output_dir/'plot_arrays.npz')}
    (output_dir/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Prepared {len(tables)} tables and {len(arrays)} arrays for figures.')


if __name__=='__main__':main()

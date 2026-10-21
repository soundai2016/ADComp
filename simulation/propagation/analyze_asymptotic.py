#!/usr/bin/env python3
"""Three-grid waveform consistency and conditional second-order extrapolation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
NAMES=['rotated_stripes','double_inclusion','smooth_checker']
LABELS=['Rotated stripes','Double inclusion','Smooth checker']
GRIDS=[256,384,512]
BANDS=[(0.,2.),(2.,4.),(4.,8.),(8.,40.)]


def norm(x):
    return float(np.linalg.norm(x))


def order_ratio(p):
    a,b,c=GRIDS
    if abs(p)<1e-7:
        return np.log(b/a)/np.log(c/b)
    return np.expm1(p*np.log(b/a))/(-np.expm1(-p*np.log(c/b)))


def observed_order(ratio):
    if order_ratio(-4.) <= ratio <= order_ratio(10.):
        return float(brentq(lambda p:order_ratio(p)-ratio,-4.,10.))
    return None


def main():
    global GRIDS
    parser=argparse.ArgumentParser()
    parser.add_argument('--grids',nargs=3,type=int,default=GRIDS)
    parser.add_argument('--suffix',default='')
    args=parser.parse_args()
    GRIDS=args.grids
    suffix=args.suffix
    if any(a>=b for a,b in zip(GRIDS[:-1],GRIDS[1:])):
        raise ValueError('Three strictly increasing grid sizes required.')
    records=[]
    band_records=[]
    extrapolations={}
    spectra={}
    for name in NAMES:
        yy={}
        for n in GRIDS:
            with np.load(RESULTS/f'{name}_n{n}.npz') as z:
                yy[n]=z['receiver_pressure'].copy()
                if n==GRIDS[0]:
                    times=z['time'].copy()
                elif not np.array_equal(times,z['time']):
                    raise ValueError('Different output times are not interpolated.')
        factor=(GRIDS[2]/GRIDS[1])**2-1.
        extrapolations[name]=yy[GRIDS[2]]+(yy[GRIDS[2]]-yy[GRIDS[1]])/factor
        for horizon in (2.,8.):
            mask=(times>0)&(times<=horizon)
            for source in ('all',0,1,2):
                selection=(mask,slice(None),slice(None)) if source=='all' else (mask,slice(None),source)
                a,b,c=[yy[n][selection] for n in GRIDS]
                d1,d2=a-b,b-c
                n1,n2=norm(d1),norm(d2)
                coeff1=d1/(float(GRIDS[0])**-2-float(GRIDS[1])**-2)
                coeff2=d2/(float(GRIDS[1])**-2-float(GRIDS[2])**-2)
                extrapolate12=b+(b-a)/((GRIDS[1]/GRIDS[0])**2-1.)
                extrapolate23=c+(c-b)/((GRIDS[2]/GRIDS[1])**2-1.)
                records.append({'material':name,'time_end':horizon,'source':source,
                    'coarse_grid':GRIDS[0],'middle_grid':GRIDS[1],'fine_grid':GRIDS[2],
                    'difference_coarse_middle_over_fine_norm':n1/norm(c),
                    'difference_middle_fine_over_fine_norm':n2/norm(c),
                    'difference_norm_ratio':n1/n2,'second_order_ratio':order_ratio(2.),
                    'observed_order_from_norm_ratio':observed_order(n1/n2),
                    'difference_direction_cosine':float(np.vdot(d1,d2).real)/(n1*n2),
                    'second_order_coefficient_relative_difference':norm(coeff1-coeff2)/norm(coeff2),
                    'conditional_second_order_finest_remainder_over_finest_norm':n2/(factor*norm(c)),
                    'conditional_extrapolate_pair_difference_over_finest_norm':norm(extrapolate12-extrapolate23)/norm(c)})
        window=np.hanning(len(times))[:,None,None]
        ff=np.fft.rfftfreq(len(times),d=float(times[1]-times[0]))
        ft={n:np.fft.rfft(yy[n]*window,axis=0) for n in GRIDS}
        weight=np.full(len(ff),2.)
        weight[0]=1.
        if len(times)%2==0: weight[-1]=1.
        finest_energy=weight*np.sum(np.abs(ft[GRIDS[2]])**2,axis=(1,2))
        difference_energy=weight*np.sum(np.abs(ft[GRIDS[1]]-ft[GRIDS[2]])**2,axis=(1,2))
        spectra[name]=(ff,finest_energy,difference_energy)
        for low,high in BANDS:
            keep=(ff>=low)&(ff<high)
            ef=float(np.sum(finest_energy[keep]));ed=float(np.sum(difference_energy[keep]))
            band_records.append({'material':name,'frequency_low':low,'frequency_high':high,
                'middle_grid':GRIDS[1],'fine_grid':GRIDS[2],
                'reference_signal_energy_fraction':ef/float(np.sum(finest_energy)),
                'relative_middle_fine_difference_in_band':np.sqrt(ed/max(ef,1e-300)),
                'difference_energy_fraction':ed/max(float(np.sum(difference_energy)),1e-300)})
    for filename,rows in [('asymptotic_comparison.csv',records),('reference_frequency_bands.csv',band_records)]:
        target=RESULTS/(Path(filename).stem+suffix+'.csv')
        with target.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(RESULTS/('conditional_second_order_extrapolation'+suffix+'.npz'),time=times,**extrapolations)
    summaries=[r for r in records if r['source']=='all' and r['time_end']==8.]
    report={'method':'Nonuniform three-grid waveform comparison','grids':GRIDS,
        'second_order_difference_ratio':order_ratio(2.),
        'assumption_for_extrapolation':'For fixed sources, receiver averages and time window, Y_h=Y+Ch^2+o(h^2), with an h-independent leading waveform C.',
        'interpretation':'The observed order, difference-vector alignment and leading-coefficient agreement diagnose this assumption. The extrapolation is conditional and supplies no rigorous error bound.',
        'frequency_analysis':'Hann-windowed recorded waveforms; fixed bands 0-2, 2-4, 4-8 and 8-40 cycles per nondimensional time; not an independent high-frequency-source experiment.',
        'time_window_for_summary':'0<t<=8','aggregate_records':summaries,
        'reference_sha256':{f'{name}_n{n}.npz':hashlib.sha256((RESULTS/f'{name}_n{n}.npz').read_bytes()).hexdigest() for name in NAMES for n in (384,512)}}
    (RESULTS/('asymptotic_summary'+suffix+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

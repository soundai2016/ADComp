#!/usr/bin/env python3
"""Draw the seven article figures from compact numerical inputs."""
from __future__ import annotations
import argparse, hashlib, json, os, re, tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'acoustic_figures'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
WIDTH=6.69
BLUE='#0072B2'; ORANGE='#D55E00'; TEAL='#009E73'; PURPLE='#AA4499'
RED='#B2182B'; BLACK='#202830'; GRAY='#77838D'; LIGHT='#DFE4E8'
MATERIALS=('rotated_stripes','double_inclusion','smooth_checker')
NAMES=('Stripes','Inclusions','Checker');MCOLORS=(BLUE,ORANGE,TEAL)
TABLES = {}
ARRAYS = {}
OUT = HERE.parent / 'results' / 'figures'
META = {}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
    'axes.labelsize':8.5,'axes.titlesize':8.8,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
    'legend.fontsize':8.5,'legend.title_fontsize':8.5,'axes.linewidth':.65,
    'axes.spines.top':False,'axes.spines.right':False,'xtick.major.width':.65,'ytick.major.width':.65,
    'xtick.minor.width':.5,'ytick.minor.width':.5,'xtick.major.size':3,'ytick.major.size':3,
    'lines.linewidth':1.2,'lines.markersize':3.8,'legend.frameon':False,
    'legend.handlelength':1.4,'legend.columnspacing':.85,'legend.labelspacing':.3,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white',
    'axes.edgecolor':BLACK,'axes.labelcolor':BLACK,'text.color':BLACK,
    'xtick.color':BLACK,'ytick.color':BLACK,'mathtext.fontset':'dejavusans'})

def one(rows,**criteria):
    r=[r for r in rows if all(r.get(k)==v for k,v in criteria.items())]
    if len(r)!=1:raise ValueError(f'Expected one row: {criteria}; got {len(r)}')
    return r[0]
def vals(rows,key):return np.array([r[key] for r in rows],dtype=float)
def title(ax,letter,label):ax.set_title(f'({letter}) {label}',loc='left',pad=6)
def grid(ax,axis='both'):
    ax.grid(axis=axis,which='major',color=LIGHT,lw=.5);ax.set_axisbelow(True)
def numeric(ax):
    for axis,scale in ((ax.xaxis,ax.get_xscale()),(ax.yaxis,ax.get_yscale())):
        if scale=='log':
            if axis is ax.yaxis and getattr(ax,'scientific_y',False):continue
            axis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
            axis.set_minor_formatter(NullFormatter())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(fig, num):
    OUT.mkdir(parents=True,exist_ok=True)
    for ax in fig.axes:numeric(ax)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    cw,ch=fig.canvas.get_width_height()
    visible=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    minsize=min(t.get_fontsize() for t in visible)
    if minsize<8.49:raise ValueError(f'Fig{num} font below 8.5 pt: {minsize}')
    audit=list(fig.texts)
    for leg in fig.legends:audit.extend(leg.get_texts())
    for ax in fig.axes:
        audit.extend([ax.title,ax._left_title,ax._right_title,ax.xaxis.label,ax.yaxis.label,*ax.texts])
        if ax.get_legend():audit.extend(ax.get_legend().get_texts())
        for axis in (ax.xaxis,ax.yaxis):
            lo,hi=sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(),*axis.get_minor_ticks()]:
                if lo<=tick.get_loc()<=hi:audit.extend([tick.label1,tick.label2])
            audit.append(axis.get_offset_text())
    outside=[]
    for t in audit:
        if not t.get_visible() or not t.get_text().strip():continue
        b=t.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1>cw+1 or b.y1>ch+1:outside.append(t.get_text())
    if outside:raise ValueError(f'Fig{num} text outside canvas: {outside}')
    for ext in ('pdf','png'):
        kw={'metadata':{'Title':f'Figure {num}','Creator':'Acoustic reduction scientific figure renderer','CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
        destination=OUT/f'Fig{num}.{ext}'
        temporary=OUT/f'Fig{num}.{ext}.tmp'
        fig.savefig(temporary,format=ext,dpi=300,**kw)
        temporary.replace(destination)
    META[f'Fig{num}']={
        'width_inches':float(fig.get_figwidth()),'height_inches':float(fig.get_figheight()),
        'minimum_text_pt':float(minsize),'bounds_check_pass':True,
        'pdf_sha256':sha(OUT/f'Fig{num}.pdf'),'png_sha256':sha(OUT/f'Fig{num}.png')}
    plt.close(fig)
    print(f'Fig{num}: PDF + PNG; all text >= 8.5 pt; canvas bounds pass',flush=True)

def fig1():
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 5.6))
    fig.subplots_adjust(left=.08, right=.98, bottom=.24, top=.91,
                        wspace=.38, hspace=.48)
    ports = (
        ('source_centres', '*', 'S', BLUE, 47),
        ('receiver_centres', 'o', 'R', ORANGE, 21),
    )
    lo = min(ARRAYS[m + '_speed'].min() for m in MATERIALS)
    hi = max(ARRAYS[m + '_speed'].max() for m in MATERIALS)
    for ax, material, name, letter in zip(axes[0], MATERIALS, NAMES, 'abc'):
        im = ax.imshow(ARRAYS[material + '_speed'].T, origin='lower',
                       extent=(0, 1, 0, 1), cmap='viridis', vmin=lo, vmax=hi,
                       rasterized=True)
        for key, symbol, prefix, color, size in ports:
            xy = np.asarray(TABLES['original_ports'][key])
            ax.scatter(*xy.T, marker=symbol, s=size, facecolors=color,
                       edgecolors='white', lw=.6)
        ax.set(xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel='$x$', ylabel='$y$')
        title(ax, letter, name)
    materials = TABLES['interface_materials']
    interface = materials['equal_speed']['interface_x']
    if interface != materials['unequal_speed']['interface_x']:
        raise ValueError('The interface layouts require a common interface position.')
    offsets = {
        'original_ports': {'S3': (-15, -11), 'R1': (4, -10),
                           'R2': (-16, 5), 'R3': (4, -10)},
        'relocated_ports': {'S3': (-17, 5), 'R4': (-18, -11)},
    }
    for ax, layout, letter, name in zip(
            axes[1, :2], ('original_ports', 'relocated_ports'), 'de',
            ('Original layout', 'Relocated layout')):
        ax.axvspan(0, interface, facecolor='#F4F6F8')
        ax.axvspan(interface, 1, facecolor='#E6EBEF')
        ax.axvline(interface, color=GRAY, lw=.8, ls='--')
        for key, symbol, prefix, color, size in ports:
            xy = np.asarray(TABLES[layout][key])
            ax.scatter(*xy.T, marker=symbol, s=size, facecolors=color,
                       edgecolors='white', lw=.6, zorder=3)
            for i, point in enumerate(xy, 1):
                label = f'{prefix}{i}'
                ax.annotate(label, point, xytext=offsets[layout].get(label, (4, 4)),
                            textcoords='offset points', fontsize=8.5, color=color)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal',
               xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel='$x$', ylabel='$y$')
        title(ax, letter, name)
    ax = axes[1, 2]
    impedance = []
    for key, color, style, label in (
        ('equal_speed', BLUE, '-', r'Equal-speed $c/c_1$'),
        ('unequal_speed', ORANGE, '--', r'Unequal-speed $c/c_1$'),
    ):
        rho = np.asarray(materials[key]['density'], dtype=float)
        bulk = np.asarray(materials[key]['bulk_modulus'], dtype=float)
        speed = np.sqrt(bulk / rho)
        impedance.append(np.sqrt(bulk * rho))
        ax.plot([0, interface, interface, 1],
                [1, 1, speed[1] / speed[0], speed[1] / speed[0]],
                color=color, ls=style, label=label)
    if not np.allclose(impedance[0] / impedance[0][0],
                       impedance[1] / impedance[1][0]):
        raise ValueError('The shared impedance profile requires equal impedance ratios.')
    ratio = impedance[0][1] / impedance[0][0]
    ax.plot([0, interface, interface, 1], [1, 1, ratio, ratio],
            color=BLACK, ls='-.', label=r'Both cases $Z/Z_1$')
    ax.set(xlim=(0, 1), ylim=(.8, 10), yscale='log',
           xticks=[0, .5, 1], yticks=[1, 1.5, 8], xlabel='$x$',
           ylabel=r'$c/c_1$ or $Z/Z_1$')
    ax.set_box_aspect(1)
    title(ax, 'f', 'Interface profiles')
    grid(ax, 'y')
    fig.legend(*ax.get_legend_handles_labels(), loc='center',
               bbox_to_anchor=(.79, .085), handlelength=1.8)
    handles = [Line2D([], [], ls='none', marker=symbol, markersize=np.sqrt(size),
                      markerfacecolor=color, markeredgecolor=color,
                      label='Sources' if prefix == 'S' else 'Receivers')
               for key, symbol, prefix, color, size in ports]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .995), ncol=2)
    cax = fig.add_axes([.12, .09, .47, .02])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('Sound speed $c$ (a-c)', labelpad=3)
    save(fig, 1)

def fig2():
    fig,axes=plt.subplots(3,2,figsize=(WIDTH,6.65),layout='constrained')
    fig.get_layout_engine().set(h_pad=.06,w_pad=.03)
    for row,key in enumerate(('interface_equal','interface_unequal')):
        data=TABLES[key];full=sorted([r for r in data if r['method']=='full_separable' and r['time_end']==8],key=lambda r:r['grid'])
        ax=axes[row,0]
        for field,label,color,marker,ls in (('receiver_total_error','Receiver',BLUE,'o','-'),('continuous_pressure_total_error','Pressure field',ORANGE,'s','-'),('pressure_cell_projection_floor','Cell projection',GRAY,'^',':')):
            ax.loglog(vals(full,'grid'),100*vals(full,field),color=color,marker=marker,ls=ls,label=label)
        ax.set(xlabel='Cells per side, $n$',ylabel='Relative error (%)',xticks=vals(full,'grid'),ylim=(.5,115))
        ax.yaxis.set_major_locator(FixedLocator([1,10,100]));ax.legend(loc='lower left',handlelength=1.)
        title(ax,'ac'[row],('Equal' if row==0 else 'Unequal')+' speed: full grid')
        ax=axes[row,1];red=sorted([r for r in data if r['grid']==128 and r['method']=='craig_bampton' and r['time_end']==8],key=lambda r:r['modes_per_patch'])
        for field,label,color,marker in (('receiver_total_error','Continuous reference',ORANGE,'s'),('receiver_same_grid_error','Same grid',BLUE,'o')):
            ax.semilogy(vals(red,'modes_per_patch'),100*vals(red,field),color=color,marker=marker,label=label)
        ax.axhline(100*one(full,grid=128)['receiver_total_error'],ls='--',color=BLACK,lw=.8,label='Full $128^2$')
        ax.set(xlabel='Internal modes per subdomain, $m$',ylabel='Receiver error (%)',xticks=[32,64,128],ylim=(.7,50))
        ax.yaxis.set_major_locator(FixedLocator([1,5,10,50]));ax.legend(loc='upper right',handlelength=1.)
        title(ax,'bd'[row],('Equal' if row==0 else 'Unequal')+' speed: CB, $n=128$')
    data=[r for r in TABLES['interface_cost'] if isinstance(r['both_total_seconds'],(int,float)) and r['time_end']==8]
    for method,label,color,marker in (('full_separable','Full',BLACK,'o'),('craig_bampton','CB',BLUE,'^'),('rational_krylov','RK',PURPLE,'s')):
        rows=sorted([r for r in data if r['method']==method],key=lambda r:(r['grid'],r['modes_per_patch'] or 0))
        axes[2,0].plot(vals(rows,'receiver_total_seconds'),100*vals(rows,'receiver_error'),color=color,marker=marker,ls='-' if method=='full_separable' else 'none',label=label)
        for r in rows:
            axes[2,1].plot(r['both_total_seconds'],100*r['continuous_field_error'],color=color,marker=marker,ls='none',mfc=color if r['receiver_error']<=.05 else 'white')
        if method=='full_separable':axes[2,1].plot(vals(rows,'both_total_seconds'),100*vals(rows,'continuous_field_error'),color=color,lw=.9)
    for col in (0,1):
        ax=axes[2,col];ax.set(xscale='log',yscale='log',xlabel='Setup + '+('receivers' if col==0 else 'receivers + fields')+' (s)',ylabel=('Receiver' if col==0 else 'Pressure-field')+' error (%)')
        ax.yaxis.set_major_locator(FixedLocator([1,5,10,50,100]));ax.xaxis.set_major_locator(LogLocator(base=10,numticks=5))
        ax.axhline(5,color=GRAY,ls=':',lw=.8)
        title(ax,'ef'[col],'Equal-speed cavity: '+('receiver cost' if col==0 else 'field cost'))
    axes[2,0].axhline(1,color=GRAY,ls=':',lw=.8);axes[2,0].legend(loc='upper right',ncol=3,handlelength=.8)
    for ax in axes.flat:grid(ax)
    save(fig, 2)

def fig3():
    waves = TABLES['waveforms']
    budgets = TABLES['budgets']
    counts = TABLES['robustness_counts']
    fig = plt.figure(figsize=(WIDTH, 6.8))
    gs = fig.add_gridspec(3, 6, left=.085, right=.985, bottom=.17,
                          top=.92, wspace=.95, hspace=.88)
    wave_axes = []
    budget_axes = []
    terms = (
        ('same_mode_term', 'Same mode', ORANGE),
        ('cross_mode_term', 'Cross mode', '#56B4E9'),
        ('source_cross_term', 'Source coupling', PURPLE),
        ('receiver_cross_term', 'Receiver coupling', TEAL),
        ('unmatched_cross_term', 'Unmatched response', GRAY),
    )
    for row, case in enumerate(('source_relocation', 'receiver_relocation')):
        name = 'Source' if row == 0 else 'Receiver'
        pair = []
        pressure = []
        for col, position in enumerate(('before', 'after')):
            ax = fig.add_subplot(gs[row, 2 * col:2 * col + 2])
            pair.append(ax)
            wave_axes.append(ax)
            rows = sorted((r for r in waves if r['case_id'] == case
                           and r['position'] == position), key=lambda r: r['time'])
            budget = one(budgets, case_id=case, position=position)
            for key, label, color, style in (
                ('y_reference', 'Reference', BLACK, '-'),
                ('y_rom', 'ROM', BLUE, '--'),
                ('y_no_ritz', 'Ritz shifts removed', ORANGE, '-.'),
            ):
                ax.plot(vals(rows, 'time'), vals(rows, key), color=color,
                        ls=style, label=label)
                pressure.extend(vals(rows, key))
            ax.set(xlim=(0, budget['time_end']), xlabel='Time $t$')
            if col == 0:
                ax.set_ylabel('Pressure')
            else:
                ax.tick_params(labelleft=False)
            increment = budget['change_percentage_points']
            ax.text(.98, .97, rf'$\Delta E={increment:+.3f}$ pp',
                    transform=ax.transAxes, ha='right', va='top',
                    color=BLUE if increment > 0 else RED,
                    bbox={'facecolor': 'white', 'edgecolor': 'none',
                          'alpha': .85, 'pad': 1})
            title(ax, chr(97 + 3 * row + col),
                  name + ': ' + ('original' if col == 0 else 'relocated'))
            grid(ax)
        low, high = min(pressure), max(pressure)
        for ax in pair:
            ax.set_ylim(low - .1 * (high - low), high + .25 * (high - low))
        ax = fig.add_subplot(gs[row, 4:6])
        box = ax.get_position()
        ax.set_position([box.x0 + .025, box.y0, box.width - .025, box.height])
        budget_axes.append(ax)
        rows = [one(budgets, case_id=case, position=p) for p in ('before', 'after')]
        positive = np.zeros(2)
        negative = np.zeros(2)
        for key, label, color in terms:
            values = 100 * vals(rows, key)
            ax.bar(range(2), values, bottom=np.where(values >= 0, positive, negative),
                   width=.52, color=color, edgecolor='white', lw=.5, label=label)
            positive += np.maximum(values, 0)
            negative += np.minimum(values, 0)
        ax.plot(range(2), 100 * vals(rows, 'delta_squared_error'), '_',
                color=BLACK, ms=12, mew=1.6, label=r'Total $\Delta Q$')
        ax.axhline(0, color=BLACK, lw=.8)
        ax.set(xlim=(-.55, 1.55), xticks=[0, 1],
               xticklabels=['Original', 'Relocated'], ylabel=r'$\Delta Q$ ($10^{-2}$)',
               xlabel=rf'$n={int(rows[0]["grid"])}$, $T={rows[0]["time_end"]:g}$')
        title(ax, 'cf'[row], name + ' budget')
        grid(ax, 'y')
    fig.legend(*wave_axes[0].get_legend_handles_labels(), ncol=3,
               loc='upper center', bbox_to_anchor=(.53, .997))
    fig.legend(*budget_axes[0].get_legend_handles_labels(), ncol=3,
               loc='center', bbox_to_anchor=(.53, .38), handlelength=1.0)
    for col, group in enumerate(('channel', 'aggregate')):
        ax = fig.add_subplot(gs[2, 3 * col:3 * col + 3])
        for changed, color, marker in (('source', BLUE, 'o'), ('receiver', ORANGE, 's')):
            rows = sorted((r for r in counts if r['group'] == group
                           and r['changed_positions'] == changed), key=lambda r: r['time_end'])
            for key, style, fill in (('default_reversals', ':', 'white'),
                                     ('all_nine_reversals', '-', color)):
                ax.plot(range(3), vals(rows, key), color=color, marker=marker,
                        ls=style, mfc=fill, ms=4)
        ax.set(xticks=range(3), xticklabels=['0.5', '2', '8'],
               xlabel='Window end $T$', ylabel='Sign reversals (count)',
               ylim=(-1, 75 if col == 0 else 7))
        title(ax, 'gh'[col], 'Channel contrasts (144)' if col == 0
              else 'Aggregate contrasts (12)')
        grid(ax)
    handles = [
        Line2D([], [], marker='o', color=BLUE, label='Source move'),
        Line2D([], [], marker='s', color=ORANGE, label='Receiver move'),
        Line2D([], [], color=GRAY, ls=':', marker='o', mfc='white', label='Default thresholds'),
        Line2D([], [], color=GRAY, ls='-', marker='o', label='All nine thresholds'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .01), ncol=2)
    save(fig, 3)

def fig4():
    fig,axes=plt.subplots(2,3,figsize=(WIDTH,4.9))
    fig.subplots_adjust(left=.092,right=.985,bottom=.145,top=.92,wspace=.36,hspace=.77)
    for col,m in enumerate(MATERIALS):
        ax=axes[0,col]
        for t,color,marker in ((2,BLUE,'o'),(8,ORANGE,'s')):
            r=sorted([r for r in TABLES['grid_comparisons'] if r['material']==m and r['comparison']=='adjacent' and r['time_end']==t],key=lambda r:r['coarse_grid'])
            ax.semilogy(range(len(r)),100*vals(r,'relative_difference_fine_denominator'),color=color,marker=marker,label=f'$T={t}$')
        ax.set(xticks=range(len(r)),xticklabels=[f'{int(v["coarse_grid"])}–{int(v["fine_grid"])}' for v in r],ylim=(.009,120),xlabel='Adjacent grid pair')
        ax.tick_params(axis='x',labelrotation=65)
        for t in ax.get_xticklabels():t.set_ha('right')
        a=one(TABLES['asymptotic'],material=m,source='all',time_end=8)
        ax.text(.03,.04,f'$p={a["observed_order_from_norm_ratio"]:.3f}$\n'+f'$\\cos\\theta={a["difference_direction_cosine"]:.4f}$',transform=ax.transAxes,fontsize=8.5,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':.5})
        title(ax,chr(97+col),NAMES[col]);grid(ax)
        if col==0:ax.set_ylabel('Receiver difference (%)');ax.legend(loc='upper right')
        ax=axes[1,col];f=ARRAYS[m+'_spectrum_frequency'];keep=f<=12
        for key,label,color,ls in (('signal','$512^2$ signal',BLACK,'-'),('difference','$384^2-512^2$ difference',ORANGE,'--')):
            y=ARRAYS[m+'_spectrum_'+key];ax.semilogy(f[keep],np.maximum(y[keep],1e-14),color=color,ls=ls,label=label)
        ax.set(xlim=(0,12),ylim=(1e-12,1),xticks=[0,4,8,12],xlabel='Frequency (cycles / time)');ax.scientific_y=True
        ax.yaxis.set_major_locator(FixedLocator([1e-12,1e-8,1e-4,1]));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:rf'$10^{{{int(round(np.log10(v)))}}}$'))
        ax.axvline(8,color=GRAY,ls=':',lw=.7);title(ax,chr(100+col),'Pressure spectrum');grid(ax)
        if col==0:ax.set_ylabel('Reference power fraction')
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.015),ncol=2)
    save(fig, 4)

def fig5():
    fig=plt.figure(figsize=(WIDTH,5.0));gs=fig.add_gridspec(2,3,left=.108,right=.978,bottom=.115,top=.895,hspace=.67,wspace=.37,height_ratios=(1,.97))
    top=[fig.add_subplot(gs[0,c]) for c in range(3)];bottom=[fig.add_subplot(gs[1,c]) for c in range(3)]
    sel=one(TABLES['mode_selection']['selected_modes'],material='smooth_checker')
    fields=[ARRAYS['smooth_checker_'+s+'_pressure_mode'] for s in ('fine','coarse','rom')];lim=max(np.max(abs(x)) for x in fields)
    labels=['Fine $128^2$','Full $32^2$','CB $32^2$, $m=32$'];freq=[sel[k] for k in ('fine_frequency','coarse_frequency','rom_frequency')]
    for i,(ax,f,label,field) in enumerate(zip(top,freq,labels,fields)):
        im=ax.imshow(field.T,origin='lower',extent=(0,1,0,1),cmap='RdBu',vmin=-lim,vmax=lim,rasterized=True)
        title(ax,chr(97+i),label+'\n'+f'$f={f:.4f}$')
        ax.set(xlabel='$x$',xticks=[0,.5,1],yticks=[0,.5,1]);ax.set_ylabel('$y$' if i==0 else '')
        if i!=0:ax.tick_params(labelleft=False)
    cax=fig.add_axes([.35,.499,.48,.022]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    fig.text(.115,.510,'Pressure mode',ha='left',va='center',fontsize=8.5)
    selected=[]
    for m in MATERIALS:
        r=one(TABLES['modal_summary'],material=m,method='craig_bampton',m=32).copy();r.update(one(TABLES['reference_evaluation'],material=m,method='craig_bampton',m=32));selected.append(r)
    x=np.arange(3);w=.23;ax=bottom[0]
    for i,key,label,color in ((-1,'weighted_grid_shift','Grid',BLUE),(0,'weighted_ritz_shift','Ritz',ORANGE),(1,'weighted_net_shift','Net',GRAY)):
        ax.bar(x+i*w,100*vals(selected,key),w*.94,color=color,label=label)
    ax.axhline(0,color=BLACK,lw=.8);ax.set(xticks=x,xticklabels=NAMES,ylabel='Frequency shift (%)',ylim=(-.8,.45));ax.legend(loc='upper left',ncol=3,handlelength=.55,columnspacing=.5)
    title(ax,'d','Frequency shifts')
    ax=bottom[1]
    for i,key,label,color in ((-1,'total_error_vs_512','ROM',BLUE),(0,'error_without_ritz_shift_vs_512','Coarse $f$',ORANGE),(1,'error_with_fine_frequencies_vs_512','Fine $f$',TEAL)):
        ax.bar(x+i*w,100*vals(selected,key),w*.94,color=color,label=label)
    ax.set(xticks=x,xticklabels=NAMES,ylabel='Receiver error (%)',ylim=(0,85));ax.legend(loc='upper right',handlelength=.8)
    title(ax,'e','Frequency replacements')
    ax=bottom[2];r=TABLES['reduction_comparison']
    for key,label,color,marker in (('same_grid_reduction_difference','Same grid',BLUE,'o'),('difference_from_fine_comparator','$512^2$ reference',ORANGE,'s')):
        ax.semilogy(vals(r,'pressure_dof'),100*vals(r,key),marker=marker,color=color,label=label)
    ax.axhline(100*r[0]['full32_difference_from_fine_comparator'],color=BLACK,ls='--',label='Full $32^2$',lw=.8)
    ax.set(xticks=[32,160,288],xlabel='Reduced dimension $r$',ylabel='Receiver error (%)',ylim=(1,180))
    title(ax,'f','Local capacity')
    for ax in bottom:grid(ax,'y');ax.tick_params(axis='x',labelrotation=20)
    height = fig.get_figheight()
    footer = .25
    positions = [ax.get_position().frozen() for ax in fig.axes]
    fig.set_figheight(height + footer)
    for ax, box in zip(fig.axes, positions):
        ax.set_position([box.x0, (box.y0 * height + footer) / (height + footer),
                         box.width, box.height * height / (height + footer)])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, (y * height + footer) / (height + footer)))
    fig.legend(*bottom[2].get_legend_handles_labels(), loc='lower center',
               bbox_to_anchor=(.55, .008), ncol=3, borderaxespad=0,
               borderpad=0, handlelength=1.2)
    save(fig, 5)

def fig6():
    fig=plt.figure(figsize=(WIDTH,5.7));gs=fig.add_gridspec(2,3,left=.075,right=.985,bottom=.115,top=.94,wspace=.31,hspace=.55,height_ratios=(1.25,1))
    data=TABLES['observation_windows'];matrices=[];weights=[]
    for m in MATERIALS:
        a=np.empty((12,4));w=np.empty(12)
        for s in range(1,4):
            for r in range(1,5):
                j=(s-1)*4+r-1
                for k,end in enumerate((2,4,6,8)):
                    row=one(data,material=m,source=s,receiver=r,window_kind='cumulative',start=0,end=end);a[j,k]=100*row['remove_ritz_error_change']
                    if end==8:w[j]=100*row['reference_energy_share']
        matrices.append(a);weights.append(w)
    lim=np.ceil(max(np.max(abs(a)) for a in matrices))
    for col,(m,a,w) in enumerate(zip(MATERIALS,matrices,weights)):
        inner=gs[0,col].subgridspec(1,2,width_ratios=(4,1.05),wspace=.07)
        ax=fig.add_subplot(inner[0]);wa=fig.add_subplot(inner[1]);im=ax.imshow(a,aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim))
        ax.set(xticks=range(4),xticklabels=['[0,2]','[0,4]','[0,6]','[0,8]'],yticks=range(12),yticklabels=[f'S{s}–R{r}' for s in range(1,4) for r in range(1,5)] if col==0 else [])
        ax.tick_params(axis='x',labelrotation=50,length=0);ax.tick_params(axis='y',length=0)
        title(ax,chr(97+col),NAMES[col])
        if m=='smooth_checker':ax.add_patch(Rectangle((2.5,4.5),1,1,fill=False,edgecolor=BLACK,lw=1.2))
        wa.imshow(w[:,None],aspect='auto',cmap='Greys',vmin=0,vmax=30)
        for j,v in enumerate(w):wa.text(0,j,f'{v:.1f}',ha='center',va='center',color='white' if v>17 else BLACK)
        wa.set(xticks=[0],xticklabels=['Share\n(%)'],yticks=[]);wa.tick_params(length=0)
        for y in (3.5,7.5):ax.axhline(y,color=BLACK,lw=.65);wa.axhline(y,color=BLACK,lw=.65)
    cax=fig.add_axes([.35,.492,.52,.022]);cb=fig.colorbar(im,cax=cax,orientation='horizontal')
    fig.text(.12,.503,r'$\Delta E$ (pp)',ha='left',va='center',fontsize=8.5)
    sens=TABLES['modal_sensitivity'];finite=TABLES['modal_finite_time'];axes=[fig.add_subplot(gs[1,i]) for i in range(3)]
    ax=axes[0];x=np.arange(3);w=.22
    for j,path,label,color in ((0,'rom','$H_r$',BLUE),(1,'fine','$H_f$',TEAL),(2,'symmetric','$H_{av}$',PURPLE)):
        y=[];alt=[]
        for m in MATERIALS:
            y.append(100*one(sens,material=m,fine_grid=128,path=path)['error_change']);alt.append(100*one(sens,material=m,fine_grid=192,path=path)['error_change'])
        ax.bar(x+(j-1)*w,y,w*.92,color=color,label=label);ax.plot(x+(j-1)*w,alt,'_',color=BLACK,ms=6,mew=1.2)
    ax.set(xticks=x,xticklabels=['Stripes','Incl.','Check.'],ylabel=r'$\Delta E$ (pp)',ylim=(0,11));ax.legend(loc='upper left',ncol=3,handlelength=.6,columnspacing=.35);ax.text(.98,.73,'$192^2$: ticks',ha='right',transform=ax.transAxes)
    title(ax,'d','Residue paths')
    rows=[one(finite,material=m,group='all',window_kind='cumulative',end=8) for m in MATERIALS]
    rows.append(one(finite,material='smooth_checker',group='channel',source=2,receiver=2,window_kind='cumulative',end=8));ax=axes[1];pos=np.zeros(4);neg=np.zeros(4)
    for key,label,color in (('exact_same_mode_frequency_terms','Same',ORANGE),('exact_cross_mode_frequency_terms','Cross','#56B4E9'),('negative_twice_ritz_remainder','$q$',GRAY)):
        y=100*vals(rows,key);ax.bar(range(4),y,bottom=np.where(y>=0,pos,neg),width=.60,color=color,edgecolor='white',lw=.5,label=label);pos+=np.maximum(y,0);neg+=np.minimum(y,0)
    ax.plot(range(4),100*vals(rows,'exact_squared_error_change'),'D',color=BLACK,ms=3,label='Total');ax.axhline(0,color=BLACK,lw=.8)
    ax.set(xticks=range(4),xticklabels=['Str.','Inc.','Ch.','S2–R2'],ylabel=r'$\Delta Q$ ($10^{-2}$)',ylim=(-3.2,12.5));ax.legend(loc='upper left',ncol=2,handlelength=.7,columnspacing=.55)
    title(ax,'e','Exact budget')
    ax=axes[2]
    for m,name,color,marker in zip(MATERIALS,NAMES,MCOLORS,('o','s','^')):
        rows=sorted([r for r in finite if r['material']==m and r['group']=='all' and r['window_kind'] in ('cumulative','short_cumulative')],key=lambda r:r['end'])
        ax.loglog(vals(rows,'end'),100*vals(rows,'ritz_linearization_error'),color=color,marker=marker,label=name)
    ax.set(xticks=[.25,1,4,8],xlabel='Window end $T$',ylabel='Ritz linearization error (%)');ax.axhline(100,color=GRAY,ls=':',lw=.7);ax.legend(loc='lower right',handlelength=.8)
    title(ax,'f','Phase approximation')
    for ax in axes:grid(ax,'y')
    save(fig, 6)

def fig7():
    fig = plt.figure(figsize=(WIDTH, 6.7))
    gs = fig.add_gridspec(3, 6, left=.10, right=.985, bottom=.09,
                          top=.90, wspace=.85, hspace=.76)
    capacity_axes = []
    for col, band in enumerate(('low_band', 'all_band')):
        ax = fig.add_subplot(gs[0, 3 * col:3 * col + 3])
        capacity_axes.append(ax)
        for method, label, color, marker in (
            ('full_FV', 'Full FV', BLACK, 'o'),
            ('fixed_modal16_tail', 'Fixed 16 modes', BLUE, 's'),
            ('growing_modal_tail', 'Increasing order', TEAL, '^'),
        ):
            rows = TABLES['layered'][band][method]
            grids = sorted(map(int, rows))
            ax.loglog(grids, [rows[str(n)] for n in grids], color=color,
                      marker=marker, label=label)
        ax.set(xticks=grids, xlabel='Extraction-grid cells', ylabel='Scattering RMS error')
        ax.scientific_y = True
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_formatter(NullFormatter())
        title(ax, 'ab'[col], r'Layered: $0<\omega\leq' + ('4' if col == 0 else '12') + r'\pi$')
        grid(ax)
    fig.legend(*capacity_axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.54, .995), ncol=3)
    data = TABLES['residual']
    low = max(min(r['actual_relative_state_error'] for r in data) * 65, 1e-5)
    high = max(r['relative_guarded_bound'] for r in data) * 140
    xmax = max(r['actual_relative_state_error'] for r in data) * 125
    for col, n in enumerate((32, 64)):
        ax = fig.add_subplot(gs[1, 3 * col:3 * col + 3])
        for method, label, color, marker in (
            ('craig_bampton', 'CB', BLUE, 'o'),
            ('rational_krylov', 'RK', PURPLE, 's'),
        ):
            rows = [r for r in data if r['grid'] == n and r['method'] == method]
            smooth = [r for r in rows if r['material'] in MATERIALS]
            interface = [r for r in rows if r['material'] not in MATERIALS]
            ax.scatter(100 * vals(smooth, 'actual_relative_state_error'),
                       100 * vals(smooth, 'relative_guarded_bound'), s=17,
                       color=color, marker=marker, alpha=.8, label=label)
            ax.scatter(100 * vals(interface, 'actual_relative_state_error'),
                       100 * vals(interface, 'relative_guarded_bound'), s=38,
                       marker='*', facecolors='white', edgecolors=color, lw=.7)
        ax.plot([low, xmax], [low, xmax], ':', color=BLACK, lw=.8)
        ax.set(xscale='log', yscale='log', xlim=(low, xmax), ylim=(low, high),
               xlabel='Endpoint state error (%)', ylabel='Residual bound (%)')
        ax.scientific_y = True
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        title(ax, 'cd'[col], f'CB and RK: $n={n}$, $T=8$')
        if col == 0:
            ax.legend(loc='upper left', ncol=2)
        grid(ax)
    cost_axes = []
    for col, material in enumerate(MATERIALS):
        ax = fig.add_subplot(gs[2, 2 * col:2 * col + 2])
        cost_axes.append(ax)
        for method, cfl, color, marker, label in (
            ('full_stormer_verlet', .25, BLACK, 'o', 'Verlet, CFL 0.25'),
            ('full_stormer_verlet', .125, GRAY, 's', 'Verlet, CFL 0.125'),
            ('craig_bampton', None, BLUE, '^', 'CB, $m=128$'),
        ):
            rows = sorted((r for r in TABLES['cost'] if r['material'] == material
                           and r['method'] == method
                           and (cfl is None or r['requested_cfl'] == cfl)), key=lambda r: r['grid'])
            seconds = vals(rows, 'total_seconds')
            errors = np.vstack((seconds - vals(rows, 'minimum_total_seconds'),
                                vals(rows, 'maximum_total_seconds') - seconds))
            ax.errorbar(seconds, 100 * vals(rows, 'total_receiver_error'), xerr=errors,
                        color=color, fmt=marker + '-', ms=3.5, lw=1.1,
                        elinewidth=.55, capsize=1, label=label)
        ax.set(xscale='log', yscale='log', xlabel='Setup + receivers (s)')
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=5))
        ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20]))
        for tolerance in (5, 10):
            ax.axhline(tolerance, color=GRAY, ls=':', lw=.6)
        if col == 0:
            ax.set_ylabel('Receiver error (%)')
        title(ax, 'efg'[col], NAMES[col])
        grid(ax)
    fig.legend(*cost_axes[0].get_legend_handles_labels(), loc='center',
               bbox_to_anchor=(.54, .335), ncol=3, handlelength=1.0)
    save(fig, 7)

def main():
    global OUT, TABLES, ARRAYS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE.parent / 'data' / 'figures')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--figures', nargs='+', type=int, choices=range(1, 8), default=list(range(1, 8)))
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    metadata = json.loads((data_dir / 'provenance.json').read_text())
    for name in ('plot_tables.json', 'plot_arrays.npz'):
        if sha(data_dir / name) != metadata['outputs'][name]['sha256']:
            raise ValueError(f'Figure input hash differs: {name}')
    TABLES = json.loads((data_dir / 'plot_tables.json').read_text())
    with np.load(data_dir / 'plot_arrays.npz', allow_pickle=False) as archive:
        ARRAYS = {key: archive[key] for key in archive.files}
    OUT = args.output_dir.resolve()
    META.clear()
    for number in args.figures:
        globals()[f'fig{number}']()
    if set(args.figures) == set(range(1, 8)):
        expected = {f'Fig{n}.{ext}' for n in range(1, 8) for ext in ('pdf', 'png')}
        for path in OUT.iterdir():
            if path.is_file() and re.fullmatch(r'Fig\d+\.(pdf|png)', path.name) and path.name not in expected:
                path.unlink()
    manifest = {
        'figures': META,
        'inputs': {name: sha(data_dir / name) for name in ('plot_tables.json', 'plot_arrays.npz')},
        'style': {'width_inches': WIDTH, 'minimum_text_pt': 8.5, 'png_dpi': 300,
                  'signed_colormap': 'RdBu: positive blue, negative red; centered at zero'},
    }
    temporary = OUT / 'figures.json.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(OUT / 'figures.json')


if __name__ == '__main__':
    main()

"""
Sprint 1 EDA: Exploratory Data Analysis
Battery RUL Research — PI-TCN
Target: Publication-quality figures for Proyek Data Mining + Journal
"""

import os, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

# ── Paths ────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW    = os.path.join(BASE,"data","raw","LG Dataset",
                      "LG_HG2_Original_Dataset")
PROC   = os.path.join(BASE,"data","processed")
FIG    = os.path.join(BASE,"outputs","figures")
os.makedirs(FIG, exist_ok=True)

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi'       : 150,
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'lines.linewidth'  : 1.2,
    'legend.framealpha': 0.8,
})

COLORS = {
    '25degC': '#185FA5',
    '40degC': '#993C1D',
    'neutral': '#444441',
    'accent' : '#1D9E75',
}

# ── CSV reader (same as preprocessing.py) ────────────────────
def find_header_row(filepath):
    with open(filepath,'r',encoding='utf-8',errors='replace') as f:
        for i,line in enumerate(f):
            if 'Voltage' in line and 'Current' in line:
                return i
    raise ValueError(f"Header not found: {filepath}")

def read_csv(filepath):
    hrow = find_header_row(filepath)
    df = pd.read_csv(filepath, skiprows=hrow,
                     encoding='utf-8', encoding_errors='replace',
                     low_memory=False)
    df.columns = df.columns.str.strip()
    if 'Time Stamp' in df.columns:
        df['Time Stamp'] = pd.to_datetime(
            df['Time Stamp'],
            format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        if df['Time Stamp'].isna().all():
            df['Time Stamp'] = pd.to_datetime(
                df['Time Stamp'], errors='coerce',
                infer_datetime_format=True)
        t0 = df['Time Stamp'].dropna().iloc[0]
        df['time_sec'] = (df['Time Stamp'] - t0).dt.total_seconds()
    else:
        df['time_sec'] = np.arange(len(df)) * 0.1
    for col in ['Voltage','Current','Temperature','Capacity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['Voltage','Current','time_sec']).reset_index(drop=True)

FILE_MAP = {
    "25degC": {"udds":"551_UDDS.csv",  "c20":"549_C20DisCh.csv"},
    "40degC": {"udds":"556_UDDS.csv",  "c20":"555_C20DisCh.csv"},
}

print("Loading data...")
dfs = {}
for temp in ("25degC","40degC"):
    dfs[temp] = {
        "udds": read_csv(os.path.join(RAW, temp, FILE_MAP[temp]["udds"])),
        "c20" : read_csv(os.path.join(RAW, temp, FILE_MAP[temp]["c20"])),
    }
    print(f"  {temp}: UDDS {len(dfs[temp]['udds']):,} rows | "
          f"C20 {len(dfs[temp]['c20']):,} rows")

with open(os.path.join(PROC,"metadata.json")) as f:
    meta = json.load(f)

X_tr = np.load(os.path.join(PROC,"X_train.npy"))
y_tr = np.load(os.path.join(PROC,"y_train.npy"))


# ════════════════════════════════════════════════════════════
# FIGURE 1: Raw Signal Overview (4 subplots)
# ════════════════════════════════════════════════════════════
print("\n[Fig 1] Raw signal overview...")
fig, axes = plt.subplots(4, 1, figsize=(14,10), sharex=False)

for temp, ax_row_offset, sig_key in [("25degC",0,"udds"),("40degC",2,"udds")]:
    df  = dfs[temp][sig_key]
    col = COLORS[temp]
    t   = df['time_sec'] / 3600  # hours

    # Voltage
    ax = axes[ax_row_offset]
    ax.plot(t, df['Voltage'], color=col, lw=0.6, alpha=0.85)
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'UDDS Drive Cycle — {temp}', fontweight='bold')
    ax.set_ylim(2.3, 4.4)
    ax.axhline(4.2, color='gray', lw=0.8, ls=':', label='V_max=4.2V')
    ax.axhline(2.5, color='gray', lw=0.8, ls=':', label='V_min=2.5V')
    ax.legend(fontsize=9, loc='lower right')

    # Current
    ax = axes[ax_row_offset + 1]
    ax.plot(t, df['Current'], color=col, lw=0.5, alpha=0.75)
    ax.axhline(0, color='black', lw=0.6, ls='-')
    ax.set_ylabel('Current (A)')
    ax.set_xlabel('Time (hours)')
    ax.fill_between(t, df['Current'], 0,
                    where=(df['Current'] > 0),
                    alpha=0.15, color='#1D9E75', label='Charging')
    ax.fill_between(t, df['Current'], 0,
                    where=(df['Current'] < 0),
                    alpha=0.15, color=col, label='Discharging')
    ax.legend(fontsize=9)

plt.suptitle('LG HG2 18650 — Raw Signal Comparison (25°C vs 40°C)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
path = os.path.join(FIG,"fig01_raw_signals.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 2: Statistical Distribution of Features
# ════════════════════════════════════════════════════════════
print("\n[Fig 2] Feature distributions...")
fig, axes = plt.subplots(2, 3, figsize=(15,8))
feat_names = ['Voltage (V)', 'Current (A)', 'Temperature (°C)']

for row, temp in enumerate(("25degC","40degC")):
    df = dfs[temp]["udds"]
    cols_raw = ['Voltage','Current','Temperature']
    for col_idx, (col, fname) in enumerate(zip(cols_raw, feat_names)):
        ax = axes[row][col_idx]
        if col not in df.columns:
            ax.text(0.5,0.5,'N/A',ha='center',va='center',
                    transform=ax.transAxes)
            continue
        data = df[col].dropna()
        ax.hist(data, bins=80, color=COLORS[temp],
                edgecolor='white', linewidth=0.3, alpha=0.85)
        ax.axvline(data.mean(), color='black', lw=1.5,
                   ls='--', label=f'μ={data.mean():.2f}')
        ax.axvline(data.median(), color='red', lw=1.2,
                   ls=':', label=f'med={data.median():.2f}')
        ax.set_title(f'{fname} — {temp}')
        ax.set_xlabel(fname)
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)

plt.suptitle('Feature Distribution — UDDS Profile (25°C vs 40°C)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIG,"fig02_feature_distributions.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 3: C/20 Discharge — SoH Comparison
# ════════════════════════════════════════════════════════════
print("\n[Fig 3] C20 discharge + SoH...")
fig, axes = plt.subplots(1, 2, figsize=(13,5))

for temp in ("25degC","40degC"):
    df  = dfs[temp]["c20"]
    col = COLORS[temp]
    Q   = meta[temp]["Q_actual"]
    SoH = meta[temp]["SoH"]

    # Discharge only
    df_dis = df[df['Current'] < -0.01].copy() if len(
        df[df['Current'] < -0.01]) > 0 else df
    t_h = (df_dis['time_sec'] - df_dis['time_sec'].iloc[0]) / 3600

    axes[0].plot(t_h, df_dis['Voltage'], color=col, lw=1.4,
                 label=f'{temp} (SoH={SoH*100:.1f}%, Q={Q:.3f}Ah)')

axes[0].set_xlabel('Time (hours)')
axes[0].set_ylabel('Voltage (V)')
axes[0].set_title('C/20 Discharge Curve — Capacity Measurement')
axes[0].legend(fontsize=10)
axes[0].axhline(2.5, color='gray', ls='--', lw=0.8, label='Cutoff 2.5V')

# Bar chart: Q_actual vs Q_nominal
labels_bar = ['Q nominal\n(3.0 Ah)',
              f'Q actual\n25°C ({meta["25degC"]["Q_actual"]:.3f} Ah)',
              f'Q actual\n40°C ({meta["40degC"]["Q_actual"]:.3f} Ah)']
vals  = [Q_NOMINAL := 3.0,
         meta["25degC"]["Q_actual"],
         meta["40degC"]["Q_actual"]]
bars  = axes[1].bar(labels_bar, vals,
                    color=['#888780', COLORS['25degC'], COLORS['40degC']],
                    edgecolor='white', width=0.5)
axes[1].set_ylim(0, 3.5)
axes[1].set_ylabel('Capacity (Ah)')
axes[1].set_title('Capacity Comparison: Nominal vs Actual')
for bar, val in zip(bars, vals):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.04,
                 f'{val:.3f} Ah', ha='center', fontsize=11,
                 fontweight='bold')
soh_25 = meta["25degC"]["SoH"] * 100
soh_40 = meta["40degC"]["SoH"] * 100
axes[1].axhline(3.0 * 0.80, color='#E24B4A', ls='--', lw=1.2,
                label='EOL threshold (80%)')
axes[1].legend(fontsize=10)

plt.suptitle('State of Health Analysis — LG HG2 18650',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIG,"fig03_soh_analysis.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 4: Correlation Heatmap
# ════════════════════════════════════════════════════════════
print("\n[Fig 4] Correlation heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(13,5))

for ax, temp in zip(axes, ("25degC","40degC")):
    df = dfs[temp]["udds"]
    cols_corr = [c for c in
                 ['Voltage','Current','Temperature','Capacity']
                 if c in df.columns]
    corr = df[cols_corr].corr()

    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols_corr)))
    ax.set_yticks(range(len(cols_corr)))
    ax.set_xticklabels(cols_corr, rotation=30, ha='right')
    ax.set_yticklabels(cols_corr)
    ax.set_title(f'Correlation Matrix — {temp}')

    for i in range(len(cols_corr)):
        for j in range(len(cols_corr)):
            ax.text(j, i, f'{corr.values[i,j]:.2f}',
                    ha='center', va='center', fontsize=10,
                    color='white' if abs(corr.values[i,j]) > 0.6
                    else 'black', fontweight='bold')

plt.colorbar(im, ax=axes, shrink=0.8, label='Pearson r')
plt.suptitle('Feature Correlation — UDDS Profile',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIG,"fig04_correlation.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 5: RUL Label Distribution & Dataset Summary
# ════════════════════════════════════════════════════════════
print("\n[Fig 5] RUL distribution + dataset summary...")
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# RUL histogram
axes[0].hist(y_tr, bins=60, color=COLORS['accent'],
             edgecolor='white', linewidth=0.3)
axes[0].set_xlabel('RUL (normalized, 0=EOL, 1=new)')
axes[0].set_ylabel('Sequence count')
axes[0].set_title('RUL Label Distribution\n(training set)')
axes[0].axvline(y_tr.mean(), color='black', ls='--', lw=1.5,
                label=f'mean={y_tr.mean():.3f}')
axes[0].legend(fontsize=10)

# Sequence count per source
temps_label = ['25°C UDDS','25°C Mixed','40°C UDDS','40°C Mixed']
counts = [15955, sum([7713,7908,7373,8075,7209,7801,7314,8582]),
          15886, sum([7525,7886,6753,1123,1299,1167,775,1429])]
bar_colors = [COLORS['25degC'], '#5DCAA5', COLORS['40degC'], '#F09975']
bars = axes[1].bar(temps_label, counts,
                   color=bar_colors, edgecolor='white')
axes[1].set_ylabel('Number of sequences')
axes[1].set_title('Sequence Count per Source File Group')
axes[1].tick_params(axis='x', rotation=15)
for bar, cnt in zip(bars, counts):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 200,
                 f'{cnt:,}', ha='center', fontsize=10)

# Dataset stats table
stats_data = [
    ['Total sequences',    f'{len(y_tr)+len(np.load(os.path.join(PROC,"y_val.npy"))):,}'],
    ['Training set',       f'{len(y_tr):,}'],
    ['Validation set',     f'{len(np.load(os.path.join(PROC,"y_val.npy"))):,}'],
    ['Window length',      '100 timesteps'],
    ['Stride',             '10 timesteps'],
    ['Features',           '6'],
    ['Q_actual 25°C',      f'{meta["25degC"]["Q_actual"]:.4f} Ah'],
    ['Q_actual 40°C',      f'{meta["40degC"]["Q_actual"]:.4f} Ah'],
    ['SoH 25°C',           f'{meta["25degC"]["SoH"]*100:.1f}%'],
    ['SoH 40°C',           f'{meta["40degC"]["SoH"]*100:.1f}%'],
]
axes[2].axis('off')
tbl = axes[2].table(cellText=stats_data,
                    colLabels=['Parameter','Value'],
                    cellLoc='left', loc='center',
                    colWidths=[0.55, 0.45])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
for (r,c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#185FA5')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#f0f4fa')
    cell.set_edgecolor('#cccccc')
axes[2].set_title('Dataset Summary', fontweight='bold')

plt.suptitle('RUL Label Analysis & Dataset Overview',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIG,"fig05_rul_dataset_summary.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


# ════════════════════════════════════════════════════════════
# FIGURE 6: Voltage-Current Phase Portrait (25°C vs 40°C)
# ════════════════════════════════════════════════════════════
print("\n[Fig 6] Phase portrait...")
fig, axes = plt.subplots(1, 2, figsize=(12,5))

for ax, temp in zip(axes, ("25degC","40degC")):
    df  = dfs[temp]["udds"]
    col = COLORS[temp]
    # Subsample for plot performance
    step = max(1, len(df)//5000)
    ax.scatter(df['Current'].iloc[::step],
               df['Voltage'].iloc[::step],
               c=col, s=1.5, alpha=0.3)
    ax.set_xlabel('Current (A)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title(f'V–I Phase Portrait — {temp}')
    ax.set_ylim(2.3, 4.5)
    # Annotate operating region
    ax.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
    ax.text(0.05, 0.95, 'Charge region', transform=ax.transAxes,
            color='#1D9E75', fontsize=9, va='top')
    ax.text(0.55, 0.05, 'Discharge region', transform=ax.transAxes,
            color=col, fontsize=9)

plt.suptitle('Voltage–Current Phase Portrait\n'
             'Dynamic Load Profile (UDDS)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
path = os.path.join(FIG,"fig06_phase_portrait.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {path}")


print("\n" + "="*60)
print("EDA Complete. All figures saved to outputs/figures/")
print("="*60)
fig_files = [f for f in os.listdir(FIG) if f.startswith('fig0')]
for f in sorted(fig_files):
    size = os.path.getsize(os.path.join(FIG,f))
    print(f"  {f}  ({size//1024} KB)")

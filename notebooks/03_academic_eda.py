"""
═══════════════════════════════════════════════════════════════
Sprint 1.5 — Academic-Grade Exploratory Data Analysis (EDA)
═══════════════════════════════════════════════════════════════
File        : notebooks/03_academic_eda.py
Project     : PI-TCN SOC Estimator Research
Target      : Publication-quality figures for Sinta 2–3 / Scopus
Date        : 2026-04-07

Outputs (saved to outputs/figures/):
  fig_eda01_thermal_voltage_shift  (.png, .pdf)
  fig_eda02_kinetics_features      (.png, .pdf)
  fig_eda03_physics_violation      (.png, .pdf)
  fig_eda04_correlation_matrix     (.png, .pdf)
"""

import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

# ── Add src to path for reusing preprocessing helpers ────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))
from preprocessing import read_csv, engineer_features, measure_capacity, DATA_RAW

# ── Paths ────────────────────────────────────────────────────
PROC_A = os.path.join(BASE, 'data', 'processed', 'scenario_A')
FIG    = os.path.join(BASE, 'outputs', 'figures')
os.makedirs(FIG, exist_ok=True)

# ═════════════════════════════════════════════════════════════
# PUBLICATION STYLE CONFIGURATION
# ═════════════════════════════════════════════════════════════
plt.rcParams.update({
    # --- Resolution ---
    'figure.dpi'            : 300,
    'savefig.dpi'           : 300,
    # --- Font (Serif / academic standard) ---
    'font.family'           : 'serif',
    'font.serif'            : ['Times New Roman', 'DejaVu Serif',
                               'Palatino Linotype', 'Georgia'],
    'font.size'             : 11,
    'axes.titlesize'        : 13,
    'axes.labelsize'        : 12,
    'xtick.labelsize'       : 10,
    'ytick.labelsize'       : 10,
    'legend.fontsize'       : 10,
    # --- Spines & Grid ---
    'axes.spines.top'       : False,
    'axes.spines.right'     : False,
    'axes.grid'             : True,
    'grid.alpha'            : 0.25,
    'grid.linestyle'        : '--',
    'grid.linewidth'        : 0.6,
    # --- Lines ---
    'lines.linewidth'       : 1.5,
    # --- Legend ---
    'legend.framealpha'     : 0.92,
    'legend.edgecolor'      : '0.75',
    'legend.fancybox'       : True,
    # --- Math text ---
    'mathtext.fontset'      : 'cm',
})

# Colorblind-friendly palette (Seaborn "colorblind")
CB = sns.color_palette("colorblind", 10)
C_TRAIN  = CB[0]   # blue   — 25°C (Train distribution)
C_OOD1   = CB[1]   # orange — 40°C (OOD Test)
C_OOD2   = CB[2]   # green  — −20°C (OOD Test)
C_DISC   = CB[3]   # red    — discharge regions
C_ACCENT = CB[4]   # purple — SOC / accent
C_DV     = CB[5]   # brown  — dV/dt
C_DI     = CB[8]   # yellow — dI/dt

TEMP_LABELS = {
    '25degC' : '25 °C  (Train)',
    '40degC' : '40 °C  (OOD Test)',
    'n20degC': '−20 °C (OOD Test)',
}

# ── Helpers ──────────────────────────────────────────────────
def load_temp_raw(temp_name, profile='udds'):
    """Load and feature-engineer raw CSV data for one temperature.
    
    Parameters
    ----------
    temp_name : str   e.g. '25degC', 'n20degC'
    profile   : str   'udds' for UDDS only, 'all' for all drive cycles
    
    Returns
    -------
    (pd.DataFrame, float)  — engineered dataframe + Q_actual
    """
    temp_dir = os.path.join(DATA_RAW, temp_name)
    Q_actual = measure_capacity(temp_dir)
    csvs = sorted(glob.glob(os.path.join(temp_dir, '*.csv')))
    dfs = []
    for c in csvs:
        fname = os.path.basename(c).lower()
        if profile == 'udds':
            if 'udds' not in fname:
                continue
        elif profile == 'all':
            if not any(k in fname for k in ['udds','la92','hwfet','us06','mixed']):
                continue
        try:
            df = read_csv(c)
            df = engineer_features(df, Q_actual)
            dfs.append(df)
        except Exception:
            pass
    if dfs:
        return pd.concat(dfs, ignore_index=True), Q_actual
    return pd.DataFrame(), Q_actual


def save_fig(fig, name):
    """Save a figure as both high-res PNG and vector PDF."""
    for ext in ('png', 'pdf'):
        path = os.path.join(FIG, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ {path}")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════
print("=" * 64)
print("  Sprint 1.5 — Academic-Grade EDA")
print("  Target: Sinta 2–3 / Scopus Publication Figures")
print("=" * 64)

print("\n[DATA] Loading raw sensor data for 3 temperature groups...")
df_25,  Q_25  = load_temp_raw('25degC',  'udds')
df_40,  Q_40  = load_temp_raw('40degC',  'udds')
df_n20, Q_n20 = load_temp_raw('n20degC', 'udds')

for name, df, Q in [('25°C', df_25, Q_25),
                     ('40°C', df_40, Q_40),
                     ('−20°C', df_n20, Q_n20)]:
    print(f"  {name:>5s} : {len(df):>8,} rows  |  Q_actual = {Q:.3f} Ah")

# Load processed sequences for correlation analysis
print("\n[DATA] Loading processed Scenario A sequences...")
X_train_A = np.load(os.path.join(PROC_A, 'X_train.npy'))
y_train_A = np.load(os.path.join(PROC_A, 'y_train.npy'))
print(f"  X_train : {X_train_A.shape}")
print(f"  y_train : {y_train_A.shape}")


# ═════════════════════════════════════════════════════════════
# FIGURE 1 — Thermal Voltage Shift (OOD Distribution)
# ═════════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("[FIG 1] Thermal Voltage Shift — Out-of-Distribution Analysis")
print("─" * 64)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# ── (a) KDE Overlay ─────────────────────────────────────────
ax = axes[0]
kde_data = [
    (df_25,  '25 °C (Train)',    C_TRAIN, '-',  2.0),
    (df_40,  '40 °C (OOD Test)', C_OOD1,  '--', 2.0),
    (df_n20, '−20 °C (OOD Test)',C_OOD2,  '-.', 2.0),
]
for df_t, label, color, ls, lw in kde_data:
    v = df_t['Voltage'].dropna()
    sns.kdeplot(v, ax=ax, color=color, linewidth=lw, linestyle=ls,
                label=label, fill=True, alpha=0.12)

ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Probability Density')
ax.set_title('(a) Voltage Distribution Across Temperatures',
             fontweight='bold')
ax.legend(frameon=True, loc='upper left')
ax.set_xlim(2.3, 4.5)

# Annotate the distribution shift
med_25  = df_25['Voltage'].median()
med_n20 = df_n20['Voltage'].median()
y_top = ax.get_ylim()[1]
ax.annotate(
    'Domain\nShift',
    xy=((med_25 + med_n20) / 2, y_top * 0.15),
    xytext=(3.05, y_top * 0.72),
    fontsize=10, ha='center', fontstyle='italic', color='0.35',
    arrowprops=dict(arrowstyle='->', color='0.45', lw=1.4,
                    connectionstyle='arc3,rad=-0.2'),
)

# ── (b) Box Plots ───────────────────────────────────────────
ax = axes[1]
box_sets = [
    (df_25,  '25 °C\n(Train)',    C_TRAIN),
    (df_40,  '40 °C\n(OOD)',      C_OOD1),
    (df_n20, '−20 °C\n(OOD)',     C_OOD2),
]
box_data = []
for df_t, _, _ in box_sets:
    v = df_t['Voltage'].dropna().values
    if len(v) > 15000:
        rng = np.random.default_rng(42)
        v = rng.choice(v, 15000, replace=False)
    box_data.append(v)

bp = ax.boxplot(
    box_data,
    labels=[b[1] for b in box_sets],
    patch_artist=True,
    widths=0.45,
    showfliers=False,
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)
for patch, (_, _, color) in zip(bp['boxes'], box_sets):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
    patch.set_edgecolor('0.3')
    patch.set_linewidth(1.2)

ax.set_ylabel('Voltage (V)')
ax.set_title('(b) Operating Voltage Range Comparison', fontweight='bold')

# Reference lines
ax.axhline(4.2, color='0.5', ls=':', lw=1.0)
ax.axhline(2.5, color='0.5', ls=':', lw=1.0)
ax.text(3.42, 4.23, r'$V_{\max}$ = 4.2 V', fontsize=9, color='0.4')
ax.text(3.42, 2.38, r'$V_{\min}$ = 2.5 V', fontsize=9, color='0.4')

fig.suptitle(
    'Thermal Domain Shift Analysis — LG HG2 18650\n'
    'Evidence of Out-of-Distribution Voltage Behavior Across Temperature',
    fontsize=13, fontweight='bold', y=1.03,
)
plt.tight_layout()
save_fig(fig, 'fig_eda01_thermal_voltage_shift')


# ═════════════════════════════════════════════════════════════
# FIGURE 2 — Kinetics Feature Analysis (dV/dt & dI/dt)
# ═════════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("[FIG 2] Kinetics Feature Analysis — Why Derivatives Matter")
print("─" * 64)

# Extract a discharge window of 100 timesteps from 25°C UDDS
df_kin = df_25.copy()
dis_mask = df_kin['Current'] < -0.01

# Find contiguous discharge blocks
block_id = dis_mask.ne(dis_mask.shift()).cumsum()
dis_groups = df_kin[dis_mask].groupby(block_id[dis_mask])

# Pick the largest block for best visual
largest_key = max(dis_groups.groups.keys(), key=lambda k: len(dis_groups.get_group(k)))
block_df = dis_groups.get_group(largest_key).copy()

# Select 100 points from the middle
mid = len(block_df) // 2
s = max(0, mid - 50)
e = min(len(block_df), s + 100)
if e - s < 100:
    s = max(0, e - 100)

win = block_df.iloc[s:e].copy().reset_index(drop=True)
t_ix = np.arange(len(win))

fig = plt.figure(figsize=(11, 7.5))
gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 0.85, 0.85], hspace=0.38)

# ── (a) Voltage ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_ix, win['Voltage'], color=C_TRAIN, lw=2.2, zorder=3)
ax1.fill_between(t_ix, win['Voltage'],
                 win['Voltage'].min() - 0.02,
                 alpha=0.08, color=C_TRAIN, zorder=1)
ax1.set_ylabel('Voltage (V)')
ax1.set_title('(a)  Voltage Signal — Single Discharge Window (25 °C, UDDS)',
              fontweight='bold')
ax1.set_xlim(0, len(t_ix) - 1)

# ── (b) dV/dt ───────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(t_ix, win['dV_dt'], color=C_DV, lw=1.6, zorder=3)
ax2.axhline(0, color='0.5', lw=0.7, ls='-', zorder=1)

# Highlight top 10% transients
thr = win['dV_dt'].abs().quantile(0.90)
spike_mask = win['dV_dt'].abs() > thr
for idx in t_ix[spike_mask.values]:
    ax2.axvline(idx, color=C_DISC, alpha=0.18, lw=5, zorder=0)

ax2.set_ylabel('dV/dt  (V · s⁻¹)')
ax2.set_title('(b)  Voltage Rate of Change — Captures Transient Dynamics',
              fontweight='bold')
# Add custom legend
ax2.legend(
    handles=[
        plt.Line2D([0], [0], color=C_DV, lw=1.6, label='dV/dt'),
        Patch(facecolor=C_DISC, alpha=0.25,
              label=f'|dV/dt| > P90 ({thr:.4f} V/s)'),
    ],
    loc='upper right', fontsize=9,
)

# ── (c) dI/dt ───────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(t_ix, win['dI_dt'], color=C_DI, lw=1.6, zorder=3)
ax3.axhline(0, color='0.5', lw=0.7, ls='-', zorder=1)
ax3.set_ylabel('dI/dt  (A · s⁻¹)')
ax3.set_xlabel('Timestep Index within Window')
ax3.set_title('(c)  Current Rate of Change — Detects Load Transitions',
              fontweight='bold')

fig.suptitle(
    'Kinetics Feature Analysis — Why Temporal Derivatives Matter\n'
    'Single Discharge Sequence (100 timesteps, 25 °C UDDS)',
    fontsize=13, fontweight='bold', y=1.01,
)
plt.tight_layout()
save_fig(fig, 'fig_eda02_kinetics_features')


# ═════════════════════════════════════════════════════════════
# FIGURE 3 — Physics Violation Identifier
# ═════════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("[FIG 3] Physics Violation Identifier — Discharge Monotonicity")
print("─" * 64)

# Take a representative segment from 25°C UDDS (~3000 rows ≈ 5 min)
n_rows = 3000
cyc = df_25.iloc[:n_rows].copy().reset_index(drop=True)
t_min = (cyc['time_sec'] - cyc['time_sec'].iloc[0]) / 60.0

fig, (ax_i, ax_soc) = plt.subplots(
    2, 1, figsize=(12, 6.5), sharex=True,
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.12},
)

# ── (a) Current + discharge shading ─────────────────────────
ax_i.plot(t_min, cyc['Current'], color=C_TRAIN, lw=0.9,
          alpha=0.95, label='Current (A)', zorder=3)
ax_i.axhline(0, color='black', lw=0.5, ls='-')
ax_i.axhline(-0.01, color=C_DISC, lw=1.0, ls='--', alpha=0.6,
             label='Discharge threshold ($I$ = −0.01 A)')

# Shade discharge regions
dis = cyc['Current'] < -0.01
in_dis = False
seg_s = 0
for i in range(len(dis)):
    if dis.iloc[i] and not in_dis:
        seg_s = i
        in_dis = True
    elif (not dis.iloc[i]) and in_dis:
        ax_i.axvspan(t_min.iloc[seg_s], t_min.iloc[i - 1],
                     alpha=0.12, color=C_DISC, zorder=0)
        in_dis = False
if in_dis:
    ax_i.axvspan(t_min.iloc[seg_s], t_min.iloc[-1],
                 alpha=0.12, color=C_DISC, zorder=0)

ax_i.set_ylabel('Current (A)')
ax_i.set_title(
    '(a)  Current Profile with Discharge Regions Highlighted',
    fontweight='bold',
)
ax_i.legend(loc='lower left', fontsize=9)

# ── (b) SOC + same discharge shading ────────────────────────
ax_soc.plot(t_min, cyc['SOC_cc'], color=C_ACCENT, lw=1.8,
            zorder=3)

# Re-shade discharge regions on SOC axis
in_dis = False
for i in range(len(dis)):
    if dis.iloc[i] and not in_dis:
        seg_s = i
        in_dis = True
    elif (not dis.iloc[i]) and in_dis:
        ax_soc.axvspan(t_min.iloc[seg_s], t_min.iloc[i - 1],
                       alpha=0.12, color=C_DISC, zorder=0)
        in_dis = False
if in_dis:
    ax_soc.axvspan(t_min.iloc[seg_s], t_min.iloc[-1],
                   alpha=0.12, color=C_DISC, zorder=0)

# Annotation: "SOC must decrease here"
disc_idx = np.where(dis.values)[0]
if len(disc_idx) > 80:
    ann_i = disc_idx[len(disc_idx) // 3]
    ann_t = t_min.iloc[ann_i]
    ann_s = cyc['SOC_cc'].iloc[ann_i]
    t_range = t_min.iloc[-1] - t_min.iloc[0]
    ax_soc.annotate(
        'SOC must decrease\n(Physics constraint)',
        xy=(ann_t, ann_s),
        xytext=(ann_t + t_range * 0.18, min(ann_s + 0.08, 1.05)),
        fontsize=10, ha='center', fontweight='bold', color=C_DISC,
        arrowprops=dict(arrowstyle='->', color=C_DISC, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor=C_DISC, alpha=0.92),
    )

ax_soc.set_ylabel('State of Charge (SOC)')
ax_soc.set_xlabel('Time (minutes)')
ax_soc.set_title(
    '(b)  SOC Trajectory — Monotonicity Constraint During Discharge',
    fontweight='bold',
)
ax_soc.set_ylim(-0.02, 1.08)

# Custom legend
legend_items = [
    plt.Line2D([0], [0], color=C_ACCENT, lw=1.8,
               label='SOC (Coulomb Counting)'),
    Patch(facecolor=C_DISC, alpha=0.2,
          label='Discharge region ($I$ < −0.01 A)'),
]
ax_soc.legend(handles=legend_items, loc='upper right', fontsize=9)

fig.suptitle(
    'Physics Violation Identification — Discharge Monotonicity Constraint\n'
    'Foundation for Physics-Informed Loss Penalty (25 °C, UDDS)',
    fontsize=13, fontweight='bold', y=1.02,
)
plt.tight_layout()
save_fig(fig, 'fig_eda03_physics_violation')


# ═════════════════════════════════════════════════════════════
# FIGURE 4 — Pearson Correlation Matrix (5 features + SOC)
# ═════════════════════════════════════════════════════════════
print("\n" + "─" * 64)
print("[FIG 4] Pearson Correlation Matrix — Features vs SOC")
print("─" * 64)

# Use raw 25°C data (all drive cycles) for comprehensive correlation
df_corr, _ = load_temp_raw('25degC', 'all')

feat_cols   = ['Voltage', 'Current', 'Temperature', 'dV_dt', 'dI_dt', 'SOC_cc']
feat_labels = ['Voltage\n(V)', 'Current\n(A)', 'Temp.\n(°C)',
               'dV/dt\n(V/s)', 'dI/dt\n(A/s)', 'SOC']

available = [c for c in feat_cols if c in df_corr.columns]
print(f"  Columns available: {len(available)}/{len(feat_cols)}")

if len(available) == len(feat_cols):
    corr = df_corr[feat_cols].corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Mask for upper triangle (optional: set to None for full)
    mask = None  # show full matrix

    hm = sns.heatmap(
        corr, annot=True, fmt='.3f',
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        xticklabels=feat_labels,
        yticklabels=feat_labels,
        square=True,
        linewidths=0.8, linecolor='white',
        cbar_kws={
            'label': 'Pearson Correlation Coefficient ($r$)',
            'shrink': 0.82,
        },
        annot_kws={'size': 11, 'fontweight': 'bold'},
        mask=mask,
        ax=ax,
    )

    ax.set_title(
        'Pearson Correlation Matrix\n'
        'Input Features & Target SOC  (25 °C, All Drive Cycles)',
        fontweight='bold', pad=18,
    )

    # Highlight SOC row and column with a border
    n = len(feat_cols)
    soc_idx = n - 1  # last column/row
    ax.add_patch(plt.Rectangle(
        (soc_idx, 0), 1, n,
        fill=False, edgecolor='black', lw=2.5, clip_on=False,
    ))
    ax.add_patch(plt.Rectangle(
        (0, soc_idx), n, 1,
        fill=False, edgecolor='black', lw=2.5, clip_on=False,
    ))

    plt.tight_layout()
    save_fig(fig, 'fig_eda04_correlation_matrix')
else:
    missing = set(feat_cols) - set(available)
    print(f"  ⚠ MISSING COLUMNS: {missing}")
    print(f"    Cannot generate correlation figure.")


# ═════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  ✅ Sprint 1.5 — Academic EDA Complete!")
print("=" * 64)

eda_files = sorted([f for f in os.listdir(FIG) if f.startswith('fig_eda')])
total_kb = 0
for f in eda_files:
    fpath = os.path.join(FIG, f)
    sz = os.path.getsize(fpath)
    total_kb += sz // 1024
    ext = os.path.splitext(f)[1]
    print(f"  {f:<46s}  {sz // 1024:>5,} KB")

print(f"\n  Total figures : {len(eda_files)} files ({total_kb:,} KB)")
print(f"  Formats       : PNG (300 DPI) + PDF (vector)")
print(f"  Palette       : Seaborn 'colorblind' (CVD-safe)")
print(f"  Font          : Serif (Times New Roman / fallback)")
print("=" * 64)

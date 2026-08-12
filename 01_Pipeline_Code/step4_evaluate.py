"""
LTAF STEP 4: EVALUATION + VISUALIZATION
=========================================
Run this after step3 completes.
Reads results CSVs + raw window data.

Produces 8 figures + paper-ready results table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import kendalltau, mannwhitneyu, wilcoxon
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

OUTPUT_PATH  = r"C:\Users\HOME\Desktop\ltaf_project"
WINDOW_PATH  = os.path.join(OUTPUT_PATH, "data", "windows")
RR_PATH      = os.path.join(OUTPUT_PATH, "data", "rr")
RESULTS_PATH = os.path.join(OUTPUT_PATH, "results")
FIGURE_PATH  = os.path.join(OUTPUT_PATH, "figures")
os.makedirs(FIGURE_PATH, exist_ok=True)

HORIZON_MINS = [5, 10, 15, 20, 30, 45, 60]
NORMAL_IDX   = len(HORIZON_MINS)
ALL_RECORDS  = [f"{i:02d}" for i in range(84)]
CSD_NAMES    = ['variance', 'lag1_ac', 'ar1_coeff', 'skewness', 'kurtosis']
HRV_NAMES    = ['rr_mean', 'rr_std', 'rmssd', 'pnn50', 'lf_hf_ratio',
                'sample_ent', 'dfa_alpha', 'poincare_sd1', 'poincare_sd2']
ALL_FEAT     = CSD_NAMES + HRV_NAMES

sns.set_theme(style='whitegrid', palette='husl')
BLUE  = '#2874A6'
GREEN = '#1E8449'
RED   = '#C0392B'
ORG   = '#D68910'

def savefig(fig, name):
    p = os.path.join(FIGURE_PATH, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅  {name}")


def load_windows(rec):
    f = os.path.join(WINDOW_PATH, f'{rec}_windows.npz')
    if not os.path.exists(f): return None
    d = np.load(f, allow_pickle=True)
    X = np.hstack([d['csd_features'], d['hrv_features']]).astype(np.float32)
    return {'X': X, 'horizon': d['horizon_label'],
            'binary': d['binary_label'], 'times': d['window_times']}


def load_rr(rec):
    f = os.path.join(RR_PATH, f'{rec}_rr.npz')
    if not os.path.exists(f): return None
    d = np.load(f, allow_pickle=True)
    return {'rr_ms': d['rr_ms'], 'rr_times': d['rr_times'],
            'af_onsets': d['af_onsets'], 'paroxysmal_af': d['paroxysmal_af']}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
all_windows, all_rr = {}, {}
for rec in ALL_RECORDS:
    w = load_windows(rec)
    r = load_rr(rec)
    if w is not None: all_windows[rec] = w
    if r is not None: all_rr[rec]     = r
print(f"  {len(all_windows)} patients with windows")
print(f"  {len(all_rr)} patients with RR data")

# Load step3 results if they exist
lopo_df   = pd.DataFrame()
iph_df    = pd.DataFrame()
curve_df  = pd.DataFrame()
lopo_path = os.path.join(RESULTS_PATH, 'lopo_results.csv')
iph_path  = os.path.join(RESULTS_PATH, 'iph_results.csv')
curve_path= os.path.join(RESULTS_PATH, 'horizon_auc_curves.csv')

if os.path.exists(lopo_path):
    lopo_df  = pd.read_csv(lopo_path)
    print(f"  Loaded LOPO results: {len(lopo_df)} patients")
if os.path.exists(iph_path):
    iph_df   = pd.read_csv(iph_path)
if os.path.exists(curve_path):
    curve_df = pd.read_csv(curve_path)


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE CSD STATISTICS (Kendall tau before AF onsets)
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing Kendall tau CSD statistics...")
tau_rows = []
for rec, rdata in all_rr.items():
    if rec not in all_windows: continue
    wdata      = all_windows[rec]
    par_af     = rdata['paroxysmal_af']
    if par_af.shape[0] == 0: continue

    for ep_idx, (af_start, af_end) in enumerate(par_af):
        # Use 60-minute lookback before each AF onset
        t_start = af_start - 3600
        mask    = (wdata['times'] >= t_start) & (wdata['times'] < af_start)
        if mask.sum() < 10: continue

        var_s = wdata['X'][mask, 0]
        ac_s  = wdata['X'][mask, 1]
        t_idx = np.arange(mask.sum())

        tv, pv = kendalltau(t_idx, var_s)
        ta, pa = kendalltau(t_idx, ac_s)
        csd    = bool(tv > 0 and pv < 0.05 and ta > 0 and pa < 0.05)

        tau_rows.append({
            'record': rec, 'episode': ep_idx,
            'af_start_min': af_start / 60,
            'tau_var': float(tv), 'p_var': float(pv),
            'tau_ac':  float(ta), 'p_ac':  float(pa),
            'mean_tau': float((tv + ta) / 2),
            'csd_detected': csd,
            'n_windows': int(mask.sum()),
        })

tau_df = pd.DataFrame(tau_rows)
tau_df.to_csv(os.path.join(RESULTS_PATH, 'tau_stats.csv'), index=False)
print(f"  {len(tau_df)} AF episodes analyzed")
if len(tau_df) > 0:
    print(f"  CSD detected: {tau_df['csd_detected'].mean()*100:.1f}%")
    print(f"  Mean τ (var): {tau_df['tau_var'].mean():.4f}")
    print(f"  Mean τ (ac):  {tau_df['tau_ac'].mean():.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

# ── FIG 1: CSD indicators before best AF episode ─────────────────────────────
if len(tau_df) > 0:
    best_row = tau_df.nlargest(1, 'mean_tau').iloc[0]
    brec     = best_row['record']
    bep      = int(best_row['episode'])
    rdata    = all_rr.get(brec)
    wdata    = all_windows.get(brec)

    if rdata is not None and wdata is not None:
        par_af   = rdata['paroxysmal_af']
        af_start = par_af[bep, 0]

        mask    = (wdata['times'] >= af_start - 3600) & \
                  (wdata['times'] < af_start)
        relt    = (wdata['times'][mask] - af_start) / 60  # minutes

        var_s = wdata['X'][mask, 0]
        ac_s  = wdata['X'][mask, 1]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(
            f'CSD Indicators in the Hour Before AF Onset — Record {brec}\n'
            f'(τ_var={best_row["tau_var"]:.3f}, '
            f'τ_ac={best_row["tau_ac"]:.3f})',
            fontsize=13, fontweight='bold'
        )

        for ax, series, name, color in zip(
            axes,
            [var_s, ac_s],
            ['RR Variance (ms²) — CSD Indicator 1',
             'Lag-1 Autocorrelation — CSD Indicator 2'],
            [BLUE, ORG]
        ):
            ax.plot(relt, series, color=color, linewidth=1.8, alpha=0.9)
            ax.fill_between(relt, series, alpha=0.15, color=color)
            # Linear trend line
            z = np.polyfit(relt, series, 1)
            ax.plot(relt, np.polyval(z, relt), '--', color='black',
                    linewidth=1.5, label=f'Trend (slope={z[0]:.2f}/min)')
            ax.axvline(0, color=RED, linewidth=2,
                       linestyle='--', label='AF onset')
            ax.set_ylabel(name, fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[1].set_xlabel('Time relative to AF onset (minutes)', fontsize=11)
        plt.tight_layout()
        savefig(fig, 'fig1_csd_before_af_onset.png')


# ── FIG 2: Tau distribution ───────────────────────────────────────────────────
if len(tau_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Kendall's τ Distribution Across All AF Episodes",
                 fontsize=13, fontweight='bold')

    for ax, col, name, color in zip(
        axes,
        ['tau_var', 'tau_ac'],
        ['τ — RR Variance', 'τ — Lag-1 Autocorrelation'],
        [BLUE, ORG]
    ):
        ax.hist(tau_df[col], bins=30, color=color, edgecolor='white',
                alpha=0.85)
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5,
                   label='τ=0')
        ax.axvline(tau_df[col].mean(), color=RED, linewidth=2,
                   label=f'Mean={tau_df[col].mean():.3f}')
        ax.set_xlabel("Kendall's τ", fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(name, fontsize=11)
        ax.legend(fontsize=9)

    pct = tau_df['csd_detected'].mean() * 100
    fig.text(0.5, -0.02,
             f'CSD detected in {pct:.1f}% of AF episodes '
             f'(both τ > 0, p < 0.05)',
             ha='center', fontsize=11, style='italic')
    plt.tight_layout()
    savefig(fig, 'fig2_tau_distribution.png')


# ── FIG 3: AUC vs Horizon curves ─────────────────────────────────────────────
if len(curve_df) > 0:
    pivot = curve_df.pivot(index='record', columns='horizon_min', values='auc')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title('AUC vs Prediction Horizon (minutes before AF onset)\n'
                 'LOPO-CV — each line = one patient',
                 fontsize=13, fontweight='bold')

    # Individual patient lines
    for rec in pivot.index:
        row   = pivot.loc[rec].dropna()
        if len(row) < 3: continue
        auc_mean = row.mean()
        col  = GREEN if auc_mean >= 0.70 else \
               ORG   if auc_mean >= 0.60 else RED
        ax.plot(row.index, row.values, color=col, alpha=0.35,
                linewidth=1.0)

    # Mean lines
    mean_vals = pivot.mean()
    ax.plot(mean_vals.index, mean_vals.values, color='black',
            linewidth=3.5, label='Population mean', zorder=10)

    good  = pivot[pivot.mean(axis=1) >= 0.70]
    if len(good) > 0:
        ax.plot(good.mean().index, good.mean().values,
                color=GREEN, linewidth=2.5,
                linestyle='--', label=f'AUC≥0.70 subgroup (n={len(good)})',
                zorder=9)

    ax.axhline(0.5,  color='gray',  linestyle=':',  linewidth=1.5,
               label='Chance (0.5)')
    ax.axhline(0.65, color=ORG,    linestyle='--', linewidth=1.2,
               label='IPH threshold (0.65)', alpha=0.7)
    ax.axhline(0.70, color=GREEN,  linestyle='--', linewidth=1.2,
               label='Good (0.70)', alpha=0.7)

    ax.set_xlabel('Prediction horizon (minutes before AF onset)', fontsize=12)
    ax.set_ylabel('AUC-ROC (LOPO-CV)', fontsize=12)
    ax.set_xticks(HORIZON_MINS)
    ax.set_xticklabels([f'{h}min' for h in HORIZON_MINS])
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    savefig(fig, 'fig3_auc_vs_horizon.png')


# ── FIG 4: Per-patient AUC bar chart ─────────────────────────────────────────
if len(lopo_df) > 0:
    lopo_s = lopo_df.sort_values('auc', ascending=False)
    colors = [GREEN if a >= 0.70 else ORG if a >= 0.60 else RED
              for a in lopo_s['auc']]

    fig, ax = plt.subplots(figsize=(max(14, len(lopo_s)*0.35), 5))
    bars = ax.bar(range(len(lopo_s)), lopo_s['auc'],
                  color=colors, edgecolor='white', alpha=0.88)
    ax.errorbar(range(len(lopo_s)), lopo_s['auc'],
                yerr=[lopo_s['auc']-lopo_s['ci_lo'],
                      lopo_s['ci_hi']-lopo_s['auc']],
                fmt='none', color='black', capsize=2, linewidth=0.8)
    ax.axhline(0.50, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(0.70, color=GREEN,   linestyle=':', linewidth=1.5)
    ax.set_xticks(range(len(lopo_s)))
    ax.set_xticklabels(lopo_s['record'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('AUC-ROC (LOPO-CV)', fontsize=12)
    ax.set_title('Per-Patient AF Onset Prediction AUC\n'
                 '(error bars = 95% bootstrap CI)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0.3, 1.05)
    ax.text(0.99, 0.96,
            f'Mean AUC = {lopo_df["auc"].mean():.3f} ± '
            f'{lopo_df["auc"].std():.3f}',
            transform=ax.transAxes, ha='right', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=GREEN, label='AUC ≥ 0.70'),
        Patch(facecolor=ORG,   label='0.60 ≤ AUC < 0.70'),
        Patch(facecolor=RED,   label='AUC < 0.60'),
    ], fontsize=9, loc='lower right')
    plt.tight_layout()
    savefig(fig, 'fig4_per_patient_auc.png')


# ── FIG 5: IPH distribution ───────────────────────────────────────────────────
if len(iph_df) > 0 and 'iph_min' in iph_df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Individual Predictability Horizon — LTAF',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.hist(iph_df['iph_min'], bins=range(0, 75, 5),
            color=BLUE, edgecolor='white', alpha=0.85)
    ax.axvline(iph_df['iph_min'].mean(), color=RED, linewidth=2,
               linestyle='--',
               label=f'Mean = {iph_df["iph_min"].mean():.0f} min')
    ax.axvline(iph_df['iph_min'].median(), color=ORG, linewidth=2,
               linestyle='-.',
               label=f'Median = {iph_df["iph_min"].median():.0f} min')
    ax.set_xlabel('IPH (minutes before AF onset)', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('IPH Distribution', fontsize=11)
    ax.legend(fontsize=10)

    ax2 = axes[1]
    iph_df['category'] = pd.cut(
        iph_df['iph_min'],
        bins=[-1, 0, 15, 30, 60],
        labels=['0 (unpredictable)', '1–15 min', '16–30 min', '31–60 min']
    )
    cat_counts = iph_df['category'].value_counts().sort_index()
    cat_colors = [RED, ORG, BLUE, GREEN]
    ax2.bar(range(len(cat_counts)), cat_counts.values,
            color=cat_colors[:len(cat_counts)], edgecolor='white', alpha=0.85)
    ax2.set_xticks(range(len(cat_counts)))
    ax2.set_xticklabels(cat_counts.index, rotation=20, ha='right')
    ax2.set_ylabel('Number of Patients', fontsize=12)
    ax2.set_title('Patients by Predictability Category', fontsize=11)
    for i, v in enumerate(cat_counts.values):
        ax2.text(i, v + 0.3, str(v), ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(fig, 'fig5_iph_distribution.png')


# ── FIG 6: Sensitivity / Specificity trade-off ───────────────────────────────
if len(lopo_df) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(lopo_df['specificity'], lopo_df['sensitivity'],
                    c=lopo_df['auc'], cmap='RdYlGn',
                    vmin=0.4, vmax=1.0, s=90, edgecolors='white',
                    linewidth=0.8, zorder=4)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('AUC-ROC', fontsize=11)
    for _, row in lopo_df.iterrows():
        ax.annotate(row['record'],
                    (row['specificity'], row['sensitivity']),
                    fontsize=7, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')
    ax.axvline(0.70, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.70, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Specificity (at balanced threshold)', fontsize=12)
    ax.set_ylabel('Sensitivity (at balanced threshold)', fontsize=12)
    ax.set_title('Sensitivity vs Specificity per Patient\n'
                 '(balanced threshold: Spec ≥ 70%)',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, 'fig6_sens_spec_scatter.png')


# ── FIG 7: CSD detection rate ─────────────────────────────────────────────────
if len(tau_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('CSD Detection in LTAF — AF Episodes',
                 fontsize=13, fontweight='bold')

    pct_yes = tau_df['csd_detected'].mean() * 100
    pct_no  = 100 - pct_yes
    axes[0].pie(
        [pct_yes, pct_no],
        labels=[f'CSD Detected\n{pct_yes:.1f}%',
                f'No CSD\n{pct_no:.1f}%'],
        colors=[GREEN, RED], autopct='%1.1f%%',
        startangle=90, textprops={'fontsize': 12}
    )
    axes[0].set_title('CSD Detection Rate\n(all AF episodes)', fontsize=11)

    yes_tau = tau_df[tau_df['csd_detected']]['mean_tau']
    no_tau  = tau_df[~tau_df['csd_detected']]['mean_tau']
    axes[1].hist(yes_tau, bins=20, alpha=0.75, color=GREEN,
                 label=f'CSD detected (n={len(yes_tau)})')
    axes[1].hist(no_tau,  bins=20, alpha=0.75, color=RED,
                 label=f'No CSD (n={len(no_tau)})')
    if len(yes_tau) > 0 and len(no_tau) > 0:
        stat, p = mannwhitneyu(yes_tau, no_tau, alternative='two-sided')
        axes[1].set_title(f'τ by CSD Status (p={p:.4f})', fontsize=11)
    axes[1].set_xlabel("Mean Kendall's τ", fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].legend(fontsize=10)
    plt.tight_layout()
    savefig(fig, 'fig7_csd_detection.png')


# ── FIG 8: Mean RR + variance time series for best patient ───────────────────
if len(lopo_df) > 0 and len(all_windows) > 0:
    best_rec_row = lopo_df.nlargest(1, 'auc')
    if len(best_rec_row) > 0:
        brec  = str(best_rec_row.iloc[0]['record']).zfill(2)
        wdata = all_windows.get(brec)
        rdata = all_rr.get(brec)

        if wdata is not None and rdata is not None:
            par_af = rdata['paroxysmal_af']
            times_hr = wdata['times'] / 3600
            rr_mean  = wdata['X'][:, 5]   # rr_mean feature
            variance = wdata['X'][:, 0]   # variance feature

            fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            fig.suptitle(
                f'Full 24-Hour RR Dynamics — Best Patient (Record {brec}, '
                f'AUC={best_rec_row.iloc[0]["auc"]:.3f})',
                fontsize=13, fontweight='bold'
            )

            pre_mask = wdata['binary'] == 1

            for ax, feat, name, col in zip(
                axes,
                [rr_mean, variance],
                ['Mean RR Interval (ms)', 'RR Variance (ms²)'],
                [BLUE, ORG]
            ):
                ax.plot(times_hr, feat, color=col, linewidth=0.8, alpha=0.8)
                if pre_mask.sum() > 0:
                    ax.scatter(times_hr[pre_mask], feat[pre_mask],
                               color=RED, s=6, alpha=0.7, zorder=5,
                               label='Pre-AF windows')
                for ep_i, (af_s, af_e) in enumerate(par_af):
                    ax.axvspan(af_s/3600, af_e/3600,
                               alpha=0.18, color=RED,
                               label='AF episode' if ep_i == 0 else '_')
                ax.set_ylabel(name, fontsize=11)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)

            axes[1].set_xlabel('Time (hours)', fontsize=11)
            plt.tight_layout()
            savefig(fig, 'fig8_full_24h_dynamics.png')


# ═════════════════════════════════════════════════════════════════════════════
# PAPER-READY RESULTS TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PAPER-READY RESULTS SUMMARY — LTAF")
print("=" * 65)

total_wins = sum(len(w['binary']) for w in all_windows.values())
total_pos  = sum(w['binary'].sum() for w in all_windows.values())
total_neg  = total_wins - total_pos

print(f"\n📊 DATASET")
print(f"  Patients processed    : {len(all_windows)}")
print(f"  Total windows         : {total_wins:,}")
print(f"  Pre-AF windows        : {int(total_pos):,} "
      f"({total_pos/max(total_wins,1)*100:.1f}%)")
print(f"  Normal sinus windows  : {int(total_neg):,} "
      f"({total_neg/max(total_wins,1)*100:.1f}%)")

print(f"\n📊 CSD / KENDALL TAU")
if len(tau_df) > 0:
    print(f"  AF episodes analyzed  : {len(tau_df)}")
    print(f"  CSD detected          : {tau_df['csd_detected'].mean()*100:.1f}%")
    print(f"  Mean τ (variance)     : {tau_df['tau_var'].mean():.4f} "
          f"± {tau_df['tau_var'].std():.4f}")
    print(f"  Mean τ (AC)           : {tau_df['tau_ac'].mean():.4f} "
          f"± {tau_df['tau_ac'].std():.4f}")

print(f"\n📊 LOPO-CV PREDICTION")
if len(lopo_df) > 0:
    print(f"  Patients evaluated    : {len(lopo_df)}")
    print(f"  Mean AUC              : {lopo_df['auc'].mean():.4f} "
          f"± {lopo_df['auc'].std():.4f}")
    print(f"  Median AUC            : {lopo_df['auc'].median():.4f}")
    print(f"  Mean Sensitivity      : {lopo_df['sensitivity'].mean():.1%}")
    print(f"  Mean Specificity      : {lopo_df['specificity'].mean():.1%}")
    print(f"  Mean Precision        : {lopo_df['precision'].mean():.1%}")
    print(f"  Mean F1               : {lopo_df['f1'].mean():.3f}")
    print(f"  Mean False Alarm Rate : {lopo_df['far'].mean():.1%}")
    print(f"  AUC ≥ 0.80            : "
          f"{(lopo_df['auc']>=0.80).sum()}/{len(lopo_df)} patients")
    print(f"  AUC ≥ 0.70            : "
          f"{(lopo_df['auc']>=0.70).sum()}/{len(lopo_df)} patients")
    print(f"  Best patient AUC      : "
          f"{lopo_df['auc'].max():.4f} "
          f"(Record {lopo_df.loc[lopo_df['auc'].idxmax(),'record']})")

    # Comparison with MIT-BIH
    print(f"\n  📈 IMPROVEMENT OVER MIT-BIH:")
    print(f"     MIT-BIH AUC  : 0.5261 (near chance)")
    print(f"     LTAF AUC     : {lopo_df['auc'].mean():.4f}")
    delta = lopo_df['auc'].mean() - 0.5261
    print(f"     Δ AUC        : +{delta:.4f} "
          f"({'significant' if delta > 0.05 else 'modest'})")

print(f"\n📊 IPH RESULTS")
if len(iph_df) > 0 and 'iph_min' in iph_df.columns:
    print(f"  Mean IPH              : {iph_df['iph_min'].mean():.0f} min")
    print(f"  Median IPH            : {iph_df['iph_min'].median():.0f} min")
    print(f"  IPH ≥ 30 min          : "
          f"{(iph_df['iph_min']>=30).sum()}/{len(iph_df)} patients")
    print(f"  IPH ≥ 60 min          : "
          f"{(iph_df['iph_min']>=60).sum()}/{len(iph_df)} patients")

print(f"\n✅ Figures → {FIGURE_PATH}")
print(f"✅ Results → {RESULTS_PATH}")

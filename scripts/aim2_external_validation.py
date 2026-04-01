#!/usr/bin/env python3
"""
Aim 2 (continued): External validation on 5 independent cohorts using branch dropout.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import brier_score_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/bfentaw2/system_biology/hcc_project/scripts')
from aim2_attention_model import (BranchEncoder, MultiHeadOmicsAttention,
                                   MultiOmicsAttentionSurvival, cox_partial_likelihood_loss)

DATA_DIR = "/Users/bfentaw2/system_biology/hcc_project/data"
RESULTS_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/aim2"
FIG_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/figures"

print("="*60)
print("AIM 2: External Validation")
print("="*60, flush=True)

# Load model config and state
with open(f"{RESULTS_DIR}/aim2_results.pkl", 'rb') as f:
    aim2_results = pickle.load(f)

config = aim2_results['model_config']

# Load feature dimensions from processed data
mrna_proc = pd.read_csv(f"{DATA_DIR}/processed/mrna_processed.csv", index_col=0)
mirna_proc = pd.read_csv(f"{DATA_DIR}/processed/mirna_processed.csv", index_col=0)
methyl_proc = pd.read_csv(f"{DATA_DIR}/processed/methyl_processed.csv", index_col=0)

model = MultiOmicsAttentionSurvival(
    mrna_proc.shape[1], mirna_proc.shape[1], methyl_proc.shape[1],
    config['latent_dim'], config['n_heads'], config['dropout'], config['branch_drop']
)
model.load_state_dict(torch.load(f"{RESULTS_DIR}/attention_model.pt", weights_only=True))
model.eval()

# Load external cohorts
with open(f"{DATA_DIR}/external/external_cohorts.pkl", 'rb') as f:
    external_cohorts = pickle.load(f)

# Chaudhary et al. published C-indices for reference
chaudhary_cindices = {
    'LIRI-JP': 0.75, 'NCI': 0.67, 'Chinese': 0.69,
    'E-TABM-36': 0.77, 'Hawaiian': 0.82
}

print("\nExternal validation results:", flush=True)
ext_results = {}

# Map omics types to full feature lists
mrna_varfilt = pd.read_csv(f"{DATA_DIR}/processed/mrna_varfiltered.csv", index_col=0)
mirna_varfilt = pd.read_csv(f"{DATA_DIR}/processed/mirna_varfiltered.csv", index_col=0)
methyl_varfilt = pd.read_csv(f"{DATA_DIR}/processed/methyl_varfiltered.csv", index_col=0)

feature_maps = {
    'mrna': list(mrna_proc.columns),
    'mirna': list(mirna_proc.columns),
    'methylation': list(methyl_proc.columns),
}

for name, cohort in external_cohorts.items():
    otype = cohort['omics_type']
    data = cohort['data']
    clin = cohort['clinical']

    # Align features with model's expected features
    target_features = feature_maps[otype]
    aligned_data = pd.DataFrame(0.0, index=data.index, columns=target_features)
    common_feats = set(data.columns) & set(target_features)
    for feat in common_feats:
        aligned_data[feat] = data[feat].values

    X_ext = torch.FloatTensor(aligned_data.values)
    T_ext = clin['OS_time'].values
    E_ext = clin['OS_event'].values

    with torch.no_grad():
        risk, attn = model.predict_single_omics(X_ext, otype)
        risk_np = risk.numpy()

    # C-index
    try:
        ci = concordance_index(T_ext, -risk_np, E_ext)
    except:
        ci = 0.5

    # Brier score (at median time)
    median_t = np.median(T_ext)
    pred_surv_prob = 1.0 / (1.0 + np.exp(risk_np))  # crude survival probability
    brier = brier_score_loss(E_ext[T_ext <= median_t],
                              1 - pred_surv_prob[T_ext <= median_t]) if (T_ext <= median_t).sum() > 5 else None

    # Log-rank test
    median_risk = np.median(risk_np)
    high = risk_np > median_risk
    try:
        lr = logrank_test(T_ext[high], T_ext[~high], E_ext[high], E_ext[~high])
        lr_p = lr.p_value
    except:
        lr_p = 1.0

    chaud_ci = chaudhary_cindices.get(name, 'N/A')
    ext_results[name] = {
        'c_index': ci, 'brier': brier, 'logrank_p': lr_p,
        'n': len(T_ext), 'events': int(E_ext.sum()),
        'omics': otype, 'chaudhary_cindex': chaud_ci
    }

    print(f"\n  {name} (n={len(T_ext)}, {otype}):")
    print(f"    C-index: {ci:.4f} (Chaudhary: {chaud_ci})")
    print(f"    Log-rank p: {lr_p:.4f}")
    if brier is not None:
        print(f"    Brier score: {brier:.4f}")

# ============================================================
# Comparison table and plot
# ============================================================
print("\n\nExternal Validation Comparison Table:")
print("-"*75)
print(f"{'Cohort':<15} {'n':>5} {'Omics':<12} {'Our C-index':>12} {'Chaudhary':>12} {'p-value':>12}")
print("-"*75)
for name, r in ext_results.items():
    print(f"{name:<15} {r['n']:>5} {r['omics']:<12} {r['c_index']:>12.4f} {r['chaudhary_cindex']:>12} {r['logrank_p']:>12.4f}")
print("-"*75)

# Bar plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
cohort_names = list(ext_results.keys())
our_ci = [ext_results[n]['c_index'] for n in cohort_names]
chaud_ci_list = [chaudhary_cindices[n] for n in cohort_names]

x = np.arange(len(cohort_names))
width = 0.35
bars1 = ax.bar(x - width/2, our_ci, width, label='Attention Model (Ours)',
               color='#FF5722', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, chaud_ci_list, width, label='Chaudhary et al.',
               color='#2196F3', edgecolor='black', linewidth=0.5)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('External Cohort', fontsize=12)
ax.set_ylabel('C-index', fontsize=12)
ax.set_title('External Validation: Attention Model vs Chaudhary et al.', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels([f"{n}\n(n={ext_results[n]['n']}, {ext_results[n]['omics']})"
                     for n in cohort_names], fontsize=9)
ax.legend(fontsize=11)
ax.set_ylim(0.3, 1.0)
for bar, ci in zip(bars1, our_ci):
    ax.text(bar.get_x() + bar.get_width()/2, ci + 0.01, f'{ci:.2f}', ha='center', fontsize=8)
for bar, ci in zip(bars2, chaud_ci_list):
    ax.text(bar.get_x() + bar.get_width()/2, ci + 0.01, f'{ci:.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_external_validation.png", dpi=300, bbox_inches='tight')
plt.close()

# KM curves for external cohorts
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (name, cohort) in enumerate(external_cohorts.items()):
    ax = axes[idx]
    clin = cohort['clinical']
    T_ext = clin['OS_time'].values
    E_ext = clin['OS_event'].values

    otype = cohort['omics_type']
    data = cohort['data']
    target_features = feature_maps[otype]
    aligned_data = pd.DataFrame(0.0, index=data.index, columns=target_features)
    common_feats = set(data.columns) & set(target_features)
    for feat in common_feats:
        aligned_data[feat] = data[feat].values

    X_ext = torch.FloatTensor(aligned_data.values)
    with torch.no_grad():
        risk, _ = model.predict_single_omics(X_ext, otype)

    risk_np = risk.numpy()
    med = np.median(risk_np)
    high = risk_np > med

    kmf_h = KaplanMeierFitter()
    kmf_l = KaplanMeierFitter()
    kmf_h.fit(T_ext[high], E_ext[high], label='High Risk')
    kmf_l.fit(T_ext[~high], E_ext[~high], label='Low Risk')

    kmf_h.plot_survival_function(ax=ax, color='red', linewidth=2)
    kmf_l.plot_survival_function(ax=ax, color='blue', linewidth=2)

    ci = ext_results[name]['c_index']
    p = ext_results[name]['logrank_p']
    ax.set_title(f"{name} (n={len(T_ext)}, {otype})\nC={ci:.3f}, p={p:.3f}", fontsize=10)
    ax.set_xlabel('Days')
    ax.legend(fontsize=8)

axes[-1].axis('off')
plt.suptitle('External Validation KM Curves', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_external_km_curves.png", dpi=300, bbox_inches='tight')
plt.close()

# Save results
with open(f"{RESULTS_DIR}/external_validation_results.pkl", 'wb') as f:
    pickle.dump(ext_results, f)

print("\nExternal validation complete!")
print(f"Figures saved to {FIG_DIR}")

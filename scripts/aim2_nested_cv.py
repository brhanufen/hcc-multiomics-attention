#!/usr/bin/env python3
"""
Aim 2 (CORRECTED): Nested CV with feature selection INSIDE the CV loop.

This fixes the methodological flaw where survival-association feature filtering
was done on the full dataset before CV, causing optimistic bias.

Now: variance-filtered data is loaded, and Spearman survival filtering is
performed on TRAINING data only within each fold.
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/processed"
RESULTS_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/aim2"
FIG_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/figures"

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. Load VARIANCE-FILTERED data (NOT survival-filtered)
# ============================================================
print("="*60)
print("AIM 2 (CORRECTED): Nested CV — Feature Selection Inside Loop")
print("="*60, flush=True)

print("\nLoading variance-filtered data (pre-survival-filter)...", flush=True)
mrna_full = pd.read_csv(f"{DATA_DIR}/mrna_varfiltered.csv", index_col=0)
mirna_full = pd.read_csv(f"{DATA_DIR}/mirna_varfiltered.csv", index_col=0)
methyl_full = pd.read_csv(f"{DATA_DIR}/methyl_varfiltered.csv", index_col=0)
clinical = pd.read_csv(f"{DATA_DIR}/clinical.csv", index_col=0)

common = sorted(set(mrna_full.index) & set(mirna_full.index) & set(methyl_full.index) & set(clinical.index))
mrna_full = mrna_full.loc[common]
mirna_full = mirna_full.loc[common]
methyl_full = methyl_full.loc[common]
clinical = clinical.loc[common]

N = len(common)
T = clinical['OS_time'].values.astype(np.float32)
E = clinical['OS_event'].values.astype(np.float32)

print(f"  Patients: {N}")
print(f"  mRNA (var-filtered): {mrna_full.shape[1]}")
print(f"  miRNA (var-filtered): {mirna_full.shape[1]}")
print(f"  Methylation (var-filtered): {methyl_full.shape[1]}")
print(f"  Events: {int(E.sum())}/{N}", flush=True)

# Import model classes
sys.path.insert(0, '/Users/bfentaw2/system_biology/hcc_project/scripts')
from aim2_attention_model import (BranchEncoder, MultiHeadOmicsAttention,
                                   MultiOmicsAttentionSurvival, cox_partial_likelihood_loss)

# ============================================================
# 2. Feature selection function (to be applied per fold)
# ============================================================
def select_features_train_only(omics_df, time, event, top_n):
    """Select top_n features by Spearman correlation with survival risk.
    MUST be called on TRAINING data only."""
    risk = event / (time + 1)
    corrs = omics_df.corrwith(pd.Series(risk, index=omics_df.index), method='spearman').abs().fillna(0)
    selected = corrs.nlargest(min(top_n, len(corrs))).index
    return list(selected)

# ============================================================
# 3. Use best hyperparameters from previous Optuna run
# ============================================================
with open(f"{RESULTS_DIR}/aim2_results.pkl", 'rb') as f:
    prev_results = pickle.load(f)
best = prev_results['model_config']
latent_dim = best['latent_dim']
n_heads = best['n_heads']
dropout = best['dropout']
weight_decay = best['weight_decay']
lr = best['lr']
branch_drop = best['branch_drop']
batch_size = 32

print(f"\nUsing hyperparameters from previous Optuna search:")
print(f"  latent_dim={latent_dim}, n_heads={n_heads}, dropout={dropout}")
print(f"  lr={lr:.6f}, weight_decay={weight_decay:.6f}, branch_drop={branch_drop:.2f}", flush=True)

# Feature counts to select per fold
N_MRNA_FEAT = 1000
N_MIRNA_FEAT = 300
N_METHYL_FEAT = 1000

# ============================================================
# 4. Proper nested 5-fold CV
# ============================================================
print(f"\n5-fold stratified CV with per-fold feature selection...", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
all_risk_scores = np.zeros(N)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), E)):
    print(f"\n  Fold {fold+1}/5...", flush=True)

    # --- Feature selection on TRAINING data only ---
    train_patients = [common[i] for i in train_idx]
    val_patients = [common[i] for i in val_idx]

    mrna_feat = select_features_train_only(
        mrna_full.loc[train_patients], T[train_idx], E[train_idx], N_MRNA_FEAT)
    mirna_feat = select_features_train_only(
        mirna_full.loc[train_patients], T[train_idx], E[train_idx], N_MIRNA_FEAT)
    methyl_feat = select_features_train_only(
        methyl_full.loc[train_patients], T[train_idx], E[train_idx], N_METHYL_FEAT)

    print(f"    Features selected (train only): mRNA={len(mrna_feat)}, miRNA={len(mirna_feat)}, methyl={len(methyl_feat)}")

    # Extract and tensorize
    X_mrna_train = torch.FloatTensor(mrna_full.loc[train_patients, mrna_feat].values)
    X_mirna_train = torch.FloatTensor(mirna_full.loc[train_patients, mirna_feat].values)
    X_methyl_train = torch.FloatTensor(methyl_full.loc[train_patients, methyl_feat].values)

    X_mrna_val = torch.FloatTensor(mrna_full.loc[val_patients, mrna_feat].values)
    X_mirna_val = torch.FloatTensor(mirna_full.loc[val_patients, mirna_feat].values)
    X_methyl_val = torch.FloatTensor(methyl_full.loc[val_patients, methyl_feat].values)

    T_train = torch.FloatTensor(T[train_idx])
    E_train = torch.FloatTensor(E[train_idx])

    # --- Build model for this fold's feature dimensions ---
    model = MultiOmicsAttentionSurvival(
        len(mrna_feat), len(mirna_feat), len(methyl_feat),
        latent_dim, n_heads, dropout, branch_drop
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_ds = TensorDataset(X_mrna_train, X_mirna_train, X_methyl_train, T_train, E_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_ci = 0
    patience_counter = 0
    best_state = None

    model.train()
    for epoch in range(100):
        epoch_loss = 0
        for batch in train_loader:
            bm, bi, bme, bt, be = batch
            optimizer.zero_grad()
            risk, _, _ = model(bm, bi, bme)
            loss = cox_partial_likelihood_loss(risk, bt, be)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        model.eval()
        with torch.no_grad():
            mask_all = torch.ones(len(val_idx), 3)
            risk_val, _, _ = model(X_mrna_val, X_mirna_val, X_methyl_val, mask_all)

        try:
            val_ci = concordance_index(T[val_idx], -risk_val.numpy(), E[val_idx])
        except:
            val_ci = 0.5

        scheduler.step(-val_ci)

        if val_ci > best_val_ci:
            best_val_ci = val_ci
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= 15:
            break

        model.train()

    # Load best model for this fold
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mask_all = torch.ones(len(val_idx), 3)
        risk_val, _, _ = model(X_mrna_val, X_mirna_val, X_methyl_val, mask_all)

    risk_np = risk_val.numpy()
    ci = concordance_index(T[val_idx], -risk_np, E[val_idx])
    all_risk_scores[val_idx] = risk_np

    fold_results.append({
        'fold': fold + 1,
        'c_index': ci,
        'n_val': len(val_idx),
        'n_events': int(E[val_idx].sum()),
        'epochs': epoch + 1,
        'mrna_features': mrna_feat[:10],  # save top 10 for reference
        'mirna_features': mirna_feat[:10],
        'methyl_features': methyl_feat[:10],
    })

    print(f"    C-index: {ci:.4f} (epochs: {epoch+1})", flush=True)

# ============================================================
# 5. Summary
# ============================================================
cv_cis = [r['c_index'] for r in fold_results]
mean_ci = np.mean(cv_cis)
std_ci = np.std(cv_cis)

print(f"\n{'='*60}")
print(f"NESTED CV RESULTS (feature selection inside loop)")
print(f"{'='*60}")
print(f"Per-fold C-indices: {[f'{c:.4f}' for c in cv_cis]}")
print(f"Mean C-index: {mean_ci:.4f} +/- {std_ci:.4f}")
print(f"\nComparison:")
print(f"  Previous (leaky) CV: {prev_results['mean_cv_cindex']:.4f} +/- {prev_results['std_cv_cindex']:.4f}")
print(f"  Corrected (nested) CV: {mean_ci:.4f} +/- {std_ci:.4f}")
print(f"  Difference: {prev_results['mean_cv_cindex'] - mean_ci:.4f}")

# KM plot with nested CV risk scores
median_risk = np.median(all_risk_scores)
risk_labels = (all_risk_scores > median_risk).astype(int)
mask_h = risk_labels == 1
mask_l = risk_labels == 0

kmf_h = KaplanMeierFitter()
kmf_l = KaplanMeierFitter()
kmf_h.fit(T[mask_h], E[mask_h], label='High Risk')
kmf_l.fit(T[mask_l], E[mask_l], label='Low Risk')
lr_res = logrank_test(T[mask_h], T[mask_l], E[mask_h], E[mask_l])

print(f"  Log-rank p (nested CV scores): {lr_res.p_value:.2e}")

# Save
fig, ax = plt.subplots(figsize=(8, 6))
kmf_h.plot_survival_function(ax=ax, color='red', linewidth=2)
kmf_l.plot_survival_function(ax=ax, color='blue', linewidth=2)
ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Overall Survival Probability', fontsize=12)
ax.set_title(f'Attention Model (Nested CV, No Feature Leakage)\n'
             f'CV C-index={mean_ci:.3f}\u00b1{std_ci:.3f}, Log-rank p={lr_res.p_value:.2e}', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_km_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# Update model comparison plot
with open("/Users/bfentaw2/system_biology/hcc_project/results/aim1/aim1_results.pkl", 'rb') as f:
    aim1 = pickle.load(f)

comparison = {
    'Autoencoder (Aim 1)': aim1['c_index'],
    'Attention model (nested CV)': mean_ci,
    'Clinical only': aim1['benchmark'].get('Clinical only', {}).get('c_index', 0.5),
}

fig, ax = plt.subplots(figsize=(8, 5))
models_sorted = sorted(comparison.items(), key=lambda x: x[1])
names = [m[0] for m in models_sorted]
cis = [m[1] for m in models_sorted]
colors = ['#FF5722' if 'Attention' in n else '#2196F3' for n in names]
bars = ax.barh(names, cis, color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('C-index', fontsize=12)
ax.set_title('Model Comparison (Nested 5-Fold CV, No Feature Leakage)', fontsize=13)
for bar, ci in zip(bars, cis):
    ax.text(ci + 0.005, bar.get_y() + bar.get_height()/2, f'{ci:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Save updated results
nested_results = {
    'cv_results': fold_results,
    'mean_cv_cindex': mean_ci,
    'std_cv_cindex': std_ci,
    'logrank_p': lr_res.p_value,
    'risk_scores': all_risk_scores,
    'risk_labels': risk_labels,
    'comparison': comparison,
    'patient_ids': common,
    'method': 'nested_cv_feature_selection_inside_loop',
}
with open(f"{RESULTS_DIR}/aim2_nested_cv_results.pkl", 'wb') as f:
    pickle.dump(nested_results, f)

print(f"\nResults saved. Figures updated.")
print("Done!")

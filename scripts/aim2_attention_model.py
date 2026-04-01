#!/usr/bin/env python3
"""
Aim 2: Interpretable attention-based multi-branch deep learning model
for multi-omics survival prediction in HCC.

Architecture:
- 3 branch encoders (mRNA, miRNA, methylation): FC -> BN -> ReLU -> FC -> BN -> ReLU
- Multi-head attention fusion across branches
- Cox partial likelihood survival layer
- Branch dropout for handling missing omics at inference

Training:
- Optuna hyperparameter optimization (100 trials)
- 5-fold stratified cross-validation
- SHAP + integrated gradients for interpretability
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
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/processed"
RESULTS_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/aim2"
FIG_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. Load data
# ============================================================
print("="*60)
print("AIM 2: Attention-Based Multi-Branch Survival Model")
print("="*60, flush=True)

print("\nLoading data...", flush=True)
mrna = pd.read_csv(f"{DATA_DIR}/mrna_processed.csv", index_col=0)
mirna = pd.read_csv(f"{DATA_DIR}/mirna_processed.csv", index_col=0)
methyl = pd.read_csv(f"{DATA_DIR}/methyl_processed.csv", index_col=0)
clinical = pd.read_csv(f"{DATA_DIR}/clinical.csv", index_col=0)

common = sorted(set(mrna.index) & set(mirna.index) & set(methyl.index) & set(clinical.index))
mrna = mrna.loc[common]
mirna = mirna.loc[common]
methyl = methyl.loc[common]
clinical = clinical.loc[common]

N = len(common)
print(f"  Patients: {N}")
print(f"  mRNA: {mrna.shape[1]}, miRNA: {mirna.shape[1]}, Methylation: {methyl.shape[1]}")
print(f"  Events: {int(clinical['OS_event'].sum())}/{N}", flush=True)

X_mrna = torch.FloatTensor(mrna.values)
X_mirna = torch.FloatTensor(mirna.values)
X_methyl = torch.FloatTensor(methyl.values)
T = clinical['OS_time'].values.astype(np.float32)
E = clinical['OS_event'].values.astype(np.float32)

# ============================================================
# 2. Model Architecture
# ============================================================

class BranchEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate=0.4):
        super().__init__()
        hidden = max(input_dim // 4, latent_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadOmicsAttention(nn.Module):
    def __init__(self, latent_dim, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = latent_dim // n_heads
        assert latent_dim % n_heads == 0

        self.W_q = nn.Linear(latent_dim, latent_dim)
        self.W_k = nn.Linear(latent_dim, latent_dim)
        self.W_v = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, branch_outputs, branch_mask=None):
        """
        branch_outputs: (batch, n_branches, latent_dim)
        branch_mask: (batch, n_branches) - 1 for active, 0 for dropped
        Returns: fused (batch, latent_dim), attention_weights (batch, n_heads, n_branches, n_branches)
        """
        B, S, D = branch_outputs.shape

        Q = self.W_q(branch_outputs).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(branch_outputs).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(branch_outputs).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if branch_mask is not None:
            mask = branch_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)

        # Pool across branches (mean of active branches)
        if branch_mask is not None:
            mask_expand = branch_mask.unsqueeze(-1)
            out = (out * mask_expand).sum(dim=1) / (mask_expand.sum(dim=1) + 1e-8)
        else:
            out = out.mean(dim=1)

        return out, attn


class MultiOmicsAttentionSurvival(nn.Module):
    def __init__(self, mrna_dim, mirna_dim, methyl_dim,
                 latent_dim=64, n_heads=2, dropout_rate=0.4, branch_drop_prob=0.2):
        super().__init__()
        self.branch_drop_prob = branch_drop_prob
        self.latent_dim = latent_dim

        self.mrna_encoder = BranchEncoder(mrna_dim, latent_dim, dropout_rate)
        self.mirna_encoder = BranchEncoder(mirna_dim, latent_dim, dropout_rate)
        self.methyl_encoder = BranchEncoder(methyl_dim, latent_dim, dropout_rate)

        self.attention = MultiHeadOmicsAttention(latent_dim, n_heads)

        self.risk_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_mrna, x_mirna, x_methyl, branch_mask=None):
        z_mrna = self.mrna_encoder(x_mrna)
        z_mirna = self.mirna_encoder(x_mirna)
        z_methyl = self.methyl_encoder(x_methyl)

        branches = torch.stack([z_mrna, z_mirna, z_methyl], dim=1)  # (B, 3, D)

        # Branch dropout during training
        if self.training and branch_mask is None:
            mask = (torch.rand(x_mrna.size(0), 3) > self.branch_drop_prob).float()
            # Ensure at least one branch active
            all_zero = mask.sum(dim=1) == 0
            if all_zero.any():
                idx = torch.randint(0, 3, (all_zero.sum(),))
                mask[all_zero, idx] = 1.0
            branch_mask = mask.to(x_mrna.device)

        fused, attn_weights = self.attention(branches, branch_mask)
        risk = self.risk_head(fused)

        return risk.squeeze(-1), attn_weights, branch_mask

    def predict_single_omics(self, x, omics_type):
        """Predict using a single omics branch (for external validation)."""
        B = x.size(0)
        device = x.device
        dummy = torch.zeros(B, self.latent_dim).to(device)

        if omics_type == 'mrna':
            z = self.mrna_encoder(x)
            branches = torch.stack([z, dummy, dummy], dim=1)
            mask = torch.tensor([[1, 0, 0]]).float().expand(B, -1).to(device)
        elif omics_type == 'mirna':
            z = self.mirna_encoder(x)
            branches = torch.stack([dummy, z, dummy], dim=1)
            mask = torch.tensor([[0, 1, 0]]).float().expand(B, -1).to(device)
        else:
            z = self.methyl_encoder(x)
            branches = torch.stack([dummy, dummy, z], dim=1)
            mask = torch.tensor([[0, 0, 1]]).float().expand(B, -1).to(device)

        fused, attn = self.attention(branches, mask)
        risk = self.risk_head(fused)
        return risk.squeeze(-1), attn


def cox_partial_likelihood_loss(risk_scores, times, events):
    """Negative Cox partial log-likelihood."""
    sorted_idx = torch.argsort(times, descending=True)
    risk_sorted = risk_scores[sorted_idx]
    events_sorted = events[sorted_idx]

    log_risk = risk_sorted
    cumsum_risk = torch.logcumsumexp(log_risk, dim=0)

    loss = -torch.mean((log_risk - cumsum_risk) * events_sorted)
    return loss


# ============================================================
# 3. Optuna Hyperparameter Optimization
# ============================================================
print("\nHyperparameter optimization with Optuna (100 trials)...", flush=True)

def objective(trial):
    latent_dim = trial.suggest_categorical('latent_dim', [32, 64, 128])
    n_heads = trial.suggest_categorical('n_heads', [1, 2, 4])
    if latent_dim % n_heads != 0:
        n_heads = 1
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_categorical('dropout', [0.3, 0.4, 0.5])
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    branch_drop = trial.suggest_float('branch_drop', 0.1, 0.3)

    # 3-fold CV for speed during optimization
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_cindices = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), E)):
        model = MultiOmicsAttentionSurvival(
            mrna.shape[1], mirna.shape[1], methyl.shape[1],
            latent_dim, n_heads, dropout, branch_drop
        )
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_dataset = TensorDataset(
            X_mrna[train_idx], X_mirna[train_idx], X_methyl[train_idx],
            torch.FloatTensor(T[train_idx]), torch.FloatTensor(E[train_idx])
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(30):
            for batch in train_loader:
                bm, bi, bme, bt, be = batch
                optimizer.zero_grad()
                risk, _, _ = model(bm, bi, bme)
                loss = cox_partial_likelihood_loss(risk, bt, be)
                if torch.isnan(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            mask_all = torch.ones(len(val_idx), 3)
            risk_val, _, _ = model(X_mrna[val_idx], X_mirna[val_idx], X_methyl[val_idx], mask_all)
            risk_np = risk_val.numpy()

        try:
            ci = concordance_index(T[val_idx], -risk_np, E[val_idx])
            cv_cindices.append(ci)
        except:
            cv_cindices.append(0.5)

    return np.mean(cv_cindices)


study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=False)

best = study.best_params
print(f"\nBest hyperparameters (C-index={study.best_value:.4f}):")
for k, v in best.items():
    print(f"  {k}: {v}")

# ============================================================
# 4. Final 5-fold CV with best hyperparameters
# ============================================================
print("\n5-fold stratified cross-validation with best params...", flush=True)

latent_dim = best['latent_dim']
n_heads = best['n_heads']
if latent_dim % n_heads != 0:
    n_heads = 1
lr = best['lr']
dropout = best['dropout']
weight_decay = best['weight_decay']
batch_size = best['batch_size']
branch_drop = best['branch_drop']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
all_risk_scores = np.zeros(N)
all_attn_weights = []
best_model = None
best_ci = 0

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(N), E)):
    print(f"\n  Fold {fold+1}/5...", flush=True)

    model = MultiOmicsAttentionSurvival(
        mrna.shape[1], mirna.shape[1], methyl.shape[1],
        latent_dim, n_heads, dropout, branch_drop
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_ds = TensorDataset(
        X_mrna[train_idx], X_mirna[train_idx], X_methyl[train_idx],
        torch.FloatTensor(T[train_idx]), torch.FloatTensor(E[train_idx])
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_ci = 0
    patience_counter = 0
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
            risk_val, attn_val, _ = model(X_mrna[val_idx], X_mirna[val_idx], X_methyl[val_idx], mask_all)

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
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mask_all = torch.ones(len(val_idx), 3)
        risk_val, attn_val, _ = model(X_mrna[val_idx], X_mirna[val_idx], X_methyl[val_idx], mask_all)

    risk_np = risk_val.numpy()
    ci = concordance_index(T[val_idx], -risk_np, E[val_idx])

    all_risk_scores[val_idx] = risk_np
    all_attn_weights.append(attn_val.numpy())

    fold_results.append({
        'fold': fold + 1,
        'c_index': ci,
        'val_idx': val_idx,
        'n_val': len(val_idx),
        'n_events': int(E[val_idx].sum()),
    })

    if ci > best_ci:
        best_ci = ci
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"    C-index: {ci:.4f} (epochs: {epoch+1})", flush=True)

mean_ci = np.mean([r['c_index'] for r in fold_results])
std_ci = np.std([r['c_index'] for r in fold_results])
print(f"\n  Mean C-index: {mean_ci:.4f} +/- {std_ci:.4f}", flush=True)

# ============================================================
# 5. Train final model on full data
# ============================================================
print("\nTraining final model on full dataset...", flush=True)

final_model = MultiOmicsAttentionSurvival(
    mrna.shape[1], mirna.shape[1], methyl.shape[1],
    latent_dim, n_heads, dropout, branch_drop
)
optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)

full_ds = TensorDataset(X_mrna, X_mirna, X_methyl, torch.FloatTensor(T), torch.FloatTensor(E))
full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)

final_model.train()
for epoch in range(80):
    for batch in full_loader:
        bm, bi, bme, bt, be = batch
        optimizer.zero_grad()
        risk, _, _ = final_model(bm, bi, bme)
        loss = cox_partial_likelihood_loss(risk, bt, be)
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        optimizer.step()

# Get final predictions and attention weights
final_model.eval()
with torch.no_grad():
    mask_all = torch.ones(N, 3)
    final_risk, final_attn, _ = final_model(X_mrna, X_mirna, X_methyl, mask_all)

final_risk_np = final_risk.numpy()
final_attn_np = final_attn.numpy()

# Full data C-index
full_ci = concordance_index(T, -final_risk_np, E)
print(f"  Full-data C-index: {full_ci:.4f}", flush=True)

# ============================================================
# 6. Risk stratification with attention model
# ============================================================
print("\nRisk stratification...", flush=True)

median_risk = np.median(final_risk_np)
risk_labels = (final_risk_np > median_risk).astype(int)

kmf_h = KaplanMeierFitter()
kmf_l = KaplanMeierFitter()
mask_h = risk_labels == 1
mask_l = risk_labels == 0

kmf_h.fit(T[mask_h], E[mask_h], label='High Risk')
kmf_l.fit(T[mask_l], E[mask_l], label='Low Risk')
lr_res = logrank_test(T[mask_h], T[mask_l], E[mask_h], E[mask_l])
lr_p = lr_res.p_value

print(f"  High-risk: {mask_h.sum()}, Low-risk: {mask_l.sum()}")
print(f"  Log-rank p: {lr_p:.2e}")
print(f"  Full-data C-index: {full_ci:.4f}", flush=True)

# KM plot
fig, ax = plt.subplots(figsize=(8, 6))
kmf_h.plot_survival_function(ax=ax, color='red', linewidth=2)
kmf_l.plot_survival_function(ax=ax, color='blue', linewidth=2)
ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Overall Survival Probability', fontsize=12)
ax.set_title(f'Aim 2: Attention-Based Model Risk Stratification\n'
             f'CV C-index={mean_ci:.3f}±{std_ci:.3f}, Log-rank p={lr_p:.2e}', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_km_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 7. Attention weight analysis
# ============================================================
print("\nAnalyzing attention weights...", flush=True)

# Extract omics-level attention (mean over heads and query/key dimensions)
# final_attn shape: (N, n_heads, 3, 3)
branch_importance = final_attn_np.mean(axis=(1, 2))  # (N, 3)
branch_names = ['mRNA', 'miRNA', 'Methylation']

mean_importance = branch_importance.mean(axis=0)
std_importance = branch_importance.std(axis=0)
print(f"  Branch importance (mean ± std):")
for i, name in enumerate(branch_names):
    print(f"    {name}: {mean_importance[i]:.4f} ± {std_importance[i]:.4f}")

# Attention heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Population-level attention
ax1.bar(branch_names, mean_importance, yerr=std_importance,
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', capsize=5)
ax1.set_ylabel('Mean Attention Weight', fontsize=12)
ax1.set_title('Omics Branch Importance (Population)', fontsize=13)

# Per-patient attention heatmap (top 50 high-risk vs 50 low-risk)
high_idx = np.where(risk_labels == 1)[0][:50]
low_idx = np.where(risk_labels == 0)[0][:50]
display_idx = np.concatenate([high_idx, low_idx])
attn_display = branch_importance[display_idx]
group_labels_display = ['High Risk']*len(high_idx) + ['Low Risk']*len(low_idx)

sns.heatmap(attn_display.T, ax=ax2, cmap='YlOrRd', xticklabels=False,
            yticklabels=branch_names)
ax2.set_title('Per-Patient Attention Weights', fontsize=13)
ax2.axvline(x=len(high_idx), color='black', linewidth=2, linestyle='--')
ax2.text(len(high_idx)//2, -0.3, 'High Risk', ha='center', fontsize=10)
ax2.text(len(high_idx)+len(low_idx)//2, -0.3, 'Low Risk', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_attention_weights.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 8. Feature-level importance (Integrated Gradients)
# ============================================================
print("\nComputing feature-level importance (Integrated Gradients)...", flush=True)

def integrated_gradients(model, x_mrna, x_mirna, x_methyl, n_steps=50):
    """Compute integrated gradients for each input feature."""
    baseline_mrna = torch.zeros_like(x_mrna)
    baseline_mirna = torch.zeros_like(x_mirna)
    baseline_methyl = torch.zeros_like(x_methyl)

    ig_mrna = torch.zeros_like(x_mrna)
    ig_mirna = torch.zeros_like(x_mirna)
    ig_methyl = torch.zeros_like(x_methyl)

    for step in range(n_steps + 1):
        alpha = step / n_steps
        interp_mrna = baseline_mrna + alpha * (x_mrna - baseline_mrna)
        interp_mirna = baseline_mirna + alpha * (x_mirna - baseline_mirna)
        interp_methyl = baseline_methyl + alpha * (x_methyl - baseline_methyl)

        interp_mrna.requires_grad_(True)
        interp_mirna.requires_grad_(True)
        interp_methyl.requires_grad_(True)

        mask = torch.ones(x_mrna.size(0), 3)
        risk, _, _ = model(interp_mrna, interp_mirna, interp_methyl, mask)
        risk_sum = risk.sum()
        risk_sum.backward()

        ig_mrna += interp_mrna.grad / (n_steps + 1)
        ig_mirna += interp_mirna.grad / (n_steps + 1)
        ig_methyl += interp_methyl.grad / (n_steps + 1)

    ig_mrna = (x_mrna - baseline_mrna) * ig_mrna
    ig_mirna = (x_mirna - baseline_mirna) * ig_mirna
    ig_methyl = (x_methyl - baseline_methyl) * ig_methyl

    return ig_mrna.detach(), ig_mirna.detach(), ig_methyl.detach()

final_model.eval()
ig_mrna, ig_mirna, ig_methyl = integrated_gradients(
    final_model, X_mrna, X_mirna, X_methyl, n_steps=30
)

# Average absolute importance across patients
mrna_importance = ig_mrna.abs().mean(dim=0).numpy()
mirna_importance = ig_mirna.abs().mean(dim=0).numpy()
methyl_importance = ig_methyl.abs().mean(dim=0).numpy()

# Top features per omics
top_mrna_idx = np.argsort(mrna_importance)[-100:][::-1]
top_mirna_idx = np.argsort(mirna_importance)[-50:][::-1]
top_methyl_idx = np.argsort(methyl_importance)[-100:][::-1]

top_mrna_genes = [(mrna.columns[i], mrna_importance[i]) for i in top_mrna_idx]
top_mirna_features = [(mirna.columns[i], mirna_importance[i]) for i in top_mirna_idx]
top_methyl_cpgs = [(methyl.columns[i], methyl_importance[i]) for i in top_methyl_idx]

print(f"\n  Top 10 mRNA features:")
for gene, imp in top_mrna_genes[:10]:
    print(f"    {gene}: {imp:.6f}")

print(f"\n  Top 10 miRNA features:")
for mir, imp in top_mirna_features[:10]:
    print(f"    {mir}: {imp:.6f}")

print(f"\n  Top 10 methylation features:")
for cpg, imp in top_methyl_cpgs[:10]:
    print(f"    {cpg}: {imp:.6f}")

# Feature importance plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (ax, name, feats) in enumerate(zip(axes,
    ['mRNA (Top 20)', 'miRNA (Top 20)', 'Methylation (Top 20)'],
    [top_mrna_genes[:20], top_mirna_features[:20], top_methyl_cpgs[:20]])):
    names_list = [f[0][:20] for f in feats]
    vals = [f[1] for f in feats]
    colors_feat = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax.barh(range(len(names_list)), vals, color=colors_feat[idx])
    ax.set_yticks(range(len(names_list)))
    ax.set_yticklabels(names_list, fontsize=8)
    ax.set_xlabel('Mean |IG|', fontsize=11)
    ax.set_title(name, fontsize=12)
    ax.invert_yaxis()

plt.suptitle('Feature Importance (Integrated Gradients)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 9. Comparison with Aim 1 and AUTOSurv-like baseline
# ============================================================
print("\nModel comparison...", flush=True)

# Load Aim 1 results
with open("/Users/bfentaw2/system_biology/hcc_project/results/aim1/aim1_results.pkl", 'rb') as f:
    aim1 = pickle.load(f)

# Simple AUTOSurv-like baseline (concatenated + attention-like weighting)
# Simplified: PCA per omics + weighted combination + Cox
from sklearn.decomposition import PCA

def autosurv_baseline():
    """Simplified AUTOSurv: PCA per omics + learned weights + Cox."""
    pca_mrna = PCA(n_components=20).fit_transform(mrna.values)
    pca_mirna = PCA(n_components=10).fit_transform(mirna.values)
    pca_methyl = PCA(n_components=20).fit_transform(methyl.values)

    combined = np.hstack([pca_mrna, pca_mirna, pca_methyl])
    skf_auto = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cis = []
    for tr, va in skf_auto.split(combined, E):
        df_tr = pd.DataFrame(combined[tr], columns=[f'f{i}' for i in range(50)])
        df_tr['T'] = T[tr]; df_tr['E'] = E[tr]
        df_va = pd.DataFrame(combined[va], columns=[f'f{i}' for i in range(50)])
        df_va['T'] = T[va]; df_va['E'] = E[va]
        try:
            cph = CoxPHFitter(penalizer=0.5)
            cph.fit(df_tr, duration_col='T', event_col='E')
            pred = cph.predict_partial_hazard(df_va)
            ci = concordance_index(T[va], -pred.values.flatten(), E[va])
            cis.append(ci)
        except:
            cis.append(0.5)
    return np.mean(cis)

autosurv_ci = autosurv_baseline()
print(f"  AUTOSurv-like baseline CV C-index: {autosurv_ci:.4f}")

comparison = {
    'Autoencoder (Aim 1)': aim1['c_index'],
    'Attention model (Aim 2)': mean_ci,
    'AUTOSurv-like baseline': autosurv_ci,
    'Clinical only': aim1['benchmark'].get('Clinical only', {}).get('c_index', 0.5),
}

print(f"\n  Model Comparison:")
for model_name, ci in comparison.items():
    print(f"    {model_name}: C-index={ci:.4f}")

# Comparison bar plot
fig, ax = plt.subplots(figsize=(8, 5))
models_sorted = sorted(comparison.items(), key=lambda x: x[1])
names_list = [m[0] for m in models_sorted]
cis_list = [m[1] for m in models_sorted]
colors = ['#FF5722' if 'Attention' in n else '#2196F3' for n in names_list]
bars = ax.barh(names_list, cis_list, color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('C-index', fontsize=12)
ax.set_title('Model Comparison (5-Fold CV)', fontsize=13)
for bar, ci in zip(bars, cis_list):
    ax.text(ci + 0.005, bar.get_y() + bar.get_height()/2, f'{ci:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 10. Save all results
# ============================================================
print("\nSaving results...", flush=True)

torch.save(final_model.state_dict(), f"{RESULTS_DIR}/attention_model.pt")

results = {
    'cv_results': fold_results,
    'mean_cv_cindex': mean_ci,
    'std_cv_cindex': std_ci,
    'full_cindex': full_ci,
    'logrank_p': lr_p,
    'best_params': best,
    'risk_scores': final_risk_np,
    'risk_labels': risk_labels,
    'branch_importance': branch_importance,
    'mrna_importance': mrna_importance,
    'mirna_importance': mirna_importance,
    'methyl_importance': methyl_importance,
    'top_mrna_genes': top_mrna_genes,
    'top_mirna_features': top_mirna_features,
    'top_methyl_cpgs': top_methyl_cpgs,
    'comparison': comparison,
    'patient_ids': common,
    'model_config': {
        'latent_dim': latent_dim, 'n_heads': n_heads,
        'dropout': dropout, 'branch_drop': branch_drop,
        'lr': lr, 'weight_decay': weight_decay
    }
}
with open(f"{RESULTS_DIR}/aim2_results.pkl", 'wb') as f:
    pickle.dump(results, f)

print("\n" + "="*60)
print("AIM 2 RESULTS SUMMARY")
print("="*60)
print(f"5-Fold CV C-index: {mean_ci:.4f} ± {std_ci:.4f}")
print(f"Full-data C-index: {full_ci:.4f}")
print(f"Log-rank p-value: {lr_p:.2e}")
print(f"Best hyperparameters: latent_dim={latent_dim}, n_heads={n_heads}, "
      f"dropout={dropout}, branch_drop={branch_drop:.2f}")
print(f"\nBranch importance: mRNA={mean_importance[0]:.3f}, "
      f"miRNA={mean_importance[1]:.3f}, Methyl={mean_importance[2]:.3f}")
print(f"\nComparison with baselines:")
for m, ci in comparison.items():
    marker = " <-- PROPOSED" if 'Attention' in m else ""
    print(f"  {m}: {ci:.4f}{marker}")
print("\nAim 2 complete!")

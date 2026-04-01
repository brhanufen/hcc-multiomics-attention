#!/usr/bin/env python3
"""
Aim 1: Reproduce and benchmark the Chaudhary et al. deep autoencoder survival model.

Architecture (from Chaudhary et al. 2018):
- Input: concatenated mRNA + miRNA + methylation features
- Encoder: Input -> 500 (tanh) -> 100 (tanh, bottleneck)
- Decoder: 100 -> 500 (tanh) -> Input_dim
- Regularization: L1/L2, 50% dropout
- Training: 10 epochs, SGD
- Post-training: extract bottleneck features, Cox PH, clustering, KM curves
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/processed"
RESULTS_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/aim1"
FIG_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)

# ============================================================
# 1. Load preprocessed data
# ============================================================
print("=" * 60)
print("AIM 1: Reproduce Chaudhary et al. Autoencoder Survival Model")
print("=" * 60)

print("\nLoading data...")
concat = pd.read_csv(f"{DATA_DIR}/concat_omics_aim1.csv", index_col=0)
clinical = pd.read_csv(f"{DATA_DIR}/clinical.csv", index_col=0)

# Ensure same patients
common = sorted(set(concat.index) & set(clinical.index))
concat = concat.loc[common]
clinical = clinical.loc[common]

N, D = concat.shape
print(f"  Patients: {N}, Features: {D}")
print(f"  Events: {int(clinical['OS_event'].sum())}/{N}")

X = concat.values.astype(np.float32)
os_time = clinical['OS_time'].values.astype(np.float32)
os_event = clinical['OS_event'].values.astype(np.float32)

# ============================================================
# 2. Deep Autoencoder (Chaudhary et al. architecture)
# ============================================================
print("\nBuilding autoencoder...")

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden1=500, bottleneck=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden1, bottleneck),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden1),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def get_bottleneck(self, x):
        with torch.no_grad():
            return self.encoder(x)


model = DeepAutoencoder(D, hidden1=500, bottleneck=100)
print(f"  Architecture: {D} -> 500 -> 100 -> 500 -> {D}")

# L1/L2 regularization via optimizer weight_decay (L2) + manual L1
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Training
print("\nTraining autoencoder (10 epochs)...")
X_tensor = torch.FloatTensor(X)
dataset = TensorDataset(X_tensor, X_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

train_losses = []
model.train()
for epoch in range(10):
    epoch_loss = 0
    for batch_x, _ in loader:
        optimizer.zero_grad()
        recon, z = model(batch_x)
        loss = criterion(recon, batch_x)

        # L1 regularization
        l1_lambda = 1e-5
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = loss + l1_lambda * l1_norm

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    epoch_loss /= N
    train_losses.append(epoch_loss)
    print(f"  Epoch {epoch+1}/10, Loss: {epoch_loss:.6f}")

# ============================================================
# 3. Extract bottleneck features
# ============================================================
print("\nExtracting bottleneck features...")
model.eval()
Z = model.get_bottleneck(X_tensor).numpy()
print(f"  Bottleneck features shape: {Z.shape}")

# ============================================================
# 4. Identify survival-associated features via Cox PH
# ============================================================
print("\nIdentifying survival-associated latent features...")

cox_pvals = []
cox_coefs = []
for i in range(Z.shape[1]):
    try:
        df_cox = pd.DataFrame({
            'T': os_time,
            'E': os_event,
            'z': Z[:, i]
        })
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df_cox, duration_col='T', event_col='E', formula='z')
        cox_pvals.append(cph.summary['p'].values[0])
        cox_coefs.append(cph.summary['coef'].values[0])
    except:
        cox_pvals.append(1.0)
        cox_coefs.append(0.0)

cox_pvals = np.array(cox_pvals)
cox_coefs = np.array(cox_coefs)
sig_features = np.where(cox_pvals < 0.05)[0]
print(f"  Significant features (p<0.05): {len(sig_features)}/{Z.shape[1]}")

# Use significant features for clustering (or all if too few)
if len(sig_features) >= 5:
    Z_surv = Z[:, sig_features]
else:
    # Use top 20 features by p-value
    top20 = np.argsort(cox_pvals)[:20]
    Z_surv = Z[:, top20]
    print(f"  Using top 20 features by p-value instead")

# ============================================================
# 5. Cluster patients into risk subgroups
# ============================================================
print("\nClustering patients into risk subgroups...")

km = KMeans(n_clusters=2, random_state=42, n_init=50)
clusters = km.fit_predict(Z_surv)

# Determine which cluster is high-risk (lower median survival)
median_surv = [np.median(os_time[clusters == c]) for c in [0, 1]]
if median_surv[0] > median_surv[1]:
    risk_labels = clusters  # cluster 1 is high risk
    high_risk_cluster = 1
else:
    risk_labels = 1 - clusters  # flip
    high_risk_cluster = 0

risk_group = np.where(risk_labels == 1, 'High Risk', 'Low Risk')
n_high = (risk_labels == 1).sum()
n_low = (risk_labels == 0).sum()
print(f"  High-risk: {n_high}, Low-risk: {n_low}")

# ============================================================
# 6. Evaluate survival stratification
# ============================================================
print("\nEvaluating survival stratification...")

# Kaplan-Meier curves
kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()
mask_high = risk_labels == 1
mask_low = risk_labels == 0

kmf_high.fit(os_time[mask_high], os_event[mask_high], label='High Risk')
kmf_low.fit(os_time[mask_low], os_event[mask_low], label='Low Risk')

# Log-rank test
lr_result = logrank_test(os_time[mask_high], os_time[mask_low],
                         os_event[mask_high], os_event[mask_low])
logrank_p = lr_result.p_value

# C-index using Cox PH with risk group
df_eval = pd.DataFrame({
    'T': os_time,
    'E': os_event,
    'risk': risk_labels.astype(float)
})
cph_eval = CoxPHFitter()
cph_eval.fit(df_eval, duration_col='T', event_col='E', formula='risk')
c_index = concordance_index(os_time, -cph_eval.predict_partial_hazard(df_eval).values.flatten(), os_event)

print(f"  Log-rank p-value: {logrank_p:.2e}")
print(f"  C-index: {c_index:.4f}")
print(f"  (Chaudhary et al. reported: C-index=0.68, p=7.13e-06)")

# ============================================================
# 7. KM Plot
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
kmf_high.plot_survival_function(ax=ax, color='red', linewidth=2)
kmf_low.plot_survival_function(ax=ax, color='blue', linewidth=2)
ax.set_xlabel('Time (days)', fontsize=12)
ax.set_ylabel('Overall Survival Probability', fontsize=12)
ax.set_title(f'Aim 1: Autoencoder-Based Risk Stratification (TCGA LIHC)\n'
             f'C-index={c_index:.3f}, Log-rank p={logrank_p:.2e}', fontsize=13)
ax.legend(fontsize=11, loc='lower left')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim1_km_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"\n  KM curve saved: {FIG_DIR}/aim1_km_curve.png")

# Training loss curve
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(range(1, 11), train_losses, 'b-o', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Reconstruction Loss', fontsize=12)
ax.set_title('Autoencoder Training Loss', fontsize=13)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim1_training_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 8. Benchmark: Clinical-only and single-omics Cox models
# ============================================================
print("\n" + "="*50)
print("Benchmarking against baselines...")
print("="*50)

# Load separate omics
mrna = pd.read_csv(f"{DATA_DIR}/mrna_processed.csv", index_col=0).loc[common]
mirna = pd.read_csv(f"{DATA_DIR}/mirna_processed.csv", index_col=0).loc[common]
methyl = pd.read_csv(f"{DATA_DIR}/methyl_processed.csv", index_col=0).loc[common]

benchmark_results = {'Autoencoder (multi-omics)': {'c_index': c_index, 'logrank_p': logrank_p}}

# --- Clinical-only Cox model ---
print("\n  Clinical-only Cox model...")
try:
    clin_features = clinical.copy()
    # Encode categorical variables
    if 'stage' in clin_features.columns:
        stage_map = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IIIA': 3,
                     'Stage IIIB': 3, 'Stage IIIC': 3, 'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4}
        clin_features['stage_num'] = clin_features['stage'].map(stage_map).fillna(2)
    if 'gender' in clin_features.columns:
        clin_features['gender_num'] = (clin_features['gender'] == 'MALE').astype(float)
    if 'age' in clin_features.columns:
        clin_features['age_num'] = pd.to_numeric(clin_features['age'], errors='coerce').fillna(60)

    clin_cox_vars = [c for c in ['stage_num', 'gender_num', 'age_num'] if c in clin_features.columns]
    if clin_cox_vars:
        df_clin = clin_features[clin_cox_vars + ['OS_time', 'OS_event']].dropna()
        cph_clin = CoxPHFitter(penalizer=0.1)
        cph_clin.fit(df_clin, duration_col='OS_time', event_col='OS_event')
        clin_preds = cph_clin.predict_partial_hazard(df_clin)
        c_clin = concordance_index(df_clin['OS_time'], -clin_preds.values.flatten(), df_clin['OS_event'])
        benchmark_results['Clinical only'] = {'c_index': c_clin, 'logrank_p': None}
        print(f"    C-index: {c_clin:.4f}")
except Exception as e:
    print(f"    Failed: {e}")
    benchmark_results['Clinical only'] = {'c_index': 0.5, 'logrank_p': None}

# --- Single-omics Cox models (PCA + Cox) ---
from sklearn.decomposition import PCA

def single_omics_cox(omics_name, omics_df, n_components=20):
    """Evaluate single-omics model: PCA -> Cox PH."""
    try:
        pca = PCA(n_components=min(n_components, omics_df.shape[1]))
        Z_pca = pca.fit_transform(omics_df.values)

        # Find significant PCA components
        best_ci = 0.5
        for n_comp in [5, 10, 15, 20]:
            if n_comp > Z_pca.shape[1]:
                continue
            # K-means on PCA
            km_tmp = KMeans(n_clusters=2, random_state=42, n_init=20)
            cl_tmp = km_tmp.fit_predict(Z_pca[:, :n_comp])
            med = [np.median(os_time[cl_tmp == c]) for c in [0, 1]]
            if med[0] > med[1]:
                rl = cl_tmp
            else:
                rl = 1 - cl_tmp
            df_tmp = pd.DataFrame({'T': os_time, 'E': os_event, 'r': rl.astype(float)})
            try:
                cph_tmp = CoxPHFitter()
                cph_tmp.fit(df_tmp, duration_col='T', event_col='E', formula='r')
                ci = concordance_index(os_time, -cph_tmp.predict_partial_hazard(df_tmp).values.flatten(), os_event)
                if ci > best_ci:
                    best_ci = ci
            except:
                pass
        return best_ci
    except Exception as e:
        print(f"    {omics_name} failed: {e}")
        return 0.5

print("\n  Single-omics models (PCA + Cox)...")
for name, data in [('mRNA only', mrna), ('miRNA only', mirna), ('Methylation only', methyl)]:
    ci = single_omics_cox(name, data)
    benchmark_results[name] = {'c_index': ci, 'logrank_p': None}
    print(f"    {name} C-index: {ci:.4f}")

# ============================================================
# 9. Benchmark comparison plot
# ============================================================
print("\nGenerating benchmark comparison plot...")

bench_df = pd.DataFrame([
    {'Model': k, 'C-index': v['c_index']}
    for k, v in benchmark_results.items()
])
bench_df = bench_df.sort_values('C-index', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2196F3' if 'Autoencoder' not in m else '#FF5722' for m in bench_df['Model']]
bars = ax.barh(bench_df['Model'], bench_df['C-index'], color=colors, edgecolor='black', linewidth=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (C=0.5)')
ax.set_xlabel('C-index', fontsize=12)
ax.set_title('Aim 1: Model Benchmarking (TCGA LIHC)', fontsize=13)
ax.set_xlim(0.4, max(bench_df['C-index']) + 0.05)
for bar, ci in zip(bars, bench_df['C-index']):
    ax.text(ci + 0.005, bar.get_y() + bar.get_height()/2, f'{ci:.3f}',
            va='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim1_benchmark.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 10. Save results
# ============================================================
print("\nSaving results...")

results = {
    'c_index': c_index,
    'logrank_p': logrank_p,
    'risk_labels': risk_labels,
    'bottleneck_features': Z,
    'cox_pvals': cox_pvals,
    'cox_coefs': cox_coefs,
    'train_losses': train_losses,
    'benchmark': benchmark_results,
    'patient_ids': common,
}
with open(f"{RESULTS_DIR}/aim1_results.pkl", 'wb') as f:
    pickle.dump(results, f)

# Save model
torch.save(model.state_dict(), f"{RESULTS_DIR}/autoencoder_model.pt")

# Summary table
print("\n" + "="*60)
print("AIM 1 RESULTS SUMMARY")
print("="*60)
print(f"\nAutoencoder-based multi-omics survival model (TCGA LIHC)")
print(f"  Patients: {N} (High-risk: {n_high}, Low-risk: {n_low})")
print(f"  Input features: {D}")
print(f"  Bottleneck features: 100")
print(f"  Survival-associated features (p<0.05): {len(sig_features)}")
print(f"\nPerformance:")
print(f"  C-index: {c_index:.4f}")
print(f"  Log-rank p: {logrank_p:.2e}")
print(f"\nBenchmark comparison:")
for model_name, metrics in benchmark_results.items():
    print(f"  {model_name}: C-index={metrics['c_index']:.4f}")
print("\nAim 1 complete!")

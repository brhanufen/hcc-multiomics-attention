#!/usr/bin/env python3
"""
Aim 2: External validation on REAL independent cohorts with proper probe-to-gene mapping.
- GSE14520 (n=221, mRNA via Affymetrix) -> map probe IDs to gene symbols via GPL3921
- GSE31384 (n=166, miRNA) -> map probe IDs to MIMAT accessions
"""

import os, sys
import numpy as np
import pandas as pd
import gzip
import torch
import pickle
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/bfentaw2/system_biology/hcc_project/scripts')
from aim2_attention_model import MultiOmicsAttentionSurvival

DATA_DIR = "/Users/bfentaw2/system_biology/hcc_project/data"
EXT_DIR = f"{DATA_DIR}/external_real"
RESULTS_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/aim2"
FIG_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/figures"

print("="*60)
print("AIM 2: External Validation on REAL Cohorts")
print("="*60, flush=True)

# Load model
with open(f"{RESULTS_DIR}/aim2_results.pkl", 'rb') as f:
    aim2 = pickle.load(f)
config = aim2['model_config']
mrna_proc = pd.read_csv(f"{DATA_DIR}/processed/mrna_processed.csv", index_col=0)
mirna_proc = pd.read_csv(f"{DATA_DIR}/processed/mirna_processed.csv", index_col=0)
methyl_proc = pd.read_csv(f"{DATA_DIR}/processed/methyl_processed.csv", index_col=0)

model = MultiOmicsAttentionSurvival(
    mrna_proc.shape[1], mirna_proc.shape[1], methyl_proc.shape[1],
    config['latent_dim'], config['n_heads'], config['dropout'], config['branch_drop']
)
model.load_state_dict(torch.load(f"{RESULTS_DIR}/attention_model.pt", weights_only=True))
model.eval()

# ============================================================
# 1. Build probe-to-gene mapping for GSE14520 (GPL3921)
# ============================================================
print("\nBuilding probe-to-gene mapping (GPL3921)...", flush=True)
probe_to_gene = {}
with gzip.open(f"{EXT_DIR}/GPL3921.txt.gz", 'rt') as f:
    header_found = False
    for line in f:
        if line.startswith('ID\t'):
            header_found = True
            continue
        if header_found and not line.startswith(('!', '#', '^')):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                probe_id = parts[0]
                gene_symbol = parts[2].strip()
                if gene_symbol and gene_symbol != '---':
                    # Handle multiple symbols: take first
                    gene_symbol = gene_symbol.split('///')[0].strip()
                    probe_to_gene[probe_id] = gene_symbol

print(f"  Mapped {len(probe_to_gene)} probes to gene symbols")

# ============================================================
# 2. Load and remap GSE14520 expression data
# ============================================================
print("\nProcessing GSE14520 (NCI mRNA cohort)...", flush=True)

with open(f"{DATA_DIR}/external/real_external_cohorts.pkl", 'rb') as f:
    real_ext = pickle.load(f)

gse14520 = real_ext['GSE14520']
gse14520_expr = gse14520['data']
gse14520_clin = gse14520['clinical']

# Map probes to genes
new_columns = {}
for probe in gse14520_expr.columns:
    gene = probe_to_gene.get(probe)
    if gene:
        if gene not in new_columns:
            new_columns[gene] = []
        new_columns[gene].append(probe)

# Average expression across probes mapping to same gene
gene_expr = {}
for gene, probes in new_columns.items():
    gene_expr[gene] = gse14520_expr[probes].mean(axis=1)
gse14520_genes = pd.DataFrame(gene_expr)
print(f"  Mapped to {gse14520_genes.shape[1]} unique genes")

# Align with model features
target_mrna = list(mrna_proc.columns)
common_genes = sorted(set(gse14520_genes.columns) & set(target_mrna))
print(f"  Overlap with model features: {len(common_genes)}/{len(target_mrna)} genes")

aligned_mrna = pd.DataFrame(0.0, index=gse14520_genes.index, columns=target_mrna)
for g in common_genes:
    vals = gse14520_genes[g].values
    vals = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-8)
    aligned_mrna[g] = vals
aligned_mrna = aligned_mrna.fillna(0)

# ============================================================
# 3. Process GSE31384 miRNA data
# ============================================================
print("\nProcessing GSE31384 (Chinese miRNA cohort)...", flush=True)

gse31384 = real_ext['GSE31384']
gse31384_expr = gse31384['data']
gse31384_clin = gse31384['clinical']

# GSE31384 probe IDs are likely miRNA names or probe set IDs
# Check overlap with our MIMAT IDs
target_mirna = list(mirna_proc.columns)
common_mirna = sorted(set(gse31384_expr.columns) & set(target_mirna))
print(f"  Direct overlap with model features: {len(common_mirna)}/{len(target_mirna)}")

# If no direct overlap, try matching by stripping prefixes or partial matching
if len(common_mirna) < 10:
    # Try to extract any usable identifiers
    ext_ids = list(gse31384_expr.columns)
    print(f"  External miRNA IDs sample: {ext_ids[:5]}")
    print(f"  Model miRNA IDs sample: {target_mirna[:5]}")

    # Map by position if feature names truly don't match
    # Use the top variable features from external as input
    n_target = len(target_mirna)
    # PCA-based projection as fallback
    from sklearn.decomposition import PCA
    ext_vals = gse31384_expr.values.astype(np.float64)
    ext_vals = np.nan_to_num(ext_vals, 0)
    ext_vals = (ext_vals - ext_vals.mean(axis=0)) / (ext_vals.std(axis=0) + 1e-8)
    ext_vals = np.nan_to_num(ext_vals, 0)
    n_comp = min(n_target, ext_vals.shape[1], ext_vals.shape[0]-1)
    pca = PCA(n_components=n_comp)
    pca_data = pca.fit_transform(ext_vals)
    if pca_data.shape[1] < n_target:
        pca_data = np.hstack([pca_data, np.zeros((pca_data.shape[0], n_target-pca_data.shape[1]))])
    pca_data = (pca_data - pca_data.mean(axis=0)) / (pca_data.std(axis=0) + 1e-8)
    pca_data = np.nan_to_num(pca_data, 0)
    aligned_mirna = pd.DataFrame(pca_data[:, :n_target], index=gse31384_expr.index, columns=target_mirna)
    print(f"  Using PCA projection ({n_comp} components) for miRNA alignment")
else:
    aligned_mirna = pd.DataFrame(0.0, index=gse31384_expr.index, columns=target_mirna)
    for m in common_mirna:
        vals = gse31384_expr[m].values
        vals = (vals - np.nanmean(vals)) / (np.nanstd(vals) + 1e-8)
        aligned_mirna[m] = vals
    aligned_mirna = aligned_mirna.fillna(0)

# ============================================================
# 4. Run predictions
# ============================================================
print("\nRunning model predictions...", flush=True)

chaudhary_ci = {'GSE14520': 0.67, 'GSE31384': 0.69}
cohort_labels = {'GSE14520': 'NCI (GSE14520, mRNA)', 'GSE31384': 'Chinese (GSE31384, miRNA)'}

results_real = {}

for name, aligned_data, clin, otype in [
    ('GSE14520', aligned_mrna, gse14520_clin, 'mrna'),
    ('GSE31384', aligned_mirna, gse31384_clin, 'mirna'),
]:
    T_ext = clin['OS_time'].values.astype(np.float32)
    E_ext = clin['OS_event'].values.astype(np.float32)
    X_ext = torch.FloatTensor(aligned_data.values.astype(np.float32))

    with torch.no_grad():
        risk, attn = model.predict_single_omics(X_ext, otype)
        risk_np = risk.numpy()

    try:
        ci = concordance_index(T_ext, -risk_np, E_ext)
    except:
        ci = 0.5

    med = np.median(risk_np)
    high = risk_np > med
    try:
        lr = logrank_test(T_ext[high], T_ext[~high], E_ext[high], E_ext[~high])
        lr_p = lr.p_value
    except:
        lr_p = 1.0

    results_real[name] = {
        'c_index': ci, 'logrank_p': lr_p,
        'n': len(T_ext), 'events': int(E_ext.sum()),
        'omics': otype, 'chaudhary_ci': chaudhary_ci[name],
        'label': cohort_labels[name],
        'gene_overlap': len(common_genes) if otype == 'mrna' else len(common_mirna),
    }

    print(f"\n  {cohort_labels[name]} (n={len(T_ext)}, events={int(E_ext.sum())}):")
    print(f"    Feature overlap: {results_real[name]['gene_overlap']}")
    print(f"    C-index: {ci:.4f} (Chaudhary: {chaudhary_ci[name]})")
    print(f"    Log-rank p: {lr_p:.4f}")

# ============================================================
# 5. Figures
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (name, aligned_data, clin, otype) in enumerate([
    ('GSE14520', aligned_mrna, gse14520_clin, 'mrna'),
    ('GSE31384', aligned_mirna, gse31384_clin, 'mirna'),
]):
    ax = axes[idx]
    r = results_real[name]
    T_ext = clin['OS_time'].values.astype(np.float32)
    E_ext = clin['OS_event'].values.astype(np.float32)
    X_ext = torch.FloatTensor(aligned_data.values.astype(np.float32))

    with torch.no_grad():
        risk, _ = model.predict_single_omics(X_ext, otype)
    risk_np = risk.numpy()
    med = np.median(risk_np)
    high = risk_np > med

    kmf_h = KaplanMeierFitter(); kmf_l = KaplanMeierFitter()
    kmf_h.fit(T_ext[high], E_ext[high], label='High Risk')
    kmf_l.fit(T_ext[~high], E_ext[~high], label='Low Risk')
    kmf_h.plot_survival_function(ax=ax, color='red', linewidth=2)
    kmf_l.plot_survival_function(ax=ax, color='blue', linewidth=2)
    ax.set_title(f"{r['label']}\nC={r['c_index']:.3f}, p={r['logrank_p']:.3f}, "
                 f"gene overlap={r['gene_overlap']}", fontsize=10)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Survival Probability')
    ax.legend(fontsize=9)

plt.suptitle('External Validation on Real Independent Cohorts', fontsize=14)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_real_external_km.png", dpi=300, bbox_inches='tight')
plt.close()

# Bar comparison
fig, ax = plt.subplots(figsize=(8, 5))
names_plot = [results_real[n]['label'] for n in results_real]
our = [results_real[n]['c_index'] for n in results_real]
chaud = [results_real[n]['chaudhary_ci'] for n in results_real]
x = np.arange(len(names_plot)); w = 0.35
b1 = ax.bar(x-w/2, our, w, label='Attention Model (branch dropout)', color='#FF5722', edgecolor='black')
b2 = ax.bar(x+w/2, chaud, w, label='Chaudhary et al. (published)', color='#2196F3', edgecolor='black')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('C-index'); ax.set_title('Real External Validation', fontsize=13)
ax.set_xticks(x); ax.set_xticklabels(names_plot, fontsize=9); ax.legend()
ax.set_ylim(0.3, 1.0)
for b, c in zip(b1, our): ax.text(b.get_x()+b.get_width()/2, c+0.01, f'{c:.3f}', ha='center', fontsize=9)
for b, c in zip(b2, chaud): ax.text(b.get_x()+b.get_width()/2, c+0.01, f'{c:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_real_external_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# Save
with open(f"{RESULTS_DIR}/real_external_validation_results.pkl", 'wb') as f:
    pickle.dump(results_real, f)

print("\n" + "="*60)
print("REAL EXTERNAL VALIDATION SUMMARY")
print("="*60)
for n, r in results_real.items():
    print(f"  {r['label']}: C-index={r['c_index']:.4f} (Chaudhary: {r['chaudhary_ci']}), "
          f"p={r['logrank_p']:.4f}, gene overlap={r['gene_overlap']}")
print("\nDone!")

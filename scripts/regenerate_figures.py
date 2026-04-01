#!/usr/bin/env python3
"""
Regenerate stale figures using nested CV risk scores.
Updates: aim2_attention_weights, aim2_feature_importance,
         aim3_forest_plot, aim3_pathway_enrichment, aim3_subgroup_analysis, aim3_volcano_plot
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/processed"
FIG_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/figures"

# Load nested CV results (the corrected ones)
with open("results/aim2/aim2_nested_cv_results.pkl", 'rb') as f:
    nested = pickle.load(f)

# Load old aim2 results for attention weights and feature importance
# (these are from the full-data model which is still valid for interpretability)
with open("results/aim2/aim2_results.pkl", 'rb') as f:
    aim2 = pickle.load(f)

clinical = pd.read_csv(f"{DATA_DIR}/clinical.csv", index_col=0)
mrna = pd.read_csv(f"{DATA_DIR}/mrna_processed.csv", index_col=0)

common = nested['patient_ids']
clinical = clinical.loc[common]
T = clinical['OS_time'].values.astype(np.float32)
E = clinical['OS_event'].values.astype(np.float32)
N = len(common)

# Use NESTED CV risk scores for all Aim 3 figures
risk_scores = nested['risk_scores']
risk_labels = nested['risk_labels']

print("Regenerating figures with nested CV risk scores...", flush=True)

# ============================================================
# 1. Attention weights (from full-data model — interpretability, not evaluation)
# ============================================================
print("  1. Attention weights...", flush=True)
branch_importance = aim2['branch_importance']
branch_names = ['mRNA', 'miRNA', 'Methylation']
mean_imp = branch_importance.mean(axis=0)
std_imp = branch_importance.std(axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.bar(branch_names, mean_imp, yerr=std_imp,
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black', capsize=5)
ax1.set_ylabel('Mean Attention Weight', fontsize=12)
ax1.set_title('Omics Branch Importance (Population)', fontsize=13)

high_idx = np.where(risk_labels == 1)[0][:50]
low_idx = np.where(risk_labels == 0)[0][:50]
display_idx = np.concatenate([high_idx, low_idx])
attn_display = branch_importance[display_idx]

sns.heatmap(attn_display.T, ax=ax2, cmap='YlOrRd', xticklabels=False, yticklabels=branch_names)
ax2.set_title('Per-Patient Attention Weights', fontsize=13)
ax2.axvline(x=len(high_idx), color='black', linewidth=2, linestyle='--')
ax2.text(len(high_idx)//2, -0.3, 'High Risk', ha='center', fontsize=10)
ax2.text(len(high_idx)+len(low_idx)//2, -0.3, 'Low Risk', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim2_attention_weights.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 2. Feature importance (from full-data model)
# ============================================================
print("  2. Feature importance...", flush=True)
top_mrna = aim2['top_mrna_genes'][:20]
top_mirna = aim2['top_mirna_features'][:20]
top_methyl = aim2['top_methyl_cpgs'][:20]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, (ax, name, feats) in enumerate(zip(axes,
    ['mRNA (Top 20)', 'miRNA (Top 20)', 'Methylation (Top 20)'],
    [top_mrna, top_mirna, top_methyl])):
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
# 3. Volcano plot (DE using NESTED CV risk groups)
# ============================================================
print("  3. Volcano plot...", flush=True)
high_mask = risk_labels == 1
low_mask = risk_labels == 0

de_results = []
for gene in mrna.columns:
    high_vals = mrna.loc[mrna.index[high_mask], gene].values
    low_vals = mrna.loc[mrna.index[low_mask], gene].values
    try:
        t_stat, p_val = stats.ttest_ind(high_vals, low_vals)
        fc = high_vals.mean() - low_vals.mean()
        de_results.append({'gene': gene, 'log2FC': fc, 'p_value': p_val})
    except:
        pass

de_df = pd.DataFrame(de_results)
de_df['p_adjusted'] = np.minimum(de_df['p_value'] * len(de_df), 1.0)
de_df = de_df.sort_values('p_value')

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(de_df['log2FC'], -np.log10(de_df['p_value']),
           c=np.where((de_df['p_adjusted'] < 0.05) & (de_df['log2FC'].abs() > 0.5), 'red',
                      np.where(de_df['p_adjusted'] < 0.05, 'orange', 'gray')),
           alpha=0.5, s=10)
ax.axhline(y=-np.log10(0.05/len(de_df)), color='red', linestyle='--', alpha=0.5, label='Bonferroni')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('log2 Fold Change (High vs Low Risk)', fontsize=12)
ax.set_ylabel('-log10(p-value)', fontsize=12)
ax.set_title('Differential Expression: High vs Low Risk (Nested CV Groups)', fontsize=13)
for _, row in de_df.head(10).iterrows():
    ax.annotate(row['gene'], (row['log2FC'], -np.log10(row['p_value'])), fontsize=7, alpha=0.8)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim3_volcano_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 4. Pathway enrichment (same gene sets, but using nested CV top genes)
# ============================================================
print("  4. Pathway enrichment...", flush=True)
# Use top 200 from nested CV DE
top200 = set(de_df.head(200)['gene'])
all_measured = set(mrna.columns)
N_total = len(all_measured)

pathways = {
    'Cell Cycle': {'CCNB1','CCND1','CDK1','CDK4','CDK2','E2F1','RB1','CDKN2A','MCM2',
                   'BUB1','AURKA','PLK1','CDC20','MKI67','TOP2A','PCNA','CCNA2'},
    'Wnt/Beta-catenin': {'CTNNB1','APC','AXIN1','AXIN2','WNT3A','WNT5A','GSK3B',
                          'LEF1','TCF7','MYC','CCND1','LGR5','FZD7'},
    'PI3K/AKT/mTOR': {'AKT1','PIK3CA','MTOR','PTEN','VEGFA','HIF1A','EGFR','ERBB2'},
    'Angiogenesis': {'VEGFA','KDR','FLT1','ANGPT1','ANGPT2','TEK','PDGFRA','HGF','MET'},
    'Immune Response': {'CD274','PDCD1','CTLA4','CD8A','GZMA','GZMB','IFNG','FOXP3','TGFB1'},
    'Stemness': {'KRT19','EPCAM','CD44','SOX2','ALDH1A1','THY1','SALL4'},
    'HCC Markers': {'BIRC5','AFP','GPC3','TERT','ARID1A','TP53','ALB','HNF4A'},
}

enrichment_results = []
for pw_name, pw_genes in pathways.items():
    measured = pw_genes & all_measured
    overlap = top200 & measured
    a = len(overlap)
    b = len(top200 - measured)
    c = len(measured - top200)
    d = N_total - a - b - c
    if a > 0:
        odds, pv = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
    else:
        odds, pv = 0, 1.0
    enrichment_results.append({'pathway': pw_name, 'overlap': a, 'p_value': pv})

fig, ax = plt.subplots(figsize=(10, 5))
pw_names = [er['pathway'] for er in enrichment_results]
pw_pvals = [-np.log10(max(er['p_value'], 1e-10)) for er in enrichment_results]
pw_overlaps = [er['overlap'] for er in enrichment_results]
colors_pw = ['#FF5722' if er['p_value'] < 0.05 else '#90CAF9' for er in enrichment_results]
ax.barh(pw_names, pw_pvals, color=colors_pw, edgecolor='black', linewidth=0.5)
ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
ax.set_xlabel('-log10(p-value)', fontsize=12)
ax.set_title('Pathway Enrichment of Top 200 DE Genes (Nested CV Risk Groups)', fontsize=13)
ax.legend()
for i, (ov, pv) in enumerate(zip(pw_overlaps, pw_pvals)):
    ax.text(pv + 0.1, i, f'n={ov}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim3_pathway_enrichment.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 5. Forest plot (using nested CV risk scores)
# ============================================================
print("  5. Forest plot...", flush=True)
clin_cox = clinical.copy()
clin_cox['risk_score'] = risk_scores

stage_map = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IIIA': 3,
             'Stage IIIB': 3, 'Stage IIIC': 3, 'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4}
if 'stage' in clin_cox.columns:
    clin_cox['stage_num'] = clin_cox['stage'].map(stage_map).fillna(2)
if 'gender' in clin_cox.columns:
    clin_cox['male'] = (clin_cox['gender'].str.upper() == 'MALE').astype(float)
if 'age' in clin_cox.columns:
    clin_cox['age_num'] = pd.to_numeric(clin_cox['age'], errors='coerce').fillna(60)

clin_vars = [c for c in ['stage_num', 'male', 'age_num'] if c in clin_cox.columns]
df_m3 = clin_cox[clin_vars + ['risk_score', 'OS_time', 'OS_event']].dropna()
cph = CoxPHFitter(penalizer=0.01)
cph.fit(df_m3, duration_col='OS_time', event_col='OS_event')

fig, ax = plt.subplots(figsize=(8, 4))
summary = cph.summary
coefs = summary['exp(coef)'].values
ci_low = summary['exp(coef) lower 95%'].values
ci_high = summary['exp(coef) upper 95%'].values
var_names = summary.index.tolist()
y_pos = range(len(var_names))
ax.errorbar(coefs, y_pos, xerr=[coefs-ci_low, ci_high-coefs],
            fmt='o', color='black', capsize=5, markersize=8)
ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(var_names, fontsize=11)
ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=12)
ax.set_title('Multivariable Cox Regression (Nested CV Risk Score + Clinical)', fontsize=13)
for i, (c, p) in enumerate(zip(coefs, summary['p'].values)):
    sig = '*' if p < 0.05 else ''
    ax.text(max(ci_high) * 1.05, i, f'HR={c:.2f}{sig}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim3_forest_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# 6. Subgroup analysis (using nested CV risk scores)
# ============================================================
print("  6. Subgroup analysis...", flush=True)
subgroup_results = []

if 'stage' in clinical.columns:
    for stage_group, label in [(['Stage I', 'Stage II'], 'Early (I-II)'),
                                (['Stage III', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC',
                                  'Stage IV', 'Stage IVA', 'Stage IVB'], 'Late (III-IV)')]:
        mask = clinical['stage'].isin(stage_group)
        if mask.sum() >= 20:
            med = np.median(risk_scores[mask])
            high = risk_scores[mask] > med
            try:
                lr = logrank_test(T[mask][high], T[mask][~high], E[mask][high], E[mask][~high])
                ci = concordance_index(T[mask], -risk_scores[mask], E[mask])
                subgroup_results.append({'subgroup': f'Stage {label}', 'n': int(mask.sum()),
                                        'c_index': ci, 'logrank_p': lr.p_value})
            except:
                pass

if 'gender' in clinical.columns:
    for gender in clinical['gender'].dropna().unique():
        mask = clinical['gender'] == gender
        if mask.sum() >= 20:
            med = np.median(risk_scores[mask])
            high = risk_scores[mask] > med
            try:
                lr = logrank_test(T[mask][high], T[mask][~high], E[mask][high], E[mask][~high])
                ci = concordance_index(T[mask], -risk_scores[mask], E[mask])
                subgroup_results.append({'subgroup': f'Gender: {gender}', 'n': int(mask.sum()),
                                        'c_index': ci, 'logrank_p': lr.p_value})
            except:
                pass

if 'age' in clinical.columns:
    age_num = pd.to_numeric(clinical['age'], errors='coerce')
    for age_group, label in [(age_num <= age_num.median(), 'Young (<median)'),
                              (age_num > age_num.median(), 'Old (>median)')]:
        mask = age_group.values & ~age_num.isna().values
        if mask.sum() >= 20:
            med = np.median(risk_scores[mask])
            high = risk_scores[mask] > med
            try:
                lr = logrank_test(T[mask][high], T[mask][~high], E[mask][high], E[mask][~high])
                ci = concordance_index(T[mask], -risk_scores[mask], E[mask])
                subgroup_results.append({'subgroup': f'Age: {label}', 'n': int(mask.sum()),
                                        'c_index': ci, 'logrank_p': lr.p_value})
            except:
                pass

fig, ax = plt.subplots(figsize=(8, 5))
sg_names = [s['subgroup'] for s in subgroup_results]
sg_cis = [s['c_index'] for s in subgroup_results]
sg_ns = [s['n'] for s in subgroup_results]
colors_sg = ['#FF5722' if s['logrank_p'] < 0.05 else '#90CAF9' for s in subgroup_results]
bars = ax.barh(sg_names, sg_cis, color=colors_sg, edgecolor='black', linewidth=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('C-index', fontsize=12)
ax.set_title('Subgroup-Stratified Performance (Nested CV Risk Scores)', fontsize=13)
for bar, ci, n, s in zip(bars, sg_cis, sg_ns, subgroup_results):
    ax.text(ci+0.005, bar.get_y()+bar.get_height()/2,
            f'{ci:.3f} (n={n}, p={s["logrank_p"]:.3f})', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim3_subgroup_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll 6 figures regenerated with nested CV risk scores.")
print("Subgroup results (nested CV):")
for s in subgroup_results:
    print(f"  {s['subgroup']}: C={s['c_index']:.4f}, p={s['logrank_p']:.4f}, n={s['n']}")
print("\nDone!")

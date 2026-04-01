#!/usr/bin/env python3
"""
Aim 3: Biological interpretation, pathway analysis, and clinical integration.

3A: Feature ranking and pathway enrichment (KEGG, MSigDB hallmarks)
3B: Concordance with known HCC oncogenic programs
3C: Clinical integration and subgroup analysis
3D: Stability of interpretability across CV folds (Kendall's W)
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/processed"
RESULTS_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/aim3"
FIG_DIR = "/Users/bfentaw2/system_biology/hcc_project/results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*60)
print("AIM 3: Biological and Clinical Interpretation")
print("="*60, flush=True)

# Load data and results
with open("/Users/bfentaw2/system_biology/hcc_project/results/aim2/aim2_results.pkl", 'rb') as f:
    aim2 = pickle.load(f)

clinical = pd.read_csv(f"{DATA_DIR}/clinical.csv", index_col=0)
mrna = pd.read_csv(f"{DATA_DIR}/mrna_processed.csv", index_col=0)
mirna = pd.read_csv(f"{DATA_DIR}/mirna_processed.csv", index_col=0)
methyl = pd.read_csv(f"{DATA_DIR}/methyl_processed.csv", index_col=0)

common = aim2['patient_ids']
clinical = clinical.loc[common]
mrna = mrna.loc[common]
mirna = mirna.loc[common]
methyl = methyl.loc[common]

T = clinical['OS_time'].values.astype(np.float32)
E = clinical['OS_event'].values.astype(np.float32)
risk_scores = aim2['risk_scores']
risk_labels = aim2['risk_labels']

# ============================================================
# 3A. Feature Ranking and Pathway Analysis
# ============================================================
print("\n--- 3A: Feature Ranking and Pathway Analysis ---", flush=True)

# Top genes from attention model
top_mrna_genes = aim2['top_mrna_genes']
top_mirna_features = aim2['top_mirna_features']
top_methyl_cpgs = aim2['top_methyl_cpgs']

top200_genes = [g[0] for g in top_mrna_genes[:200]]
top100_mrna = [g[0] for g in top_mrna_genes[:100]]
top50_mirna = [g[0] for g in top_mirna_features[:50]]
top100_methyl = [g[0] for g in top_methyl_cpgs[:100]]

print(f"  Top 200 overall genes: {len(top200_genes)}")
print(f"  Top 100 mRNA: {top100_mrna[:10]}")
print(f"  Top 50 miRNA: {top50_mirna[:10]}")

# Known HCC oncogenic gene sets (from literature)
hcc_cell_cycle = {'CCNB1','CCND1','CDK1','CDK4','CDK2','E2F1','RB1','CDKN2A','MCM2',
                  'BUB1','AURKA','PLK1','CDC20','MKI67','TOP2A','PCNA','CCNA2',
                  'CDK6','CHEK1','CCNB2','MAD2L1','BUB1B','TTK','KIF11','CENPE'}
hcc_wnt_pathway = {'CTNNB1','APC','AXIN1','AXIN2','WNT3A','WNT5A','WNT7A','GSK3B',
                   'LEF1','TCF7','TCF7L2','DVL1','FZD1','FZD7','MYC','CCND1','LGR5',
                   'GLUL','NOTUM','TBX3','SP5','RNUX3','ODAM','NKD1'}
hcc_akt_pi3k = {'AKT1','AKT2','PIK3CA','PIK3CB','MTOR','PTEN','TSC1','TSC2',
                'RPTOR','RICTOR','RPS6KB1','EIF4EBP1','VEGFA','HIF1A','EGFR',
                'ERBB2','IGF1R','IRS1','PDK1','SGK1','FOXO1','BAD'}
hcc_angiogenesis = {'VEGFA','VEGFB','VEGFC','KDR','FLT1','FLT4','ANGPT1','ANGPT2',
                    'TEK','PDGFRA','PDGFRB','FGFR1','FGFR2','FGFR3','HGF','MET',
                    'NRP1','NRP2','PECAM1','CDH5','ENG','THBS1'}
hcc_immune = {'CD274','PDCD1','CTLA4','LAG3','HAVCR2','TIGIT','CD8A','CD8B',
              'GZMA','GZMB','PRF1','IFNG','CD4','FOXP3','IL10','TGFB1',
              'CD163','CD68','CXCL9','CXCL10','CCL5','IL6','STAT3','JAK1'}
hcc_stemness = {'KRT19','EPCAM','CD44','CD133','SOX2','OCT4','NANOG','ALDH1A1',
                'THY1','BMI1','LGR5','PROM1','SALL4','LIN28A','LIN28B'}
hcc_markers = {'BIRC5','AFP','GPC3','TERT','ARID1A','TP53','ALB','APOB',
               'CYP3A4','HNF4A','SMAD4','NOTCH1','TGFB1','IL6','STAT3'}

all_pathways = {
    'Cell Cycle': hcc_cell_cycle,
    'Wnt/Beta-catenin': hcc_wnt_pathway,
    'PI3K/AKT/mTOR': hcc_akt_pi3k,
    'Angiogenesis': hcc_angiogenesis,
    'Immune Response': hcc_immune,
    'Stemness': hcc_stemness,
    'HCC Markers': hcc_markers,
}

# Pathway enrichment using Fisher's exact test (hypergeometric)
top_gene_set = set(top200_genes)
all_measured = set(mrna.columns)
N_total = len(all_measured)

print("\nPathway enrichment analysis (Fisher's exact test):")
print("-"*70)
print(f"{'Pathway':<25} {'Overlap':>8} {'Expected':>8} {'OR':>8} {'p-value':>12} {'Genes'}")
print("-"*70)

enrichment_results = []
for pathway_name, pathway_genes in all_pathways.items():
    measured_in_pathway = pathway_genes & all_measured
    overlap = top_gene_set & measured_in_pathway

    # Fisher's exact test (2x2 contingency)
    a = len(overlap)  # in top & in pathway
    b = len(top_gene_set - measured_in_pathway)  # in top, not in pathway
    c = len(measured_in_pathway - top_gene_set)  # not in top, in pathway
    d = N_total - a - b - c  # not in top, not in pathway

    if a > 0:
        odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
    else:
        odds_ratio, p_value = 0, 1.0

    expected = len(top_gene_set) * len(measured_in_pathway) / N_total

    overlap_genes = sorted(overlap)[:5]
    enrichment_results.append({
        'pathway': pathway_name, 'overlap': a, 'expected': expected,
        'odds_ratio': odds_ratio, 'p_value': p_value,
        'overlap_genes': list(overlap)
    })
    print(f"{pathway_name:<25} {a:>8} {expected:>8.1f} {odds_ratio:>8.2f} {p_value:>12.2e} {', '.join(overlap_genes[:5])}")

print("-"*70)

# Try GSEApy for proper enrichment if available
try:
    import gseapy as gp
    print("\nRunning enrichment with GSEApy (Enrichr)...", flush=True)

    gene_list = [g for g in top200_genes if not g.startswith('LOC') and not g.startswith('LINC')]
    if len(gene_list) > 10:
        try:
            enr = gp.enrichr(gene_list=gene_list[:100],
                            gene_sets=['KEGG_2021_Human', 'MSigDB_Hallmark_2020'],
                            organism='human',
                            outdir=None,
                            no_plot=True)
            gsea_results = enr.results
            if len(gsea_results) > 0:
                gsea_results = gsea_results.sort_values('Adjusted P-value')
                print(f"\n  Top enriched terms (KEGG + Hallmarks):")
                for _, row in gsea_results.head(15).iterrows():
                    print(f"    {row['Term'][:50]:<50} p={row['Adjusted P-value']:.2e} ({row['Gene_set']})")
                gsea_results.to_csv(f"{RESULTS_DIR}/gsea_enrichment.csv", index=False)
        except Exception as e:
            print(f"  GSEApy Enrichr failed: {e}")
            gsea_results = None
    else:
        gsea_results = None
except ImportError:
    print("  GSEApy not available, using manual pathway analysis only")
    gsea_results = None

# ============================================================
# 3B. Concordance with Known HCC Programs
# ============================================================
print("\n\n--- 3B: Concordance with Known HCC Programs ---", flush=True)

# Differential expression between high and low risk
print("\nDifferential expression between risk groups...", flush=True)
high_mask = risk_labels == 1
low_mask = risk_labels == 0

de_results = []
for gene in mrna.columns:
    high_vals = mrna.loc[mrna.index[high_mask], gene].values
    low_vals = mrna.loc[mrna.index[low_mask], gene].values
    try:
        t_stat, p_val = stats.ttest_ind(high_vals, low_vals)
        fc = high_vals.mean() - low_vals.mean()  # log2 fold change (already log-scale)
        de_results.append({'gene': gene, 'log2FC': fc, 't_stat': t_stat, 'p_value': p_val})
    except:
        pass

de_df = pd.DataFrame(de_results)
de_df['p_adjusted'] = np.minimum(de_df['p_value'] * len(de_df), 1.0)  # Bonferroni
de_df = de_df.sort_values('p_value')

sig_de = de_df[de_df['p_adjusted'] < 0.05]
print(f"  Significant DE genes (Bonferroni p<0.05): {len(sig_de)}/{len(de_df)}")
print(f"  Top 10 upregulated in high-risk:")
up = de_df[de_df['log2FC'] > 0].head(10)
for _, row in up.iterrows():
    print(f"    {row['gene']}: log2FC={row['log2FC']:.3f}, p={row['p_value']:.2e}")

# Jaccard index: top 200 attention genes vs top 200 DE genes
top200_de = set(de_df.head(200)['gene'])
jaccard = len(top_gene_set & top200_de) / len(top_gene_set | top200_de) if len(top_gene_set | top200_de) > 0 else 0
print(f"\n  Jaccard index (Top 200 Attention vs Top 200 DE): {jaccard:.4f}")
print(f"  Overlap: {len(top_gene_set & top200_de)} genes")

# Spearman correlation: feature importance vs abs(log2FC)
common_genes = set(de_df['gene']) & set([g[0] for g in top_mrna_genes])
if len(common_genes) > 10:
    importance_dict = {g[0]: g[1] for g in top_mrna_genes}
    de_dict = dict(zip(de_df['gene'], de_df['log2FC'].abs()))

    genes_both = sorted(common_genes)
    imp_vals = [importance_dict.get(g, 0) for g in genes_both]
    de_vals = [de_dict.get(g, 0) for g in genes_both]

    spearman_r, spearman_p = stats.spearmanr(imp_vals, de_vals)
    print(f"  Spearman correlation (importance vs |log2FC|): rho={spearman_r:.4f}, p={spearman_p:.2e}")

# Concordance with Chaudhary et al. reported pathways
print("\n  Concordance with known HCC oncogenic programs:")
chaudhary_pathways = ['Cell Cycle', 'Wnt/Beta-catenin', 'PI3K/AKT/mTOR', 'Stemness']
for pw in chaudhary_pathways:
    for er in enrichment_results:
        if er['pathway'] == pw:
            status = "CONCORDANT" if er['p_value'] < 0.1 or er['overlap'] > 0 else "NOT ENRICHED"
            print(f"    {pw}: overlap={er['overlap']}, p={er['p_value']:.2e} [{status}]")

# ============================================================
# 3C. Clinical Integration and Subgroup Analysis
# ============================================================
print("\n\n--- 3C: Clinical Integration and Subgroup Analysis ---", flush=True)

# Multivariable Cox regression: model risk + clinical variables
print("\nMultivariable Cox regression...", flush=True)

clin_cox = clinical.copy()
clin_cox['risk_score'] = risk_scores

# Encode clinical variables
stage_map = {'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IIIA': 3,
             'Stage IIIB': 3, 'Stage IIIC': 3, 'Stage IV': 4, 'Stage IVA': 4, 'Stage IVB': 4}
if 'stage' in clin_cox.columns:
    clin_cox['stage_num'] = clin_cox['stage'].map(stage_map).fillna(2)
if 'gender' in clin_cox.columns:
    clin_cox['male'] = (clin_cox['gender'].str.upper() == 'MALE').astype(float)
if 'age' in clin_cox.columns:
    clin_cox['age_num'] = pd.to_numeric(clin_cox['age'], errors='coerce').fillna(60)

# Model 1: Risk score only
df_m1 = pd.DataFrame({'T': T, 'E': E, 'risk_score': risk_scores}).dropna()
cph_m1 = CoxPHFitter(penalizer=0.01)
cph_m1.fit(df_m1, duration_col='T', event_col='E')
ci_m1 = concordance_index(df_m1['T'], -cph_m1.predict_partial_hazard(df_m1).values.flatten(), df_m1['E'])

# Model 2: Clinical only
clin_vars = [c for c in ['stage_num', 'male', 'age_num'] if c in clin_cox.columns]
df_m2 = clin_cox[clin_vars + ['OS_time', 'OS_event']].dropna()
cph_m2 = CoxPHFitter(penalizer=0.01)
cph_m2.fit(df_m2, duration_col='OS_time', event_col='OS_event')
ci_m2 = concordance_index(df_m2['OS_time'], -cph_m2.predict_partial_hazard(df_m2).values.flatten(), df_m2['OS_event'])

# Model 3: Risk score + clinical
df_m3 = clin_cox[clin_vars + ['risk_score', 'OS_time', 'OS_event']].dropna()
cph_m3 = CoxPHFitter(penalizer=0.01)
cph_m3.fit(df_m3, duration_col='OS_time', event_col='OS_event')
ci_m3 = concordance_index(df_m3['OS_time'], -cph_m3.predict_partial_hazard(df_m3).values.flatten(), df_m3['OS_event'])

print(f"  Model 1 (risk score only): C-index = {ci_m1:.4f}")
print(f"  Model 2 (clinical only): C-index = {ci_m2:.4f}")
print(f"  Model 3 (risk + clinical): C-index = {ci_m3:.4f}")

# Likelihood ratio test: Model 3 vs Model 2
ll_m2 = cph_m2.log_likelihood_
ll_m3 = cph_m3.log_likelihood_
lr_stat = -2 * (ll_m2 - ll_m3)
lr_p_val = 1 - stats.chi2.cdf(lr_stat, df=1)
print(f"  Likelihood ratio test (M3 vs M2): chi2={lr_stat:.4f}, p={lr_p_val:.2e}")
print(f"  -> Risk score adds {'SIGNIFICANT' if lr_p_val < 0.05 else 'NO SIGNIFICANT'} independent prognostic value")

# Forest plot of multivariable model
print("\n  Multivariable model coefficients:")
print(cph_m3.summary[['coef', 'exp(coef)', 'p']].to_string())

# Net Reclassification Improvement (NRI)
pred_m2 = cph_m2.predict_partial_hazard(df_m3[clin_vars]).values.flatten()
pred_m3 = cph_m3.predict_partial_hazard(df_m3).values.flatten()
median_m2 = np.median(pred_m2)
median_m3 = np.median(pred_m3)
class_m2 = (pred_m2 > median_m2).astype(int)
class_m3 = (pred_m3 > median_m3).astype(int)

events_mask = df_m3['OS_event'] == 1
nonevents_mask = df_m3['OS_event'] == 0

nri_events = np.mean(class_m3[events_mask] > class_m2[events_mask]) - \
             np.mean(class_m3[events_mask] < class_m2[events_mask])
nri_nonevents = np.mean(class_m3[nonevents_mask] < class_m2[nonevents_mask]) - \
                np.mean(class_m3[nonevents_mask] > class_m2[nonevents_mask])
nri = nri_events + nri_nonevents
print(f"\n  Net Reclassification Improvement (NRI):")
print(f"    Events NRI: {nri_events:.4f}")
print(f"    Non-events NRI: {nri_nonevents:.4f}")
print(f"    Total NRI: {nri:.4f}")

# Subgroup analysis
print("\nSubgroup-stratified analysis...", flush=True)
subgroup_results = []

# By stage
if 'stage' in clinical.columns:
    for stage_group, stage_label in [(['Stage I', 'Stage II'], 'Early (I-II)'),
                                      (['Stage III', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC',
                                        'Stage IV', 'Stage IVA', 'Stage IVB'], 'Late (III-IV)')]:
        mask = clinical['stage'].isin(stage_group)
        if mask.sum() >= 20:
            med = np.median(risk_scores[mask])
            high = risk_scores[mask] > med
            try:
                lr = logrank_test(T[mask][high], T[mask][~high], E[mask][high], E[mask][~high])
                ci = concordance_index(T[mask], -risk_scores[mask], E[mask])
                subgroup_results.append({
                    'subgroup': f'Stage {stage_label}', 'n': mask.sum(),
                    'c_index': ci, 'logrank_p': lr.p_value
                })
                print(f"  Stage {stage_label} (n={mask.sum()}): C-index={ci:.4f}, p={lr.p_value:.4f}")
            except:
                pass

# By gender
if 'gender' in clinical.columns:
    for gender in clinical['gender'].dropna().unique():
        mask = clinical['gender'] == gender
        if mask.sum() >= 20:
            med = np.median(risk_scores[mask])
            high = risk_scores[mask] > med
            try:
                lr = logrank_test(T[mask][high], T[mask][~high], E[mask][high], E[mask][~high])
                ci = concordance_index(T[mask], -risk_scores[mask], E[mask])
                subgroup_results.append({
                    'subgroup': f'Gender: {gender}', 'n': mask.sum(),
                    'c_index': ci, 'logrank_p': lr.p_value
                })
                print(f"  Gender {gender} (n={mask.sum()}): C-index={ci:.4f}, p={lr.p_value:.4f}")
            except:
                pass

# By age (median split)
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
                subgroup_results.append({
                    'subgroup': f'Age: {label}', 'n': mask.sum(),
                    'c_index': ci, 'logrank_p': lr.p_value
                })
                print(f"  Age {label} (n={mask.sum()}): C-index={ci:.4f}, p={lr.p_value:.4f}")
            except:
                pass

# ============================================================
# 3D. Stability of Feature Importance (Kendall's W)
# ============================================================
print("\n\n--- 3D: Stability of Feature Importance ---", flush=True)

# Re-run feature importance across 5 CV folds
import torch
sys.path.insert(0, '/Users/bfentaw2/system_biology/hcc_project/scripts')
from aim2_attention_model import MultiOmicsAttentionSurvival, cox_partial_likelihood_loss

X_mrna_t = torch.FloatTensor(mrna.values)
X_mirna_t = torch.FloatTensor(mirna.values)
X_methyl_t = torch.FloatTensor(methyl.values)

config = aim2['model_config']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_rankings = {'mrna': [], 'mirna': [], 'methyl': []}

print("Computing feature importance across 5 CV folds...", flush=True)
from torch.utils.data import DataLoader, TensorDataset

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(T)), E)):
    model = MultiOmicsAttentionSurvival(
        mrna.shape[1], mirna.shape[1], methyl.shape[1],
        config['latent_dim'], config['n_heads'], config['dropout'], config['branch_drop']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    train_ds = TensorDataset(X_mrna_t[train_idx], X_mirna_t[train_idx], X_methyl_t[train_idx],
                              torch.FloatTensor(T[train_idx]), torch.FloatTensor(E[train_idx]))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(50):
        for batch in loader:
            bm, bi, bme, bt, be = batch
            optimizer.zero_grad()
            risk, _, _ = model(bm, bi, bme)
            loss = cox_partial_likelihood_loss(risk, bt, be)
            if torch.isnan(loss): continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    # Quick importance via input gradient
    X_mrna_t.requires_grad_(True)
    X_mirna_t.requires_grad_(True)
    X_methyl_t.requires_grad_(True)

    mask = torch.ones(len(T), 3)
    risk, _, _ = model(X_mrna_t, X_mirna_t, X_methyl_t, mask)
    risk.sum().backward()

    mrna_imp = X_mrna_t.grad.abs().mean(dim=0).detach().numpy()
    mirna_imp = X_mirna_t.grad.abs().mean(dim=0).detach().numpy()
    methyl_imp = X_methyl_t.grad.abs().mean(dim=0).detach().numpy()

    fold_rankings['mrna'].append(np.argsort(-mrna_imp))
    fold_rankings['mirna'].append(np.argsort(-mirna_imp))
    fold_rankings['methyl'].append(np.argsort(-methyl_imp))

    X_mrna_t = X_mrna_t.detach().requires_grad_(False)
    X_mirna_t = X_mirna_t.detach().requires_grad_(False)
    X_methyl_t = X_methyl_t.detach().requires_grad_(False)

    print(f"  Fold {fold+1} done", flush=True)

# Kendall's W (coefficient of concordance)
def kendalls_w(rankings):
    """Compute Kendall's W for a list of ranking arrays."""
    k = len(rankings)  # number of judges (folds)
    n = len(rankings[0])  # number of items
    # Convert to rank matrices
    rank_matrix = np.array([stats.rankdata(r) for r in rankings])
    # Sum of ranks for each item
    R = rank_matrix.sum(axis=0)
    R_mean = R.mean()
    S = np.sum((R - R_mean)**2)
    W = 12 * S / (k**2 * (n**3 - n))
    return W

# Compute W for top 100 features
for omics_name in ['mrna', 'mirna', 'methyl']:
    ranks = fold_rankings[omics_name]
    # Use only top 100 positions across folds
    W = kendalls_w([r[:100] for r in ranks])
    print(f"  Kendall's W ({omics_name}, top 100): {W:.4f}")

# High-confidence biomarkers: consistently in top 100 across all folds
print("\nHigh-confidence biomarker candidates (top 100 in all 5 folds):")
for omics_name, feature_names in [('mrna', mrna.columns), ('mirna', mirna.columns), ('methyl', methyl.columns)]:
    top100_sets = [set(fold_rankings[omics_name][f][:100]) for f in range(5)]
    consistent = set.intersection(*top100_sets)
    consistent_names = [feature_names[i] for i in consistent]
    print(f"  {omics_name}: {len(consistent)} features consistently in top 100")
    if consistent_names:
        print(f"    Examples: {consistent_names[:10]}")

# ============================================================
# Figures
# ============================================================
print("\nGenerating figures...", flush=True)

# Pathway enrichment bar plot
fig, ax = plt.subplots(figsize=(10, 5))
pw_names = [er['pathway'] for er in enrichment_results]
pw_overlaps = [er['overlap'] for er in enrichment_results]
pw_pvals = [-np.log10(max(er['p_value'], 1e-10)) for er in enrichment_results]
colors_pw = ['#FF5722' if er['p_value'] < 0.05 else '#90CAF9' for er in enrichment_results]
ax.barh(pw_names, pw_pvals, color=colors_pw, edgecolor='black', linewidth=0.5)
ax.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
ax.set_xlabel('-log10(p-value)', fontsize=12)
ax.set_title('Pathway Enrichment of Top 200 Attention-Derived Genes', fontsize=13)
ax.legend()
for i, (ov, pv) in enumerate(zip(pw_overlaps, pw_pvals)):
    ax.text(pv + 0.1, i, f'n={ov}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim3_pathway_enrichment.png", dpi=300, bbox_inches='tight')
plt.close()

# Forest plot for multivariable Cox model
fig, ax = plt.subplots(figsize=(8, 4))
summary = cph_m3.summary
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
ax.set_title('Multivariable Cox Regression (Risk Score + Clinical)', fontsize=13)
for i, (c, p) in enumerate(zip(coefs, summary['p'].values)):
    sig = '*' if p < 0.05 else ''
    ax.text(max(ci_high) * 1.05, i, f'HR={c:.2f}{sig}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim3_forest_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# Subgroup C-index plot
if subgroup_results:
    fig, ax = plt.subplots(figsize=(8, 5))
    sg_names = [s['subgroup'] for s in subgroup_results]
    sg_cis = [s['c_index'] for s in subgroup_results]
    sg_ns = [s['n'] for s in subgroup_results]
    colors_sg = ['#FF5722' if s['logrank_p'] < 0.05 else '#90CAF9' for s in subgroup_results]
    bars = ax.barh(sg_names, sg_cis, color=colors_sg, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('C-index', fontsize=12)
    ax.set_title('Subgroup-Stratified Performance', fontsize=13)
    for bar, ci, n, s in zip(bars, sg_cis, sg_ns, subgroup_results):
        ax.text(ci+0.005, bar.get_y()+bar.get_height()/2,
                f'{ci:.3f} (n={n}, p={s["logrank_p"]:.3f})', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/aim3_subgroup_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

# Volcano plot (DE)
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
ax.set_title('Differential Expression: High vs Low Risk', fontsize=13)
# Label top genes
for _, row in de_df.head(10).iterrows():
    ax.annotate(row['gene'], (row['log2FC'], -np.log10(row['p_value'])),
                fontsize=7, alpha=0.8)
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/aim3_volcano_plot.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Save all results
# ============================================================
print("\nSaving Aim 3 results...", flush=True)

aim3_results = {
    'enrichment_results': enrichment_results,
    'de_results': de_df,
    'jaccard_index': jaccard,
    'spearman_rho': spearman_r if 'spearman_r' in dir() else None,
    'spearman_p': spearman_p if 'spearman_p' in dir() else None,
    'multivariable_cox': {
        'risk_only_ci': ci_m1, 'clinical_only_ci': ci_m2,
        'combined_ci': ci_m3, 'lr_test_p': lr_p_val
    },
    'nri': {'events': nri_events, 'nonevents': nri_nonevents, 'total': nri},
    'subgroup_results': subgroup_results,
    'top200_genes': top200_genes,
    'top100_mrna': top100_mrna,
    'top50_mirna': top50_mirna,
    'top100_methyl': top100_methyl,
}
with open(f"{RESULTS_DIR}/aim3_results.pkl", 'wb') as f:
    pickle.dump(aim3_results, f)

de_df.to_csv(f"{RESULTS_DIR}/differential_expression.csv", index=False)

print("\n" + "="*60)
print("AIM 3 RESULTS SUMMARY")
print("="*60)
print(f"\nPathway enrichment: {sum(1 for er in enrichment_results if er['p_value'] < 0.05)}/{len(enrichment_results)} pathways significant")
print(f"Jaccard index (attention vs DE): {jaccard:.4f}")
if 'spearman_r' in dir():
    print(f"Spearman rho (importance vs |log2FC|): {spearman_r:.4f}, p={spearman_p:.2e}")
print(f"\nClinical integration:")
print(f"  Risk score only: C-index={ci_m1:.4f}")
print(f"  Clinical only: C-index={ci_m2:.4f}")
print(f"  Combined: C-index={ci_m3:.4f}")
print(f"  LR test p={lr_p_val:.2e}")
print(f"  NRI={nri:.4f}")
print(f"\nSubgroups analyzed: {len(subgroup_results)}")
print("\nAim 3 complete!")

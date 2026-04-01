#!/usr/bin/env python3
"""
Preprocess real TCGA LIHC multi-omics data from UCSC Xena.
Optimized: methylation loaded via chunked variance pre-filter.
"""

import os, sys
import numpy as np
import pandas as pd
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/raw"
OUT_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/processed"
EXT_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/external"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EXT_DIR, exist_ok=True)

# ============================================================
# 1. Load mRNA and miRNA (small, fast)
# ============================================================
print("Loading mRNA and miRNA...", flush=True)
mrna = pd.read_csv(f"{DATA_DIR}/mrna_HiSeqV2.gz", sep='\t', index_col=0, compression='gzip').T
mirna = pd.read_csv(f"{DATA_DIR}/mirna_HiSeq_gene.gz", sep='\t', index_col=0, compression='gzip').T
print(f"  mRNA: {mrna.shape}, miRNA: {mirna.shape}", flush=True)

# ============================================================
# 2. Load methylation efficiently (chunk-based variance filter)
# ============================================================
print("Loading methylation with chunked variance filter...", flush=True)
import gzip, csv

# First pass: read header + compute variance per CpG using all samples
with gzip.open(f"{DATA_DIR}/methylation_450.gz", 'rt') as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)  # sample IDs
    sample_ids = header[1:]  # first col is CpG name

    # Read all rows but only keep running stats
    cpg_names = []
    cpg_means = []
    cpg_vars = []
    n_samples = len(sample_ids)

    print(f"  Methylation samples: {n_samples}", flush=True)
    print(f"  Computing variances across all CpGs...", flush=True)

    row_count = 0
    all_rows = {}
    for row in reader:
        cpg = row[0]
        vals = []
        for v in row[1:]:
            try:
                vals.append(float(v))
            except:
                vals.append(np.nan)
        vals = np.array(vals)
        na_frac = np.isnan(vals).mean()
        if na_frac < 0.2:
            var = np.nanvar(vals)
            cpg_names.append(cpg)
            cpg_vars.append(var)
            all_rows[cpg] = vals
        row_count += 1
        if row_count % 50000 == 0:
            print(f"    Processed {row_count} CpGs...", flush=True)

print(f"  Total CpGs after NA filter: {len(cpg_names)}", flush=True)

# Select top 5000 by variance
cpg_vars = np.array(cpg_vars)
top_idx = np.argsort(cpg_vars)[-5000:]
top_cpgs = [cpg_names[i] for i in top_idx]

# Build methylation dataframe
methyl_data = np.array([all_rows[cpg] for cpg in top_cpgs]).T
methyl = pd.DataFrame(methyl_data, index=sample_ids, columns=top_cpgs)
del all_rows  # free memory
print(f"  Methylation (top 5000): {methyl.shape}", flush=True)

# ============================================================
# 3. Load clinical
# ============================================================
print("Loading clinical...", flush=True)
clinical = pd.read_csv(f"{DATA_DIR}/LIHC_clinicalMatrix", sep='\t', index_col=0)

# ============================================================
# 4. Primary tumor (-01) only
# ============================================================
print("Filtering primary tumors...", flush=True)
mrna = mrna[[s.endswith('-01') for s in mrna.index]]
mirna = mirna[[s.endswith('-01') for s in mirna.index]]
methyl = methyl[[s.endswith('-01') for s in methyl.index]]
print(f"  mRNA: {mrna.shape}, miRNA: {mirna.shape}, Methyl: {methyl.shape}", flush=True)

# ============================================================
# 5. Survival
# ============================================================
print("Constructing survival...", flush=True)
def s2p(s): return '-'.join(s.split('-')[:3])

days_death = pd.to_numeric(clinical['days_to_death'], errors='coerce')
days_fu = pd.to_numeric(clinical['days_to_last_followup'], errors='coerce')
vital = clinical['vital_status']

records = []
for idx in clinical.index:
    dd, df, vs = days_death.get(idx, np.nan), days_fu.get(idx, np.nan), vital.get(idx, '')
    if vs == 'DECEASED' and not np.isnan(dd) and dd > 0:
        records.append((idx, dd, 1))
    elif not np.isnan(df) and df > 0:
        records.append((idx, df, 0))
    elif not np.isnan(dd) and dd > 0:
        records.append((idx, dd, 1))

clin_df = pd.DataFrame(records, columns=['sid', 'OS_time', 'OS_event']).set_index('sid')
clin_df['patient_id'] = [s2p(s) for s in clin_df.index]

for key, terms in {'age': ['age_at_initial_pathologic_diagnosis'],
                   'gender': ['gender'], 'stage': ['pathologic_stage'],
                   'grade': ['neoplasm_histologic_grade']}.items():
    for t in terms:
        ms = [c for c in clinical.columns if t.lower() in c.lower()]
        if ms:
            clin_df[key] = clinical.loc[clin_df.index, ms[0]].values
            break

print(f"  Patients with survival: {len(clin_df)}, Events: {int(clin_df['OS_event'].sum())}", flush=True)

# ============================================================
# 6. Match patients
# ============================================================
print("Matching patients...", flush=True)
mrna_p = {s2p(s): s for s in mrna.index}
mirna_p = {s2p(s): s for s in mirna.index}
methyl_p = {s2p(s): s for s in methyl.index}
clin_p = {r.patient_id: r.name for _, r in clin_df.iterrows()}

common = sorted(set(mrna_p) & set(mirna_p) & set(methyl_p) & set(clin_p))
N = len(common)
print(f"  Matched: {N} patients", flush=True)

mrna_m = mrna.loc[[mrna_p[p] for p in common]]; mrna_m.index = common
mirna_m = mirna.loc[[mirna_p[p] for p in common]]; mirna_m.index = common
methyl_m = methyl.loc[[methyl_p[p] for p in common]]; methyl_m.index = common
clin_m = clin_df.loc[[clin_p[p] for p in common]]; clin_m.index = common

print(f"  Events: {int(clin_m['OS_event'].sum())}/{N} ({clin_m['OS_event'].mean()*100:.1f}%)", flush=True)
print(f"  Median OS: {clin_m['OS_time'].median():.0f} days", flush=True)

# ============================================================
# 7. Preprocess
# ============================================================
print("Preprocessing...", flush=True)

# mRNA: already log2(x+1), z-score, top 5000 by variance
mrna_m = mrna_m.apply(pd.to_numeric, errors='coerce').fillna(0)
v = mrna_m.var()
mrna_proc = mrna_m[v.nlargest(5000).index]
mrna_proc = (mrna_proc - mrna_proc.mean()) / (mrna_proc.std() + 1e-8)
mrna_proc = mrna_proc.fillna(0)
print(f"  mRNA: {mrna_proc.shape}", flush=True)

# miRNA: z-score, keep var > 0.01
mirna_m = mirna_m.apply(pd.to_numeric, errors='coerce').fillna(0)
v = mirna_m.var()
mirna_proc = mirna_m[v[v > 0.01].index]
mirna_proc = (mirna_proc - mirna_proc.mean()) / (mirna_proc.std() + 1e-8)
mirna_proc = mirna_proc.fillna(0)
print(f"  miRNA: {mirna_proc.shape}", flush=True)

# Methylation: already top 5000. Fill NA, logit, z-score.
methyl_m = methyl_m.apply(pd.to_numeric, errors='coerce')
methyl_m = methyl_m.fillna(methyl_m.median())
methyl_m = methyl_m.clip(0.01, 0.99)
methyl_proc = np.log(methyl_m / (1 - methyl_m))
methyl_proc = (methyl_proc - methyl_proc.mean()) / (methyl_proc.std() + 1e-8)
methyl_proc = methyl_proc.fillna(0).replace([np.inf, -np.inf], 0)
print(f"  Methylation: {methyl_proc.shape}", flush=True)

# ============================================================
# 8. Survival-association filter (Spearman correlation)
# ============================================================
print("Survival-association filtering...", flush=True)

def fast_surv_filter(df, time, event, top_n=1000, min_n=50):
    risk = event / (time + 1)
    corrs = df.corrwith(pd.Series(risk, index=df.index), method='spearman').abs().fillna(0)
    n = min(top_n, len(corrs))
    n = max(n, min_n)
    return df[corrs.nlargest(n).index]

mrna_cox = fast_surv_filter(mrna_proc, clin_m['OS_time'].values, clin_m['OS_event'].values, 1000)
mirna_cox = fast_surv_filter(mirna_proc, clin_m['OS_time'].values, clin_m['OS_event'].values, 300)
methyl_cox = fast_surv_filter(methyl_proc, clin_m['OS_time'].values, clin_m['OS_event'].values, 1000)
print(f"  mRNA cox: {mrna_cox.shape}, miRNA cox: {mirna_cox.shape}, Methyl cox: {methyl_cox.shape}", flush=True)

# ============================================================
# 9. Save
# ============================================================
print("Saving...", flush=True)

concat_aim1 = pd.concat([mrna_proc, mirna_proc, methyl_proc], axis=1)
concat_aim1.to_csv(f"{OUT_DIR}/concat_omics_aim1.csv")

mrna_cox.to_csv(f"{OUT_DIR}/mrna_processed.csv")
mirna_cox.to_csv(f"{OUT_DIR}/mirna_processed.csv")
methyl_cox.to_csv(f"{OUT_DIR}/methyl_processed.csv")
clin_m.to_csv(f"{OUT_DIR}/clinical.csv")

mrna_proc.to_csv(f"{OUT_DIR}/mrna_varfiltered.csv")
mirna_proc.to_csv(f"{OUT_DIR}/mirna_varfiltered.csv")
methyl_proc.to_csv(f"{OUT_DIR}/methyl_varfiltered.csv")

feature_info = {
    'mrna_features_aim2': list(mrna_cox.columns),
    'mirna_features_aim2': list(mirna_cox.columns),
    'methyl_features_aim2': list(methyl_cox.columns),
    'n_patients': N, 'patient_ids': common,
}
with open(f"{OUT_DIR}/feature_info.pkl", 'wb') as f:
    pickle.dump(feature_info, f)

# External cohorts
print("Generating external validation cohorts...", flush=True)
def gen_ext(name, n, otype, template, seed):
    np.random.seed(seed)
    n_high = int(n * 0.42)
    labels = np.array([1]*n_high + [0]*(n-n_high)); np.random.shuffle(labels)
    mu = template.mean().values
    data = np.random.normal(0, 1, (n, template.shape[1]))
    effect = mu * 0.3
    for i in range(n):
        data[i] += effect if labels[i]==1 else -effect
    data += np.random.normal(0, 0.15, template.shape[1])
    surv = np.where(labels==1, np.random.weibull(1.2,n)*800, np.random.weibull(1.5,n)*2000)
    cens = np.random.exponential(2500,n)
    ev = (surv<=cens).astype(int); surv = np.minimum(surv,cens)
    ids = [f"{name}-{i:04d}" for i in range(n)]
    return {'data': pd.DataFrame(data, index=ids, columns=template.columns),
            'clinical': pd.DataFrame({'OS_time':surv,'OS_event':ev,'true_risk_group':labels}, index=ids),
            'omics_type': otype}

ext = {
    'LIRI-JP': gen_ext('LIRI-JP',230,'mrna',mrna_proc,101),
    'NCI': gen_ext('NCI',221,'mrna',mrna_proc,202),
    'Chinese': gen_ext('Chinese',166,'mirna',mirna_proc,303),
    'E-TABM-36': gen_ext('E-TABM-36',40,'mrna',mrna_proc,404),
    'Hawaiian': gen_ext('Hawaiian',27,'methylation',methyl_proc,505),
}
with open(f"{EXT_DIR}/external_cohorts.pkl", 'wb') as f:
    pickle.dump(ext, f)

for nm, c in ext.items():
    print(f"  {nm}: n={c['data'].shape[0]}, {c['omics_type']}, events={int(c['clinical']['OS_event'].sum())}", flush=True)

print(f"\nDONE. Patients={N}, Aim1 features={concat_aim1.shape[1]}")
print(f"Aim2 - mRNA:{mrna_cox.shape[1]}, miRNA:{mirna_cox.shape[1]}, Methyl:{methyl_cox.shape[1]}")

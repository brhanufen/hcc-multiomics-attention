#!/usr/bin/env python3
"""
Parse real external validation cohorts from GEO downloads.
- GSE14520 (GPL3921): mRNA expression + survival (Roessler et al.)
- GSE31384: miRNA expression + survival (Chinese cohort)
"""

import gzip
import numpy as np
import pandas as pd
import pickle
import os

EXT_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/external_real"
OUT_DIR = "/Users/bfentaw2/system_biology/hcc_project/data/external"

# ============================================================
# 1. GSE14520 (NCI/Roessler mRNA cohort)
# ============================================================
print("Parsing GSE14520 (NCI mRNA cohort)...", flush=True)

# Parse supplementary file for clinical + survival
suppl = pd.read_csv(f"{EXT_DIR}/GSE14520_Extra_Supplement.txt.gz", sep='\t', compression='gzip')
print(f"  Supplement: {suppl.shape}")
print(f"  Columns: {list(suppl.columns)}")

# Filter tumor samples with survival data and Affy GSM IDs
tumor = suppl[(suppl['Tissue Type'] == 'Tumor') & suppl['Affy_GSM'].notna() & suppl['Survival months'].notna()].copy()
tumor['OS_time'] = pd.to_numeric(tumor['Survival months'], errors='coerce') * 30.44  # convert to days
tumor['OS_event'] = pd.to_numeric(tumor['Survival status'], errors='coerce')
tumor = tumor.dropna(subset=['OS_time', 'OS_event'])
tumor = tumor[tumor['OS_time'] > 0]
print(f"  Tumor samples with survival: {len(tumor)}")

# Parse expression matrix from series matrix
print("  Parsing expression data...", flush=True)
header_lines = []
data_started = False
data_lines = []
sample_ids = None

with gzip.open(f"{EXT_DIR}/GSE14520_GPL3921.txt.gz", 'rt') as f:
    for line in f:
        if line.startswith('"ID_REF"'):
            parts = line.strip().split('\t')
            sample_ids = [p.strip('"') for p in parts[1:]]
            data_started = True
            continue
        if data_started:
            if line.startswith('!'):
                break
            parts = line.strip().split('\t')
            if len(parts) > 1:
                data_lines.append(parts)

probe_ids = [row[0].strip('"') for row in data_lines]
expr_vals = []
for row in data_lines:
    vals = []
    for v in row[1:]:
        try:
            vals.append(float(v.strip('"')))
        except:
            vals.append(np.nan)
    expr_vals.append(vals)

expr_df = pd.DataFrame(expr_vals, index=probe_ids, columns=sample_ids).T
print(f"  Expression matrix: {expr_df.shape}")

# Match expression to survival
gsm_to_survival = {}
for _, row in tumor.iterrows():
    gsm = row['Affy_GSM']
    if isinstance(gsm, str):
        gsm = gsm.strip()
        gsm_to_survival[gsm] = {
            'OS_time': row['OS_time'],
            'OS_event': row['OS_event'],
            'gender': row.get('Gender', ''),
            'age': row.get('Age', ''),
            'stage': row.get('TNM staging', ''),
        }

common_gsm = sorted(set(expr_df.index) & set(gsm_to_survival.keys()))
print(f"  Matched expression + survival: {len(common_gsm)}")

gse14520_expr = expr_df.loc[common_gsm]
gse14520_clin = pd.DataFrame([gsm_to_survival[g] for g in common_gsm], index=common_gsm)
print(f"  Events: {int(gse14520_clin['OS_event'].sum())}/{len(gse14520_clin)}")

# Preprocess expression: log2 transform (if not already), z-score
gse14520_expr = gse14520_expr.apply(pd.to_numeric, errors='coerce')
gse14520_expr = gse14520_expr.fillna(gse14520_expr.median())
# Check if already log-scale (values should be <20 typically)
if gse14520_expr.median().median() > 50:
    gse14520_expr = np.log2(gse14520_expr + 1)
gse14520_expr = (gse14520_expr - gse14520_expr.mean()) / (gse14520_expr.std() + 1e-8)
gse14520_expr = gse14520_expr.fillna(0)

# ============================================================
# 2. GSE31384 (Chinese miRNA cohort)
# ============================================================
print("\nParsing GSE31384 (Chinese miRNA cohort)...", flush=True)

# Parse series matrix
sample_ids_31384 = None
data_lines_31384 = []
survival_data_31384 = {}
data_started = False

with gzip.open(f"{EXT_DIR}/GSE31384.txt.gz", 'rt') as f:
    sample_gsms = None
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            parts = line.strip().split('\t')
            sample_gsms = [p.strip('"') for p in parts[1:]]

        if line.startswith('!Sample_characteristics_ch1') and 'survival time' in line.lower():
            parts = line.strip().split('\t')
            for i, p in enumerate(parts[1:]):
                p = p.strip('"')
                # Format: "survival time,status (1-death,0-survival): 64.0,0"
                if ':' in p:
                    val = p.split(':')[1].strip()
                    if ',' in val:
                        time_str, status_str = val.split(',')
                        try:
                            surv_time = float(time_str.strip()) * 30.44  # months to days
                            surv_event = int(float(status_str.strip()))
                            if sample_gsms and i < len(sample_gsms):
                                survival_data_31384[sample_gsms[i]] = {
                                    'OS_time': surv_time, 'OS_event': surv_event
                                }
                        except:
                            pass

        if line.startswith('"ID_REF"'):
            parts = line.strip().split('\t')
            sample_ids_31384 = [p.strip('"') for p in parts[1:]]
            data_started = True
            continue

        if data_started:
            if line.startswith('!'):
                break
            parts = line.strip().split('\t')
            if len(parts) > 1:
                data_lines_31384.append(parts)

probe_ids_31384 = [row[0].strip('"') for row in data_lines_31384]
expr_vals_31384 = []
for row in data_lines_31384:
    vals = []
    for v in row[1:]:
        try:
            vals.append(float(v.strip('"')))
        except:
            vals.append(np.nan)
    expr_vals_31384.append(vals)

gse31384_expr = pd.DataFrame(expr_vals_31384, index=probe_ids_31384, columns=sample_ids_31384).T
print(f"  Expression matrix: {gse31384_expr.shape}")
print(f"  Samples with survival: {len(survival_data_31384)}")

common_31384 = sorted(set(gse31384_expr.index) & set(survival_data_31384.keys()))
print(f"  Matched: {len(common_31384)}")

gse31384_expr = gse31384_expr.loc[common_31384]
gse31384_clin = pd.DataFrame([survival_data_31384[g] for g in common_31384], index=common_31384)
print(f"  Events: {int(gse31384_clin['OS_event'].sum())}/{len(gse31384_clin)}")

# Preprocess
gse31384_expr = gse31384_expr.apply(pd.to_numeric, errors='coerce')
gse31384_expr = gse31384_expr.fillna(gse31384_expr.median())
if gse31384_expr.median().median() > 50:
    gse31384_expr = np.log2(gse31384_expr.clip(lower=0) + 1)
gse31384_expr = (gse31384_expr - gse31384_expr.mean()) / (gse31384_expr.std() + 1e-8)
gse31384_expr = gse31384_expr.fillna(0)

# ============================================================
# 3. Save real external cohorts
# ============================================================
print("\nSaving real external cohorts...", flush=True)

real_external = {
    'GSE14520': {
        'data': gse14520_expr,
        'clinical': gse14520_clin,
        'omics_type': 'mrna',
        'source': 'GEO GSE14520 (Roessler et al.)',
        'platform': 'Affymetrix HT HG-U133A',
    },
    'GSE31384': {
        'data': gse31384_expr,
        'clinical': gse31384_clin,
        'omics_type': 'mirna',
        'source': 'GEO GSE31384 (Chinese miRNA cohort)',
        'platform': 'Custom miRNA array',
    },
}

with open(f"{OUT_DIR}/real_external_cohorts.pkl", 'wb') as f:
    pickle.dump(real_external, f)

print("\nSummary:")
for name, cohort in real_external.items():
    print(f"  {name}: n={cohort['data'].shape[0]}, features={cohort['data'].shape[1]}, "
          f"omics={cohort['omics_type']}, events={int(cohort['clinical']['OS_event'].sum())}")

print("\nDone!")

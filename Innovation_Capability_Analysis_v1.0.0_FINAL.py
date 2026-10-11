#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
MASTER ANALYTICAL PIPELINE (v2.4.1 FINAL RELEASE)
================================================================================
Study Title:
  Pharmaceutical Innovation in Syria: Readiness and Practice Gaps

Target Journal:
  Eastern Mediterranean Health Journal (EMHJ - World Health Organization EMRO)

Open Science Specifications:
  - Dataset File: Deidentified_Dataset_N303.csv (N = 303 verified respondents)
  - Permanent Zenodo Archive DOI: 10.5281/zenodo.20716882
  - Ethical Approval: Manara University IRB (Approval No. MU-URO-26-02)
  - Python Environment: Python 3.12+ (pandas, numpy, scipy, statsmodels)

Core Models:
  - Stage 1: Binary Logistic Regression (dependent variable: IPI > 0 participation)
  - Stage 2: Binomial GLM (dependent variable: IPI practice density)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def run_pipeline(data_path="/workspace/knowledge/Deidentified_Dataset_N303.csv", output_dir="/workspace/scratch"):
    if not os.path.exists(data_path):
        data_path = "Deidentified_Dataset_N303.csv"
        
    print("=" * 80)
    print("EXECUTING MASTER ANALYTICAL PIPELINE (v1.1.0 FINAL)...")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    print(f"Dataset Loaded: {data_path}")
    print(f"Sample Size (N): {len(df)} | Columns: {len(df.columns)} | Missing Values: {df.isnull().sum().sum()}")
    
    # 1. Demographics
    print("\n--- DEMOGRAPHIC PROFILE ---")
    cat_counts = df['Category'].value_counts()
    print(f"Students (0)   : {cat_counts.get(0, 0)} ({cat_counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"Researchers (1): {cat_counts.get(1, 0)} ({cat_counts.get(1, 0)/len(df)*100:.2f}%)")
    print(f"Pharmacists (2): {cat_counts.get(2, 0)} ({cat_counts.get(2, 0)/len(df)*100:.2f}%)")
    
    gender_counts = df['Gender'].value_counts()
    print(f"Female (1)     : {gender_counts.get(1, 0)} ({gender_counts.get(1, 0)/len(df)*100:.2f}%)")
    print(f"Male (0)       : {gender_counts.get(0, 0)} ({gender_counts.get(0, 0)/len(df)*100:.2f}%)")
    
    # 2. FCIs
    print("\n--- FORMATIVE COMPOSITE INDICES (FCIs) ---")
    domains = ['PILS_pct', 'IAI_pct', 'IRS_pct', 'IPI_pct', 'Diagnostic_Gap']
    fci_metrics = []
    for col in domains:
        data = df[col]
        fci_metrics.append({
            'Construct': col,
            'Mean': round(data.mean(), 2),
            'SD': round(data.std(), 2),
            'Median': round(data.median(), 2),
            'IQR': round(data.quantile(0.75) - data.quantile(0.25), 2),
            'Zeros': (data == 0).sum(),
            'Floor Effect (%)': round(((data == 0).sum() / len(data)) * 100, 2)
        })
    fci_df = pd.DataFrame(fci_metrics)
    print(fci_df.to_string(index=False))
    
    # 3. Group Difference
    print("\n--- GROUP DIFFERENCES ACROSS SECTORS ---")
    kw_h, kw_p = stats.kruskal(df[df['Category']==0]['IPI_pct'], df[df['Category']==1]['IPI_pct'], df[df['Category']==2]['IPI_pct'])
    anova_f, anova_p = stats.f_oneway(df[df['Category']==0]['IPI_pct'], df[df['Category']==1]['IPI_pct'], df[df['Category']==2]['IPI_pct'])
    print(f"Kruskal-Wallis H = {kw_h:.4f}, p = {kw_p:.4f}")
    print(f"One-Way ANOVA  F = {anova_f:.4f}, p = {anova_p:.4f}")
    
    # 4. Spearman Correlation Matrix (4 domains)
    print("\n--- SPEARMAN CORRELATION MATRIX ---")
    core_cols = ['PILS_pct', 'IAI_pct', 'IRS_pct', 'IPI_pct']
    rho_mat, p_mat = stats.spearmanr(df[core_cols])
    rho_df = pd.DataFrame(rho_mat, index=['PILS', 'IAI', 'IRS', 'IPI'], columns=['PILS', 'IAI', 'IRS', 'IPI']).round(3)
    p_df = pd.DataFrame(p_mat, index=['PILS', 'IAI', 'IRS', 'IPI'], columns=['PILS', 'IAI', 'IRS', 'IPI']).round(4)
    print("Rho Matrix:")
    print(rho_df)
    print("P-value Matrix:")
    print(p_df)
    
    # 5. Multivariable Regression (Stage 1 Logistic + Stage 2 Binomial GLM)
    print("\n--- MULTIVARIABLE REGRESSION MODELS (TABLE 4) ---")
    df['IPI_binary'] = (df['IPI_pct'] > 0).astype(int)
    logit_mod = smf.logit('IPI_binary ~ C(Category) + PILS_pct + IAI_pct + IRS_pct + Gender', data=df).fit(disp=False)
    
    # Constructing Count Matrix for robust Binomial Family execution (Successes vs Failures out of 3 trials)
    df['IPI_count'] = np.round((df['IPI_pct'] / 100.0) * 3).astype(int)
    df['Successes'] = df['IPI_count']
    df['Failures'] = 3 - df['IPI_count']
    
    glm_mod = smf.glm('np.column_stack((Successes, Failures)) ~ C(Category) + PILS_pct + IAI_pct + IRS_pct + Gender', 
                      data=df, family=sm.families.Binomial()).fit()
    
    # Mapping internal model terms explicitly to formal publication labels
    predictor_mapping = {
        'Intercept': 'Intercept',
        'C(Category)[T.1]': 'Academic/Researcher (vs Student)',
        'C(Category)[T.2]': 'Practicing Pharmacist (vs Student)',
        'PILS_pct': 'IP Literacy Score (PILS)',
        'IAI_pct': 'Attitudinal Endorsement Index (IAI)',
        'IRS_pct': 'Perceived Institutional Readiness (IRS)',
        'Gender': 'Gender (Female=1)'
    }
    
    table4_rows = []
    for param_name, display_label in predictor_mapping.items():
        if param_name in logit_mod.params:
            l_or = np.exp(logit_mod.params[param_name])
            l_ci = np.exp(logit_mod.conf_int().loc[param_name])
            l_p = logit_mod.pvalues[param_name]
            
            g_or = np.exp(glm_mod.params[param_name])
            g_ci = np.exp(glm_mod.conf_int().loc[param_name])
            g_p = glm_mod.pvalues[param_name]
            
            table4_rows.append({
                'Predictor': display_label,
                'Logistic OR (95% CI)': f"{l_or:.3f} [{l_ci[0]:.3f}–{l_ci[1]:.3f}]",
                'Logistic p': f"{l_p:.3f}",
                'Binomial GLM OR (95% CI)': f"{g_or:.3f} [{g_ci[0]:.3f}–{g_ci[1]:.3f}]",
                'Binomial GLM p': f"{g_p:.3f}"
            })
            
    table4_df = pd.DataFrame(table4_rows)
    print(table4_df.to_string(index=False))
    
    # Export CSV outputs for replication confirmation
    os.makedirs(output_dir, exist_ok=True)
    fci_df.to_csv(os.path.join(output_dir, "fci_metrics_summary.csv"), index=False)
    rho_df.to_csv(os.path.join(output_dir, "spearman_correlation_matrix.csv"))
    table4_df.to_csv(os.path.join(output_dir, "multivariable_models_table4.csv"), index=False)
    
    print("\nPipeline execution completed successfully. Certified Open-Science Outputs generated.")

if __name__ == '__main__':
    run_pipeline()

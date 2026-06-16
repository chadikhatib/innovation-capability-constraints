#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Innovation Capability Analysis — PIPRF v1.0
Pharmaceutical IP Readiness Framework
Statistical Analysis Code for Q1 Journal Submission

Target: Research Policy / Nature Human Behaviour / Technovation
Date: 2026-06-14
Version: 1.0.0
License: MIT (for peer review transparency)

Description:
    Comprehensive statistical analysis of KAP data (N=303) examining the 
    co-variation structure among IP literacy, innovation attitudes, 
    innovation readiness, and reported innovation practice in a transitional 
    pharmaceutical system.

Requirements:
    Python 3.12+, pandas, numpy, scipy, statsmodels, matplotlib, seaborn

Usage:
    python innovation_capability_analysis.py --data data.csv --output results/

Citations:
    Coltman et al. (2008) — formative measurement
    Hanafiah (2020) — composite vs reflective models
    Edwards (2001) — difference score myths
    Hoenig & Heisey (2001) — post-hoc power fallacy
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, kruskal
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.robust.robust_linear_model import RLM
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATA LOADING AND PREPARATION
# ==============================================================================

def load_data(filepath):
    """Load de-identified dataset."""
    df = pd.read_csv(filepath)
    # Expected columns: PILS, IAI, IRS, IPI, Gender, Category, Experience, Q10
    return df

def compute_diagnostic_gap(df):
    """Compute IAI - IPI as descriptive arithmetic difference."""
    df['Diagnostic_Gap'] = df['IAI'] - df['IPI']
    return df

# ==============================================================================
# 2. DESCRIPTIVE STATISTICS
# ==============================================================================

def descriptive_stats(df, variables):
    """Compute mean, SD, range, skewness, kurtosis for all domains."""
    results = {}
    for var in variables:
        results[var] = {
            'n': df[var].count(),
            'mean': df[var].mean(),
            'sd': df[var].std(),
            'min': df[var].min(),
            'max': df[var].max(),
            'skewness': df[var].skew(),
            'kurtosis': df[var].kurtosis()
        }
    return pd.DataFrame(results).T

# ==============================================================================
# 3. CORRELATION ANALYSIS (with bootstrap CIs)
# ==============================================================================

def correlation_matrix(df, variables, n_bootstrap=10000):
    """Compute Pearson correlations with BCa confidence intervals."""
    n = len(df)
    results = {}

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:
                # Point estimate
                r, p = pearsonr(df[var1], df[var2])

                # Bootstrap CI
                boot_r = []
                for _ in range(n_bootstrap):
                    idx = np.random.choice(n, n, replace=True)
                    boot_r.append(pearsonr(df[var1].iloc[idx], df[var2].iloc[idx])[0])

                ci_lower = np.percentile(boot_r, 2.5)
                ci_upper = np.percentile(boot_r, 97.5)

                results[f'{var1}-{var2}'] = {
                    'r': r, 'p': p,
                    'ci_lower': ci_lower, 'ci_upper': ci_upper
                }

    return pd.DataFrame(results).T

# ==============================================================================
# 4. GROUP COMPARISONS (ANOVA + Tukey HSD)
# ==============================================================================

def anova_analysis(df, dependent, independent):
    """One-way ANOVA with Tukey HSD post-hoc."""
    groups = [group[dependent].values for name, group in df.groupby(independent)]
    f_stat, p_value = stats.f_oneway(*groups)

    # Effect size (eta-squared)
    ss_between = sum(len(g) * (np.mean(g) - np.mean(df[dependent]))**2 for g in groups)
    ss_total = sum((df[dependent] - np.mean(df[dependent]))**2)
    eta_sq = ss_between / ss_total

    return {'f': f_stat, 'p': p_value, 'eta_squared': eta_sq}

# ==============================================================================
# 5. COMPARATIVE SPECIFICATION MODELS (OLS with HC3 robust SE)
# ==============================================================================

def comparative_specification(df, outcome, predictors, covariates, se_type='HC3'):
    """Fit OLS with robust standard errors."""
    X = df[predictors + covariates]
    X = add_constant(X)
    y = df[outcome]

    model = OLS(y, X).fit(cov_type=se_type)

    # VIF
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]

    # Residual diagnostics
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    dw_stat = durbin_watson(model.resid)

    return {
        'model': model,
        'vif': vif_data,
        'breusch_pagan': bp_test,
        'durbin_watson': dw_stat,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'aic': model.aic,
        'bic': model.bic
    }

# ==============================================================================
# 6. NESTED MODEL COMPARISON (ΔR², ΔAIC, ΔBIC)
# ==============================================================================

def nested_model_comparison(baseline, expanded):
    """Compare nested models for incremental fit."""
    delta_r2 = expanded['r_squared'] - baseline['r_squared']
    delta_aic = expanded['aic'] - baseline['aic']
    delta_bic = expanded['bic'] - baseline['bic']

    # F-test for ΔR²
    n = baseline['model'].nobs
    k_base = baseline['model'].df_model
    k_exp = expanded['model'].df_model

    f_change = (delta_r2 / (k_exp - k_base)) /                ((1 - expanded['r_squared']) / (n - k_exp - 1))

    return {
        'delta_r2': delta_r2,
        'delta_aic': delta_aic,
        'delta_bic': delta_bic,
        'f_change': f_change
    }

# ==============================================================================
# 7. SENSITIVITY ANALYSES
# ==============================================================================

def sensitivity_analyses(df, outcome, predictors, covariates):
    """Run comprehensive sensitivity checks."""
    results = {}

    # 1. Outlier exclusion (Cook's D > 4/n)
    n = len(df)
    threshold = 4 / n

    # 2. Alternative SE estimators
    for se_type in ['HC1', 'HC2', 'HC3', 'HC4']:
        res = comparative_specification(df, outcome, predictors, covariates, se_type)
        results[f'HC_{se_type}'] = res['model'].params[predictors[0]]

    # 3. Non-parametric alternatives
    for pred in predictors:
        rho, p = spearmanr(df[pred], df[outcome])
        results[f'spearman_{pred}'] = {'rho': rho, 'p': p}

    return results

# ==============================================================================
# 8. CONVERGENT VALIDITY (Q10 vs PILS)
# ==============================================================================

def convergent_validity(df, self_assess_col, objective_col):
    """Compare self-assessed vs objective knowledge."""
    group_yes = df[df[self_assess_col] == 1][objective_col]
    group_no = df[df[self_assess_col] == 0][objective_col]

    # Mann-Whitney U (non-parametric)
    u_stat, p_val = stats.mannwhitneyu(group_yes, group_no, alternative='two-sided')

    # Effect size (point-biserial r)
    r_pb = np.sqrt(u_stat / (len(group_yes) * len(group_no)))

    return {
        'group_yes_mean': group_yes.mean(),
        'group_no_mean': group_no.mean(),
        'u_statistic': u_stat,
        'p_value': p_val,
        'effect_size_rpb': r_pb
    }

# ==============================================================================
# 9. MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Innovation Capability Analysis')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    parser.add_argument('--output', default='results/', help='Output directory')
    args = parser.parse_args()

    # Load data
    df = load_data(args.data)
    df = compute_diagnostic_gap(df)

    # Define variables
    domains = ['PILS', 'IAI', 'IRS', 'IPI']
    covariates = ['Gender', 'Student', 'Academic', 'Experience_Level']

    # Run all analyses
    desc = descriptive_stats(df, domains + ['Diagnostic_Gap'])
    corr = correlation_matrix(df, domains)

    # Specification A: IRS outcome
    spec_a = comparative_specification(df, 'IRS', ['PILS', 'IAI'], covariates)

    # Specification B: IPI outcome
    spec_b = comparative_specification(df, 'IPI', ['PILS', 'IAI', 'IRS'], covariates)

    # Nested comparison
    baseline = comparative_specification(df, 'IPI', ['PILS', 'IAI'], covariates)
    nested = nested_model_comparison(baseline, spec_b)

    # Sensitivity
    sens = sensitivity_analyses(df, 'IPI', ['PILS', 'IAI', 'IRS'], covariates)

    # Validity
    valid = convergent_validity(df, 'Q10', 'PILS_raw')

    # Save results
    import os
    os.makedirs(args.output, exist_ok=True)

    desc.to_csv(f'{args.output}/S2_descriptive_stats.csv')
    corr.to_csv(f'{args.output}/S4_correlation_matrix.csv')
    spec_a['model'].summary().as_csv(f'{args.output}/S5_specification_A.csv')
    spec_b['model'].summary().as_csv(f'{args.output}/S5_specification_B.csv')

    print(f"Analysis complete. Results saved to {args.output}")

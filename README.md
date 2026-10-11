# Pharmaceutical Innovation in Syria: Readiness and Practice Gaps

## 1. Overview & Executive Summary
This open-science reproducibility repository contains the complete dataset, statistical analysis plan, compliance checklists, and fully executable Python analytical codebase for the empirical study investigating Pharmaceutical Innovation Readiness and Practice Gaps among pharmacy students, academic researchers, and practicing pharmacists in Syria (\(N = 303\)).

The study is strictly contextualized within the following global frameworks:
* **WHO Global Strategy on Human Resources for Health (Workforce 2030)**
* **International Pharmaceutical Federation (FIP) Development Goals** (Goal 4: Advanced Practice & Goal 20: Digital Health & Innovation)

### 1.1 Key Empirical Findings
* **Attitudinal Decoupling & Passion-Practice Gap:** High professional endorsement of IP and innovation (\(IAI = 71.30 \pm 8.27\%\)) is statistically decoupled from reported innovation practice (\(IPI = 10.45 \pm 17.91\%\)), yielding a profound Implementation Diagnostic Gap of \(60.84\) points (\(r_s = -0.031\), \(p = 0.587\)).
* **Severe Floor Effects & Multivariable Regression:** Operational constructs exhibited pronounced floor effects (\(48.51\%\) zeros in IRS, \(71.95\%\) zeros in IPI). Multivariable regression models—including Binary Logistic Regression for practice participation (\(IPI > 0\): \(OR_{IRS} = 0.992\), \(95\%\text{ CI } [0.979, 1.005]\), \(p = 0.242\)) and Binomial GLM (\(OR_{IRS} = 0.992\), \(95\%\text{ CI } [0.981, 1.003]\), \(p = 0.170\))—confirmed that individual-level knowledge, attitudes, and perceived readiness do not statistically predict practice participation. This highlights systemic institutional bottlenecks rather than individual deficiencies.
* **Sectoral Floor Effect:** Practice participation remains uniformly depressed across professional sectors (Undergraduate Students: \(11.01\%\), Academic Researchers: \(9.88\%\), Practicing Pharmacists: \(6.67\%\); Kruskal-Wallis \(H = 0.996\), \(p = 0.608\); One-Way ANOVA \(F = 0.694\), \(p = 0.500\)).

---

## 2. Repository Structure & File Index

| File / Folder | Type | Description |
| :--- | :--- | :--- |
| `reproducible_statistical_analysis.py` | Python Script | Primary executable Python 3.12 script performing all data validation, FCIs calculation, Spearman correlations, Stage-1 Logistic regression, and Stage-2 Binomial GLM. |
| `Deidentified_Dataset_N303.csv` | Dataset (CSV) | Complete de-identified dataset (\(N = 303\), 40 variables) with synthetic participant IDs (P001–P303). |
| `SAP_STATISTICAL_ANALYSIS_PLAN_v1.0.txt` | Documentation | Formal Statistical Analysis Plan (v1.0) defining research questions, variable operationalization, and model specifications for EMHJ. |
| `STROBE_CHECKLIST_v1.1.txt` | Compliance | STROBE observational study compliance checklist (100% full compliance). |
| `SUPPLEMENTARY_MATERIALS_INDEX.txt` | Documentation | Index mapping supplementary tables (S1–S8) and visual figures (Figures 1, 2, 3). |
| `requirements.txt` | Environment | Exact Python package version specifications. |
| `CITATION.cff` | Metadata | Machine-readable citation file format metadata. |
| `LICENSE` | Legal | MIT Open Source License terms. |

---

## 3. Dataset Characteristics & Methodology (N = 303)
The final analytical sample (\(N = 303\)) reflects verified responses collected electronically between 10 March and 15 April 2026.

### 3.1 Sample Stratification & Demographics

| Attribute | Category / Stratum | Metrics | Percentage |
| :--- | :--- | :--- | :--- |
| **Gender** | Female (Gender = 1)<br>Male (Gender = 0) | \(n = 215\)<br>\(n = 88\) | \(70.96\%\)<br>\(29.04\%\) |
| **Professional Sector** | Undergraduate Students<br>Academic/Graduate Researchers<br>Licensed Practicing Pharmacists | \(n = 224\)<br>\(n = 54\)<br>\(n = 25\) | \(73.93\%\)<br>\(17.82\%\)<br>\(8.25\%\) |
| **Experience Level** | < 1 year<br>1–5 years<br>6–10 years<br> > 10 years | \(n = 80\)<br>\(n = 153\)<br>\(n = 45\)<br>\(n = 25\) | \(26.40\%\)<br>\(50.50\%\)<br>\(14.85\%\)<br>\(8.25\%\) |

### 3.2 Core Construct Specifications (Formative Composite Indices, 0–100 Scale)

| Variable Name | Construct | Operational Metrics & Items | Descriptive Statistics |
| :--- | :--- | :--- | :--- |
| **PILS_pct** | Pharmaceutical IP Literacy Score | 9 objective binary legal items | Mean = \(38.87 \pm 15.55\%\), Median = \(44.44\%\) |
| **IAI_pct** | Intellectual Property Attitude Index | 10 Likert items (1–5 scale) | Mean = \(71.30 \pm 8.27\%\), Median = \(72.50\%\) |
| **IRS_pct** | Perceived Institutional Readiness Score | 4 objective binary items (TTOs, funding, legal support) | Mean = \(17.82 \pm 20.35\%\), \(48.51\%\) absolute zeros |
| **IPI_pct** | Innovation Practice Index | 3 behavioral binary items (patents, R&D, tech transfer) | Mean = \(10.45 \pm 17.91\%\), \(71.95\%\) absolute zeros |
| **Diagnostic_Gap** | Implementation Diagnostic Gap | Arithmetic difference (IAI_pct - IPI_pct) | Mean = \(60.84 \pm 19.93\) points |

---

## 4. Instructions for Replication
Execution steps for full statistical audit and table reproduction:

1. **Environment Setup:** Install the required open-science ecosystem dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Analysis Pipeline:** Execute the main reproducible validation codebase:
   ```bash
   python reproducible_statistical_analysis.py
   ```

---

## 5. Ethical Compliance & Archival
* **IRB Ethical Approval:** Formal approval granted by the Institutional Review Board (IRB) of Manara University, Syria (Approval No. MU-URO-26-02).
* **Data Archiving:** Permanently archived on Zenodo (DOI: [10.5281/zenodo.23290550](https://doi.org)).
* **Standard Citation:**
  Khatib, C., Alkozy, H., Hamdan, Z., Isber, M., & Mlhem, J. (2026). *Pharmaceutical Innovation in Syria: Readiness and Practice Gaps*. Eastern Mediterranean Health Journal (EMHJ / WHO EMRO). Dataset & Code Repository. DOI: 10.5281/zenodo.23290550.

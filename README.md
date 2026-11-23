# Machine Learning Model Drift Audit | EarlyAlert

**A reproducible analysis of model performance degradation, population drift, and subgroup risk in an early-warning classification system.**

## Project Overview

EarlyAlert is a fictional college early-warning system designed to identify first-year students at risk of not returning for their second year. A classifier developed using a Fall 2023 cohort was applied unchanged to Fall 2025, despite changes in enrollment patterns, the learning management system (LMS), and financial administration policies.

This project audits whether the original model remained effective, identifies where the data changed, and translates the findings into practical monitoring and model-governance recommendations. The analysis uses two cohort datasets comprising **8,100 records** (4,200 in 2023; 3,900 in 2025).

**Central finding:** The model's accuracy increased, but its ability to identify actual dropouts deteriorated sharply. In Fall 2025, **689 of 1,417 students who did not return (48.6%) were not flagged** for intervention.

## Business Questions

- Did the unchanged model retain its ability to identify at-risk students in 2025?
- Which student groups experienced the largest gaps between predicted risk and observed outcomes?
- Which features shifted between cohorts, and how might institutional changes explain the shifts?
- What should be monitored, and when should the organization intervene, repair data pipelines, or retrain the model?

## Analytical Approach

### 1. Baseline model and out-of-time evaluation

- Explored cohort characteristics, missingness, feature distributions, and relationships.
- Engineered a `has_balance` indicator and used a scikit-learn preprocessing pipeline with numerical scaling, binary-feature passthrough, and categorical one-hot encoding.
- Trained a **class-weighted logistic regression** classifier on Fall 2023 with a **stratified 75/25 train/test split**.
- Evaluated the 2023 holdout and the entire 2025 cohort using the same **0.30 alert threshold**, without retraining or refitting the prediction model on 2025 data.
- Compared ROC AUC, accuracy, recall, precision, alert rate, average predicted risk, and observed dropout rate.

### 2. Subgroup performance

Evaluated 2025 outcomes across **delivery mode, age group, international status, and outstanding-balance status**, comparing recall, mean predicted risk, observed dropout rate, and missed cases. The difference between observed dropout rate and mean predicted risk is reported as a **group-level risk gap**; it is not a complete probability-calibration assessment.

### 3. Data drift detection

Applied complementary techniques to the 2023 and 2025 feature distributions:

| Method | Purpose |
| --- | --- |
| Population Stability Index (PSI) | Quantify feature-level distribution changes, including categorical features |
| Kolmogorov–Smirnov test | Compare numerical distributions |
| Wasserstein distance | Measure the magnitude of numerical distribution shifts in original units |
| Chi-square test | Compare categorical and binary feature distributions |
| Domain classifier | Test how well a separate logistic regression distinguishes 2023 from 2025 records |

PSI monitoring bands used in this analysis: **GREEN < 0.10**, **AMBER 0.10–< 0.25**, and **RED ≥ 0.25**. These are screening thresholds, not proof of a causal change or model failure.

## Key Results

### Model performance

| Metric | Fall 2023 holdout (n=1,050) | Fall 2025 (n=3,900) |
| --- | ---: | ---: |
| Accuracy | 57.62% | 71.77% |
| ROC AUC | 0.8029 | 0.7536 |
| **Recall** | **89.72%** | **51.38%** |
| Precision | 35.14% | 63.86% |
| Students flagged | 61.52% | 29.23% |
| Mean predicted risk | 42.15% | 22.44% |
| Actual dropout rate | 24.10% | 36.33% |

Accuracy alone would have given a misleading impression of improvement. While the observed dropout rate rose by **12.24 percentage points**, the alert rate fell by **32.29 percentage points** and recall dropped by **38.35 percentage points**. The model missed **689 actual dropouts** in the 2025 cohort.

### Subgroup findings

The most pronounced operational gap appeared among students with an outstanding balance:

| Fall 2025 subgroup | Actual dropout rate | Mean predicted risk | Recall |
| --- | ---: | ---: | ---: |
| Balance owing | 58.24% | 23.40% | 44.65% |
| No balance | 27.81% | 22.07% | 56.85% |
| Under age 25 | 33.79% | 18.79% | 44.00% |
| Age 25 and over | 42.14% | 30.76% | 64.87% |

For students with a balance owing, the observed dropout rate exceeded the group's mean predicted risk by approximately **34.8 percentage points**. The 2025 policy of deregistering students with unpaid balances after week six is a plausible contributor, although the analysis does not establish causality.

### Distribution shifts

| Feature | PSI | Status |
| --- | ---: | --- |
| `lms_logins_first4wks` | 3.4637 | RED |
| `delivery_mode` | 0.4468 | RED |
| `commute_minutes` | 0.4099 | RED |
| `international` | 0.1575 | AMBER |
| `age` | 0.1427 | AMBER |

The remaining nine assessed features were GREEN. The largest shift occurred in LMS logins, consistent with a potential change in measurement or data mapping following an LMS vendor migration. Changes in delivery mode and commute time align with the expansion of online offerings.

A separate **domain classifier achieved a held-out ROC AUC of 0.9709**, indicating that the two cohorts were readily distinguishable based on their features. This supports strong covariate drift; it does not, on its own, establish which changes caused the prediction model's performance decline.

## Recommendations

1. **Pause automated alert decisions** until the data pipeline and model behavior have been reviewed; retain human-led student support and escalation routes.
2. **Investigate and correct LMS feature mapping**, validating definitions, units, distributions, and missingness across the vendor migration.
3. **Retrain and revalidate** on representative, recent data, with particular attention to recall, threshold selection, subgroup outcomes, and the effects of the balance-related policy.
4. **Implement ongoing monitoring** for recall, missed cases, alert volume, average risk scores, PSI, source-data quality, and subgroup performance, with explicit escalation thresholds and assigned owners.
5. **Require human oversight and documented approval** before redeployment, including periodic review of intervention outcomes and potential unintended impacts.

These are audit recommendations; this repository does **not** implement a production alerting or monitoring service.

## Repository Structure

```text
ml-model-drift-audit/
├── data/
│   ├── XY_fall2023_cohort.csv
│   └── XY_fall2025_cohort.csv
├── notebooks/
│   └── model_drift_audit.ipynb
├── docs/
│   └── model_drift_audit_report.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

**Prerequisites:** Python 3.14.7 and Jupyter support in VS Code (or JupyterLab).

1. Clone or download the repository and open its root folder in VS Code.
2. Create and activate a virtual environment, then install dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. Ensure both cohort CSV files are available in `data/`.
4. Open `notebooks/model_drift_audit.ipynb`, select the project's Python environment, and run the notebook from top to bottom.

**Working-directory note:** The notebook currently uses `Path.cwd().parent / "data"`; it expects the Jupyter working directory to be `notebooks/`. If your Jupyter kernel starts in the repository root, adjust the path or the working directory before running.

## Tools and Skills Demonstrated

**Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn**

Exploratory data analysis, feature engineering, classification, model validation, cohort comparison, statistical hypothesis testing, data drift detection, subgroup performance analysis, and responsible AI monitoring.

## Scope and Limitations

- This is a **fictional educational case study**, not an audit of a real institution or a live student-support system.
- The 2023 metrics use a held-out sample, whereas 2025 metrics use the complete later cohort; the sample sizes and cohort compositions differ.
- Distribution shift, statistical association, and changes in model performance do not by themselves establish causation.
- The group-level risk gap is descriptive; formal calibration analysis and post-retraining validation are outside the implemented scope.
- Data redistribution should follow the original dataset's license or publication terms.

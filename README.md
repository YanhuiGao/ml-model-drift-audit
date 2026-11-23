# Machine Learning Model Drift Audit | EarlyAlert

**Evaluating model reliability across changing populations, data systems, and institutional policies.**

A reproducible Python case study combining **classification performance analysis, statistical drift detection, subgroup evaluation, and model-risk monitoring**. The focus is not simply whether the model's inputs changed, but whether those changes undermined its ability to identify people who needed support.

> **Key finding:** In Fall 2025, the unchanged early-warning model failed to flag **689 of 1,417 students who did not return (48.6%)**, even as its overall accuracy increased. The audit recommends pausing automated alerts, validating the LMS data feed, and retraining with appropriate human oversight.

## Business Context

EarlyAlert is a fictional college early-warning system that assigns first-year students a probability of not returning for their second year. Students with predicted risk **≥ 0.30** are flagged for advisor outreach. The system was developed on a Fall 2023 cohort and applied unchanged to Fall 2025, when fewer alerts coincided with more students leaving.

This analysis uses **8,100 cohort records**: 4,200 from Fall 2023 and 3,900 from Fall 2025. The institution and scenario are fictional; the work is an educational model-auditing case study, not an evaluation of a live student-support service.

### Questions addressed

- Did the model still identify students at risk when applied to a later cohort?
- Which groups experienced the largest missed-case and predicted-risk gaps?
- Which input distributions changed, and which institutional events could explain them?
- What monitoring controls and human decisions would be needed before redeployment?

## Analytical Workflow

| Stage | Approach |
| --- | --- |
| Exploratory analysis | Profile cohort sizes, missing values, numerical/categorical distributions, and feature relationships. |
| Baseline modeling | Engineer a `has_balance` indicator; preprocess with `ColumnTransformer` (numerical scaling, binary passthrough, categorical one-hot encoding); train **class-weighted logistic regression** using a **stratified 75/25 split** of Fall 2023. |
| Out-of-time validation | Apply the **unchanged prediction pipeline** to the full Fall 2025 cohort, using the same **0.30** intervention threshold. |
| Subgroup analysis | Compare recall, observed dropout rate, mean predicted risk, and missed cases by delivery mode, age, international status, and balance owing. |
| Feature drift | Compute PSI for **14 features**, including categorical/binary attributes; supplement with **Kolmogorov–Smirnov, Wasserstein distance, and chi-square tests**. |
| Model-based drift | Train a separate logistic-regression **domain classifier** to distinguish 2023 from 2025; evaluate held-out ROC AUC and rank feature coefficients. |
| Operational response | Translate performance and drift evidence into proposed monitoring thresholds, escalation actions, and human review. |

## Key Findings

### 1. Accuracy improved while detection deteriorated

| Metric | Fall 2023 holdout (n = 1,050) | Fall 2025 (n = 3,900) |
| --- | ---: | ---: |
| Accuracy | 57.62% | 71.77% |
| ROC AUC | 0.8029 | 0.7536 |
| **Recall (dropouts flagged)** | **89.72%** | **51.38%** |
| Precision | 35.14% | 63.86% |
| Students flagged | 61.52% | 29.23% |
| Mean predicted risk | 42.15% | 22.44% |
| Actual dropout rate | 24.10% | 36.33% |

The observed dropout rate increased by **12.23 percentage points**, while the share of students flagged fell by **32.29 points**. Recall declined by **38.34 points**. Of the **1,417** students who did not return in 2025, **689** were never flagged for outreach.

**Interpretation:** Accuracy alone would suggest an improvement, but recall and alert coverage reveal a failure against the system's primary operational purpose. The 2023 figures are based on a holdout sample; the 2025 figures cover the full later cohort.

### 2. The highest-risk financial subgroup was under-identified

| Fall 2025 group | Students | Actual dropout rate | Mean predicted risk | Recall |
| --- | ---: | ---: | ---: | ---: |
| Balance owing | 1,092 | 58.24% | 23.40% | **44.65%** |
| No balance | 2,808 | 27.81% | 22.07% | 56.85% |
| Under 25 | 2,711 | 33.79% | 18.79% | 44.00% |
| Age 25+ | 1,189 | 42.14% | 30.76% | 64.87% |
| In-person | 1,728 | 31.89% | 18.47% | 43.01% |
| Online | 1,380 | 45.22% | 30.55% | 62.34% |

Students with an outstanding balance had **more than twice** the observed dropout rate of those without one, yet received nearly the same average predicted risk. Their group-level difference between observed dropout and mean predicted risk was approximately **34.8 percentage points**. This is a descriptive group-level risk gap, **not** a full calibration analysis.

The notebook also evaluates hybrid delivery and domestic/international status. Subgroup differences motivate targeted investigation; they do not establish that group membership caused the errors.

### 3. Multiple features drifted substantially

PSI was calculated against the Fall 2023 reference distribution. Screening bands: **GREEN < 0.10**, **AMBER 0.10–< 0.25**, **RED ≥ 0.25**.

| Feature | PSI | Band | Supporting evidence |
| --- | ---: | --- | --- |
| `lms_logins_first4wks` | **3.4637** | RED | KS D = 0.669; Wasserstein = 34.6 logins |
| `delivery_mode` | **0.4468** | RED | Chi-square = 834.15, p < 0.001 |
| `commute_minutes` | **0.4099** | RED | KS D = 0.254; Wasserstein = 7.8 minutes |
| `international` | 0.1575 | AMBER | Chi-square = 303.59, p < 0.001 |
| `age` | 0.1427 | AMBER | KS D = 0.156; Wasserstein = 2.04 years |

Across the 14 evaluated features, **3 were RED, 2 AMBER, and 9 GREEN**. The extremely large LMS-login shift warrants checking feature definitions and integration quality before treating the values as comparable. Small p-values alone do not imply a practically important shift; effect size and business context matter.

A separate **domain classifier achieved a held-out ROC AUC of 0.9709**, showing that the two cohorts are highly distinguishable. Its leading coefficients included LMS logins, assignment submission percentage, online delivery, and international status. This corroborates substantial population/data change but does **not** prove which shifts caused the prediction model's decline.

## Institutional Changes and Drift Interpretation

The scenario's change log provides plausible explanations for several observed shifts. These are **contextual hypotheses**, not causal findings.

| Change in 2024–2025 | Evidence in the audit | Interpretation / follow-up |
| --- | --- | --- |
| International study-permit restrictions (2024) | `international` PSI = 0.1575 | A change in cohort composition could affect generalizability; review segment coverage. |
| Expansion of online offerings to working adults (2024) | `delivery_mode` PSI = 0.4468; `commute_minutes` PSI = 0.4099; `age` PSI = 0.1427 | A shift toward online and older students is consistent with the observed changes in delivery, commuting, and age distributions. |
| LMS vendor migration (January 2025) | `lms_logins_first4wks` PSI = 3.4637 | Check whether login counting, units, extraction, or field mapping changed; validate the feed before retraining. |
| Automatic deregistration for unpaid balances (Fall 2025) | Balance-owing dropout rate = 58.24%; recall = 44.65%, while `outstanding_balance` PSI = 0.0044 and `has_balance` PSI = 0.0010 | The feature distributions stayed relatively stable, but the policy may have changed how financial status relates to the outcome: **potential concept drift**. |

**Critical distinction:** PSI detects changes in the *distribution of inputs*. It cannot by itself detect a change in the *relationship between an input and the target*. The balance-related findings illustrate why outcome monitoring must accompany feature-drift monitoring.

## Monitoring and Response Framework

The following is a **proposed governance plan derived from the audit**, not an implemented production monitoring system.

| Monitor | Proposed trigger | Review cadence | Response |
| --- | --- | --- | --- |
| Overall recall at 0.30 | AMBER: < 75%; RED: < 60% | Each term, once outcomes are available | Investigate missed cases; pause automated alerts and retrain at RED. |
| Alert rate and mean predicted risk | Alert rate outside 40–70%, or substantial shift from the 2023 baseline | Each term and after source changes | Verify upstream data first; then reassess threshold, calibration, or retraining. |
| Key-feature PSI | AMBER: 0.10–< 0.25; RED: ≥ 0.25 | Each term and after system changes | Review drivers; validate mappings and retrain where warranted. |
| LMS data quality | Missingness > 5%, undocumented vendor change, or PSI ≥ 0.10 | Weekly after an LMS change, then monthly | Suspend use of the affected feed until validated. |
| Subgroup recall and risk gaps | Subgroup recall < 75%; escalate if balance-group recall < 50% or between-group recall gap > 15 points | Each term | Prioritize targeted human outreach and evaluate subgroup performance before redeployment. |
| Balance-owing outcomes | Recall < 50%, dropout-rate ratio > 2×, or risk gap > 0.20 | Each term and after policy changes | Provide financial-support review regardless of model flag; validate on post-policy outcomes. |
| Domain-classifier AUC | AMBER: 0.60–0.69; RED: ≥ 0.70 | Each term | Investigate separating features; assess whether continued use is appropriate. |

Thresholds are **case-study proposals**, not externally validated industry standards. Monitoring outcome-based metrics requires labels to become available; leading indicators and data-quality checks can be monitored earlier.

## Audit Recommendation

**Pause automated alerts, repair the data feed, then retrain and validate before the next intake.**

1. **Validate the LMS integration:** confirm that login counts have consistent definitions, units, and extraction logic across vendors.
2. **Retrain on representative recent data:** account for the changed cohort and financial policy, and re-evaluate recall, precision, threshold choice, and subgroup outcomes.
3. **Maintain human-led support during the pause:** advisors should be able to identify students with financial or academic concerns independently of model flags.
4. **Require accountable redeployment:** document validation evidence, provide interpretable risk factors, allow advisor overrides, and obtain formal governance approval.
5. **Monitor after launch:** track data quality, cohort drift, outcome-based performance, and subgroup disparities rather than relying on accuracy alone.

## Repository Structure

```text
ml-model-drift-audit/
├── data/
│   ├── XY_fall2023_cohort.csv
│   └── XY_fall2025_cohort.csv
├── notebooks/
│   └── model_drift_audit.ipynb
├── docs/
│   └── model_drift_audit_report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

*If the report is retained as `.docx`, update the path above accordingly. Publish only files whose redistribution is permitted.*

## Reproduce the Analysis

**Environment used:** Python 3.14.7; pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.9.1, SciPy 1.18.1, Matplotlib 3.11.2, Seaborn 0.13.2.

From the repository root, on Windows PowerShell:

```powershell
git clone https://github.com/YOUR_USERNAME/ml-model-drift-audit.git
cd ml-model-drift-audit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Open `notebooks/model_drift_audit.ipynb` in VS Code with the Python/Jupyter extensions, select the `.venv` kernel, and run all cells. The two cohort CSVs must be present in `data/`.

**Path note:** The notebook uses `Path.cwd().parent / "data"`, so it expects the kernel's working directory to be `notebooks/`. If the kernel starts in the repository root, change the working directory or adjust the path before running. A fresh-environment installation has not been independently verified as part of this README.

## Technology and Analytical Skills

**Python · pandas · NumPy · scikit-learn · SciPy · Matplotlib · Seaborn · Jupyter**

Classification and model validation · Feature engineering and preprocessing pipelines · Statistical testing · Data drift and domain classification · Subgroup diagnostics · Model risk monitoring · Evidence-based business recommendations

## Limitations

- The college, system, and events are part of a **fictional educational scenario**; no production deployment or live monitoring is included.
- The 2023 performance figures come from a held-out test split, while 2025 results use the full later cohort; their sample sizes and underlying populations differ.
- Group-level mean-risk gaps are descriptive and do not substitute for individual-level calibration curves or causal inference.
- Observed policy and platform changes provide plausible explanations, not verified causal effects.
- Retraining, deployment, and the proposed monitoring controls are **recommendations**, not completed implementations.

## References

1. S. Rabanser, S. Günnemann, and Z. C. Lipton, “Failing loudly: An empirical study of methods for detecting dataset shift,” *Advances in Neural Information Processing Systems*, vol. 32, 2019. [Paper](https://papers.nips.cc/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html).
2. T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” *PLOS ONE*, vol. 10, no. 3, e0118432, 2015. [DOI](https://doi.org/10.1371/journal.pone.0118432).
3. E. Tabassi, *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1, 2023. [DOI](https://doi.org/10.6028/NIST.AI.100-1).

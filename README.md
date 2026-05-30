# Patient Survival Records — ML Competition (KASDD 2025/2026)

> **Group A05 — KASDUY**
> Kecerdasan Artifisial dan Sains Data Dasar | Fakultas Ilmu Komputer, Universitas Indonesia

**Team Members**
| Name | NPM | Contribution |
|---|---|---|
| Clarissa Indriana Pramesti | 2306211660 | EDA (disease class profiling, cost drivers), GMM Clustering, Feature Engineering |
| Indah Cahya Puspitahati | 2306245453 | EDA (organ function trends, cost factors), TP2 Regression |
| Kadek Savitri | 2306203236 | EDA (demographics, hospitalization), K-Means Clustering |
| Nisa Najla Hanina Hasanah | 2306240055 | EDA (ADL-cost analysis, vital signs severity), K-Means Clustering |

---

## Overview

A unified end-to-end ML pipeline covering three tasks on the **Patient Survival Records** dataset — a clinical dataset of ICU patients with physiological parameters, demographics, and functional status across 7,000+ records and 34 features.

| Task | Problem | Target | Metric | Best Score |
|---|---|---|---|---|
| **TP1** | Classification | `death` (patient mortality) | F1 Score | **0.851** (public) |
| **TP2** | Regression | `hospital_charges` (total billing) | R² | **0.670** (public) |
| **TP3** | Clustering | Patient segmentation | Silhouette / BIC | k=4 clusters |

---

## Repository Structure

```
├── Final_TP1_3_A05_Kasduy.ipynb   # Main notebook (all 3 tasks)
├── train_tp1.csv                   # TP1 training data (7,284 samples)
├── test_tp1.csv                    # TP1 test data (1,821 samples)
├── train_tp2.csv                   # TP2 training data (7,146 samples)
├── test_tp2.csv                    # TP2 test data (1,787 samples)
└── README.md
```

---

## Notebook Structure

```
1. Setup & Load Data
2. Exploratory Data Analysis (12 analytical questions)
3. Data Preprocessing & Cleaning
4. Feature Engineering (Shared Pipeline — TP1 & TP2)
5. Feature Selection (TP3 Clustering)
6. Modelling TP1 — Classification
7. Modelling TP2 — Regression
8. Clustering TP3 — GMM + PCA & K-Means + PCA
9. Profiling & Clinical Insights
```

---

## Methodology

### Data Preprocessing
- **Domain-aware cleaning:** Clinically invalid zero values (heart rate, respiratory rate, mean blood pressure, glucose) replaced with `NaN` before imputation — zeros are physiologically impossible for these vital signs
- **MNAR handling:** Lab values (`urine`, `glucose`, `bun`, `albumin`, `bilirubin`, `pafi_ratio`, `ph`) treated as Missing Not At Random; missing indicator flags added as predictive features
- **Imputation:** Median imputation for numeric, `"missing"` category for categorical
- **Categorical encoding:** Ordinal encoding for `income_level` and `dnr_status` (natural order); frequency encoding + label encoding for nominal features with rare category handling
- **Outlier handling (TP3):** IQR Capping / Winsorization — capping instead of removal to preserve all critical patient records
- **Log transformation:** `log1p` applied to skewed features (`|skew| > 1`); Yeo-Johnson power transform applied in TP2 pipeline

### Feature Engineering (TP1 & TP2)
| Feature | Formula | Rationale |
|---|---|---|
| `age_squared` | age² | Captures non-linear mortality risk with age |
| `log_hosp_day` | log1p(hosp_day_pre_study) | Stabilizes right-skewed distribution |
| `comorbidity_per_day` | comorbidity_count / (hosp_day + 1) | Disease burden relative to stay length |
| `hosp_day × age` | interaction | Elderly long-stay patients carry compounded risk |
| `pressure_per_heartrate` | mean_blood_pressure / (heart_rate + 1) | Hemodynamic status in a single value |
| `has_dnr` | 1 if dnr_status ≠ 'no dnr' | Binary flag — highly predictive of mortality |
| `age_bin` | 5 buckets: 0–30, 30–50, 50–65, 65–80, 80+ | Sharpens age signal for tree models |

### TP1 — Classification (Mortality Prediction)
- **Models:** XGBoost, LightGBM, CatBoost, Logistic Regression (base learners)
- **Approach:** Stacked ensemble → Logistic Regression meta-learner on OOF predictions → threshold optimization for F1
- **Validation:** 5-fold stratified cross-validation
- **Results:**

| Model | OOF F1 | OOF AUC |
|---|---|---|
| XGBoost | 0.8338 | 0.8284 |
| LightGBM | 0.8363 | 0.8302 |
| CatBoost | 0.8389 | 0.8346 |
| Logistic Regression | 0.8395 | 0.8338 |
| **Stacked Ensemble** | **0.851 (public)** | — |

### TP2 — Regression (Hospital Charges Estimation)
- **Models:** XGBoost + LightGBM (base) → RidgeCV (meta-learner)
- **Approach:** log1p target transformation (highly right-skewed distribution) + 5-fold CV target encoding for high-cardinality categoricals + Yeo-Johnson power transform
- **Evaluation:** R² computed on original scale (expm1 back-transform)
- **Results:**

| Model | CV R² |
|---|---|
| XGBoost | 0.7697 |
| LightGBM | 0.7699 |
| CatBoost | 0.7704 |
| **Stacking (LGBM + XGB → RidgeCV)** | **0.7717 (CV) / 0.670 (public)** |

**Key finding from EDA:** `hosp_day_pre_study` is the dominant cost driver (Pearson r = +0.50), followed by `dnr_day_issued` (r = +0.61) and `bilirubin`. Counter-intuitively, higher comorbidity burden correlates with *lower* costs — likely because multi-comorbid patients receive palliative care earlier or have shorter stays.

### TP3 — Clustering (Patient Segmentation)
Two independent approaches compared:

**Approach 1 — GMM + PCA (Clarissa)**
- PCA dimensionality reduction: 43 → 23 features (85% variance retained)
- 4 clinical interaction features added pre-GMM: `c_comorbidity_x_hospday`, `c_metabolic_stress` (bilirubin + creatinine − albumin), `c_resp_cardiac_stress` (−pafi_ratio + resp_rate + heart_rate), `c_socioeco_burden`
- Model selection via BIC + AIC + Silhouette; k=4 chosen for clinical interpretability
- Soft (probabilistic) assignment — more realistic for medical settings where cluster boundaries are not crisp

**Approach 2 — K-Means + PCA (Nisa)**
- PCA dimensionality reduction: 43 → 12 features (~68% variance)
- Model selection via Elbow method + Silhouette + Davies-Bouldin; optimal k=4
- Hard assignment — more efficient, outperforms GMM on all quantitative metrics

**4 Patient Segments Identified (consistent across both approaches):**

| Cluster | Profile | Mortality | Median Charges |
|---|---|---|---|
| Critically Ill — Renal Failure | ARF/MOSF dominant, highest creatinine & BUN | ~71–80% | $31–56K |
| Elderly — Multi-Comorbid Chronic | Oldest patients, COPD/CHF/Cirrhosis, highest comorbidity burden | ~73–89% | $9–11K (low — fast deterioration) |
| Acute Critical — Active Response | Youngest, ARF/MOSF, highest heart & resp rate | ~63–65% | Highest |
| Stable — Heterogeneous | Mixed disease types, metastatic cancer proportion high, best prognosis | ~60% | Low |

---

## Key References

The following papers and clinical frameworks informed methodological decisions:

**Clustering & Model Selection**
- **Chen et al. (2024)** — GMM cluster selection using BIC as primary criterion; soft probabilistic assignment for medical patient segmentation. Directly referenced in Section 8.2.3 for k-selection methodology.

**Clinical Feature Interpretation**
- **McCabe, D. (2019).** *Katz Index of Independence in Activities of Daily Living (ADL).* The Hartford Institute for Geriatric Nursing, NYU Rory Meyers College of Nursing. — Basis for interpreting `adl_calibrated` scores (0 = fully dependent, 7 = fully independent) used in clustering feature selection.

**Ensemble & Boosting Methods**
- **Chen & Guestrin (2016).** *XGBoost: A Scalable Tree Boosting System.* KDD. — XGBoost base learner for both TP1 and TP2.
- **Ke et al. (2017).** *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS. — LightGBM base learner; implicit feature selection via gradient-based leaf splitting used in TP1 & TP2 (no explicit feature selection needed).

**Missing Data Handling**
- MNAR (Missing Not At Random) framework — lab values absent because tests were not ordered (not random absence), justifying missing indicator flags as informative predictive signals rather than noise.

**Statistical Methods**
- Winsorization / IQR Capping — outlier treatment that preserves all patient records, critical when data loss could remove rare high-severity cases.
- Yeo-Johnson Power Transform — chosen over Box-Cox for TP2 because it handles zero and negative values in features.

---

## Key Findings

1. **Hospitalization duration (`hosp_day_pre_study`)** is the strongest predictor of hospital charges (r = +0.50), more than disease class or comorbidity count alone.
2. **DNR status timing** matters: patients with DNR set *before* study entry have the highest median charges ($30.4K) — they had already accumulated more interventions before the decision was made.
3. **Comorbidity burden ↑ → charges ↓** (counterintuitive): patients with multiple comorbidities may transition to palliative care faster or deteriorate sooner, reducing total billing.
4. **Tree-based models outperform linear models** significantly because most feature-target relationships are non-linear — confirmed by low Pearson correlations but high tree model performance.
5. **4 clinically meaningful patient segments** were consistently recovered by both GMM and K-Means, aligning with the 4 disease classes in the dataset.

---

*Submitted as Final Project — KASDD Genap 2025/2026, Fakultas Ilmu Komputer, Universitas Indonesia*

# Predicting In-Hospital Mortality in Seriously Ill Hospitalized Patients

## Project Overview

This project develops a reproducible health data science workflow for predicting in-hospital mortality among seriously ill hospitalized patients using the SUPPORT2 dataset from the UCI Machine Learning Repository.

The analysis combines data validation, preprocessing, exploratory data analysis, inferential statistics, supervised machine learning, model evaluation, and model interpretation. Particular attention is given to missing data, class imbalance, temporal information leakage, and clinically meaningful interpretation of predictive performance.

## Research Question

**Can routinely available demographic, clinical, physiologic, and functional characteristics predict in-hospital mortality among seriously ill hospitalized patients?**

The prediction point was defined as SUPPORT study entry, approximately hospital day 3. Variables containing information occurring after this point were excluded from the final predictive analysis.

## Dataset

The project uses the **SUPPORT2** dataset from the UCI Machine Learning Repository.

The dataset contains **9,105 seriously ill hospitalized patients** and includes demographic characteristics, diagnoses, physiologic measurements, laboratory values, functional status measures, and clinical outcomes.

The primary outcome is:

- `hospdead = 0`: Survived hospitalization
- `hospdead = 1`: Died during hospitalization

Approximately **25.9%** of patients experienced in-hospital mortality.

The notebook supports two data-loading pathways:

1. Automatic retrieval of SUPPORT2 from the UCI Machine Learning Repository.
2. Local upload of a SUPPORT2-compatible CSV file.

## Data Validation and Preprocessing

The workflow includes automated checks for:

- Dataset dimensions
- Target-variable availability
- Missing target values
- Binary target coding
- Duplicate observations
- Missing predictor values
- Constant predictors

Predictor types were assigned according to their clinical meaning rather than relying only on their stored data type.

Variables with more than **40% missingness** were excluded using a predefined pragmatic threshold. Implausible zero values in selected physiologic measurements were recoded as missing.

Numerical predictors were processed using median imputation with missingness indicators followed by standardization. Categorical predictors were imputed with an explicit missing category and one-hot encoded.

All preprocessing was incorporated into scikit-learn pipelines to reduce data leakage between the training and evaluation datasets.

## Information Leakage Review

Potential information leakage was evaluated according to the defined prediction point.

Hospital cost variables, future treatment-intensity information, existing survival estimates, physician prognostic estimates, and DNR timing information were excluded when they contained information unavailable or inappropriate for prediction at study entry.

Initial model interpretation also identified DNR status as disproportionately influential. Further investigation showed that the variable included a category representing DNR orders occurring after study admission. DNR status was therefore excluded and the complete modeling pipeline was rerun.

The selected model's ROC-AUC decreased from approximately **0.923** in the preliminary analysis to **0.865** after this exclusion, demonstrating how temporal information leakage can inflate apparent predictive performance.

## Exploratory and Statistical Analysis

Exploratory analysis examined:

- In-hospital mortality prevalence
- Mortality across disease classes
- APACHE III physiology scores by mortality outcome
- Missing-data patterns
- Spearman correlations among numerical predictors
- Clinical data quality and physiologic measurements

Two inferential statistical analyses were conducted.

A **Mann-Whitney U test** demonstrated a significant difference in APACHE III physiology score distributions between patients who survived and patients who died in the hospital (`p < .001`). The rank-biserial effect size was approximately **0.581**.

A **chi-square test of independence** identified a significant association between disease class and in-hospital mortality (`p < .001`). Cramér's V was approximately **0.361**.

## Machine Learning Models

Ten supervised classification approaches were evaluated using a consistent preprocessing and evaluation framework:

- Logistic Regression
- Support Vector Machine
- Decision Tree
- Random Forest
- Bagging
- AdaBoost
- Gradient Boosting
- XGBoost
- Stacking Classifier
- Multilayer Perceptron Neural Network

A stratified 80/20 training and evaluation split was used with a fixed random state for reproducibility.

## Model Evaluation

Models were evaluated using multiple complementary metrics:

- Accuracy
- ROC-AUC
- PR-AUC
- Sensitivity
- Specificity
- Precision
- F1 score
- Brier score

**ROC-AUC was defined as the primary model-selection metric before model comparison.**

ROC curves were also used to compare discrimination across classification thresholds, and a confusion matrix was used to examine classification performance for the selected model at the default threshold.

## Final Model

The **Stacking Classifier** achieved the highest ROC-AUC and was selected as the final model.

Final evaluation performance:

| Metric | Stacking Classifier |
| --- | ---: |
| ROC-AUC | 0.865 |
| PR-AUC | 0.726 |
| Accuracy | 0.822 |
| Sensitivity | 0.551 |
| Specificity | 0.916 |
| Precision | 0.697 |
| F1 Score | 0.615 |
| Brier Score | 0.124 |

Although stacking achieved the highest predefined primary metric, XGBoost and gradient boosting produced very similar ROC-AUC values. The results therefore should not be interpreted as evidence that stacking was substantially superior to the other leading ensemble models.

Logistic regression also demonstrated an important clinical tradeoff. Its ROC-AUC was slightly lower, but its sensitivity was substantially higher, illustrating how the preferred model or classification threshold may depend on the intended clinical application.

## Model Interpretation

Permutation importance was used to evaluate the contribution of individual predictors to the final stacking classifier.

The most influential predictors included:

- SUPPORT physiology score (`sps`)
- APACHE III physiology score (`aps`)
- Coma score (`scoma`)
- Functional status (`adlsc`)
- Age
- Hospital study day (`hday`)

These findings suggest that physiologic severity, neurologic status, functional status, age, and timing of study entry contributed substantially to mortality discrimination.

Permutation importance represents predictive contribution rather than causality. Importance values may also be affected by correlated predictors.

## Limitations

Several limitations should be considered when interpreting this analysis.

SUPPORT2 is a historical dataset and may not represent current clinical populations or treatment practices. The analysis uses a single dataset and one stratified train/evaluation split without external validation.

Missing data required both feature exclusion and imputation, and these decisions may affect model performance. Although variables were reviewed for temporal consistency, retrospective clinical data can still contain complex relationships between measurement timing, treatment, and outcome.

The default classification threshold also produced a tradeoff between sensitivity and specificity. A model intended for clinical use would require further threshold evaluation based on the consequences of false-negative and false-positive predictions.

Finally, this project does not provide comprehensive external validation, subgroup fairness evaluation, calibration assessment, or prospective clinical evaluation. The models should therefore be interpreted as an analytical demonstration rather than clinical decision-support tools.

## Reproducibility

Several design choices were used to improve reproducibility:

- Fixed random state (`101`)
- Automated data validation
- Reusable preprocessing and evaluation functions
- Scikit-learn pipelines
- Explicit missing-data handling
- Consistent evaluation across models
- Predefined primary model-selection metric
- Local-upload and UCI data-loading pathways
- Model interpretation as part of pipeline validation

The final notebook is designed to run sequentially from data loading through model evaluation and interpretation.

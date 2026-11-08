# Car Insurance Claim Classification

## a. Problem statement
Predict whether a policyholder has a recorded car insurance claim (OUTCOME = 1) or no recorded claim (OUTCOME = 0). Compare five classification algorithms using the same data splits and deploy an interactive evaluation app called ClaimScope. This is a binary classification problem

## b. Dataset description
Dataset: Car Insurance Data, published by Sagnik Roy (sagnik1511), Kaggle version 1.
Source: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
File: data/Car_Insurance_Claim.csv. Downloaded 8 November 2026 through Kaggle's public download endpoint.

The downloaded CSV has 10,000 rows and 19 columns: 17 input predictors, ID, and OUTCOME. ID is excluded from modeling. The assignment minimum of 12 features and 500 instances is satisfied. OUTCOME has 6,867 no-claim records (68.67%) and 3,133 claim records (31.33%). CREDIT_SCORE has 982 missing values; ANNUAL_MILEAGE has 957. The source describes a mixture of real and generated records; the observation period and sampling design are not established here. The source card uses the word loan in its outcome description; this project follows the car insurance context and uses the supplied binary claim label without redefining it. Kaggle API metadata returned an unknown license; check the dataset's current reuse terms before public redistribution. No dataset license is inferred by this project.

Categorical predictors: AGE, GENDER, RACE, DRIVING_EXPERIENCE, EDUCATION, INCOME, VEHICLE_YEAR, POSTAL_CODE, VEHICLE_TYPE.
Numeric predictors: CREDIT_SCORE, VEHICLE_OWNERSHIP, MARRIED, CHILDREN, ANNUAL_MILEAGE, SPEEDING_VIOLATIONS, DUIS, PAST_ACCIDENTS.

## c. Github Repository Link
NOT YET PUBLISHED - replace this line with your actual GitHub repository URL after upload.

Live Streamlit App Link: NOT YET DEPLOYED - replace with your actual Community Cloud URL after deployment.

## d. Models used
All models use a scikit-learn Pipeline that includes the fitted preprocessing. Numeric values are median-imputed and standardized. Categorical values are mode-imputed and one-hot encoded; unknown categories are ignored. POSTAL_CODE is nominal, not a continuous magnitude. Dense output supports Gaussian Naive Bayes. The raw CSV schema is validated before prediction.

Split: stratified 60% training (6,000), 20% validation (2,000), 20% test (2,000), random seed 42 at both split stages. No oversampling is applied. Pipelines are fitted only on training data. The exact split indices, dataset SHA256, environment versions and model filenames are stored in model/manifest.json. The saved models remain fitted on the 6,000 training rows; validation and test rows are never used for fitting.

Fixed model settings are baseline design choices, not an exhaustive hyperparameter search: LogisticRegression C=1, max_iter=2000; DecisionTree max_depth=6, min_samples_leaf=15; KNN k=25 and distance weights; GaussianNB var_smoothing=0.01; RBF SVC C=1, gamma=scale. Other parameters use the pinned library defaults. Logistic regression, tree and SVM use seed 42 where applicable. Tree constraints, a moderately large k, and NB smoothing regularize the baselines. Thresholds use each estimator's default predict method.

Validation F1 ranks models before test reporting; AUC breaks ties. Validation selected Logistic Regression. No settings are changed after seeing the test scores. SVM is the descriptive test-set winner, while logistic regression remains the validation-selected model. The difference is not evidence of a statistically reliable or universal winner.

### Held-out test comparison
Positive class = 1. Accuracy, precision, recall, F1 and MCC use predicted labels. AUC uses positive-class probabilities for four models and the decision_function score for SVM; hard labels are never used for AUC. Values below are proportions, not percentages. Precision, recall and F1 use binary averaging; zero divisions return zero. MCC uses all four confusion-matrix cells.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8450 | 0.9183 | 0.7642 | 0.7300 | 0.7467 | 0.6355 |
| Decision Tree | 0.8410 | 0.8992 | 0.7437 | 0.7508 | 0.7472 | 0.6313 |
| KNN | 0.8180 | 0.8843 | 0.7259 | 0.6725 | 0.6982 | 0.5690 |
| Naive Bayes | 0.7655 | 0.8627 | 0.5940 | 0.7923 | 0.6790 | 0.5130 |
| Support Vector Machine | 0.8480 | 0.9048 | 0.7597 | 0.7524 | 0.7560 | 0.6457 |

### Observations about model performance

**Logistic Regression.** Highest test AUC (0.9183) and precision (0.7642); F1 0.7467. Selected by validation F1 before test evaluation. A useful interpretable baseline; coefficients describe associations after encoding, not causal effects.

**Decision Tree.** Test F1 0.7472 and recall 0.7508, close to logistic regression. The depth and leaf constraints limit overfitting. Its AUC 0.8992 is lower than logistic regression.

**KNN.** Test F1 0.6982 and recall 0.6725 trail the leading models. Distance in the scaled, one-hot feature space may be less informative; this is an explanation to investigate, not a proven cause.

**Naive Bayes.** Highest test recall (0.7923), but lowest precision (0.5940) and accuracy (0.7655). It identifies more claims while producing more false positives. Correlated predictors and one-hot inputs can challenge Gaussian independence assumptions.

**Support Vector Machine.** Highest test accuracy (0.8480), F1 (0.7560) and MCC (0.6457). Its nonlinear RBF boundary performs well in this split. The test F1 advantage over logistic regression is small and has no significance test.

**Overall winner for this test split:** Support Vector Machine on test F1, accuracy and MCC. Logistic regression leads AUC and was selected on validation. A majority-class no-claim baseline gives 0.6870 test accuracy and zero positive-class F1, demonstrating why accuracy alone is insufficient.

### Validation comparison
| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8565 | 0.9200 | 0.7681 | 0.7767 | 0.7724 | 0.6676 |
| Decision Tree | 0.8465 | 0.9052 | 0.7432 | 0.7799 | 0.7611 | 0.6485 |
| KNN | 0.8180 | 0.8855 | 0.7271 | 0.6715 | 0.6982 | 0.5691 |
| Naive Bayes | 0.7950 | 0.8777 | 0.6348 | 0.8150 | 0.7137 | 0.5684 |
| Support Vector Machine | 0.8455 | 0.9073 | 0.7469 | 0.7671 | 0.7569 | 0.6438 |

## Repository files
- app.py: customized Streamlit evaluation frontend, no fitting calls.
- core.py: shared schema, preprocessing, model constructors and metrics.
- train.py: reproducible split, five-model training, evaluation and persistence.
- model/: five compressed .joblib fitted pipelines and manifest.json.
- test_data.csv: the untouched 2,000-row test split, with inputs and OUTCOME.
- data/Car_Insurance_Claim.csv: the source CSV used for reproducibility.
- results/: held-out metrics, validation metrics, observations and class reports.
- requirements.txt: pinned app and training dependencies.
- create_submission.py and requirements-report.txt: regenerate the final PDF after inserting live links.
- .streamlit/config.toml: app theme and upload limit.

## Run locally
Use Python 3.12. Open a terminal in this project folder.
```bash
python -m venv .venv
```
Windows PowerShell activation:
```powershell
.venv\Scripts\Activate.ps1
```
macOS/Linux activation:
```bash
source .venv/bin/activate
```
Install dependencies and launch the already-trained app:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Open the local URL shown in the terminal, normally http://localhost:8501. Use the bundled test data or upload test_data.csv. Choose a model to see its six metrics, confusion matrix and classification report. The comparison table shows all five models on the same CSV. Predictions and metrics can be downloaded.

To reproduce training locally (not inside the app):
```bash
python train.py --data data/Car_Insurance_Claim.csv
```
Only load the trusted model files supplied with this project. No pickle/model upload is provided in the app.

## Upload and deploy
Create your own GitHub repository, upload the extracted project contents with the model directory intact, then deploy app.py from branch main on Streamlit Community Cloud. Choose Python 3.12 in advanced settings. Use requirements.txt at the repository root. The separate step-by-step guide covers browser upload, Git commands, local development, deployment and troubleshooting.

## Finalize the required single submission PDF
After publishing, replace both link lines in this README. Install the report-only dependency, then run:
```bash
python -m pip install -r requirements-report.txt
python create_submission.py --github https://github.com/YOUR_USERNAME/YOUR_REPO --app https://YOUR_APP.streamlit.app
```
Replace both example URLs with your actual working URLs. The script updates the README link lines and produces Car_Insurance_Assignment_Submission.pdf, with GitHub link first, Streamlit link second and the complete README content after them. Open and verify both links in an incognito window before submission. The ZIP and guide support your work; the assignment asks for the single finalized PDF.

## Limitations and academic ownership
Results describe this random holdout, not a real deployment population. No temporal validation or confidence interval is provided. Race and gender are retained for academic comparison; no fairness assessment or production underwriting approval is implied. Review and understand the source, make your own documented changes, and preserve genuine GitHub commits as you develop. Do not manufacture a history of work. The source target timing is insufficiently documented to prove all predictors precede each claim.

## References
- Kaggle source: https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
- GitHub repositories: https://docs.github.com/en/repositories/creating-and-managing-repositories/quickstart-for-repositories
- scikit-learn pipelines: https://scikit-learn.org/stable/modules/compose.html
- scikit-learn metrics: https://scikit-learn.org/stable/modules/model_evaluation.html

"""Shared schema, preprocessing, classifiers and evaluation for claim classification."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

TARGET = 'OUTCOME'
CATEGORICAL = ['AGE', 'GENDER', 'RACE', 'DRIVING_EXPERIENCE', 'EDUCATION', 'INCOME', 'VEHICLE_YEAR', 'POSTAL_CODE', 'VEHICLE_TYPE']
NUMERIC = ['CREDIT_SCORE', 'VEHICLE_OWNERSHIP', 'MARRIED', 'CHILDREN', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS']
FEATURES = CATEGORICAL + NUMERIC
SEED = 42

def validate_frame(frame):
    if len(frame) == 0:
        raise ValueError('The CSV contains no rows.')
    missing = sorted(set(FEATURES + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    frame = frame.copy()
    y = pd.to_numeric(frame[TARGET], errors='coerce')
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('OUTCOME must contain only 0 or 1, with no missing labels.')
    for col in NUMERIC:
        values = pd.to_numeric(frame[col], errors='coerce')
        if (frame[col].notna() & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f'{col} contains invalid numeric values.')
        frame[col] = values
    # Postal codes are nominal categories, never a continuous numerical scale.
    for col in CATEGORICAL:
        if col == 'POSTAL_CODE':
            frame[col] = frame[col].map(lambda v: str(int(v)) if pd.notna(v) and str(v).replace('.', '', 1).isdigit() and float(v).is_integer() else v)
        frame[col] = frame[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    return frame[FEATURES], y.astype(int)

def make_models():
    estimators = {
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=25, weights='distance', n_jobs=1),
        'Naive Bayes': GaussianNB(var_smoothing=1e-2),
        'Support Vector Machine': SVC(C=1.0, kernel='rbf', gamma='scale', random_state=SEED),
    }
    result = {}
    for name, estimator in estimators.items():
        preprocess = ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC),
            ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL)
        ])
        result[name] = Pipeline([('preprocess', preprocess), ('classifier', estimator)])
    return result

def evaluate(model, X, y):
    prediction = model.predict(X)
    score = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X)
    auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
    values = {'Accuracy': accuracy_score(y, prediction), 'AUC': auc,
              'Precision': precision_score(y, prediction, zero_division=0),
              'Recall': recall_score(y, prediction, zero_division=0),
              'F1': f1_score(y, prediction, zero_division=0),
              'MCC': matthews_corrcoef(y, prediction)}
    return values, prediction, score

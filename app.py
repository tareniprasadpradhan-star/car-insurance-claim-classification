"""ClaimScope: evaluate saved insurance classifiers on labelled test CSVs only."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from core import validate_frame, evaluate

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='ClaimScope | Car Insurance', page_icon='🚗', layout='wide')
st.title('ClaimScope')
st.subheader('Car insurance claim model comparison')
st.caption('Sagnik Roy dataset | 5 saved classifiers | Positive class: claim (OUTCOME = 1)')
st.info('Upload labelled test data to compare the saved models. Predictions use training-time preprocessing; the app does not train models.')

@st.cache_resource
def load_models():
    metadata = json.loads((ROOT / 'model/manifest.json').read_text())
    return {name: joblib.load(ROOT / 'model' / filename) for name, filename in metadata['models'].items()}, metadata

models, metadata = load_models()
with st.sidebar:
    st.header('Evaluation controls')
    selected = st.selectbox('Choose a classifier', list(models))
    file = st.file_uploader('Upload test CSV including OUTCOME', type=['csv'])
    use_sample = st.checkbox('Use bundled held-out test data', value=True, disabled=file is not None)
    st.download_button('Download test data', (ROOT / 'test_data.csv').read_bytes(), file_name='test_data.csv', mime='text/csv')
    st.caption('Validation-selected model: ' + metadata['validation_winner'])
    st.caption('Maximum upload: 5,000 rows and 5 MB. Target values must be 0 or 1.')
if file is not None and file.size > 5 * 1024 * 1024:
    st.error('Please upload a test CSV smaller than 5 MB.'); st.stop()
if file is None and not use_sample:
    st.warning('Upload your test CSV or enable the bundled test data.'); st.stop()
try:
    frame = pd.read_csv(file if file is not None else ROOT / 'test_data.csv')
    if len(frame) > 5000:
        raise ValueError('Please upload at most 5,000 test rows.')
    X, y = validate_frame(frame)
except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(str(exc)); st.stop()

st.write(f'**{len(frame):,} test rows** · **{int(y.sum()):,} claims** · **{y.mean():.1%} claim rate**')
if y.nunique() < 2:
    st.warning('Only one target class is present. ROC AUC is undefined and appears as N/A.')
rows, selected_output = [], None
for name, model in models.items():
    metrics, pred, score = evaluate(model, X, y)
    rows.append({'Model': name, **metrics})
    if name == selected:
        selected_output = (metrics, pred, score)
comparison = pd.DataFrame(rows).set_index('Model')
metrics, prediction, score = selected_output
st.markdown('### ' + selected)
for column, (name, value) in zip(st.columns(6), metrics.items()):
    column.metric(name, 'N/A' if pd.isna(value) else f'{value:.4f}')
left, right = st.columns([1, 1.4])
with left:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y, prediction, labels=[0, 1], display_labels=['No claim', 'Claim'], cmap='Blues', ax=ax, colorbar=False)
    ax.set_title('Actual versus predicted claims')
    st.pyplot(fig); plt.close(fig)
with right:
    st.markdown('#### Classification report')
    report = classification_report(y, prediction, labels=[0, 1], target_names=['No claim', 'Claim'], output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), width='stretch')
st.markdown('### All five models on the same uploaded data')
st.dataframe(comparison.style.format('{:.4f}', na_rep='N/A'), width='stretch')
st.download_button('Download comparison metrics', comparison.to_csv().encode(), 'comparison_metrics.csv', 'text/csv')
output = frame.copy(); output['PREDICTED_OUTCOME'] = prediction; output['CLAIM_SCORE'] = score
st.download_button('Download selected model predictions', output.to_csv(index=False).encode(), 'claim_predictions.csv', 'text/csv')
with st.expander('Data preview and interpretation'):
    st.dataframe(frame.head(20), width='stretch')
    st.write('Precision, recall and F1 refer to claim = 1. AUC uses positive-class probabilities, except SVM which uses its decision score. A decision score is not a calibrated probability. ID is ignored. Unknown categories are accepted by the fitted encoder.')
    st.write('This is an academic demonstration. The source contains partly generated records; results do not establish real-world underwriting performance or causal effects.')

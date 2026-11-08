"""Train all five models once; save fitted pipelines and reproducible holdout results."""
from pathlib import Path
import argparse, json, hashlib, platform
import pandas as pd
import numpy as np
import sklearn, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from core import validate_frame, make_models, evaluate, SEED, FEATURES, TARGET

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data/Car_Insurance_Claim.csv')
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
    X, y = validate_frame(raw)
    train_idx, held_idx = train_test_split(np.arange(len(raw)), test_size=0.4, stratify=y, random_state=SEED)
    val_idx, test_idx = train_test_split(held_idx, test_size=0.5, stratify=y.iloc[held_idx], random_state=SEED)
    for directory in ['model', 'results']:
        (ROOT / directory).mkdir(exist_ok=True)
    raw.iloc[test_idx].to_csv(ROOT / 'test_data.csv', index=False)
    validation, testing, manifest, details, fitted = [], [], {}, {}, {}
    for number, (name, model) in enumerate(make_models().items(), 1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        v, _, _ = evaluate(model, X.iloc[val_idx], y.iloc[val_idx])
        validation.append({'Model': name, **v})
        filename = f'classifier_{number}.joblib'
        joblib.dump(model, ROOT / 'model' / filename, compress=3)
        manifest[name] = filename
        fitted[name] = model
    validation = pd.DataFrame(validation)
    winner = validation.sort_values(['F1', 'AUC'], ascending=False).iloc[0]['Model']
    # Model selection is complete before any test evaluation.
    for name, model in fitted.items():
        t, prediction, _ = evaluate(model, X.iloc[test_idx], y.iloc[test_idx])
        testing.append({'Model': name, **t})
        details[name] = {'confusion_matrix': confusion_matrix(y.iloc[test_idx], prediction, labels=[0, 1]).tolist(),
                         'classification_report': classification_report(y.iloc[test_idx], prediction, output_dict=True, zero_division=0)}
        print(name, t, flush=True)
    pd.DataFrame(testing).to_csv(ROOT / 'results/metrics.csv', index=False)
    validation.to_csv(ROOT / 'results/validation_metrics.csv', index=False)
    meta = {'dataset': 'sagnik1511/car-insurance-data', 'source_url': 'https://www.kaggle.com/datasets/sagnik1511/car-insurance-data',
            'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(), 'seed': SEED,
            'shape': list(raw.shape), 'features': FEATURES, 'target': TARGET,
            'split_sizes': {'train': len(train_idx), 'validation': len(val_idx), 'test': len(test_idx)},
            'split_indices': {'train': train_idx.tolist(), 'validation': val_idx.tolist(), 'test': test_idx.tolist()},
            'positive_counts': {k: int(y.iloc[ix].sum()) for k, ix in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]},
            'validation_winner': winner, 'selection_rule': 'Highest validation F1, AUC tie-break; test results reported without further tuning.',
            'models': manifest, 'versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}}
    (ROOT / 'model/manifest.json').write_text(json.dumps(meta, indent=2))
    (ROOT / 'results/details.json').write_text(json.dumps(details, indent=2))
    print('Validation-selected model:', winner)

if __name__ == '__main__':
    main()

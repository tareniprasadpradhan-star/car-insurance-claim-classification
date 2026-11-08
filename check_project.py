"""Meaningful local acceptance checks; run with python check_project.py."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from streamlit.testing.v1 import AppTest
from core import validate_frame, evaluate

root=Path(__file__).resolve().parent
meta=json.loads((root/'model/manifest.json').read_text())
raw=pd.read_csv(root/'data/Car_Insurance_Claim.csv')
frame=pd.read_csv(root/'test_data.csv')
X,y=validate_frame(frame)
assert len(frame)==2000 and len(X.columns)==17
sets=[set(meta['split_indices'][key]) for key in ['train','validation','test']]
assert len(set.union(*sets))==10000
assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
expected=pd.read_csv(root/'results/metrics.csv').set_index('Model')
for name,filename in meta['models'].items():
    model=joblib.load(root/'model'/filename)
    metrics,pred,_=evaluate(model,X,y)
    for key,value in metrics.items(): assert np.isclose(value,expected.loc[name,key])
    # Missing numeric values and unseen categories must remain predictable.
    altered=X.head(3).copy();altered['AGE']='unseen age';altered['CREDIT_SCORE']=np.nan
    assert len(model.predict(altered))==3
    one,_,_=evaluate(model,X.head(3),pd.Series([0,0,0]))
    assert np.isnan(one['AUC'])
for bad in [frame.drop(columns='OUTCOME'),frame.assign(OUTCOME=7),frame.assign(CREDIT_SCORE='bad'),frame.iloc[:0]]:
    try: validate_frame(bad)
    except ValueError: pass
    else: raise AssertionError('Invalid CSV accepted')
app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
assert not app.exception, str(app.exception)
for name in meta['models']:
    app.sidebar.selectbox[0].select(name).run()
    assert not app.exception, str(app.exception)
    assert len(app.metric)==6
    assert len(app.dataframe)>=2
app.sidebar.checkbox[0].uncheck().run()
assert not app.exception and len(app.warning)==1
print('PASS: disjoint splits, saved-model metric parity, missing/unseen inputs, invalid schema, one-class AUC, all five Streamlit selections, no-data branch.')

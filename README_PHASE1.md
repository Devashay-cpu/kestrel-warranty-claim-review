# Phase 1 quick start
```
pip install -r requirements.txt
# put the 5 CSVs (train.csv, test_unlabelled.csv, partners.csv, products.csv, sample_submission.csv) in data/raw/
python scripts/run_phase1.py                    # writes artifacts/*
python -m unittest discover -s tests -v         # 16 tests, ~35 s
```
Phase 2 usage:
```python
from kestrel.clean import load_all; from kestrel import pipeline as P, splits as S, features as F
ds = load_all(); Xtr = P.train_features(ds); Xte = P.test_features(ds)
tr, va = P.fold_data(ds, S.FOLDS[1], Xtr)       # one time-based fold
feats = F.feature_names()                       # default groups; cat columns in F.CAT_FEATURES
```

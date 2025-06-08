import os, sys, argparse, pathlib, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore', category=UserWarning)

P = argparse.ArgumentParser()
P.add_argument('--train_dir'); P.add_argument('--test_dir'); P.add_argument('--data_root')
A = P.parse_args()
root = pathlib.Path(A.data_root or os.getcwd())
TRAIN_DIR = pathlib.Path(A.train_dir) if A.train_dir else root/'39_Training_Dataset'
TEST_DIR  = pathlib.Path(A.test_dir)  if A.test_dir  else root/'39_Test_Dataset'
print('TRAIN_DIR:', TRAIN_DIR); print('TEST_DIR:', TEST_DIR)

CHAN = ['Ax','Ay','Az','Gx','Gy','Gz']; MAX_LEN = 2000

def _load(uid:int, base:pathlib.Path):
    fp = base/f'{uid}.txt'
    arr = np.loadtxt(fp, dtype=float) if fp.exists() else np.zeros((0,6))
    if arr.shape[0] < MAX_LEN:
        arr = np.pad(arr, ((0, MAX_LEN-arr.shape[0]), (0,0)))
    return arr[:MAX_LEN]

def _feat(uid:int, base:pathlib.Path):
    arr = _load(uid, base)
    d = {'unique_id': uid}
    for i,ch in enumerate(CHAN):
        col = arr[:,i]
        d[f'{ch}_mean'] = col.mean(); d[f'{ch}_std'] = col.std()
        d[f'{ch}_min']  = col.min();  d[f'{ch}_max'] = col.max()
        d[f'{ch}_q25']  = np.quantile(col,0.25); d[f'{ch}_q75'] = np.quantile(col,0.75)
    return d

def build(uids, base, cache):
    if cache.exists(): return pd.read_feather(cache).set_index('unique_id')
    subdir = base/('train_data' if 'train' in base.name.lower() else 'test_data')
    df = pd.DataFrame(Parallel(n_jobs=-1)(delayed(_feat)(uid, subdir) for uid in uids)).set_index('unique_id')
    df.reset_index().to_feather(cache)
    return df

def lgb_oof(X, y, Xt, n):
    skf = StratifiedKFold(3, shuffle=True, random_state=42)
    oof = np.zeros((len(X), n)); pred = np.zeros((len(Xt), n))
    params = dict(boosting_type='gbdt', objective='multiclass' if n>2 else 'binary', num_class=n if n>2 else 1,
                  learning_rate=0.1, n_estimators=300, num_leaves=64, feature_fraction=0.8, subsample=0.8,
                  device_type='gpu', max_depth=-1, verbose=-1)
    for tr, va in skf.split(X, y):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric='multi_logloss' if n>2 else 'logloss',
                 callbacks=[lgb.early_stopping(20, verbose=False)])
        oof[va] = clf.predict_proba(X.iloc[va], raw_score=False)
        pred += clf.predict_proba(Xt)/skf.n_splits
    return pred

def main():
    df = pd.read_csv(TRAIN_DIR/'train_info.csv')
    train_ids = df['unique_id'].tolist()
    test_ids  = sorted(int(p.stem) for p in (TEST_DIR/'test_data').glob('*.txt'))
    targets = [c for c in df.select_dtypes(include=['int','float']).columns if c!='unique_id']
    Xtr = build(train_ids, TRAIN_DIR, TRAIN_DIR/'train_fast.feather')
    Xte = build(test_ids,  TEST_DIR,  TEST_DIR/'test_fast.feather')
    Xtr, Xte = Xtr.align(Xte, join='inner', axis=1)

    label_map = {}; preds_all = {}
    for col in targets:
        le = LabelEncoder().fit(df[col]); y = le.transform(df[col]); label_map[col] = le.classes_
        preds_all[col] = lgb_oof(Xtr, y, Xte, len(le.classes_))

    sub = pd.DataFrame({'unique_id': test_ids})
    for col, p in preds_all.items():
        cls = label_map[col]
        if len(cls) == 2:
            idx1 = list(cls).index(1)
            sub[col] = p[:, idx1]
        else:
            for i, cl in enumerate(cls):
                sub[f'{col}_{cl}'] = p[:, i]
    samp = TRAIN_DIR.parent/'sample_submission.csv'
    if samp.exists():
        sample = pd.read_csv(samp)
        for c in sample.columns:
            if c not in sub.columns: sub[c] = 0.0
        sub = sub[sample.columns]
    sub.to_csv('submission_fast.csv', index=False, float_format='%.6f', line_terminator='\n')
    print('submission_fast.csv saved', sub.shape)

if __name__ == '__main__':
    main()

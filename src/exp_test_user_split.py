from cmath import exp
import pickle
import os
import gc
from re import sub
import joblib
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from argparse import Namespace
from collections import defaultdict
import argparse
import datetime as dt
import mlflow
import sweetviz as sv
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, GroupKFold, train_test_split 
from sklearn.metrics import auc, roc_curve
from collections import Counter, defaultdict

import lightgbm as lgb

import sys
sys.path.append('../src')
from utils import seed_everything
from preprocess import get_train_test

import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 64)

SAVE_PATH = '../save/'
DATA_PATH = '../input/'
EXPERIMENT_ID = 0

def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--seed', type=int, default=55)
parser.add_argument('--folds', type=int, default=5)
parser.add_argument('--n_estimators', type=int, default=1000)
parser.add_argument('--cv_method', type=str, default='stratified_group')
parser.add_argument('--save_name', type=str, default='tmp')
args = parser.parse_args()

seed_everything(args.seed)
setattr(args, 'save_name', SAVE_PATH+args.save_name)
if not os.path.isdir(args.save_name):
    os.makedirs(args.save_name)
if args.debug:
    setattr(args, 'folds', 2)
    setattr(args, 'n_estimators', 10)
assert args.cv_method in {"group", "stratified", "time", "group_time", "stratified_group"}, "unknown cv method"

print("Read data.")
# assert data.isnull().any().sum() == 0, "null exists."
data = get_train_test(args)
group_feature = data.loc[data['test']==False, 'start_at__date']

train = data[data['test']==False].reset_index(drop=True)
test = data[data['test']==True].reset_index(drop=True)
if args.cv_method=="time":
    train = data[data['test']==False].sort_values('start_at__date').reset_index(drop=True)
    test = data[data['test']==True].sort_values('start_at__date').reset_index(drop=True)

features = train.drop(['target', 'session_id', 'test', 'user_id', 'registor_number', 'start_at__date'], axis=1).columns.tolist()
features = [col for col in features if 'user_' not in col]
print('feature: ', features)

# report = sv.compare([train.drop('target', axis=1), "train"], [test.drop('target', axis=1), "test"])
# report.show_html(os.path.join('../save', "train_vs_test.html"))

def run():    
    # hyperparams from: https://www.kaggle.com/valleyzw/ubiquant-lgbm-optimization
    params = {
        'learning_rate':0.05,
        'objective': 'binary',
        "metric": "auc",
        'boosting_type': "gbdt",
        'verbosity': -1,
        'n_jobs': -1, 
        'seed': args.seed,
        # 'lambda_l1': 6.610898817934583, 
        # 'lambda_l2': 1.2572931636397838e-07, 
        # 'num_leaves': 31, 
        'feature_fraction': 0.9, 
        'bagging_fraction': 0.9, 
        'bagging_freq': 3, 
        'max_depth': 5, 
        # 'max_bin': 214, 
        # 'min_data_in_leaf': 450,
        'n_estimators': args.n_estimators, 
        # 'extra_trees': True,
    }
    
    y = train['target']
    train['preds'] = -1000
    group_feature = train['user_id']
    
    def run_single_fold(fold, trn_ind, val_ind):
        with mlflow.start_run(experiment_id=EXPERIMENT_ID, nested=True):
            train_dataset = lgb.Dataset(train.loc[trn_ind, features], y.loc[trn_ind])
            valid_dataset = lgb.Dataset(train.loc[val_ind, features], y.loc[val_ind])
            model = lgb.train(
                params,
                train_set = train_dataset, 
                valid_sets = [train_dataset, valid_dataset], 
                verbose_eval=50,
                early_stopping_rounds=50,
            )
            joblib.dump(model, args.save_name+f'/lgbm_seed{args.seed}_{fold}.pkl')
            preds = model.predict(train.loc[val_ind, features])
            train.loc[val_ind, "preds"] = preds
            del train_dataset, valid_dataset, model
            gc.collect()
            
    if args.cv_method=="stratified":
        stkf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for fold, (trn_ind, val_ind) in enumerate(stkf.split(train[features], y)):
            print(f"=====================fold: {fold}=====================")
            print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
            run_single_fold(fold, trn_ind, val_ind)
    elif args.cv_method=="time":
        tscv = TimeSeriesSplit(args.folds)
        for fold, (trn_ind, val_ind) in enumerate(tscv.split(train[features])):
            print(f"=====================fold: {fold}=====================")
            print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
            run_single_fold(fold, trn_ind, val_ind)
    elif args.cv_method=="group":
        kfold = GroupKFold(args.folds)
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], y, group_feature)):
            print(f"=====================fold: {fold}=====================")
            print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
            run_single_fold(fold, trn_ind, val_ind)
    elif args.cv_method=="stratified_group":
        for fold, (trn_ind, val_ind) in enumerate(stratified_group_k_fold(train[features], y, group_feature, args.folds, seed=args.seed)):
            print(f"=====================fold: {fold}=====================")
            print(f"train length: {len(trn_ind)}, valid length: {len(val_ind)}")
            run_single_fold(fold, trn_ind, val_ind)

        
    train.filter(regex=r"^(?!f_).*").to_csv(args.save_name+"/preds.csv", index=False)


mlflow.lightgbm.autolog()
with mlflow.start_run(experiment_id=EXPERIMENT_ID):
    mlflow.log_params(vars(args))
    run()
    df = train[["target", "preds"]].query("preds!=-1000")
    fpr, tpr, thresholds = roc_curve(df.target, df.preds)

    print(f"lgbm {args.cv_method} {args.folds} auc: {auc(fpr, tpr):.4f}")
    mlflow.log_metric('folds_auc', auc(fpr, tpr))
    del df
    gc.collect()

    models = [joblib.load(args.save_name+f'/lgbm_seed{args.seed}_{fold}.pkl') for fold in range(args.folds)]

    submit_df = pd.read_csv(DATA_PATH+'/atmaCup13_sample_submission.csv')
    submit_df['target'] = np.mean(np.stack([models[fold].predict(test[features]) for fold in range(args.folds)]), axis=0)
    
    submit_df.to_csv(args.save_name+f'/submit{args.seed}.csv', index=False)

sns.distplot(train['preds'], label='oof')
sns.distplot(submit_df['target'], label='Test')

plt.savefig(args.save_name+'/oof_test_distplot.png')
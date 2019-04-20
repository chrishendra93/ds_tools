import pandas as pd
import numpy as np
from .cv_utils import run_kfold_cv
from functools import reduce


def find_missing_data_stats(df, print_stats=True, return_result=False):
    missing_dat = df.isnull().sum()
    if print_stats:
        print("Number of columns with missing entries: {}".format(len(missing_dat[missing_dat > 0])))
        print("Total number of missing entries: {}".format(missing_dat.sum()))
    if return_result:
        cols_with_missing_dat = missing_dat[missing_dat > 0]
        return cols_with_missing_dat if len(cols_with_missing_dat) > 0 else None


def generate_correlation_df(df, features, label=None, n_fold=10, imbalanced=False, verbose=True):
    if verbose:
        print("class label will be treated as integer to calculate correlation")
    df[label] = df[label].astype('int')
    if not imbalanced:
        return df[features].corr()
    elif imbalanced and not label:
        raise ValueError("Label variable is not given")
    else:
        return calculate_balanced_corr(df, label, n_fold, verbose)


def calculate_balanced_corr(df, label, n_fold, verbose):
    ''' label is assumed to be at least valid string that can be converted to integers '''
    if verbose:
        print("calculating correlation through {} folds split of undersampled majority classes".format(n_fold))
    grouped_df = df.groupby(label)
    n_minorities = grouped_df.size().min()
    tmp = []
    for i in range(n_fold):
        tmp.append(grouped_df.apply(lambda x: x.sample(n_minorities)).reset_index(drop=True).corr())
    return reduce(lambda x, y: x + y, tmp) / len(tmp)


def generate_features_importance(model, X, y, generate_std=False, cv=10):
    features_importance = []
    for train_index, _ in StratifiedKFold(n_splits=cv).split(X, y):
        x_train, y_train = X.values[train_index], y.values[train_index]
        features_score = model.fit(x_train, y_train).feature_importances_
        features_importance = np.vstack([features_importance, features_score]) if len(features_importance) > 0 \
            else features_score
    return features_importance


def generate_cumulative_top_features_scores(model, X, y, metric, n_features, ranked_features, n_fold=10,
                                            n_jobs=1):
    scores = []
    for i in range(1, n_features + 1):
        top_features = ranked_features[0:i]
        cv_result = run_kfold_cv(model, X[top_features], y, n_fold, [metric], n_jobs,
                                 **{'print_result':False, 'return_output':True})
        scores.append(np.mean(cv_result['test_{}'.format(metric)]))
    return scores

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from .data_exploration_utils import generate_correlation_df, generate_cumulative_top_features_scores
from .data_exploration_utils import generate_features_importance
''' this module is intended for data exploration with jupyter notebook '''
''' make sure that %matplotlib inline is in function '''
''' the functions in this module expect a dataframe input '''


def plot_bar_features_importance(X, y, model=None, figsize=(40, 40), cv=10, vline=None,
                                 features_importance=None):
    ''' we assume that the column names of X represents the features '''
    if (not model) and (features_importance is None or len(features_importance) == 0):
        raise ValueError("Either model must be specified along with training data or features importance must be given")

    features_importance = generate_features_importance(model, X, y, generate_std=False, cv=cv) \
        if (features_importance is None) else features_importance
    mean_importance = np.mean(features_importance, axis=0)
    fig = plt.figure(figsize=figsize)
    indices = np.argsort(mean_importance)
    plt.title('Feature Importances', fontsize=30)
    plt.barh(range(X.shape[1]), mean_importance[indices], color='r', align='center')
    if not vline:
        vline = np.mean(mean_importance)
    plt.axvline(x=vline)
    plt.yticks(range(X.shape[1]), X.columns[indices], fontsize=30)
    plt.xticks(fontsize=30)
    plt.ylim([-1, X.shape[1]])
    plt.show()
    return


def plot_box_features_importance(X, y, model=None, figsize=(40, 40), cv=10, hline=None,
                                 features_importance=None):
    ''' we assume that the column names of X represents the features '''
    if (not model) and (features_importance is None or len(features_importance) == 0):
        raise ValueError("Either model must be specified along with training data or features importance must be given")

    features_importance = generate_features_importance(model, X, y, generate_std=False, cv=cv) \
        if (features_importance is None) else features_importance
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sns.boxplot(x="features", y="value", data=pd.melt(pd.DataFrame(features_importance, columns=X.columns),
                                                      var_name="features", value_name="value"), ax=ax)
    if not hline:
        hline = np.median(np.median(features_importance, axis=0))

    plt.axhline(hline)
    plt.show()


def plot_score_by_top_features(X, y, n_features, metric, model=None, figsize=(20, 20), cv=10, vline=None,
                               mean_importance=None, scores=None):
    if (scores is None or len(scores) == 0) and (mean_importance is None or len(mean_importance) == 0):
        raise ValueError("Either metric scores must be specified or features importance must be given")
    elif scores is None:
        ranked_features = X.columns.values[np.argsort(-mean_importance)]
        scores = generate_cumulative_top_features_scores(model, X, y, metric, n_features,
                                                         ranked_features, n_fold=10, n_jobs=1)

    fig = plt.figure(figsize=figsize)
    plt.title('test_{} vs Top Features'.format(metric), fontsize=30)
    plt.plot(np.arange(1, n_features + 1), scores)


def plot_correlation_matrix(df, features, label=None, n_fold=10, imbalanced=False, verbose=True, figsize=(12, 8)):
    corr_df = generate_correlation_df(df, features, label, n_fold, imbalanced, verbose)
    plt.figure(figsize=figsize)
    sns.heatmap(data=corr_df)
    plt.show()
    plt.gcf().clear()

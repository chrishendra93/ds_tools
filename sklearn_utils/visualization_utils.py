import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

''' this module is intended for data exploration with jupyter notebook '''
''' make sure that %matplotlib inline is in function '''
''' the functions in this module expect a dataframe input '''


def plot_bar_features_importance(model, X, y, figsize=(40, 40), cv=10, features_importance=None):
    ''' we assume that the column names of X represents the features '''
    features_importance = generate_features_importance(model, X, y, generate_std=False, cv=cv) \
        if not features_importance else features_importance
    mean_importance = np.mean(features_importance, axis=0)
    fig = plt.figure(figsize=figsize)
    indices = np.argsort(mean_importance)
    plt.title('Feature Importances under XGBoost', fontsize=30)
    plt.barh(range(X.shape[1]), mean_importance[indices], color='r', align='center')
    plt.axvline(x=0.01)
    plt.yticks(range(X.shape[1]), X.columns[indices], fontsize=30)
    plt.xticks(fontsize=30)
    plt.ylim([-1, X.shape[1]])
    plt.show()
    return


def plot_box_features_importance(model, X, y, figsize=(40, 40), cv=10, features_importance=None, axhline=None):
    ''' we assume that the column names of X represents the features '''
    features_importance = generate_features_importance(model, X, y, generate_std=False, cv=cv) \
        if not features_importance else features_importance
    fig, ax = plt.subplots(1, 1, figsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    sns.boxplot(x="features", y="value", data=pd.melt(pd.DataFrame(features_importance, columns=X.columns),
                                                      var_name="features", value_name="value"), ax=ax)
    if axhline:
        plt.axhline(y=0.01)
    plt.show()


def generate_features_importance(model, X, y, generate_std=False, cv=10):
    features_importance = []
    for train_index, _ in StratifiedKFold(n_splits=cv).split(X, y):
        x_train, y_train = X.values[train_index], y.values[train_index]
        features_score = model.fit(x_train, y_train).feature_importances_
        features_importance = np.vstack([features_importance, features_score]) if len(features_importance) > 0 \
            else features_score
    return features_importance

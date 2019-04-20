import numpy as np
from sklearn import warnings
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, PredefinedSplit, cross_validate
from imblearn.pipeline import Pipeline


def generate_gridsearch_cv(model, x_train, y_train, scores,
                           params, cv=10, x_test=None, y_test=None, n_jobs=1,
                           verbose=0):
    ''' gridsearch cv on training data via kfold cross validation'''
    if isinstance(scores, str):
        scores = [scores]

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf = GridSearchCV(model, params, cv=cv, scoring=score, n_jobs=n_jobs, verbose=verbose)
            clf.fit(x_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        best_mean = clf.cv_results_['mean_test_score'][clf.best_index_]
        best_std = clf.cv_results_['std_test_score'][clf.best_index_]
        print("%0.3f (+/-%0.03f) for %r" % (best_mean, best_std * 2,
                                            clf.best_params_))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        if x_test and y_test:
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(x_test)
            print(metrics.classification_report(y_true, y_pred))
            print()


def generate_gridsearch_cv_fix_fold(model, x_train, y_train, x_fix, y_fix, scores,
                                    params, cv, x_test=None, y_test=None, n_jobs=1,
                                    verbose=0):
    ''' gridsearch cv with some of the data being fixed on the training set only '''
    ''' one possible use case is when we have an additional labelled data that we are '''
    ''' unsure if it is labelled correctly. In that case, we might want to exclude them from '''
    ''' the test set '''
    X, y, test_fold = create_predefined_fold(x_train, y_train.reshape(-1, 1),
                                             x_fix, y_fix.reshape(-1, 1))
    generate_gridsearch_cv(model, X, y, scores,
                           params, PredefinedSplit(test_fold=test_fold),
                           x_test, y_test, n_jobs, verbose)


def create_predefined_fold(x_train, y_train, x_fix, y_fix, cv=10):
    split = list(StratifiedKFold(n_splits=cv, shuffle=True).split(x_train, y_train))
    test_fold = np.zeros(len(x_train) + len(x_fix))
    for i in range(len(split)):
        test_index = split[i][1]
        test_fold[test_index] = i
    test_fold[len(x_fix):] = -1
    return np.vstack((x_train, x_fix)), np.vstack((y_train, y_fix)), test_fold


def pretty_print_kfold_result(cv_result, prefixes, model_name, return_train_score):
    ''' print mean and standard deviation of the result from sklearn.model_selection.cross_validate '''
    print("printing result for model {}:".format(model_name))
    print()
    val_type = ["test_"] if not return_train_score else ["train_", "test_"]
    for prefix in prefixes:
        for val in val_type:
            res_type = val + prefix
            mean = np.mean(cv_result[res_type])
            std = np.std(cv_result[res_type])
            print("\t {}: {} +/- {}".format(res_type, mean, std))
        print()
    print("========================================================")


def run_kfold_cv(model, X, y, cv_fold=10, metrics_list=['accuracy', 'f1'],
                 n_jobs=None, fit_params=None, sampler=None, task_type='classification',
                 return_train_score=True, return_output=False, print_result=True):
    ''' when specified, sampler must be able to work with imblearn.pipeline Pipeline class'''

    model_name = type(model).__name__
    prefixes = ["score"] if isinstance(metrics_list, str) else metrics_list

    if fit_params:
        model = model.set_params(**fit_params)

    if sampler:
        model = Pipeline([('sampling', sampler),
                          (task_type, model)])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cv_result = cross_validate(model, X, y, scoring=metrics_list, n_jobs=n_jobs, cv=cv_fold,
                                   return_train_score=return_train_score)

    if print_result:
        pretty_print_kfold_result(cv_result, prefixes, model_name, return_train_score)

    if return_output:
        return cv_result

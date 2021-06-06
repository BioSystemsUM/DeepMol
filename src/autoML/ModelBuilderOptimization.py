from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from models.sklearnModels import SklearnModel
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier


class ModelBuilderOptimization(object):

    def __init__(self):
        pass
        
    def get_model(self, model):
        if model['name'] == 'RandomForestClassifier':
            return rf_model_builder
        elif model['name'] == 'SVM':
            return svm_model_builder
        elif model['name'] == 'KNeighborsClassifier':
            return knc_model_builder
        elif model['name'] == 'DecisionTreeClassifier':
            return dt_model_builder
        elif model['name'] == 'RidgeClassifier':
            return ridge_model_builder


def rf_model_builder(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
        warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None, model_dir=None):

    rf_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, 
                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features, max_leaf_nodes=None, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split,
                bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)

    return SklearnModel(rf_model, model_dir)


def svm_model_builder(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False,
        tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
        break_ties=False, random_state=None, model_dir=None):

    svm_model = SVC(C=C, kernel=kernel, degree=degree, gamma='scale', coef0=coef0, shrinking=shrinking, 
        probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, verbose=verbose,
        max_iter=max_iter, decision_function_shape=decision_function_shape, break_ties=break_ties, random_state=random_state)

    return SklearnModel(svm_model, model_dir)


def knc_model_builder(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
        metric_params=None, n_jobs=None, model_dir=None, **kwargs):

    knc_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p,
        metric=metric, metric_params=metric_params, n_jobs=n_jobs, model_dir=model_dir, **kwargs)

    return SklearnModel(knc_model, model_dir)


def dt_model_builder(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, ccp_alpha=0.0, model_dir=None):

        dt_model = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
        min_impurity_split=min_impurity_split, class_weight=class_weight, ccp_alpha=ccp_alpha)

        return SklearnModel(dt_model, model_dir)

def ridge_model_builder(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001,
        class_weight=None, solver='auto', random_state=None, model_dir=None):

    ridge_model = RidgeClassifier(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X,
        max_iter=max_iter, tol=tol, class_weight=class_weight, solver=solver, random_state=random_state)

    return SklearnModel(ridge_model, model_dir)


def sgd_model_builder(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
        tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal',
        eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None,
        warm_start=False, average=False, model_dir=None):

    sgd_model = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
        max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon, n_jobs=n_jobs,
        random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping,
        validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, class_weight=class_weight,
        warm_start=warm_start, average=average)

    return SklearnModel(sgd_model, model_dir)


def ab_model_builder(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None,
        model_dir=None):

        ab_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
            algorithm=algorithm, random_state=random_state)

        return SklearnModel(ab_model, model_dir)

# TODO adicionar outros
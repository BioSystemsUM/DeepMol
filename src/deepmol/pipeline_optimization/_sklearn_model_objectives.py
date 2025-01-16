from deepmol.base import Predictor
from deepmol.datasets.datasets import Dataset
from deepmol.models.sklearn_model_builders import *


def linear_regression_step(trial):
    """
    Get a LinearRegression object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LinearRegression object step.
    """
    return linear_regression_model()


def ridge_step(trial):
    """
    Get a Ridge object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The Ridge object step.
    """
    alpha = trial.suggest_float('alpha_ridge_model', 0.001, 10.0)
    ridge_kwargs = {'alpha': alpha}
    return ridge_model(ridge_kwargs=ridge_kwargs)


def ridge_classifier_step(trial):
    """
    Get a RidgeClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The RidgeClassifier object step.
    """
    alpha = trial.suggest_float('alpha_ridge_classifier', 0.001, 10.0)
    ridge_classifier_kwargs = {'alpha': alpha}
    return ridge_classifier_model(ridge_classifier_kwargs=ridge_classifier_kwargs)


def ridge_cv_step(trial):
    """
    Get a RidgeCV object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The RidgeCV object step.
    """
    alpha = trial.suggest_float('alpha_ridge_cv', 0.01, 10.0)
    ridge_cv_kwargs = {'alphas': alpha}
    return ridge_cv_model(ridge_cv_kwargs=ridge_cv_kwargs)


def ridge_classifier_cv_step(trial):
    """
    Get a RidgeClassifierCV object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The RidgeClassifierCV object step.
    """
    alpha = trial.suggest_float('alpha_ridge_classifier_cv', 0.01, 10.0)
    ridge_classifier_cv_kwargs = {'alphas': alpha}
    return ridge_classifier_cv_model(ridge_classifier_cv_kwargs=ridge_classifier_cv_kwargs)


def lasso_step(trial):
    """
    Get a Lasso object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The Lasso object step.
    """
    alpha = trial.suggest_float('alpha_lasso', 0.01, 10.0)
    lasso_kwargs = {'alpha': alpha}
    return lasso_model(lasso_kwargs=lasso_kwargs)


def lasso_cv_step(trial):
    """
    Get a LassoCV object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LassoCV object step.
    """
    alpha = trial.suggest_float('alpha_lasso_cv', 0.01, 10.0)
    lasso_cv_kwargs = {'alphas': [alpha]}
    return lasso_cv_model(lasso_cv_kwargs=lasso_cv_kwargs)


def lasso_lars_cv_step(trial):
    """
    Get a LassoLarsCV object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LassoLarsCV object step.
    """
    return lasso_lars_cv_model()


def lasso_lars_ic_step(trial):
    """
    Get a LassoLarsIC object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LassoLarsIC object step.
    """
    criterion = trial.suggest_categorical('criterion_lasso_lars', ['aic', 'bic'])
    lasso_lars_ic_kwargs = {'criterion': criterion}
    return lasso_lars_ic_model(lasso_lars_ic_kwargs=lasso_lars_ic_kwargs)


def elastic_net_step(trial):
    """
    Get a ElasticNet object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    ElasticNet
        The ElasticNet object step.
    """
    alpha = trial.suggest_float('alpha_elastic_net', 0.01, 10.0)
    l1_ratio = trial.suggest_float('l1_ratio_elastic_net', 0.0, 1.0)
    elastic_net_kwargs = {'alpha': alpha, 'l1_ratio': l1_ratio}
    return elastic_net_model(elastic_net_kwargs=elastic_net_kwargs)


def ortogonal_matching_pursuit_step(trial):
    """
    Get a OrtogonalMatchingPursuit object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The OrtogonalMatchingPursuit object step.
    """
    return ortogonal_matching_pursuit_model()


def bayesian_ridge_step(trial):
    """
    Get a BayesianRidge object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The BayesianRidge object step.
    """
    alpha = trial.suggest_float('alpha_bayesian_ridge', 1e-5, 1e+1)
    lambda_1 = trial.suggest_float('lambda_1_bayesian_ridge', 1e-5, 1e+1)
    lambda_2 = trial.suggest_float('lambda_2_bayesian_ridge', 1e-5, 1e+1)
    bayesian_ridge_kwargs = {'alpha_1': alpha, 'alpha_2': alpha, 'lambda_1': lambda_1, 'lambda_2': lambda_2}
    return bayesian_ridge_model(bayesian_ridge_kwargs=bayesian_ridge_kwargs)


def ard_regression_step(trial):
    """
    Get a ARDRegression object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The ARDRegression object step.
    """
    alpha_1 = trial.suggest_loguniform('alpha_1', 1e-8, 1.0)
    alpha_2 = trial.suggest_loguniform('alpha_2', 1e-8, 1.0)
    lambda_1 = trial.suggest_loguniform('lambda_1_ard_regression', 1e-8, 1.0)
    lambda_2 = trial.suggest_loguniform('lambda_2_ard_regression', 1e-8, 1.0)
    threshold_lambda = trial.suggest_loguniform('threshold_lambda', 1e-8, 1.0)
    ard_regression_kwargs = {'alpha_1': alpha_1, 'alpha_2': alpha_2, 'lambda_1': lambda_1, 'lambda_2': lambda_2,
                             'threshold_lambda': threshold_lambda}
    return ard_regression_model(ard_regression_kwargs=ard_regression_kwargs)


def logistic_regression_step(trial):
    """
    Get a LogisticRegression object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LogisticRegression object step.
    """
    C = trial.suggest_float('C_logistic_regression', 0.01, 10.0, log=True)
    logistic_regression_kwargs = {'C': C}
    return logistic_regression_model(logistic_regression_kwargs=logistic_regression_kwargs)


def logistic_regression_multiclass_step(trial):
    """
    Get a LogisticRegressionMulticlass object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LogisticRegressionMulticlass object step.
    """
    C = trial.suggest_float('C_logistic_regression_multiclass', 0.01, 10.0, log=True)
    multiclass_type = trial.suggest_categorical('multiclass_type_logistic_regression_multiclass', ['ovr', 'multinomial'])
    logistic_regression_multiclass_kwargs = {'C': C, 'multi_class': multiclass_type}
    return logistic_regression_model(logistic_regression_kwargs=logistic_regression_multiclass_kwargs)


def logistic_regression_cv_step(trial):
    """
    Get a LogisticRegressionCV object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LogisticRegressionCV object step.
    """
    Cs = trial.suggest_int('Cs_logistic_regression_cv', 1, 10)
    logistic_regression_cv_kwargs = {'Cs': Cs}
    return logistic_regression_cv_model(logistic_regression_cv_kwargs=logistic_regression_cv_kwargs)


def logistic_regression_cv_multiclass_step(trial):
    """
    Get a LogisticRegressionMulticlassCV object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LogisticRegressionMulticlassCV object step.
    """
    Cs = trial.suggest_int('Cs_logistic_regression_cv_multiclass', 1, 10)
    multiclass_type = trial.suggest_categorical('multiclass_type_logistic_regression_cv_multiclass', ['ovr', 'multinomial'])
    logistic_regression_cv_multiclass_kwargs = {'Cs': Cs, 'multi_class': multiclass_type}
    return logistic_regression_cv_model(logistic_regression_cv_kwargs=logistic_regression_cv_multiclass_kwargs)


def tweedie_regressor_step(trial):
    """
    Get a TweedieRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The TweedieRegressor object step.
    """
    power = trial.suggest_float('power_tweedie_regressor', 0.0, 1.0)
    alpha = trial.suggest_float('alpha_tweedie_regressor', 0.0, 2.0)
    tweedie_regression_kwargs = {'power': power, 'alpha': alpha}
    return tweedie_regressor_model(tweedie_regressor_kwargs=tweedie_regression_kwargs)


def poisson_regressor_step(trial):
    """
    Get a PoissonRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The PoissonRegressor object step.
    """
    alpha = trial.suggest_float('alpha_poisson_regressor', 0.0, 2.0)
    poisson_regression_kwargs = {'alpha': alpha}
    return poisson_regressor_model(poisson_regressor_kwargs=poisson_regression_kwargs)


def gamma_regressor_step(trial):
    """
    Get a GammaRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The GammaRegressor object step.
    """
    alpha = trial.suggest_float('alpha_gamma_regressor', 0.0, 2.0)
    gamma_regression_kwargs = {'alpha': alpha}
    return gamma_regressor_model(gamma_regressor_kwargs=gamma_regression_kwargs)


def perceptron_step(trial):
    """
    Get a Perceptron object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The Perceptron object step.
    """
    alpha = trial.suggest_float('alpha_perceptron', 0.0, 2.0)
    perceptron_kwargs = {'alpha': alpha}
    return perceptron_model(perceptron_kwargs=perceptron_kwargs)


def passive_aggressive_regressor_step(trial):
    """
    Get a PassiveAggressiveRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The PassiveAggressiveRegressor object step.
    """
    C = trial.suggest_float('C_passive_aggressive_regressor', 0.0, 10.0)
    passive_aggressive_regressor_kwargs = {'C': C}
    return passive_aggressive_regressor_model(passive_aggressive_regressor_kwargs=passive_aggressive_regressor_kwargs)


def passive_aggressive_classifier_step(trial):
    """
    Get a PassiveAggressiveClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The PassiveAggressiveClassifier object step.
    """
    C = trial.suggest_float('C_passive_aggressive_classifier', 0.0, 10.0)
    passive_aggressive_classifier_kwargs = {'C': C}
    return passive_aggressive_classifier_model(
        passive_aggressive_classifier_kwargs=passive_aggressive_classifier_kwargs)


def huber_regressor_step(trial):
    """
    Get a HuberRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The HuberRegressor object step.
    """
    alpha = trial.suggest_float('alpha_huber_regressor', 0.0, 2.0)
    epsilon = trial.suggest_float('epsilon_huber_regressor', 1.0, 2.0)
    huber_regressor_kwargs = {'alpha': alpha, 'epsilon': epsilon}
    return huber_regressor_model(huber_regressor_kwargs=huber_regressor_kwargs)


def ransac_regressor_step(trial):
    """
    Get a RANSACRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The RANSACRegressor object step.
    """
    base_estimator = trial.suggest_categorical('base_estimator_ransac_regressor', ['linear', 'ridge', 'lasso'])
    if base_estimator == 'linear':
        model = LinearRegression()
    elif base_estimator == 'ridge':
        model = Ridge()
    else:
        model = Lasso()
    min_samples = trial.suggest_float('min_samples', 0.0, 1.0)
    ransac_regressor_kwargs = {'base_estimator': model, 'min_samples': min_samples}
    return ransac_regressor_model(ransac_regressor_kwargs=ransac_regressor_kwargs)


def theil_sen_regressor_step(trial):
    """
    Get a TheilSenRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The TheilSenRegressor object step.
    """
    theil_sen_regressor_kwargs = {}
    return theil_sen_regressor_model(theil_sen_regressor_kwargs=theil_sen_regressor_kwargs)


def quantile_regressor_step(trial):
    """
    Get a QuantileRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The QuantileRegressor object step.
    """
    alpha = trial.suggest_float('alpha_quantile_regressor', 0.0, 1.0)
    quantile = trial.suggest_float('quantile', 0.1, 0.9)
    quantile_regressor_kwargs = {'alpha': alpha, 'quantile': quantile}
    return quantile_regressor_model(quantile_regressor_kwargs=quantile_regressor_kwargs)


def linear_discriminat_analysis_step(trial):
    """
    Get a LinearDiscriminantAnalysis object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The LinearDiscriminantAnalysis object step.
    """
    solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'])
    linear_discriminant_analysis_kwargs = {'solver': solver}
    return linear_discriminant_analysis_model(linear_discriminant_analysis_kwargs=linear_discriminant_analysis_kwargs)


def quadratic_discriminant_analysis_step(trial):
    """
    Get a QuadraticDiscriminantAnalysis object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The QuadraticDiscriminantAnalysis object step.
    """
    reg_param = trial.suggest_float('reg_param', 0.0, 1.0)
    quadratic_discriminant_analysis_kwargs = {'reg_param': reg_param}
    return quadratic_discriminant_analysis_model(
        quadratic_discriminant_analysis_kwargs=quadratic_discriminant_analysis_kwargs)


def kernel_ridge_step(trial):
    """
    Get a KernelRidge object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The KernelRidge object step.
    """
    alpha = trial.suggest_float('alpha_kernel_ridge', 0.0, 1.0)
    kernel = trial.suggest_categorical('kernel_ridge', ['linear', 'poly', 'rbf'])
    gamma = trial.suggest_float('gamma_kernel_ridge', 0.0, 1.0)
    kernel_ridge_kwargs = {'alpha': alpha, 'kernel': kernel, 'gamma': gamma}
    return kernel_ridge_regressor_model(kernel_ridge_regressor_kwargs=kernel_ridge_kwargs)


def svc_step(trial):
    """
    Get a SVC object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The SVC object step.
    """
    C = trial.suggest_float('C_svc', 0.1, 10.0)
    kernel = trial.suggest_categorical('kernel_svc', ['linear', 'poly', 'rbf'])
    gamma = trial.suggest_float('gamma_svc', 0.0, 1.0)
    degree = trial.suggest_int('degree_svc', 2, 5)
    svc_kwargs = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
    return svc_model(svc_kwargs=svc_kwargs)


def nu_svc_step(trial):
    """
    Get a NuSVC object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The NuSVC object step.
    """
    nu = trial.suggest_float('nu_nu_svc', 0.0, 1.0)
    kernel = trial.suggest_categorical('kernel_nu_svc', ['linear', 'poly', 'rbf'])
    gamma = trial.suggest_float('gamma_nu_svc', 0.0, 1.0)
    degree = trial.suggest_int('degree_nu_svc', 2, 5)
    nu_svc_kwargs = {'nu': nu, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
    return nu_svc_model(nu_svc_kwargs=nu_svc_kwargs)


def linear_svc_step(trial):
    """
    Get a LinearSVC object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The LinearSVC object step.
    """
    C = trial.suggest_float('C_linear_svc', 0.1, 10.0)
    linear_svc_kwargs = {'C': C}
    return linear_svc_model(linear_svc_kwargs=linear_svc_kwargs)


def linear_svc_multiclass_step(trial):
    """
    Get a LinearSVC object for the Optuna optimization for multiclass classification.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The LinearSVC object step.
    """
    C = trial.suggest_float('C_linear_svc_multiclass', 0.1, 10.0)
    multiclass_type = trial.suggest_categorical('multiclass_type_linear_svc_multiclass', ['ovr', 'crammer_singer'])
    linear_svc_multiclass_kwargs = {'C': C, 'multi_class': multiclass_type}
    return linear_svc_model(linear_svc_kwargs=linear_svc_multiclass_kwargs)


def svr_step(trial):
    """
    Get a SVR object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The SVR object step.
    """
    C = trial.suggest_float('C_svr', 0.1, 10.0)
    kernel = trial.suggest_categorical('kernel_svr', ['linear', 'poly', 'rbf'])
    gamma = trial.suggest_float('gamma_svr', 0.0, 1.0)
    degree = trial.suggest_int('degree_svr', 2, 5)
    svr_kwargs = {'C': C, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
    return svr_model(svr_kwargs=svr_kwargs)


def nu_svr_step(trial):
    """
    Get a NuSVR object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The NuSVR object step.
    """
    nu = trial.suggest_float('nu_nu_svr', 0.0, 1.0)
    C = trial.suggest_float('C_nu_svr', 0.1, 10.0)
    kernel = trial.suggest_categorical('kernel_nu_svr', ['linear', 'poly', 'rbf'])
    gamma = trial.suggest_float('gamma_nu_svr', 0.0, 1.0)
    degree = trial.suggest_int('degree_nu_svr', 2, 5)
    nu_svr_kwargs = {'nu': nu, 'kernel': kernel, 'gamma': gamma, 'degree': degree, 'C': C}
    return nu_svr_model(nu_svr_kwargs=nu_svr_kwargs)


def linear_svr_step(trial):
    """
    Get a LinearSVR object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The LinearSVR object step.
    """
    epsilon = trial.suggest_float('epsilon_linear_svr', 0.0, 1.0)
    C = trial.suggest_float('C_linear_svr', 0.1, 10.0)
    linear_svr_kwargs = {'epsilon': epsilon, 'C': C}
    return linear_svr_model(linear_svr_kwargs=linear_svr_kwargs)


def one_class_svm_step(trial):
    """
    Get a OneClassSVM object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The OneClassSVM object step.
    """
    nu = trial.suggest_float('nu_one_class_svm', 0.0, 1.0)
    kernel = trial.suggest_categorical('kernel_one_class_svm', ['linear', 'poly', 'rbf'])
    gamma = trial.suggest_float('gamma_one_class_svm', 0.0, 1.0)
    degree = trial.suggest_int('degree_one_class_svm', 2, 5)
    one_class_svm_kwargs = {'nu': nu, 'kernel': kernel, 'gamma': gamma, 'degree': degree}
    return one_class_svm_model(one_class_svm_kwargs=one_class_svm_kwargs)


def sgd_regressor_step(trial):
    """
    Get a SGDRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The SGDRegressor object step.
    """
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
    alpha = trial.suggest_float('alpha_sgd_regressor', 0.0, 1.0)
    l1_ratio = trial.suggest_float('l1_ratio_sgd_regressor', 0.0, 1.0)
    epsilon = trial.suggest_float('epsilon_sgd_regressor', 0.0, 1.0)
    sgd_regressor_kwargs = {'penalty': penalty, 'alpha': alpha, 'l1_ratio': l1_ratio, 'epsilon': epsilon}
    return sgd_regressor_model(sgd_regressor_kwargs=sgd_regressor_kwargs)


def sgd_classifier_step(trial):
    """
    Get a SGDClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The SGDClassifier object step.
    """
    penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet'])
    alpha = trial.suggest_float('alpha_sgd_classifier', 0.0, 1.0)
    l1_ratio = trial.suggest_float('l1_ratio_sgd_classifier', 0.0, 1.0)
    epsilon = trial.suggest_float('epsilon_sgd_classifier', 0.0, 1.0)
    sgd_classifier_kwargs = {'penalty': penalty, 'alpha': alpha, 'l1_ratio': l1_ratio, 'epsilon': epsilon}
    return sgd_classifier_model(sgd_classifier_kwargs=sgd_classifier_kwargs)


def sgd_one_class_svm_step(trial):
    """
    Get a SGDOneClassSVM object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    OneClassSVM
        The SGDOneClassSVM object step.
    """
    nu = trial.suggest_float('nu_sgd_one_class_svm', 0.0, 1.0)
    tol = trial.suggest_float('tol', 0.0, 1.0)
    sgd_one_class_svm_kwargs = {'nu': nu, 'tol': tol}
    return sgd_one_class_svm_model(sgd_one_class_svm_kwargs=sgd_one_class_svm_kwargs)


def kneighbors_regressor_step(trial):
    """
    Get a KNeighborsRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The KNeighborsRegressor object step.
    """
    n_neighbors = trial.suggest_int('n_neighbors_kneighbors_regressor', 2, 10)
    weights = trial.suggest_categorical('weights_kneighbors_regressor', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm_kneighbors_regressor', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size_kneighbors_regressor', 2, 10)
    p = trial.suggest_int('p', 2, 10)
    kneighbors_regressor_kwargs = {'n_neighbors': n_neighbors, 'weights': weights, 'algorithm': algorithm,
                                   'leaf_size': leaf_size, 'p': p}
    return kneighbors_regressor_model(kneighbors_regressor_kwargs=kneighbors_regressor_kwargs)


def kneighbors_classifier_step(trial):
    """
    Get a KNeighborsClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The KNeighborsClassifier object step.
    """
    n_neighbors = trial.suggest_int('n_neighbors_kneighbors_classifier', 2, 10)
    weights = trial.suggest_categorical('weights_kneighbors_classifier', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm_kneighbors_classifier', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size_kneighbors_classifier', 2, 10)
    p = trial.suggest_int('p', 2, 10)
    kneighbors_classifier_kwargs = {'n_neighbors': n_neighbors, 'weights': weights, 'algorithm': algorithm,
                                    'leaf_size': leaf_size, 'p': p}
    return kneighbors_classifier_model(kneighbors_classifier_kwargs=kneighbors_classifier_kwargs)


def radius_neighbors_regressor_step(trial):
    """
    Get a RadiusNeighborsRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The RadiusNeighborsRegressor object step.
    """
    radius_n = trial.suggest_float('radius_n_radius_neighbors_regressor', 0.0, 10.0)
    weights = trial.suggest_categorical('weights_radius_neighbors_regressor', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm_radius_neighbors_regressor', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size_radius_neighbors_regressor', 2, 10)
    p = trial.suggest_int('p', 2, 10)
    radius_neighbors_regressor_kwargs = {'radius': radius_n, 'weights': weights, 'algorithm': algorithm,
                                         'leaf_size': leaf_size, 'p': p}
    return radius_neighbors_regressor_model(radius_neighbors_regressor_kwargs=radius_neighbors_regressor_kwargs)


def radius_neighbors_classifier_step(trial):
    """
    Get a RadiusNeighborsClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The RadiusNeighborsClassifier object step.
    """
    radius_n = trial.suggest_int('radius_n_radius_neighbors_classifier', 0.0, 10.0)
    weights = trial.suggest_categorical('weights_radius_neighbors_classifier', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm_radius_neighbors_classifier', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    leaf_size = trial.suggest_int('leaf_size_radius_neighbors_classifier', 2, 10)
    p = trial.suggest_int('p', 2, 10)
    radius_neighbors_classifier_kwargs = {'radius': radius_n, 'weights': weights, 'algorithm': algorithm,
                                          'leaf_size': leaf_size, 'p': p}
    return radius_neighbors_classifier_model(radius_neighbors_classifier_kwargs=radius_neighbors_classifier_kwargs)


def nearest_centroid_step(trial):
    """
    Get a NearestCentroid object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The NearestCentroid object step.
    """
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
    nearest_centroid_kwargs = {'metric': metric}
    return nearest_centroid_model(nearest_centroid_kwargs=nearest_centroid_kwargs)


def gaussian_process_regressor_step(trial):
    """
    Get a GaussianProcessRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The GaussianProcessRegressor object step.
    """
    alpha = trial.suggest_float('alpha_gaussian_process_regressor', 0.0, 1.0)
    gaussian_process_regressor_kwargs = {'alpha': alpha}
    return gaussian_process_regressor_model(gaussian_process_regressor_kwargs=gaussian_process_regressor_kwargs)


def gaussian_process_multiclass_classifier_step(trial):
    """
    Get a GaussianProcessClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The GaussianProcessClassifier object step.
    """
    multiclass_type = trial.suggest_categorical('multiclass_type_gaussian_process_multiclass', ['one_vs_rest', 'one_vs_one'])
    gaussian_process_multiclass_classifier_kwargs = {'multi_class': multiclass_type}
    return gaussian_process_classifier_model(
        gaussian_process_classifier_kwargs=gaussian_process_multiclass_classifier_kwargs)


def gaussian_process_classifier_step(trial):
    """
    Get a GaussianProcessClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The GaussianProcessClassifier object step.
    """
    gaussian_process_classifier_kwargs = {}
    return gaussian_process_classifier_model(gaussian_process_classifier_kwargs=gaussian_process_classifier_kwargs)


def pls_regression_step(trial):
    """
    Get a PLSRegression object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The PLSRegression object step.
    """
    n_components = trial.suggest_int('n_components', 2, 10)
    scale = trial.suggest_categorical('scale', [True, False])
    pls_regression_kwargs = {'n_components': n_components, 'scale': scale}
    return pls_regression_model(pls_regression_kwargs=pls_regression_kwargs)


def gaussian_nb_step(trial):
    """
    Get a GaussianNB object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The GaussianNB object step.
    """
    gaussian_nb_kwargs = {}
    return gaussian_nb_model(gaussian_nb_kwargs=gaussian_nb_kwargs)


def multinomial_nb_step(trial):
    """
    Get a MultinomialNB object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The MultinomialNB object step.
    """
    alpha = trial.suggest_float('alpha_multinomial_nb', 0.0, 1.0)
    multinomial_nb_kwargs = {'alpha': alpha}
    return multinomial_nb_model(multinomial_nb_kwargs=multinomial_nb_kwargs)


def bernoulli_nb_step(trial):
    """
    Get a BernoulliNB object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The BernoulliNB object step.
    """
    alpha = trial.suggest_float('alpha_bernoulli_nb', 0.0, 1.0)
    bernoulli_nb_kwargs = {'alpha': alpha}
    return bernoulli_nb_model(bernoulli_nb_kwargs=bernoulli_nb_kwargs)


def categorical_nb_step(trial):
    """
    Get a CategoricalNB object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The CategoricalNB object step.
    """
    alpha = trial.suggest_float('alpha_categorical_nb', 0.0, 1.0)
    categorical_nb_kwargs = {'alpha': alpha}
    return categorical_nb_model(categorical_nb_kwargs=categorical_nb_kwargs)


def complement_nb_step(trial):
    """
    Get a ComplementNB object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The ComplementNB object step.
    """
    alpha = trial.suggest_float('alpha_complement_nb', 0.0, 1.0)
    complement_nb_kwargs = {'alpha': alpha}
    return complement_nb_model(complement_nb_kwargs=complement_nb_kwargs)


def decision_tree_regressor_step(trial):
    """
    Get a DecisionTreeRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The DecisionTreeRegressor object step.
    """
    decision_tree_regressor_kwargs = {}
    return decision_tree_regressor_model(decision_tree_regressor_kwargs=decision_tree_regressor_kwargs)


def decision_tree_classifier_step(trial):
    """
    Get a DecisionTreeClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The DecisionTreeClassifier object step.
    """
    criterion = trial.suggest_categorical('criterion_decision_tree', ['gini', 'entropy'])
    decision_tree_classifier_kwargs = {'criterion': criterion}
    return decision_tree_classifier_model(decision_tree_classifier_kwargs=decision_tree_classifier_kwargs)


def extra_tree_classifier_step(trial):
    """
    Get a ExtraTreeClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The ExtraTreeClassifier object step.
    """
    criterion = trial.suggest_categorical('criterion_extra_tree', ['gini', 'entropy'])
    extra_tree_classifier_kwargs = {'criterion': criterion}
    return extra_tree_classifier_model(extra_tree_classifier_kwargs=extra_tree_classifier_kwargs)


def extra_tree_regressor_step(trial):
    """
    Get a ExtraTreeRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The ExtraTreeRegressor object step.
    """
    extra_tree_regressor_kwargs = {}
    return extra_tree_regressor_model(extra_tree_regressor_kwargs=extra_tree_regressor_kwargs)


def random_forest_regressor_step(trial):
    """
    Get a RandomForestRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The RandomForestRegressor object step.
    """
    n_estimators = trial.suggest_int('n_estimators_random_forest_regressor', 100, 1000, step=100)
    criterion = trial.suggest_categorical('criterion_random_forest_regressor', ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'])
    max_features = trial.suggest_categorical('max_features_random_forest_regressor', ['sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap_random_forest_regressor', [True, False])
    random_forest_regressor_kwargs = {'n_estimators': n_estimators, 'criterion': criterion,
                                      'max_features': max_features, 'bootstrap': bootstrap}
    return random_forest_regressor_model(random_forest_regressor_kwargs=random_forest_regressor_kwargs)


def random_forest_classifier_step(trial):
    """
    Get a RandomForestClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The RandomForestClassifier object step.
    """
    n_estimators = trial.suggest_int('n_estimators_random_forest_classifier', 100, 1000, step=100)
    criterion = trial.suggest_categorical('criterion_random_forest_classifier', ['gini', 'entropy'])
    max_features = trial.suggest_categorical('max_features_random_forest_classifier', ['sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap_random_forest_classifier', [True, False])
    random_forest_classifier_kwargs = {'n_estimators': n_estimators, 'criterion': criterion,
                                       'max_features': max_features, 'bootstrap': bootstrap}
    return random_forest_classifier_model(random_forest_classifier_kwargs=random_forest_classifier_kwargs)


def extra_trees_regressor_step(trial):
    """
    Get a ExtraTreesRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The ExtraTreesRegressor object step.
    """
    n_estimators = trial.suggest_int('n_estimators_extra_trees_regressor', 100, 1000, step=100)
    criterion = trial.suggest_categorical('criterion_extra_trees_regressor', ['friedman_mse', 'squared_error', 'absolute_error', 'poisson'])
    max_features = trial.suggest_categorical('max_features_extra_trees_regressor', ['sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap_extra_trees_regressor', [True, False])
    extra_trees_regressor_kwargs = {'n_estimators': n_estimators, 'criterion': criterion,
                                    'max_features': max_features, 'bootstrap': bootstrap}
    return extra_trees_regressor_model(extra_trees_regressor_kwargs=extra_trees_regressor_kwargs)


def extra_trees_classifier_step(trial):
    """
    Get a ExtraTreesClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The ExtraTreesClassifier object step.
    """
    n_estimators = trial.suggest_int('n_estimators_extra_trees_classifier', 100, 1000, step=100)
    criterion = trial.suggest_categorical('criterion_extra_trees_classifier', ['gini', 'entropy'])
    max_features = trial.suggest_categorical('max_features_extra_trees_classifier', ['sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap_extra_trees_classifier', [True, False])
    extra_trees_classifier_kwargs = {'n_estimators': n_estimators, 'criterion': criterion,
                                     'max_features': max_features, 'bootstrap': bootstrap}
    return extra_trees_classifier_model(extra_trees_classifier_kwargs=extra_trees_classifier_kwargs)


def ada_boost_regressor_step(trial):
    """
    Get a AdaBoostRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The AdaBoostRegressor object step.
    """
    n_estimators = trial.suggest_int('n_estimators_ada_boost_regressor', 50, 500, step=50)
    learning_rate = trial.suggest_float('learning_rate_ada_boost_regressor', 0.01, 1.0)
    loss = trial.suggest_categorical('loss_ada_boost_regressor', ['linear', 'square', 'exponential'])
    ada_boost_regressor_kwargs = {'n_estimators': n_estimators, 'learning_rate': learning_rate, 'loss': loss}
    return ada_boost_regressor_model(ada_boost_regressor_kwargs=ada_boost_regressor_kwargs)


def ada_boost_classifier_step(trial):
    """
    Get a AdaBoostClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The AdaBoostClassifier object step.
    """
    n_estimators = trial.suggest_int('n_estimators_ada_boost_classifier', 50, 500, step=50)
    learning_rate = trial.suggest_float('learning_rate_ada_boost_classifier', 0.01, 1.0)
    algorithm = trial.suggest_categorical('algorithm_ada_boost_classifier', ['SAMME', 'SAMME.R'])
    ada_boost_classifier_kwargs = {'n_estimators': n_estimators, 'learning_rate': learning_rate,
                                   'algorithm': algorithm}
    return ada_boost_classifier_model(ada_boost_classifier_kwargs=ada_boost_classifier_kwargs)


def gradient_boosting_regressor_step(trial):
    """
    Get a GradientBoostingRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The GradientBoostingRegressor object step.
    """
    loss = trial.suggest_categorical('loss_gradient_boosting_regressor', ['ls', 'lad', 'huber'])
    n_estimators = trial.suggest_int('n_estimators_gradient_boosting_regressor', 50, 500, step=50)
    learning_rate = trial.suggest_float('learning_rate_gradient_boosting_regressor', 0.01, 1.0)
    criterion = trial.suggest_categorical('criterion_gradient_boosting_regressor', ['friedman_mse', 'squared_error'])
    max_features = trial.suggest_categorical('max_features_gradient_boosting_regressor', ['sqrt', 'log2'])
    gradient_boosting_regressor_kwargs = {'loss': loss, 'n_estimators': n_estimators,
                                          'learning_rate': learning_rate, 'criterion': criterion,
                                          'max_features': max_features}
    return gradient_boosting_regressor_model(gradient_boosting_regressor_kwargs=gradient_boosting_regressor_kwargs)


def gradient_boosting_classifier_step(trial):
    """
    Get a GradientBoostingClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The GradientBoostingClassifier object step.
    """
    loss = trial.suggest_categorical('loss_gradient_boosting_classifier', ['deviance', 'exponential'])
    n_estimators = trial.suggest_int('n_estimators_gradient_boosting_classifier', 50, 500, step=50)
    learning_rate = trial.suggest_float('learning_rate_gradient_boosting_classifier', 0.01, 1.0)
    criterion = trial.suggest_categorical('criterion_gradient_boosting_classifier', ['friedman_mse', 'squared_error'])
    max_features = trial.suggest_categorical('max_features_gradient_boosting_classifier', ['sqrt', 'log2'])
    gradient_boosting_classifier_kwargs = {'loss': loss, 'n_estimators': n_estimators,
                                           'learning_rate': learning_rate, 'criterion': criterion,
                                           'max_features': max_features}
    return gradient_boosting_classifier_model(gradient_boosting_classifier_kwargs=gradient_boosting_classifier_kwargs)


def gradient_boosting_multiclass_classifier_step(trial):
    """
    Get a GradientBoostingClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The GradientBoostingClassifier object step.
    """
    loss = trial.suggest_categorical('loss_gradient_boosting_multiclass_classifier', ['deviance', 'log_loss'])
    n_estimators = trial.suggest_int('n_estimators_gradient_boosting_multiclass_classifier', 50, 500, step=50)
    learning_rate = trial.suggest_float('learning_rate_gradient_boosting_multiclass_classifier', 0.01, 1.0)
    criterion = trial.suggest_categorical('criterion_gradient_boosting_multiclass', ['friedman_mse', 'squared_error'])
    max_features = trial.suggest_categorical('max_features_gradient_boosting_multiclass_classifier', ['sqrt', 'log2'])
    gradient_boosting_classifier_kwargs = {'loss': loss, 'n_estimators': n_estimators,
                                           'learning_rate': learning_rate, 'criterion': criterion,
                                           'max_features': max_features}
    return gradient_boosting_classifier_model(gradient_boosting_classifier_kwargs=gradient_boosting_classifier_kwargs)


def hist_gradient_boosting_regressor_step(trial):
    """
    Get a HistGradientBoostingRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The HistGradientBoostingRegressor object step.
    """
    loss = trial.suggest_categorical('loss_hist_gradient_boosting_regressor', ['poisson', 'absolute_error', 'squared_error'])
    learning_rate = trial.suggest_float('learning_rate_hist_gradient_boosting_regressor', 0.01, 1.0)
    hist_gradient_boosting_regressor_kwargs = {'loss': loss, 'learning_rate': learning_rate}
    return hist_gradient_boosting_regressor_model(
        hist_gradient_boosting_regressor_kwargs=hist_gradient_boosting_regressor_kwargs)


def hist_gradient_boosting_classifier_step(trial):
    """
    Get a HistGradientBoostingClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The HistGradientBoostingClassifier object step.
    """
    learning_rate = trial.suggest_float('learning_rate_hist_gradient_boosting_classifier', 0.01, 1.0)
    hist_gradient_boosting_classifier_kwargs = {'learning_rate': learning_rate}
    return hist_gradient_boosting_classifier_model(
        hist_gradient_boosting_classifier_kwargs=hist_gradient_boosting_classifier_kwargs)


def voting_regressor_step(trial):
    """
    Get a VotingRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The VotingRegressor object step.
    """
    estimators = [('lr', LinearRegression()), ('svr', SVR()), ('rfr', RandomForestRegressor()),
                  ('gbr', GradientBoostingRegressor()), ('mlpr', MLPRegressor())]
    weights = [trial.suggest_float('weight_lr', 0.0, 1.0), trial.suggest_float('weight_svr', 0.0, 1.0),
               trial.suggest_float('weight_rfr', 0.0, 1.0), trial.suggest_float('weight_gbr', 0.0, 1.0),
               trial.suggest_float('weight_mlp', 0.0, 1.0)]
    total_sum = sum(weights)
    weights = [weight / total_sum for weight in weights]
    voting_regressor_kwargs = {'estimators': estimators, 'weights': weights}
    return voting_regressor_model(voting_regressor_kwargs=voting_regressor_kwargs)


def voting_classifier_step(trial):
    """
    Get a VotingClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The VotingClassifier object step.
    """
    estimators = [('lr', LogisticRegression()), ('svc', SVC()), ('rfr', RandomForestClassifier()),
                  ('gbr', GradientBoostingClassifier()), ('mlpr', MLPClassifier())]
    weights = [trial.suggest_float('weight_lr', 0.0, 1.0), trial.suggest_float('weight_svc', 0.0, 1.0),
               trial.suggest_float('weight_rfr', 0.0, 1.0), trial.suggest_float('weight_gbr', 0.0, 1.0),
               trial.suggest_float('weight_mlp', 0.0, 1.0)]
    total_sum = sum(weights)
    weights = [weight / total_sum for weight in weights]
    voting_classifier_kwargs = {'estimators': estimators, 'weights': weights}
    return voting_classifier_model(voting_classifier_kwargs=voting_classifier_kwargs)


def stacking_regressor_step(trial):
    """
    Get a StackingRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The StackingRegressor object step.
    """
    estimators = [('lr', LinearRegression()), ('svr', SVR()), ('rfr', RandomForestRegressor()),
                  ('gbr', GradientBoostingRegressor())]
    final_estimator = MLPRegressor()
    stacking_regressor_kwargs = {'estimators': estimators, 'final_estimator': final_estimator}
    return stacking_regressor_model(stacking_regressor_kwargs=stacking_regressor_kwargs)


def stacking_classifier_step(trial):
    """
    Get a StackingClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The StackingClassifier object step.
    """
    estimators = [('lr', LogisticRegression()), ('svc', SVC()), ('rfr', RandomForestClassifier()),
                  ('gbr', GradientBoostingClassifier())]
    final_estimator = MLPClassifier()
    stacking_classifier_kwargs = {'estimators': estimators, 'final_estimator': final_estimator}
    return stacking_classifier_model(stacking_classifier_kwargs=stacking_classifier_kwargs)


def bagging_regressor_step(trial):
    """
    Get a BaggingRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Predictor
        The BaggingRegressor object step.
    """
    base_estimator = trial.suggest_categorical('base_estimator_bagging_regressor', ['lr', 'svr', 'rfr', 'gbr', 'mlpr'])
    if base_estimator == 'lr':
        base_estimator = LinearRegression()
    elif base_estimator == 'svr':
        base_estimator = SVR()
    elif base_estimator == 'rfr':
        base_estimator = RandomForestRegressor()
    elif base_estimator == 'gbr':
        base_estimator = GradientBoostingRegressor()
    else:
        base_estimator = MLPRegressor()
    n_estimators = trial.suggest_int('n_estimators_bagging_regressor', 50, 500, step=50)
    bootstrap = trial.suggest_categorical('bootstrap_bagging_regressor', [True, False])
    bootstrap_features = trial.suggest_categorical('bootstrap_features', [True, False])
    bagging_regressor_kwargs = {'base_estimator': base_estimator, 'n_estimators': n_estimators,
                                'bootstrap': bootstrap, 'bootstrap_features': bootstrap_features}
    return bagging_regressor_model(bagging_regressor_kwargs=bagging_regressor_kwargs)


def bagging_classifier_step(trial):
    """
    Get a BaggingClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The BaggingClassifier object step.
    """
    base_estimator = trial.suggest_categorical('base_estimator_bagging_classifier', ['lr', 'svc', 'rfr', 'gbr', 'mlpr'])
    if base_estimator == 'lr':
        base_estimator = LogisticRegression()
    elif base_estimator == 'svc':
        base_estimator = SVC()
    elif base_estimator == 'rfr':
        base_estimator = RandomForestClassifier()
    elif base_estimator == 'gbr':
        base_estimator = GradientBoostingClassifier()
    else:
        base_estimator = MLPClassifier()
    n_estimators = trial.suggest_int('n_estimators_bagging_classifier', 50, 500, step=50)
    bootstrap = trial.suggest_categorical('bootstrap_bagging_classifier', [True, False])
    bootstrap_features = trial.suggest_categorical('bootstrap_features', [True, False])
    bagging_classifier_kwargs = {'base_estimator': base_estimator, 'n_estimators': n_estimators,
                                 'bootstrap': bootstrap, 'bootstrap_features': bootstrap_features}
    return bagging_classifier_model(bagging_classifier_kwargs=bagging_classifier_kwargs)


def one_vs_rest_classifier_step(trial):
    """
    Get a OneVsRestClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The OneVsRestClassifier object step.
    """
    estimator = trial.suggest_categorical('estimator_one_vs_rest_classifier', ['lr', 'svc', 'rfr', 'gbr', 'mlpr'])
    if estimator == 'lr':
        estimator = LogisticRegression()
    elif estimator == 'svc':
        estimator = SVC()
    elif estimator == 'rfr':
        estimator = RandomForestClassifier()
    elif estimator == 'gbr':
        estimator = GradientBoostingClassifier()
    else:
        estimator = MLPClassifier()
    one_vs_rest_classifier_kwargs = {'estimator': estimator}
    return one_vs_rest_classifier_model(one_vs_rest_classifier_kwargs=one_vs_rest_classifier_kwargs)


def one_vs_one_classifier_step(trial):
    """
    Get a OneVsOneClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The OneVsOneClassifier object step.
    """
    estimator = trial.suggest_categorical('estimator_one_vs_one_classifier', ['lr', 'svc', 'rfr', 'gbr', 'mlpr'])
    if estimator == 'lr':
        estimator = LogisticRegression()
    elif estimator == 'svc':
        estimator = SVC()
    elif estimator == 'rfr':
        estimator = RandomForestClassifier()
    elif estimator == 'gbr':
        estimator = GradientBoostingClassifier()
    else:
        estimator = MLPClassifier()
    one_vs_one_classifier_kwargs = {'estimator': estimator}
    return one_vs_one_classifier_model(one_vs_one_classifier_kwargs=one_vs_one_classifier_kwargs)


def output_code_classifier_step(trial):
    """
    Get a OutputCodeClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The OutputCodeClassifier object step.
    """
    estimator = trial.suggest_categorical('estimator_output_code_classifier', ['lr', 'svc', 'rfr', 'gbr', 'mlpr'])
    if estimator == 'lr':
        estimator = LogisticRegression()
    elif estimator == 'svc':
        estimator = SVC()
    elif estimator == 'rfr':
        estimator = RandomForestClassifier()
    elif estimator == 'gbr':
        estimator = GradientBoostingClassifier()
    else:
        estimator = MLPClassifier()
    code_size = trial.suggest_float('code_size', 1, 10, step=0.5)
    output_code_classifier_kwargs = {'estimator': estimator, 'code_size': code_size}
    return output_code_classifier_model(output_code_classifier_kwargs=output_code_classifier_kwargs)


def multi_output_classifier_step(trial):
    """
    Get a MultiOutputClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The MultiOutputClassifier object step.
    """
    estimator = trial.suggest_categorical('estimator_multi_output_classifier', ['lr', 'svc', 'rfr', 'gbr', 'mlpr'])
    if estimator == 'lr':
        estimator = LogisticRegression()
    elif estimator == 'svc':
        estimator = SVC()
    elif estimator == 'rfr':
        estimator = RandomForestClassifier()
    elif estimator == 'gbr':
        estimator = GradientBoostingClassifier()
    else:
        estimator = MLPClassifier()
    multi_output_classifier_kwargs = {'estimator': estimator}
    return multi_output_classifier_model(multi_output_classifier_kwargs=multi_output_classifier_kwargs)


def classifier_chain_step(trial):
    """
    Get a ClassifierChain object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The ClassifierChain object step.
    """
    estimator = trial.suggest_categorical('estimator_classifier_chain', ['lr', 'svc', 'rfr', 'gbr', 'mlpr'])
    if estimator == 'lr':
        estimator = LogisticRegression()
    elif estimator == 'svc':
        estimator = SVC()
    elif estimator == 'rfr':
        estimator = RandomForestClassifier()
    elif estimator == 'gbr':
        estimator = GradientBoostingClassifier()
    else:
        estimator = MLPClassifier()
    order = trial.suggest_categorical('order_classifier_chain', ['random', None])
    classifier_chain_kwargs = {'estimator': estimator, 'order': order}
    return classifier_chain_model(classifier_chain_kwargs=classifier_chain_kwargs)


def multi_output_regressor_step(trial):
    """
    Get a MultiOutputRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Regressor
        The MultiOutputRegressor object step.
    """
    estimator = trial.suggest_categorical('estimator_multi_output_regressor', ['lr', 'svr', 'rfr', 'gbr', 'mlpr'])
    if estimator == 'lr':
        estimator = LinearRegression()
    elif estimator == 'svr':
        estimator = SVR()
    elif estimator == 'rfr':
        estimator = RandomForestRegressor()
    elif estimator == 'gbr':
        estimator = GradientBoostingRegressor()
    else:
        estimator = MLPRegressor()
    multi_output_regressor_kwargs = {'estimator': estimator}
    return multi_output_regressor_model(multi_output_regressor_kwargs=multi_output_regressor_kwargs)


def regressor_chain_step(trial):
    """
    Get a RegressorChain object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Regressor
        The RegressorChain object step.
    """
    estimator = trial.suggest_categorical('estimator_regressor_chain', ['lr', 'svr', 'rfr', 'gbr', 'mlpr'])
    if estimator == 'lr':
        estimator = LinearRegression()
    elif estimator == 'svr':
        estimator = SVR()
    elif estimator == 'rfr':
        estimator = RandomForestRegressor()
    elif estimator == 'gbr':
        estimator = GradientBoostingRegressor()
    else:
        estimator = MLPRegressor()
    order = trial.suggest_categorical('order_regressor_chain', ['random', 'count', 'prior'])
    regressor_chain_kwargs = {'estimator': estimator, 'order': order}
    return regressor_chain_model(regressor_chain_kwargs=regressor_chain_kwargs)


def isotonic_regression_step(trial):
    """
    Get a IsotonicRegression object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Regressor
        The IsotonicRegression object step.
    """
    isotonic_regression_kwargs = {}
    return isotonic_regression_model(isotonic_regression_kwargs=isotonic_regression_kwargs)


def mlp_regressor_step(trial):
    """
    Get a MLPRegressor object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Regressor
        The MLPRegressor object step.
    """
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes_mlp_regressor",
                                                   [str(cat) for cat in [(50,), (100,), (50, 50), (100, 50)]])
    activation = trial.suggest_categorical("activation_mlp_regressor", ["relu", "tanh"])
    alpha = trial.suggest_loguniform("alpha_mlp_regressor", 1e-5, 1e-2)
    mlp_regressor_kwargs = {'hidden_layer_sizes': eval(hidden_layer_sizes), 'activation': activation, 'alpha': alpha}
    return mlp_regressor_model(mlp_regressor_kwargs=mlp_regressor_kwargs)


def mlp_classifier_step(trial):
    """
    Get a MLPClassifier object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The MLPClassifier object step.
    """
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes_mlp_classifier",
                                                   [str(cat) for cat in [(50,), (100,), (50, 50), (100, 50)]])
    activation = trial.suggest_categorical("activation_mlp_classifier", ["relu", "tanh"])
    alpha = trial.suggest_loguniform("alpha_mlp_classifier", 1e-5, 1e-2)
    mlp_classifier_kwargs = {'hidden_layer_sizes': eval(hidden_layer_sizes), 'activation': activation, 'alpha': alpha}
    return mlp_classifier_model(mlp_classifier_kwargs=mlp_classifier_kwargs)


def label_propagation_step(trial):
    """
    Get a LabelPropagation object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The LabelPropagation object step.
    """
    kernel = trial.suggest_categorical("kernel_label_propagation", ["knn", "rbf"])
    label_propagation_kwargs = {'kernel': kernel}
    return label_propagation_model(label_propagation_kwargs=label_propagation_kwargs)


def label_spreading_step(trial):
    """
    Get a LabelSpreading object for the Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object.

    Returns
    -------
    Classifier
        The LabelSpreading object step.
    """
    kernel = trial.suggest_categorical("kernel_label_spreading", ["knn", "rbf"])
    label_spreading_kwargs = {'kernel': kernel}
    return label_spreading_model(label_spreading_kwargs=label_spreading_kwargs)


_REGRESSION_MODELS = {'linear_regression_model': linear_regression_step,
                      'ridge_model': ridge_step,
                      'ridge_cv_model': ridge_cv_step,
                      'lasso_model': lasso_step,
                      'lasso_cv_model': lasso_cv_step,
                      'lasso_lars_cv_model': lasso_lars_cv_step,
                      'lasso_lars_ic_model': lasso_lars_ic_step,
                      'elastic_net_model': elastic_net_step,
                      'ortogonal_matching_pursuit_model': ortogonal_matching_pursuit_step,
                      'bayesian_ridge_model': bayesian_ridge_step,
                      'ard_regression_model': ard_regression_step,
                      'tweedie_regressor_model': tweedie_regressor_step,
                      'poisson_regressor_model': poisson_regressor_step,
                      'gamma_regressor_model': gamma_regressor_step,
                      'passive_aggressive_regressor_model': passive_aggressive_regressor_step,
                      'huber_regressor_model': huber_regressor_step,
                      'ransac_regressor_model': ransac_regressor_step,
                      'theil_sen_regressor_model': theil_sen_regressor_step,
                      'quantile_regressor_model': quantile_regressor_step,
                      'kernel_ridge_regressor_model': kernel_ridge_step,
                      'svr_model': svr_step,
                      'nu_svr_model': nu_svr_step,
                      'linear_svr_model': linear_svr_step,
                      'sgd_regressor_model': sgd_regressor_step,
                      'kneighbors_regressor_model': kneighbors_regressor_step,
                      'radius_neighbors_regressor_model': radius_neighbors_regressor_step,
                      'gaussian_process_regressor_model': gaussian_process_regressor_step,
                      'pls_regression_model': pls_regression_step,
                      'decision_tree_regressor_model': decision_tree_regressor_step,
                      'random_forest_regressor_model': random_forest_regressor_step,
                      'extra_trees_regressor_model': extra_trees_regressor_step,
                      'ada_boost_regressor_model': ada_boost_regressor_step,
                      'gradient_boosting_regressor_model': gradient_boosting_regressor_step,
                      'hist_gradient_boosting_regressor_model': hist_gradient_boosting_regressor_step,
                      'voting_regressor_model': voting_regressor_step,
                      'stacking_regressor_model': stacking_regressor_step,
                      'bagging_regressor_model': bagging_regressor_step,
                      # 'isotonic_regression_model': isotonic_regression_step, # ValueError: Isotonic regression input X should be a 1d array or 2d array with 1 feature
                      'mlp_regressor_model': mlp_regressor_step
                      }

_CLASSIFICATION_MODELS = {'ridge_classifier_model': ridge_classifier_step,
                          'ridge_classifier_cv_model': ridge_classifier_cv_step,
                          'logistic_regression_model': logistic_regression_step,
                          'logistic_regression_cv_model': logistic_regression_cv_step,
                          'perceptron_model': perceptron_step,
                          'passive_aggressive_classifier_model': passive_aggressive_classifier_step,
                          'linear_discriminant_analysis_model': linear_discriminat_analysis_step,
                          'quadradic_discriminant_analysis_model': quadratic_discriminant_analysis_step,
                          'svc_model': svc_step,
                          'nu_svc_model': nu_svc_step,
                          'linear_svc_model': linear_svc_step,
                          'one_class_svm_model': one_class_svm_step,
                          'sgd_classifier_model': sgd_classifier_step,
                          'sgd_one_class_svm_model': sgd_one_class_svm_step,
                          'kneighbors_classifier_model': kneighbors_classifier_step,
                          'radius_neighbors_classifier_model': radius_neighbors_classifier_step,
                          'nearest_centroid_model': nearest_centroid_step,
                          'gaussian_process_classifier_model': gaussian_process_classifier_step,
                          'gaussian_nb_model': gaussian_nb_step,
                          'multinomial_nb_model': multinomial_nb_step,
                          'bernoulli_nb_model': bernoulli_nb_step,
                          # 'categorical_nb_model': categorical_nb_step,
                          'complement_nb_model': complement_nb_step,
                          'decision_tree_classifier_model': decision_tree_classifier_step,
                          'random_forest_classifier_model': random_forest_classifier_step,
                          'extra_trees_classifier_model': extra_trees_classifier_step,
                          'ada_boost_classifier_model': ada_boost_classifier_step,
                          'gradient_boosting_classifier_model': gradient_boosting_classifier_step,
                          'hist_gradient_boosting_classifier_model': hist_gradient_boosting_classifier_step,
                          'voting_classifier_model': voting_classifier_step,
                          'stacking_classifier_model': stacking_classifier_step,
                          'bagging_classifier_model': bagging_classifier_step,
                          'mlp_classifier_model': mlp_classifier_step,
                          }

#############################################################################################################
# |                                       | Number of Targets | Target Cardinality | Valid Type of Target     |
# |---------------------------------------|-------------------|--------------------|--------------------------|
# | Multiclass Classification             | 1                 | >2                 | 'multiclass'             |
# | Multilabel Classification             | >1                | 2 (0 or 1)         | 'multilabel'             |
# | Multiclass-Multioutput Classification | >1                | >2                 | 'multiclass-multioutput' |
# | Multioutput Regression                | >1                | Continuous         | 'continuous-multioutput' |
#############################################################################################################

# MULTICLASS
_MULTICLASS_CLASSIFICATION_MODELS = {'bernoulli_nb_model': bernoulli_nb_step,
                                     'decision_tree_classifier_model': decision_tree_classifier_step,
                                     'extra_trees_classifier_model': extra_trees_classifier_step,
                                     'extra_tree_classifier_model': extra_tree_classifier_step,
                                     'gaussian_nb_model': gaussian_nb_step,
                                     'knneighbors_classifier_model': kneighbors_classifier_step,
                                     'label_propagation_model': label_propagation_step,
                                     'label_spreading_model': label_spreading_step,
                                     'linear_discriminant_analysis_model': linear_discriminat_analysis_step,
                                     'linear_svc_model': linear_svc_multiclass_step,
                                     'logistic_regression_model': logistic_regression_multiclass_step,
                                     'logistic_regression_cv_model': logistic_regression_cv_multiclass_step,
                                     'mlp_classifier_model': mlp_classifier_step,
                                     'nearest_centroid_model': nearest_centroid_step,
                                     'quadradic_discriminant_analysis_model': quadratic_discriminant_analysis_step,
                                     'radius_neighbors_classifier_model': radius_neighbors_classifier_step,
                                     'random_forest_classifier_model': random_forest_classifier_step,
                                     'ridge_classifier_model': ridge_classifier_step,
                                     'ridge_classifier_cv_model': ridge_classifier_cv_step,
                                     'nu_svc_model': nu_svc_step,
                                     'svc_model': svc_step,
                                     'gaussian_process_classifier_model': gaussian_process_multiclass_classifier_step,
                                     'gradient_boosting_classifier_model': gradient_boosting_multiclass_classifier_step,
                                     'sgd_classifier_model': sgd_classifier_step,
                                     'perceptron_model': perceptron_step,
                                     'passive_aggressive_classifier_model': passive_aggressive_classifier_step,
                                     'one_vs_rest_classifier_model': one_vs_rest_classifier_step,
                                     'one_vs_one_classifier_model': one_vs_one_classifier_step,
                                     'output_code_classifier_model': output_code_classifier_step,
                                     }

# MULTITASK
_MULTILABEL_CLASSIFICATION_MODELS = {'decision_tree_classifier_model': decision_tree_classifier_step,
                                     'extra_tree_classifier_model': extra_tree_classifier_step,
                                     'extra_trees_classifier_model': extra_trees_classifier_step,
                                     'knneighbors_classifier_model': kneighbors_classifier_step,
                                     'mlp_classifier_model': mlp_classifier_step,
                                     'radius_neighbors_classifier_model': radius_neighbors_classifier_step,
                                     'random_forest_classifier_model': random_forest_classifier_step,
                                     'ridge_classifier_model': ridge_classifier_step,
                                     'ridge_classifier_cv_model': ridge_classifier_cv_step,
                                     'multi_output_classifier_model': multi_output_classifier_step,
                                     'classifier_chain_model': classifier_chain_step,
                                     }

# MULTITASK
_MULTILABEL_REGRESSION_MODELS = {'multi_output_regressor_model': multi_output_regressor_step,
                                 'regressor_chain_model': regressor_chain_step,
                                 }


def _get_sk_model(trial, task_type: str) -> Predictor:
    """
    Get a sklearn model step based on the task type for the optuna optimization.

    Parameters
    ----------
    trial: optuna.trial.Trial
        The optuna trial object.
    task_type: str
        The task type of the model.

    Returns
    -------
    Predictor
        The sklearn model step.
    """
    if isinstance(task_type, str):
        if task_type == "regression":
            model = trial.suggest_categorical("regression_model", list(_REGRESSION_MODELS.keys()))
            return _REGRESSION_MODELS[model](trial)
        elif task_type == "classification_binary":
            model = trial.suggest_categorical("classification_model", list(_CLASSIFICATION_MODELS.keys()))
            return _CLASSIFICATION_MODELS[model](trial)
        elif task_type == "classification_multiclass":
            model = trial.suggest_categorical("multiclass_model", list(_MULTICLASS_CLASSIFICATION_MODELS.keys()))
            return _MULTICLASS_CLASSIFICATION_MODELS[model](trial)
    elif isinstance(task_type, list):
        task_type_sig = list(set(task_type))
        if len(task_type_sig) == 1 and task_type_sig[0] == "classification":
            model = trial.suggest_categorical("multiask_model", list(_MULTILABEL_CLASSIFICATION_MODELS.keys()))
            return _MULTILABEL_CLASSIFICATION_MODELS[model](trial)
        elif len(task_type_sig) == 1 and task_type_sig[0] == "regression":
            model = trial.suggest_categorical("multiregression_model", list(_MULTILABEL_REGRESSION_MODELS.keys()))
            return _MULTILABEL_REGRESSION_MODELS[model](trial)
        else:
            raise ValueError(f'Unknown task type: {task_type_sig}')
    else:
        raise ValueError(f'Unknown task type: {task_type}')

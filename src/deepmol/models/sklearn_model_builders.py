from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier, \
    AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier, \
    HistGradientBoostingRegressor, HistGradientBoostingClassifier, VotingRegressor, VotingClassifier, StackingRegressor, \
    StackingClassifier, BaggingRegressor, BaggingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier, RidgeCV, RidgeClassifierCV, Lasso, LassoCV, \
    LassoLarsCV, LassoLarsIC, MultiTaskLasso, ElasticNet, MultiTaskElasticNet, MultiTaskElasticNetCV, \
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, LogisticRegression, LogisticRegressionCV, \
    TweedieRegressor, PoissonRegressor, GammaRegressor, SGDRegressor, SGDClassifier, Perceptron, \
    PassiveAggressiveRegressor, PassiveAggressiveClassifier, HuberRegressor, RANSACRegressor, TheilSenRegressor, \
    QuantileRegressor, SGDOneClassSVM
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain, MultiOutputRegressor, RegressorChain
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, RadiusNeighborsRegressor, \
    RadiusNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR, OneClassSVM
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, ExtraTreeRegressor, ExtraTreeClassifier

from deepmol.models import SklearnModel
import os


#####################
### LINEAR MODELS ###
#####################

num_of_cores = os.cpu_count() // 2

def linear_regression_model(linear_regression_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.LinearRegression.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    Parameters
    ----------
    linear_regression_kwargs : dict
        Keyword arguments for sklearn.linear_model.LinearRegression
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.LinearRegression
    """
    linear_regression_kwargs = linear_regression_kwargs or {}
    linear_regression_kwargs["n_jobs"] = num_of_cores
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LinearRegression(**linear_regression_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ridge_model(ridge_kwargs: dict = None,
                sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.Ridge.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    Parameters
    ----------
    ridge_kwargs : dict
        Keyword arguments for sklearn.linear_model.Ridge
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.Ridge
    """
    ridge_kwargs = ridge_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = Ridge(**ridge_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ridge_classifier_model(ridge_classifier_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.RidgeClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html

    Parameters
    ----------
    ridge_classifier_kwargs : dict
        Keyword arguments for sklearn.linear_model.RidgeClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.RidgeClassifier
    """
    ridge_classifier_kwargs = ridge_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = RidgeClassifier(**ridge_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ridge_cv_model(ridge_cv_kwargs: dict = None,
                   sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.RidgeCV.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html

    Parameters
    ----------
    ridge_cv_kwargs : dict
        Keyword arguments for sklearn.linear_model.RidgeCV
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.RidgeCV
    """
    ridge_cv_kwargs = ridge_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = RidgeCV(**ridge_cv_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ridge_classifier_cv_model(ridge_classifier_cv_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.RidgeClassifierCV.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html

    Parameters
    ----------
    ridge_classifier_cv_kwargs : dict
        Keyword arguments for sklearn.linear_model.RidgeClassifierCV
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.RidgeClassifierCV
    """
    ridge_classifier_cv_kwargs = ridge_classifier_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = RidgeClassifierCV(**ridge_classifier_cv_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def lasso_model(lasso_kwargs: dict = None,
                sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.Lasso.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

    Parameters
    ----------
    lasso_kwargs : dict
        Keyword arguments for sklearn.linear_model.Lasso
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.Lasso
    """
    lasso_kwargs = lasso_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = Lasso(**lasso_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def lasso_cv_model(lasso_cv_kwargs: dict = None,
                   sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.LassoCV.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html

    Parameters
    ----------
    lasso_cv_kwargs : dict
        Keyword arguments for sklearn.linear_model.LassoCV
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.LassoCV
    """
    lasso_cv_kwargs = lasso_cv_kwargs or {}
    lasso_cv_kwargs["n_jobs"] = num_of_cores
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LassoCV(**lasso_cv_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def lasso_lars_cv_model(lasso_lars_cv_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.LassoLarsCV.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html

    Parameters
    ----------
    lasso_lars_cv_kwargs : dict
        Keyword arguments for sklearn.linear_model.LassoLarsCV
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.LassoLarsCV
    """
    lasso_lars_cv_kwargs = lasso_lars_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    lasso_lars_cv_kwargs["n_jobs"] = num_of_cores
    model = LassoLarsCV(**lasso_lars_cv_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def lasso_lars_ic_model(lasso_lars_ic_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.LassoLarsIC.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html

    Parameters
    ----------
    lasso_lars_ic_kwargs : dict
        Keyword arguments for sklearn.linear_model.LassoLarsIC
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.LassoLarsIC
    """
    lasso_lars_ic_kwargs = lasso_lars_ic_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LassoLarsIC(**lasso_lars_ic_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def multitask_lasso_model(multitask_lasso_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.MultiTaskLasso.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html

    Parameters
    ----------
    multitask_lasso_kwargs : dict
        Keyword arguments for sklearn.linear_model.MultiTaskLasso
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.MultiTaskLasso
    """
    multitask_lasso_kwargs = multitask_lasso_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = MultiTaskLasso(**multitask_lasso_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def elastic_net_model(elastic_net_kwargs: dict = None,
                      sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.ElasticNet.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

    Parameters
    ----------
    elastic_net_kwargs : dict
        Keyword arguments for sklearn.linear_model.ElasticNet
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.ElasticNet
    """
    elastic_net_kwargs = elastic_net_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = ElasticNet(**elastic_net_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def multitask_elastic_net_model(multitask_elastic_net_kwargs: dict = None, sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.MultiTaskElasticNet.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html

    Parameters
    ----------
    multitask_elastic_net_kwargs : dict
        Keyword arguments for sklearn.linear_model.MultiTaskElasticNet
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.MultiTaskElasticNet
    """
    multitask_elastic_net_kwargs = multitask_elastic_net_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = MultiTaskElasticNet(**multitask_elastic_net_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def multitask_elastic_net_cv_model(multitask_elastic_net_cv_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.MultiTaskElasticNetCV.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html

    Parameters
    ----------
    multitask_elastic_net_cv_kwargs : dict
        Keyword arguments for sklearn.linear_model.MultiTaskElasticNetCV
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.MultiTaskElasticNetCV
    """
    multitask_elastic_net_cv_kwargs = multitask_elastic_net_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model

    multitask_elastic_net_cv_kwargs["n_jobs"] = num_of_cores
    model = MultiTaskElasticNetCV(**multitask_elastic_net_cv_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ortogonal_matching_pursuit_model(ortogonal_matching_pursuit_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.OrthogonalMatchingPursuit.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html

    Parameters
    ----------
    ortogonal_matching_pursuit_kwargs : dict
        Keyword arguments for sklearn.linear_model.OrthogonalMatchingPursuit
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.OrthogonalMatchingPursuit
    """
    ortogonal_matching_pursuit_kwargs = ortogonal_matching_pursuit_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = OrthogonalMatchingPursuit(**ortogonal_matching_pursuit_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def bayesian_ridge_model(bayesian_ridge_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.BayesianRidge.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html

    Parameters
    ----------
    bayesian_ridge_kwargs : dict
        Keyword arguments for sklearn.linear_model.BayesianRidge
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.BayesianRidge
    """
    bayesian_ridge_kwargs = bayesian_ridge_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = BayesianRidge(**bayesian_ridge_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ard_regression_model(ard_regression_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.ARDRegression.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html

    Parameters
    ----------
    ard_regression_kwargs : dict
        Keyword arguments for sklearn.linear_model.ARDRegression
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.ARDRegression
    """
    ard_regression_kwargs = ard_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = ARDRegression(**ard_regression_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def logistic_regression_model(logistic_regression_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.LogisticRegression.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

    Parameters
    ----------
    logistic_regression_kwargs : dict
        Keyword arguments for sklearn.linear_model.LogisticRegression
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.LogisticRegression
    """
    logistic_regression_kwargs = logistic_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    logistic_regression_kwargs["n_jobs"] = num_of_cores
    model = LogisticRegression(**logistic_regression_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def logistic_regression_cv_model(logistic_regression_cv_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.LogisticRegressionCV.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

    Parameters
    ----------
    logistic_regression_cv_kwargs : dict
        Keyword arguments for sklearn.linear_model.LogisticRegressionCV
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.LogisticRegressionCV
    """
    logistic_regression_cv_kwargs = logistic_regression_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    logistic_regression_cv_kwargs["n_jobs"] = num_of_cores
    model = LogisticRegressionCV(**logistic_regression_cv_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def tweedie_regressor_model(tweedie_regressor_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.TweedieRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html

    Parameters
    ----------
    tweedie_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.TweedieRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.TweedieRegressor
    """
    tweedie_regressor_kwargs = tweedie_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = TweedieRegressor(**tweedie_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def poisson_regressor_model(poisson_regressor_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.PoissonRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html

    Parameters
    ----------
    poisson_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.PoissonRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.PoissonRegressor
    """
    poisson_regressor_kwargs = poisson_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = PoissonRegressor(**poisson_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def gamma_regressor_model(gamma_regressor_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.GammaRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html

    Parameters
    ----------
    gamma_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.GammaRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.GammaRegressor
    """
    gamma_regressor_kwargs = gamma_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = GammaRegressor(**gamma_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def perceptron_model(perceptron_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.Perceptron.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html

    Parameters
    ----------
    perceptron_kwargs : dict
        Keyword arguments for sklearn.linear_model.Perceptron
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.Perceptron
    """
    perceptron_kwargs = perceptron_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    perceptron_kwargs["n_jobs"] = num_of_cores
    model = Perceptron(**perceptron_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def passive_aggressive_regressor_model(passive_aggressive_regressor_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.PassiveAggressiveRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html

    Parameters
    ----------
    passive_aggressive_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.PassiveAggressiveRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.PassiveAggressiveRegressor
    """
    passive_aggressive_regressor_kwargs = passive_aggressive_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = PassiveAggressiveRegressor(**passive_aggressive_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def passive_aggressive_classifier_model(passive_aggressive_classifier_kwargs: dict = None,
                                        sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.PassiveAggressiveClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html

    Parameters
    ----------
    passive_aggressive_classifier_kwargs : dict
        Keyword arguments for sklearn.linear_model.PassiveAggressiveClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.PassiveAggressiveClassifier
    """
    passive_aggressive_classifier_kwargs = passive_aggressive_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    passive_aggressive_classifier_kwargs["n_jobs"] = num_of_cores
    model = PassiveAggressiveClassifier(**passive_aggressive_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def huber_regressor_model(huber_regressor_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.HuberRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html

    Parameters
    ----------
    huber_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.HuberRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.HuberRegressor
    """
    huber_regressor_kwargs = huber_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = HuberRegressor(**huber_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ransac_regressor_model(ransac_regressor_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.RANSACRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html

    Parameters
    ----------
    ransac_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.RANSACRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.RANSACRegressor
    """
    ransac_regressor_kwargs = ransac_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = RANSACRegressor(**ransac_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def theil_sen_regressor_model(theil_sen_regressor_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.TheilSenRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html

    Parameters
    ----------
    theil_sen_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.TheilSenRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.TheilSenRegressor
    """
    theil_sen_regressor_kwargs = theil_sen_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    theil_sen_regressor_kwargs["n_jobs"] = num_of_cores
    model = TheilSenRegressor(**theil_sen_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def quantile_regressor_model(quantile_regressor_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.QuantileRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html

    Parameters
    ----------
    quantile_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.QuantileRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.QuantileRegressor
    """
    quantile_regressor_kwargs = quantile_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = QuantileRegressor(**quantile_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


##################################################
### Linear and Quadratic Discriminant Analysis ###
##################################################


def linear_discriminant_analysis_model(linear_discriminant_analysis_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.discriminant_analysis.LinearDiscriminantAnalysis.
    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

    Parameters
    ----------
    linear_discriminant_analysis_kwargs : dict
        Keyword arguments for sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    """
    linear_discriminant_analysis_kwargs = linear_discriminant_analysis_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = LinearDiscriminantAnalysis(**linear_discriminant_analysis_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def quadratic_discriminant_analysis_model(quadratic_discriminant_analysis_kwargs: dict = None,
                                          sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.
    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html

    Parameters
    ----------
    quadratic_discriminant_analysis_kwargs : dict
        Keyword arguments for sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    """
    quadratic_discriminant_analysis_kwargs = quadratic_discriminant_analysis_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = QuadraticDiscriminantAnalysis(**quadratic_discriminant_analysis_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


###############################
### Kernel ridge regression ###
###############################


def kernel_ridge_regressor_model(kernel_ridge_regressor_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.kernel_ridge.KernelRidge.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html

    Parameters
    ----------
    kernel_ridge_regressor_kwargs : dict
        Keyword arguments for sklearn.kernel_ridge.KernelRidge
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.kernel_ridge.KernelRidge
    """
    kernel_ridge_regressor_kwargs = kernel_ridge_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = KernelRidge(**kernel_ridge_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


###############################
### Support Vector Machines ###
###############################


def svc_model(svc_kwargs: dict = None, sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.svm.SVC.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Parameters
    ----------
    svc_kwargs : dict
        Keyword arguments for sklearn.svm.SVC
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.svm.SVC
    """
    svc_kwargs = svc_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = SVC(**svc_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def nu_svc_model(nu_svc_kwargs: dict = None,
                 sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.svm.NuSVC.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html

    Parameters
    ----------
    nu_svc_kwargs : dict
        Keyword arguments for sklearn.svm.NuSVC
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.svm.NuSVC
    """
    nu_svc_kwargs = nu_svc_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = NuSVC(**nu_svc_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def linear_svc_model(linear_svc_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.svm.LinearSVC.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    Parameters
    ----------
    linear_svc_kwargs : dict
        Keyword arguments for sklearn.svm.LinearSVC
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.svm.LinearSVC
    """
    linear_svc_kwargs = linear_svc_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = LinearSVC(**linear_svc_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def svr_model(svr_kwargs: dict = None, sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.svm.SVR.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

    Parameters
    ----------
    svr_kwargs : dict
        Keyword arguments for sklearn.svm.SVR
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.svm.SVR
    """
    svr_kwargs = svr_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = SVR(**svr_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def nu_svr_model(nu_svr_kwargs: dict = None,
                 sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.svm.NuSVR.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html

    Parameters
    ----------
    nu_svr_kwargs : dict
        Keyword arguments for sklearn.svm.NuSVR
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.svm.NuSVR
    """
    nu_svr_kwargs = nu_svr_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = NuSVR(**nu_svr_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def linear_svr_model(linear_svr_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.svm.LinearSVR.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html

    Parameters
    ----------
    linear_svr_kwargs : dict
        Keyword arguments for sklearn.svm.LinearSVR
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.svm.LinearSVR
    """
    linear_svr_kwargs = linear_svr_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LinearSVR(**linear_svr_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def one_class_svm_model(one_class_svm_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.svm.OneClassSVM.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

    Parameters
    ----------
    one_class_svm_kwargs : dict
        Keyword arguments for sklearn.svm.OneClassSVM
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.svm.OneClassSVM
    """
    one_class_svm_kwargs = one_class_svm_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = OneClassSVM(**one_class_svm_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


###################################
### Stochastic Gradient Descent ###
###################################


def sgd_regressor_model(sgd_regressor_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.SGDRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html

    Parameters
    ----------
    sgd_regressor_kwargs : dict
        Keyword arguments for sklearn.linear_model.SGDRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.SGDRegressor
    """
    sgd_regressor_kwargs = sgd_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = SGDRegressor(**sgd_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def sgd_classifier_model(sgd_classifier_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.SGDClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

    Parameters
    ----------
    sgd_classifier_kwargs : dict
        Keyword arguments for sklearn.linear_model.SGDClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.SGDClassifier
    """
    sgd_classifier_kwargs = sgd_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    sgd_classifier_kwargs["n_jobs"] = num_of_cores
    model = SGDClassifier(**sgd_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def sgd_one_class_svm_model(sgd_one_class_svm_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.linear_model.SGDOneClassSVM.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html

    Parameters
    ----------
    sgd_one_class_svm_kwargs : dict
        Keyword arguments for sklearn.linear_model.SGDOneClassSVM
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.linear_model.SGDOneClassSVM
    """
    sgd_one_class_svm_kwargs = sgd_one_class_svm_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = SGDOneClassSVM(**sgd_one_class_svm_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


#########################
### Nearest Neighbors ###
#########################


def kneighbors_regressor_model(kneighbors_regressor_kwargs: dict = None,
                               sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.neighbors.KNeighborsRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html

    Parameters
    ----------
    kneighbors_regressor_kwargs : dict
        Keyword arguments for sklearn.neighbors.KNeighborsRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.neighbors.KNeighborsRegressor
    """
    kneighbors_regressor_kwargs = kneighbors_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    kneighbors_regressor_kwargs["n_jobs"] = num_of_cores
    model = KNeighborsRegressor(**kneighbors_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def kneighbors_classifier_model(kneighbors_classifier_kwargs: dict = None, sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.neighbors.KNeighborsClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    Parameters
    ----------
    kneighbors_classifier_kwargs : dict
        Keyword arguments for sklearn.neighbors.KNeighborsClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.neighbors.KNeighborsClassifier
    """
    kneighbors_classifier_kwargs = kneighbors_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    kneighbors_classifier_kwargs["n_jobs"] = num_of_cores
    model = KNeighborsClassifier(**kneighbors_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def radius_neighbors_regressor_model(radius_neighbors_regressor_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.neighbors.RadiusNeighborsRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html

    Parameters
    ----------
    radius_neighbors_regressor_kwargs : dict
        Keyword arguments for sklearn.neighbors.RadiusNeighborsRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.neighbors.RadiusNeighborsRegressor
    """
    radius_neighbors_regressor_kwargs = radius_neighbors_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    radius_neighbors_regressor_kwargs["n_jobs"] = num_of_cores
    model = RadiusNeighborsRegressor(**radius_neighbors_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def radius_neighbors_classifier_model(radius_neighbors_classifier_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.neighbors.RadiusNeighborsClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html

    Parameters
    ----------
    radius_neighbors_classifier_kwargs : dict
        Keyword arguments for sklearn.neighbors.RadiusNeighborsClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.neighbors.RadiusNeighborsClassifier
    """
    radius_neighbors_classifier_kwargs = radius_neighbors_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    radius_neighbors_classifier_kwargs["n_jobs"] = num_of_cores
    model = RadiusNeighborsClassifier(**radius_neighbors_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def nearest_centroid_model(nearest_centroid_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.neighbors.NearestCentroid.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html

    Parameters
    ----------
    nearest_centroid_kwargs : dict
        Keyword arguments for sklearn.neighbors.NearestCentroid
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.neighbors.NearestCentroid
    """
    nearest_centroid_kwargs = nearest_centroid_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = NearestCentroid(**nearest_centroid_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


##########################
### Gaussian Processes ###
##########################

def gaussian_process_regressor_model(gaussian_process_regressor_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.gaussian_process.GaussianProcessRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html

    Parameters
    ----------
    gaussian_process_regressor_kwargs : dict
        Keyword arguments for sklearn.gaussian_process.GaussianProcessRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.gaussian_process.GaussianProcessRegressor
    """
    gaussian_process_regressor_kwargs = gaussian_process_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = GaussianProcessRegressor(**gaussian_process_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def gaussian_process_classifier_model(gaussian_process_classifier_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.gaussian_process.GaussianProcessClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html

    Parameters
    ----------
    gaussian_process_classifier_kwargs : dict
        Keyword arguments for sklearn.gaussian_process.GaussianProcessClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.gaussian_process.GaussianProcessClassifier
    """
    gaussian_process_classifier_kwargs = gaussian_process_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = GaussianProcessClassifier(**gaussian_process_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


###########################
### Cross decomposition ###
###########################


def pls_regression_model(pls_regression_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.cross_decomposition.PLSRegression.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

    Parameters
    ----------
    pls_regression_kwargs : dict
        Keyword arguments for sklearn.cross_decomposition.PLSRegression
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.cross_decomposition.PLSRegression
    """
    pls_regression_kwargs = pls_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = PLSRegression(**pls_regression_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


###################
### Naive Bayes ###
###################


def gaussian_nb_model(gaussian_nb_kwargs: dict = None,
                      sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.naive_bayes.GaussianNB.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html

    Parameters
    ----------
    gaussian_nb_kwargs : dict
        Keyword arguments for sklearn.naive_bayes.GaussianNB
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.naive_bayes.GaussianNB
    """
    gaussian_nb_kwargs = gaussian_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = GaussianNB(**gaussian_nb_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def multinomial_nb_model(multinomial_nb_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.naive_bayes.MultinomialNB.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

    Parameters
    ----------
    multinomial_nb_kwargs : dict
        Keyword arguments for sklearn.naive_bayes.MultinomialNB
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.naive_bayes.MultinomialNB
    """
    multinomial_nb_kwargs = multinomial_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = MultinomialNB(**multinomial_nb_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def bernoulli_nb_model(bernoulli_nb_kwargs: dict = None,
                       sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.naive_bayes.BernoulliNB.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html

    Parameters
    ----------
    bernoulli_nb_kwargs : dict
        Keyword arguments for sklearn.naive_bayes.BernoulliNB
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.naive_bayes.BernoulliNB
    """
    bernoulli_nb_kwargs = bernoulli_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = BernoulliNB(**bernoulli_nb_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def categorical_nb_model(categorical_nb_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.naive_bayes.CategoricalNB.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html

    Parameters
    ----------
    categorical_nb_kwargs : dict
        Keyword arguments for sklearn.naive_bayes.CategoricalNB
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.naive_bayes.CategoricalNB
    """
    categorical_nb_kwargs = categorical_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = CategoricalNB(**categorical_nb_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def complement_nb_model(complement_nb_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.naive_bayes.ComplementNB.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html

    Parameters
    ----------
    complement_nb_kwargs : dict
        Keyword arguments for sklearn.naive_bayes.ComplementNB
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.naive_bayes.ComplementNB
    """
    complement_nb_kwargs = complement_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = ComplementNB(**complement_nb_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


######################
### Decision Trees ###
######################


def decision_tree_regressor_model(decision_tree_regressor_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.tree.DecisionTreeRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

    Parameters
    ----------
    decision_tree_regressor_kwargs : dict
        Keyword arguments for sklearn.tree.DecisionTreeRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.tree.DecisionTreeRegressor
    """
    decision_tree_regressor_kwargs = decision_tree_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = DecisionTreeRegressor(**decision_tree_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def decision_tree_classifier_model(decision_tree_classifier_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.tree.DecisionTreeClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

    Parameters
    ----------
    decision_tree_classifier_kwargs : dict
        Keyword arguments for sklearn.tree.DecisionTreeClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.tree.DecisionTreeClassifier
    """
    decision_tree_classifier_kwargs = decision_tree_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = DecisionTreeClassifier(**decision_tree_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def extra_tree_regressor_model(extra_tree_regressor_kwargs: dict = None,
                               sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.tree.ExtraTreeRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html

    Parameters
    ----------
    extra_tree_regressor_kwargs : dict
        Keyword arguments for sklearn.tree.ExtraTreeRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.tree.ExtraTreeRegressor
    """
    extra_tree_regressor_kwargs = extra_tree_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = ExtraTreeRegressor(**extra_tree_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def extra_tree_classifier_model(extra_tree_classifier_kwargs: dict = None, sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.tree.ExtraTreeClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html

    Parameters
    ----------
    extra_tree_classifier_kwargs : dict
        Keyword arguments for sklearn.tree.ExtraTreeClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.tree.ExtraTreeClassifier
    """
    extra_tree_classifier_kwargs = extra_tree_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = ExtraTreeClassifier(**extra_tree_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


########################
### Ensemble methods ###
########################


def random_forest_regressor_model(random_forest_regressor_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.RandomForestRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

    Parameters
    ----------
    random_forest_regressor_kwargs : dict
        Keyword arguments for sklearn.ensemble.RandomForestRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.RandomForestRegressor
    """
    random_forest_regressor_kwargs = random_forest_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    random_forest_regressor_kwargs["n_jobs"] = num_of_cores
    model = RandomForestRegressor(**random_forest_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def random_forest_classifier_model(random_forest_classifier_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.RandomForestClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

    Parameters
    ----------
    random_forest_classifier_kwargs : dict
        Keyword arguments for sklearn.ensemble.RandomForestClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.RandomForestClassifier
    """
    random_forest_classifier_kwargs = random_forest_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    random_forest_classifier_kwargs["n_jobs"] = num_of_cores
    model = RandomForestClassifier(**random_forest_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def extra_trees_regressor_model(extra_trees_regressor_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.ExtraTreesRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

    Parameters
    ----------
    extra_trees_regressor_kwargs : dict
        Keyword arguments for sklearn.ensemble.ExtraTreesRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.ExtraTreesRegressor
    """
    extra_trees_regressor_kwargs = extra_trees_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    extra_trees_regressor_kwargs["n_jobs"] = num_of_cores
    model = ExtraTreesRegressor(**extra_trees_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def extra_trees_classifier_model(extra_trees_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.ExtraTreesClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html

    Parameters
    ----------
    extra_trees_classifier_kwargs : dict
        Keyword arguments for sklearn.ensemble.ExtraTreesClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.ExtraTreesClassifier
    """
    extra_trees_classifier_kwargs = extra_trees_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    extra_trees_classifier_kwargs["n_jobs"] = num_of_cores
    model = ExtraTreesClassifier(**extra_trees_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ada_boost_regressor_model(ada_boost_regressor_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.AdaBoostRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html

    Parameters
    ----------
    ada_boost_regressor_kwargs : dict
        Keyword arguments for sklearn.ensemble.AdaBoostRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.AdaBoostRegressor
    """
    ada_boost_regressor_kwargs = ada_boost_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = AdaBoostRegressor(**ada_boost_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def ada_boost_classifier_model(ada_boost_classifier_kwargs: dict = None,
                               sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.AdaBoostClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

    Parameters
    ----------
    ada_boost_classifier_kwargs : dict
        Keyword arguments for sklearn.ensemble.AdaBoostClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.AdaBoostClassifier
    """
    ada_boost_classifier_kwargs = ada_boost_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = AdaBoostClassifier(**ada_boost_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def gradient_boosting_regressor_model(gradient_boosting_regressor_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.GradientBoostingRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

    Parameters
    ----------
    gradient_boosting_regressor_kwargs : dict
        Keyword arguments for sklearn.ensemble.GradientBoostingRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.GradientBoostingRegressor
    """
    gradient_boosting_regressor_kwargs = gradient_boosting_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = GradientBoostingRegressor(**gradient_boosting_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def gradient_boosting_classifier_model(gradient_boosting_classifier_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.GradientBoostingClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

    Parameters
    ----------
    gradient_boosting_classifier_kwargs : dict
        Keyword arguments for sklearn.ensemble.GradientBoostingClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.GradientBoostingClassifier
    """
    gradient_boosting_classifier_kwargs = gradient_boosting_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = GradientBoostingClassifier(**gradient_boosting_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def hist_gradient_boosting_regressor_model(hist_gradient_boosting_regressor_kwargs: dict = None,
                                           sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.HistGradientBoostingRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html

    Parameters
    ----------
    hist_gradient_boosting_regressor_kwargs : dict
        Keyword arguments for sklearn.ensemble.HistGradientBoostingRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.HistGradientBoostingRegressor
    """
    hist_gradient_boosting_regressor_kwargs = hist_gradient_boosting_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = HistGradientBoostingRegressor(**hist_gradient_boosting_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def hist_gradient_boosting_classifier_model(hist_gradient_boosting_classifier_kwargs: dict = None,
                                            sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.HistGradientBoostingClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

    Parameters
    ----------
    hist_gradient_boosting_classifier_kwargs : dict
        Keyword arguments for sklearn.ensemble.HistGradientBoostingClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.HistGradientBoostingClassifier
    """
    hist_gradient_boosting_classifier_kwargs = hist_gradient_boosting_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = HistGradientBoostingClassifier(**hist_gradient_boosting_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def voting_regressor_model(voting_regressor_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.VotingRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html

    Parameters
    ----------
    voting_regressor_kwargs : dict
        Keyword arguments for sklearn.ensemble.VotingRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.VotingRegressor
    """
    voting_regressor_kwargs = voting_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    voting_regressor_kwargs["n_jobs"] = num_of_cores
    model = VotingRegressor(**voting_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def voting_classifier_model(voting_classifier_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.VotingClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

    Parameters
    ----------
    voting_classifier_kwargs : dict
        Keyword arguments for sklearn.ensemble.VotingClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.VotingClassifier
    """
    voting_classifier_kwargs = voting_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    voting_classifier_kwargs["n_jobs"] = num_of_cores
    model = VotingClassifier(**voting_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def stacking_regressor_model(stacking_regressor_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.StackingRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html

    Parameters
    ----------
    stacking_regressor_kwargs : dict
        Keyword arguments for sklearn.ensemble.StackingRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.StackingRegressor
    """
    stacking_regressor_kwargs = stacking_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    stacking_regressor_kwargs["n_jobs"] = num_of_cores
    model = StackingRegressor(**stacking_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def stacking_classifier_model(stacking_classifier_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.StackingClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html

    Parameters
    ----------
    stacking_classifier_kwargs : dict
        Keyword arguments for sklearn.ensemble.StackingClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.StackingClassifier
    """
    stacking_classifier_kwargs = stacking_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    stacking_classifier_kwargs["n_jobs"] = num_of_cores
    model = StackingClassifier(**stacking_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def bagging_regressor_model(bagging_regressor_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.BaggingRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html

    Parameters
    ----------
    bagging_regressor_kwargs : dict
        Keyword arguments for sklearn.ensemble.BaggingRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.BaggingRegressor
    """
    bagging_regressor_kwargs = bagging_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    bagging_regressor_kwargs["n_jobs"] = num_of_cores
    model = BaggingRegressor(**bagging_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def bagging_classifier_model(bagging_classifier_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.ensemble.BaggingClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html

    Parameters
    ----------
    bagging_classifier_kwargs : dict
        Keyword arguments for sklearn.ensemble.BaggingClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.ensemble.BaggingClassifier
    """
    bagging_classifier_kwargs = bagging_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    bagging_classifier_kwargs["n_jobs"] = num_of_cores
    model = BaggingClassifier(**bagging_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


#############################################
### Multiclass and multioutput algorithms ###
#############################################


def one_vs_rest_classifier_model(one_vs_rest_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.multiclass.OneVsRestClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html

    Parameters
    ----------
    one_vs_rest_classifier_kwargs : dict
        Keyword arguments for sklearn.multiclass.OneVsRestClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.multiclass.OneVsRestClassifier
    """
    one_vs_rest_classifier_kwargs = one_vs_rest_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    one_vs_rest_classifier_kwargs["n_jobs"] = num_of_cores
    model = OneVsRestClassifier(**one_vs_rest_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def one_vs_one_classifier_model(one_vs_one_classifier_kwargs: dict = None, sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.multiclass.OneVsOneClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html

    Parameters
    ----------
    one_vs_one_classifier_kwargs : dict
        Keyword arguments for sklearn.multiclass.OneVsOneClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.multiclass.OneVsOneClassifier
    """
    one_vs_one_classifier_kwargs = one_vs_one_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    one_vs_one_classifier_kwargs["n_jobs"] = num_of_cores
    model = OneVsOneClassifier(**one_vs_one_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def output_code_classifier_model(output_code_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.multiclass.OutputCodeClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OutputCodeClassifier.html

    Parameters
    ----------
    output_code_classifier_kwargs : dict
        Keyword arguments for sklearn.multiclass.OutputCodeClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.multiclass.OutputCodeClassifier
    """
    output_code_classifier_kwargs = output_code_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    output_code_classifier_kwargs["n_jobs"] = num_of_cores
    model = OutputCodeClassifier(**output_code_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def multi_output_classifier_model(multi_output_classifier_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.multioutput.MultiOutputClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html

    Parameters
    ----------
    multi_output_classifier_kwargs : dict
        Keyword arguments for sklearn.multioutput.MultiOutputClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.multioutput.MultiOutputClassifier
    """
    multi_output_classifier_kwargs = multi_output_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    multi_output_classifier_kwargs["n_jobs"] = num_of_cores
    model = MultiOutputClassifier(**multi_output_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def classifier_chain_model(classifier_chain_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.multioutput.ClassifierChain.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html

    Parameters
    ----------
    classifier_chain_kwargs : dict
        Keyword arguments for sklearn.multioutput.ClassifierChain
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.multioutput.ClassifierChain
    """
    classifier_chain_kwargs = classifier_chain_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    estimator = classifier_chain_kwargs.pop('estimator', None)
    model = ClassifierChain(estimator, **classifier_chain_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def multi_output_regressor_model(multi_output_regressor_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.multioutput.MultiOutputRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html

    Parameters
    ----------
    multi_output_regressor_kwargs : dict
        Keyword arguments for sklearn.multioutput.MultiOutputRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.multioutput.MultiOutputRegressor
    """
    multi_output_regressor_kwargs = multi_output_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    multi_output_regressor_kwargs["n_jobs"] = num_of_cores
    model = MultiOutputRegressor(**multi_output_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def regressor_chain_model(regressor_chain_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.multioutput.RegressorChain.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html

    Parameters
    ----------
    regressor_chain_kwargs : dict
        Keyword arguments for sklearn.multioutput.RegressorChain
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.multioutput.RegressorChain
    """
    regressor_chain_kwargs = regressor_chain_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    estimator = regressor_chain_kwargs.pop('estimator', None)
    model = RegressorChain(estimator, **regressor_chain_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


###########################
### Isotonic regression ###
###########################


def isotonic_regression_model(isotonic_regression_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.isotonic.IsotonicRegression.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html

    Parameters
    ----------
    isotonic_regression_kwargs : dict
        Keyword arguments for sklearn.isotonic.IsotonicRegression
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.isotonic.IsotonicRegression
    """
    isotonic_regression_kwargs = isotonic_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = IsotonicRegression(**isotonic_regression_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


#############################
### Neural network models ###
#############################


def mlp_regressor_model(mlp_regressor_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.neural_network.MLPRegressor.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

    Parameters
    ----------
    mlp_regressor_kwargs : dict
        Keyword arguments for sklearn.neural_network.MLPRegressor
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.neural_network.MLPRegressor
    """
    mlp_regressor_kwargs = mlp_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = MLPRegressor(**mlp_regressor_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def mlp_classifier_model(mlp_classifier_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.neural_network.MLPClassifier.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

    Parameters
    ----------
    mlp_classifier_kwargs : dict
        Keyword arguments for sklearn.neural_network.MLPClassifier
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.neural_network.MLPClassifier
    """
    mlp_classifier_kwargs = mlp_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = MLPClassifier(**mlp_classifier_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def label_propagation_model(label_propagation_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.semi_supervised.LabelPropagation.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html

    Parameters
    ----------
    label_propagation_kwargs : dict
        Keyword arguments for sklearn.semi_supervised.LabelPropagation
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.semi_supervised.LabelPropagation
    """
    label_propagation_kwargs = label_propagation_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    label_propagation_kwargs["n_jobs"] = num_of_cores
    model = LabelPropagation(**label_propagation_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)


def label_spreading_model(label_spreading_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    """
    DeepMol wrapper for sklearn.semi_supervised.LabelSpreading.
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html

    Parameters
    ----------
    label_spreading_kwargs : dict
        Keyword arguments for sklearn.semi_supervised.LabelSpreading
    sklearn_kwargs : dict
        Keyword arguments for SklearnModel

    Returns
    -------
    SklearnModel
        Wrapped sklearn.semi_supervised.LabelSpreading
    """
    label_spreading_kwargs = label_spreading_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    label_spreading_kwargs["n_jobs"] = num_of_cores
    model = LabelSpreading(**label_spreading_kwargs)
    return SklearnModel(model=model, **sklearn_kwargs)

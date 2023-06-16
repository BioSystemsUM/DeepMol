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
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR, OneClassSVM
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from deepmol.models import SklearnModel


#####################
### LINEAR MODELS ###
#####################

def linear_regression_model(model_dir: str = 'linear_regression_model/',
                            linear_regression_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    linear_regression_kwargs = linear_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LinearRegression(**linear_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ridge_model(model_dir: str = 'ridge_model/',
                ridge_kwargs: dict = None,
                sklearn_kwargs: dict = None) -> SklearnModel:
    ridge_kwargs = ridge_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = Ridge(**ridge_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ridge_classifier_model(model_dir: str = 'ridge_classifier_model/',
                           ridge_classifier_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    ridge_classifier_kwargs = ridge_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = RidgeClassifier(**ridge_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ridge_cv_model(model_dir: str = 'ridge_cv_model/',
                   ridge_cv_kwargs: dict = None,
                   sklearn_kwargs: dict = None) -> SklearnModel:
    ridge_cv_kwargs = ridge_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = RidgeCV(**ridge_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ridge_classifier_cv_model(model_dir: str = 'ridge_classifier_cv_model/',
                              ridge_classifier_cv_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    ridge_classifier_cv_kwargs = ridge_classifier_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = RidgeClassifierCV(**ridge_classifier_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def lasso_model(model_dir: str = 'lasso_model/',
                lasso_kwargs: dict = None,
                sklearn_kwargs: dict = None) -> SklearnModel:
    lasso_kwargs = lasso_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = Lasso(**lasso_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def lasso_cv_model(model_dir: str = 'lasso_cv_model/',
                   lasso_cv_kwargs: dict = None,
                   sklearn_kwargs: dict = None) -> SklearnModel:
    lasso_cv_kwargs = lasso_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LassoCV(**lasso_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def lasso_lars_cv_model(model_dir: str = 'lasso_lars_cv_model/',
                        lasso_lars_cv_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    lasso_lars_cv_kwargs = lasso_lars_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LassoLarsCV(**lasso_lars_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def lasso_lars_ic_model(model_dir: str = 'lasso_lars_ic_model/',
                        lasso_lars_ic_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    lasso_lars_ic_kwargs = lasso_lars_ic_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LassoLarsIC(**lasso_lars_ic_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multitask_lasso_model(model_dir: str = 'multitask_lasso_model/',
                          multitask_lasso_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    multitask_lasso_kwargs = multitask_lasso_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = MultiTaskLasso(**multitask_lasso_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def elastic_net_model(model_dir: str = 'elastic_net_model/',
                      elastic_net_kwargs: dict = None,
                      sklearn_kwargs: dict = None) -> SklearnModel:
    elastic_net_kwargs = elastic_net_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = ElasticNet(**elastic_net_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multitask_elastic_net_model(model_dir: str = 'multitask_elastic_net_model/',
                                multitask_elastic_net_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    multitask_elastic_net_kwargs = multitask_elastic_net_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = MultiTaskElasticNet(**multitask_elastic_net_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multitask_elastic_net_cv_model(model_dir: str = 'multitask_elastic_net_cv_model/',
                                   multitask_elastic_net_cv_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    multitask_elastic_net_cv_kwargs = multitask_elastic_net_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = MultiTaskElasticNetCV(**multitask_elastic_net_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ortogonal_matching_pursuit_model(model_dir: str = 'ortogonal_matching_pursuit_model/',
                                     ortogonal_matching_pursuit_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    ortogonal_matching_pursuit_kwargs = ortogonal_matching_pursuit_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = OrthogonalMatchingPursuit(**ortogonal_matching_pursuit_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def bayesian_ridge_model(model_dir: str = 'bayesian_ridge_model/',
                         bayesian_ridge_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    bayesian_ridge_kwargs = bayesian_ridge_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = BayesianRidge(**bayesian_ridge_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ard_regression_model(model_dir: str = 'ard_regression_model/',
                         ard_regression_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    ard_regression_kwargs = ard_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = ARDRegression(**ard_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def logistic_regression_model(model_dir: str = 'logistic_regression_model/',
                              logistic_regression_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    logistic_regression_kwargs = logistic_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = LogisticRegression(**logistic_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def logistic_regression_cv_model(model_dir: str = 'logistic_regression_cv_model/',
                                 logistic_regression_cv_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    logistic_regression_cv_kwargs = logistic_regression_cv_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = LogisticRegressionCV(**logistic_regression_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def tweedie_regressor_model(model_dir: str = 'tweedie_regressor_model/',
                            tweedie_regressor_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    tweedie_regressor_kwargs = tweedie_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = TweedieRegressor(**tweedie_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def poisson_regressor_model(model_dir: str = 'poison_regressor_model/',
                            poisson_regressor_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    poisson_regressor_kwargs = poisson_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = PoissonRegressor(**poisson_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def gamma_regressor_model(model_dir: str = 'gamma_regressor_model/',
                          gamma_regressor_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    gamma_regressor_kwargs = gamma_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = GammaRegressor(**gamma_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def perceptron_model(model_dir: str = 'perceptron_model/',
                     perceptron_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    perceptron_kwargs = perceptron_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = Perceptron(**perceptron_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def passive_aggressive_regressor_model(model_dir: str = 'passive_aggressive_regressor_model/',
                                       passive_aggressive_regressor_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    passive_aggressive_regressor_kwargs = passive_aggressive_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = PassiveAggressiveRegressor(**passive_aggressive_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def passive_aggressive_classifier_model(model_dir: str = 'passive_aggressive_classifier_model/',
                                        passive_aggressive_classifier_kwargs: dict = None,
                                        sklearn_kwargs: dict = None) -> SklearnModel:
    passive_aggressive_classifier_kwargs = passive_aggressive_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = PassiveAggressiveClassifier(**passive_aggressive_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def huber_regressor_model(model_dir: str = 'huber_regressor_model/',
                          huber_regressor_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    huber_regressor_kwargs = huber_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = HuberRegressor(**huber_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ransac_regressor_model(model_dir: str = 'ransac_regressor_model/',
                           ransac_regressor_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    ransac_regressor_kwargs = ransac_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = RANSACRegressor(**ransac_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def theil_sen_regressor_model(model_dir: str = 'theil_sen_regressor_model/',
                              theil_sen_regressor_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    theil_sen_regressor_kwargs = theil_sen_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = TheilSenRegressor(**theil_sen_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def quantile_regressor_model(model_dir: str = 'quantile_regressor_model/',
                             quantile_regressor_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    quantile_regressor_kwargs = quantile_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = QuantileRegressor(**quantile_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


##################################################
### Linear and Quadratic Discriminant Analysis ###
##################################################


def linear_discriminant_analysis_model(model_dir: str = 'linear_discriminant_analysis_model/',
                                       linear_discriminant_analysis_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    linear_discriminant_analysis_kwargs = linear_discriminant_analysis_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = LinearDiscriminantAnalysis(**linear_discriminant_analysis_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def quadratic_discriminant_analysis_model(model_dir: str = 'quadratic_discriminant_analysis_model/',
                                          quadratic_discriminant_analysis_kwargs: dict = None,
                                          sklearn_kwargs: dict = None) -> SklearnModel:
    quadratic_discriminant_analysis_kwargs = quadratic_discriminant_analysis_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = QuadraticDiscriminantAnalysis(**quadratic_discriminant_analysis_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###############################
### Kernel ridge regression ###
###############################


def kernel_ridge_regressor_model(model_dir: str = 'kernel_ridge_regressor_model/',
                                 kernel_ridge_regressor_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    kernel_ridge_regressor_kwargs = kernel_ridge_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = KernelRidge(**kernel_ridge_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###############################
### Support Vector Machines ###
###############################


def svc_model(model_dir: str = 'svc_model/',
              svc_kwargs: dict = None,
              sklearn_kwargs: dict = None) -> SklearnModel:
    svc_kwargs = svc_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = SVC(**svc_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def nu_svc_model(model_dir: str = 'nu_svc_model/',
                 nu_svc_kwargs: dict = None,
                 sklearn_kwargs: dict = None) -> SklearnModel:
    nu_svc_kwargs = nu_svc_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = NuSVC(**nu_svc_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def linear_svc_model(model_dir: str = 'linear_svc_model/',
                     linear_svc_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    linear_svc_kwargs = linear_svc_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = LinearSVC(**linear_svc_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def svr_model(model_dir: str = 'svr_model/',
              svr_kwargs: dict = None,
              sklearn_kwargs: dict = None) -> SklearnModel:
    svr_kwargs = svr_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = SVR(**svr_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def nu_svr_model(model_dir: str = 'nu_svr_model/',
                 nu_svr_kwargs: dict = None,
                 sklearn_kwargs: dict = None) -> SklearnModel:
    nu_svr_kwargs = nu_svr_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = NuSVR(**nu_svr_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def linear_svr_model(model_dir: str = 'linear_svr_model/',
                     linear_svr_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    linear_svr_kwargs = linear_svr_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = LinearSVR(**linear_svr_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def one_class_svm_model(model_dir: str = 'one_class_svm_model/',
                        one_class_svm_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    one_class_svm_kwargs = one_class_svm_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = OneClassSVM(**one_class_svm_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###################################
### Stochastic Gradient Descent ###
###################################


def sgd_regressor_model(model_dir: str = 'sgd_regressor_model/',
                        sgd_regressor_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    sgd_regressor_kwargs = sgd_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = SGDRegressor(**sgd_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def sgd_classifier_model(model_dir: str = 'sgd_classifier_model/',
                         sgd_classifier_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    sgd_classifier_kwargs = sgd_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = SGDClassifier(**sgd_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def sgd_one_class_svm_model(model_dir: str = 'sgd_one_class_svm/',
                            sgd_one_class_svm_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    sgd_one_class_svm_kwargs = sgd_one_class_svm_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = SGDOneClassSVM(**sgd_one_class_svm_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


#########################
### Nearest Neighbors ###
#########################


def kneighbors_regressor_model(model_dir: str = 'kneighbors_regressor_model/',
                               kneighbors_regressor_kwargs: dict = None,
                               sklearn_kwargs: dict = None) -> SklearnModel:
    kneighbors_regressor_kwargs = kneighbors_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = KNeighborsRegressor(**kneighbors_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def kneighbors_classifier_model(model_dir: str = 'kneighbors_classifier_model/',
                                kneighbors_classifier_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    kneighbors_classifier_kwargs = kneighbors_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = KNeighborsClassifier(**kneighbors_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def radius_neighbors_regressor_model(model_dir: str = 'radius_neighbors_regressor_model/',
                                     radius_neighbors_regressor_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    radius_neighbors_regressor_kwargs = radius_neighbors_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = RadiusNeighborsRegressor(**radius_neighbors_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def radius_neighbors_classifier_model(model_dir: str = 'radius_neighbors_classifier_model/',
                                      radius_neighbors_classifier_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    radius_neighbors_classifier_kwargs = radius_neighbors_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = RadiusNeighborsClassifier(**radius_neighbors_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def nearest_centroid_model(model_dir: str = 'nearest_centroid_model/',
                           nearest_centroid_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    nearest_centroid_kwargs = nearest_centroid_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = NearestCentroid(**nearest_centroid_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


##########################
### Gaussian Processes ###
##########################

def gaussian_process_regressor_model(model_dir: str = 'gaussian_process_regressor_model/',
                                     gaussian_process_regressor_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    gaussian_process_regressor_kwargs = gaussian_process_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = GaussianProcessRegressor(**gaussian_process_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def gaussian_process_classifier_model(model_dir: str = 'gaussian_process_classifier_model/',
                                      gaussian_process_classifier_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    gaussian_process_classifier_kwargs = gaussian_process_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = GaussianProcessClassifier(**gaussian_process_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###########################
### Cross decomposition ###
###########################


def pls_regression_model(model_dir: str = 'pls_regression_model/',
                         pls_regression_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    pls_regression_kwargs = pls_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = PLSRegression(**pls_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###################
### Naive Bayes ###
###################


def gaussian_nb_model(model_dir: str = 'gaussian_nb_model/',
                      gaussian_nb_kwargs: dict = None,
                      sklearn_kwargs: dict = None) -> SklearnModel:
    gaussian_nb_kwargs = gaussian_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = GaussianNB(**gaussian_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multinomial_nb_model(model_dir: str = 'multinomial_nb_model/',
                         multinomial_nb_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    multinomial_nb_kwargs = multinomial_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = MultinomialNB(**multinomial_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def bernoulli_nb_model(model_dir: str = 'bernoulli_nb_model/',
                       bernoulli_nb_kwargs: dict = None,
                       sklearn_kwargs: dict = None) -> SklearnModel:
    bernoulli_nb_kwargs = bernoulli_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = BernoulliNB(**bernoulli_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def categorical_nb_model(model_dir: str = 'categorical_nb_model/',
                         categorical_nb_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    categorical_nb_kwargs = categorical_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = CategoricalNB(**categorical_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def complement_nb_model(model_dir: str = 'complement_nb_model/',
                        complement_nb_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    complement_nb_kwargs = complement_nb_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = ComplementNB(**complement_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


######################
### Decision Trees ###
######################


def decision_tree_regressor_model(model_dir: str = 'decision_tree_regressor_model/',
                                  decision_tree_regressor_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    decision_tree_regressor_kwargs = decision_tree_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = DecisionTreeRegressor(**decision_tree_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def decision_tree_classifier_model(model_dir: str = 'decision_tree_classifier_model/',
                                   decision_tree_classifier_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    decision_tree_classifier_kwargs = decision_tree_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = DecisionTreeClassifier(**decision_tree_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


########################
### Ensemble methods ###
########################


def random_forest_regressor_model(model_dir: str = 'random_forest_regressor_model/',
                                  random_forest_regressor_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    random_forest_regressor_kwargs = random_forest_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = RandomForestRegressor(**random_forest_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def random_forest_classifier_model(model_dir: str = 'random_forest_classifier_model/',
                                   random_forest_classifier_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    random_forest_classifier_kwargs = random_forest_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = RandomForestClassifier(**random_forest_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def extra_trees_regressor_model(model_dir: str = 'extra_trees_regressor_model/',
                                extra_trees_regressor_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    extra_trees_regressor_kwargs = extra_trees_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = ExtraTreesRegressor(**extra_trees_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def extra_trees_classifier_model(model_dir: str = 'extra_trees_classifier_model/',
                                 extra_trees_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    extra_trees_classifier_kwargs = extra_trees_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = ExtraTreesClassifier(**extra_trees_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ada_boost_regressor_model(model_dir: str = 'ada_boost_regressor_model/',
                              ada_boost_regressor_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    ada_boost_regressor_kwargs = ada_boost_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = AdaBoostRegressor(**ada_boost_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ada_boost_classifier_model(model_dir: str = 'ada_boost_classifier_model/',
                               ada_boost_classifier_kwargs: dict = None,
                               sklearn_kwargs: dict = None) -> SklearnModel:
    ada_boost_classifier_kwargs = ada_boost_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = AdaBoostClassifier(**ada_boost_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def gradient_boosting_regressor_model(model_dir: str = 'gradient_boosting_regressor_model/',
                                      gradient_boosting_regressor_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    gradient_boosting_regressor_kwargs = gradient_boosting_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = GradientBoostingRegressor(**gradient_boosting_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def gradient_boosting_classifier_model(model_dir: str = 'gradient_boosting_classifier_model/',
                                       gradient_boosting_classifier_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    gradient_boosting_classifier_kwargs = gradient_boosting_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = GradientBoostingClassifier(**gradient_boosting_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def hist_gradient_boosting_regressor_model(model_dir: str = 'hist_gradient_boosting_regressor_model/',
                                           hist_gradient_boosting_regressor_kwargs: dict = None,
                                           sklearn_kwargs: dict = None) -> SklearnModel:
    hist_gradient_boosting_regressor_kwargs = hist_gradient_boosting_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = HistGradientBoostingRegressor(**hist_gradient_boosting_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def hist_gradient_boosting_classifier_model(model_dir: str = 'hist_gradient_boosting_classifier_model/',
                                            hist_gradient_boosting_classifier_kwargs: dict = None,
                                            sklearn_kwargs: dict = None) -> SklearnModel:
    hist_gradient_boosting_classifier_kwargs = hist_gradient_boosting_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = HistGradientBoostingClassifier(**hist_gradient_boosting_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def voting_regressor_model(model_dir: str = 'voting_regressor_model/',
                           voting_regressor_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    voting_regressor_kwargs = voting_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = VotingRegressor(**voting_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def voting_classifier_model(model_dir: str = 'voting_classifier_model/',
                            voting_classifier_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    voting_classifier_kwargs = voting_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = VotingClassifier(**voting_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def stacking_regressor_model(model_dir: str = 'stacking_regressor_model/',
                             stacking_regressor_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    stacking_regressor_kwargs = stacking_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = StackingRegressor(**stacking_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def stacking_classifier_model(model_dir: str = 'stacking_classifier_model/',
                              stacking_classifier_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    stacking_classifier_kwargs = stacking_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = StackingClassifier(**stacking_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def bagging_regressor_model(model_dir: str = 'bagging_regressor_model/',
                            bagging_regressor_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    bagging_regressor_kwargs = bagging_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = BaggingRegressor(**bagging_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def bagging_classifier_model(model_dir: str = 'bagging_classifier_model/',
                             bagging_classifier_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    bagging_classifier_kwargs = bagging_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = BaggingClassifier(**bagging_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


#############################################
### Multiclass and multioutput algorithms ###
#############################################


def one_vs_rest_classifier_model(model_dir: str = 'one_vs_rest_classifier_model/',
                                 one_vs_rest_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    one_vs_rest_classifier_kwargs = one_vs_rest_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = OneVsRestClassifier(**one_vs_rest_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def one_vs_one_classifier_model(model_dir: str = 'one_vs_one_classifier_model/',
                                one_vs_one_classifier_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    one_vs_one_classifier_kwargs = one_vs_one_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = OneVsOneClassifier(**one_vs_one_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def output_code_classifier_model(model_dir: str = 'output_code_classifier_model/',
                                 output_code_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    output_code_classifier_kwargs = output_code_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = OutputCodeClassifier(**output_code_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multi_output_classifier_model(model_dir: str = 'multi_output_classifier_model/',
                                  multi_output_classifier_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    multi_output_classifier_kwargs = multi_output_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = MultiOutputClassifier(**multi_output_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def classifier_chain_model(model_dir: str = 'classifier_chain_model/',
                           classifier_chain_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    classifier_chain_kwargs = classifier_chain_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = ClassifierChain(**classifier_chain_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multi_output_regressor_model(model_dir: str = 'multi_output_regressor_model/',
                                 multi_output_regressor_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    multi_output_regressor_kwargs = multi_output_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = MultiOutputRegressor(**multi_output_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def regressor_chain_model(model_dir: str = 'regressor_chain_model/',
                          regressor_chain_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    regressor_chain_kwargs = regressor_chain_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = RegressorChain(**regressor_chain_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###########################
### Isotonic regression ###
###########################


def isotonic_regression_model(model_dir: str = 'isotonic_regression_model/',
                              isotonic_regression_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    isotonic_regression_kwargs = isotonic_regression_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = IsotonicRegression(**isotonic_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


#############################
### Neural network models ###
#############################


def mlp_regressor_model(model_dir: str = 'mlp_regressor_model/',
                        mlp_regressor_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    mlp_regressor_kwargs = mlp_regressor_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Regression model
    model = MLPRegressor(**mlp_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def mlp_classifier_model(model_dir: str = 'mlp_classifier_model/',
                         mlp_classifier_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    mlp_classifier_kwargs = mlp_classifier_kwargs or {}
    sklearn_kwargs = sklearn_kwargs or {}
    # Classification model
    model = MLPClassifier(**mlp_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)

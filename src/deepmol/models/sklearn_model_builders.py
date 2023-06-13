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
    # Regression model
    model = LinearRegression(**linear_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ridge_model(model_dir: str = 'ridge_model/',
                ridge_kwargs: dict = None,
                sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = Ridge(**ridge_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ridge_classifier_mode(model_dir: str = 'ridge_classifier_model/',
                          ridge_classifier_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = RidgeClassifier(**ridge_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ridge_cv_model(model_dir: str = 'ridge_cv_model/',
                   ridge_cv_kwargs: dict = None,
                   sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = RidgeCV(**ridge_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ridge_classifier_cv(model_dir: str = 'ridge_classifier_cv_model/',
                        ridge_classifier_cv_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = RidgeClassifierCV(**ridge_classifier_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def lasso_model(model_dir: str = 'lasso_model/',
                lasso_kwargs: dict = None,
                sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = Lasso(**lasso_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def lasso_cv_model(model_dir: str = 'lasso_cv_model/',
                   lasso_cv_kwargs: dict = None,
                   sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = LassoCV(**lasso_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def lasso_lars_cv_model(model_dir: str = 'lasso_lars_cv_model/',
                        lasso_lars_cv_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = LassoLarsCV(**lasso_lars_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def lasso_lars_ic_model(model_dir: str = 'lasso_lars_ic_model/',
                        lasso_lars_ic_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = LassoLarsIC(**lasso_lars_ic_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multitask_lasso_model(model_dir: str = 'multitask_lasso_model/',
                          multitask_lasso_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = MultiTaskLasso(**multitask_lasso_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def elastic_net_model(model_dir: str = 'elastic_net_model/',
                      elastic_net_kwargs: dict = None,
                      sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = ElasticNet(**elastic_net_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multitask_elastic_net_model(model_dir: str = 'multitask_elastic_net_model/',
                                multitask_elastic_net_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = MultiTaskElasticNet(**multitask_elastic_net_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multitask_elastic_net_cv_model(model_dir: str = 'multitask_elastic_net_cv_model/',
                                   multitask_elastic_net_cv_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = MultiTaskElasticNetCV(**multitask_elastic_net_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ortogonal_matching_pursuit_model(model_dir: str = 'ortogonal_matching_pursuit_model/',
                                     ortogonal_matching_pursuit_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = OrthogonalMatchingPursuit(**ortogonal_matching_pursuit_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def bayesian_ridge_model(model_dir: str = 'bayesian_ridge_model/',
                         bayesian_ridge_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = BayesianRidge(**bayesian_ridge_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ard_regression_model(model_dir: str = 'ard_regression_model/',
                         ard_regression_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = ARDRegression(**ard_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def logistic_regression_model(model_dir: str = 'logistic_regression_model/',
                              logistic_regression_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = LogisticRegression(**logistic_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def logistic_regression_cv_model(model_dir: str = 'logistic_regression_cv_model/',
                                 logistic_regression_cv_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = LogisticRegressionCV(**logistic_regression_cv_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def tweedie_regressor_model(model_dir: str = 'tweedie_regressor_model/',
                            tweedie_regressor_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = TweedieRegressor(**tweedie_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def poison_regressor_model(model_dir: str = 'poison_regressor_model/',
                           poison_regressor_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = PoissonRegressor(**poison_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def gamma_regressor_model(model_dir: str = 'gamma_regressor_model/',
                          gamma_regressor_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = GammaRegressor(**gamma_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def perceptron_model(model_dir: str = 'perceptron_model/',
                     perceptron_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = Perceptron(**perceptron_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def passive_aggressive_regressor_model(model_dir: str = 'passive_aggressive_regressor_model/',
                                       passive_aggressive_regressor_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = PassiveAggressiveRegressor(**passive_aggressive_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def passive_aggressive_classifier_model(model_dir: str = 'passive_aggressive_classifier_model/',
                                        passive_aggressive_classifier_kwargs: dict = None,
                                        sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = PassiveAggressiveClassifier(**passive_aggressive_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def huber_regressor_model(model_dir: str = 'huber_regressor_model/',
                          huber_regressor_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = HuberRegressor(**huber_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ransac_regressor_model(model_dir: str = 'ransac_regressor_model/',
                           ransac_regressor_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = RANSACRegressor(**ransac_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def theil_sen_regressor_model(model_dir: str = 'theil_sen_regressor_model/',
                              theil_sen_regressor_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = TheilSenRegressor(**theil_sen_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def quantile_regressor_model(model_dir: str = 'quantile_regressor_model/',
                             quantile_regressor_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = QuantileRegressor(**quantile_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


##################################################
### Linear and Quadratic Discriminant Analysis ###
##################################################


def linear_discriminant_analysis_model(model_dir: str = 'linear_discriminant_analysis_model/',
                                       linear_discriminant_analysis_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = LinearDiscriminantAnalysis(**linear_discriminant_analysis_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def quadratic_discriminant_analysis_model(model_dir: str = 'quadratic_discriminant_analysis_model/',
                                          quadratic_discriminant_analysis_kwargs: dict = None,
                                          sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = QuadraticDiscriminantAnalysis(**quadratic_discriminant_analysis_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###############################
### Kernel ridge regression ###
###############################


def kernel_ridge_regressor_model(model_dir: str = 'kernel_ridge_regressor_model/',
                                 kernel_ridge_regressor_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = KernelRidge(**kernel_ridge_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###############################
### Support Vector Machines ###
###############################


def svc_model(model_dir: str = 'svc_model/',
              svc_kwargs: dict = None,
              sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = SVC(**svc_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def nu_svc_model(model_dir: str = 'nu_svc_model/',
                 nu_svc_kwargs: dict = None,
                 sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = NuSVC(**nu_svc_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def linear_svc_model(model_dir: str = 'linear_svc_model/',
                     linear_svc_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = LinearSVC(**linear_svc_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def svr_model(model_dir: str = 'svr_model/',
              svr_kwargs: dict = None,
              sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = SVR(**svr_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def nu_svr_model(model_dir: str = 'nu_svr_model/',
                 nu_svr_kwargs: dict = None,
                 sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = NuSVR(**nu_svr_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def linear_svr_model(model_dir: str = 'linear_svr_model/',
                     linear_svr_kwargs: dict = None,
                     sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = LinearSVR(**linear_svr_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def one_class_svm_model(model_dir: str = 'one_class_svm_model/',
                        one_class_svm_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = OneClassSVM(**one_class_svm_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###################################
### Stochastic Gradient Descent ###
###################################


def sgd_regressor_model(model_dir: str = 'sgd_regressor_model/',
                        sgd_regressor_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = SGDRegressor(**sgd_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def sgd_classifier_model(model_dir: str = 'sgd_classifier_model/',
                         sgd_classifier_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = SGDClassifier(**sgd_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def sgd_one_class_svm(model_dir: str = 'sgd_one_class_svm/',
                      sgd_one_class_svm_kwargs: dict = None,
                      sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = SGDOneClassSVM(**sgd_one_class_svm_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


#########################
### Nearest Neighbors ###
#########################


def kneighbors_regressor_model(model_dir: str = 'kneighbors_regressor_model/',
                               kneighbors_regressor_kwargs: dict = None,
                               sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = KNeighborsRegressor(**kneighbors_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def kneighbors_classifier_model(model_dir: str = 'kneighbors_classifier_model/',
                                kneighbors_classifier_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = KNeighborsClassifier(**kneighbors_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def radius_neighbors_regressor_model(model_dir: str = 'radius_neighbors_regressor_model/',
                                     radius_neighbors_regressor_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = RadiusNeighborsRegressor(**radius_neighbors_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def radius_neighbors_classifier_model(model_dir: str = 'radius_neighbors_classifier_model/',
                                      radius_neighbors_classifier_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = RadiusNeighborsClassifier(**radius_neighbors_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def nearest_centroid_model(model_dir: str = 'nearest_centroid_model/',
                           nearest_centroid_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = NearestCentroid(**nearest_centroid_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


##########################
### Gaussian Processes ###
##########################

def gaussian_process_regressor_model(model_dir: str = 'gaussian_process_regressor_model/',
                                     gaussian_process_regressor_kwargs: dict = None,
                                     sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = GaussianProcessRegressor(**gaussian_process_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def gaussian_process_classifier_model(model_dir: str = 'gaussian_process_classifier_model/',
                                      gaussian_process_classifier_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = GaussianProcessClassifier(**gaussian_process_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###########################
### Cross decomposition ###
###########################


def pls_regression_model(model_dir: str = 'pls_regression_model/',
                         pls_regression_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = PLSRegression(**pls_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###################
### Naive Bayes ###
###################


def gaussian_nb_model(model_dir: str = 'gaussian_nb_model/',
                      gaussian_nb_kwargs: dict = None,
                      sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = GaussianNB(**gaussian_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multinomial_nb_model(model_dir: str = 'multinomial_nb_model/',
                         multinomial_nb_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = MultinomialNB(**multinomial_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def bernoulli_nb_model(model_dir: str = 'bernoulli_nb_model/',
                       bernoulli_nb_kwargs: dict = None,
                       sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = BernoulliNB(**bernoulli_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def categorical_nb_model(model_dir: str = 'categorical_nb_model/',
                         categorical_nb_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = CategoricalNB(**categorical_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def complement_nb_model(model_dir: str = 'complement_nb_model/',
                        complement_nb_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = ComplementNB(**complement_nb_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


######################
### Decision Trees ###
######################


def decision_tree_regressor_model(model_dir: str = 'decision_tree_regressor_model/',
                                  decision_tree_regressor_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = DecisionTreeRegressor(**decision_tree_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def decision_tree_classifier_model(model_dir: str = 'decision_tree_classifier_model/',
                                   decision_tree_classifier_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = DecisionTreeClassifier(**decision_tree_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


########################
### Ensemble methods ###
########################


def random_forest_regressor_model(model_dir: str = 'random_forest_regressor_model/',
                                  random_forest_regressor_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = RandomForestRegressor(**random_forest_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def random_forest_classifier_model(model_dir: str = 'random_forest_classifier_model/',
                                   random_forest_classifier_kwargs: dict = None,
                                   sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = RandomForestClassifier(**random_forest_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def extra_trees_regressor_model(model_dir: str = 'extra_trees_regressor_model/',
                                extra_trees_regressor_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = ExtraTreesRegressor(**extra_trees_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def extra_trees_classifier_model(model_dir: str = 'extra_trees_classifier_model/',
                                 extra_trees_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = ExtraTreesClassifier(**extra_trees_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ada_boost_regressor_model(model_dir: str = 'ada_boost_regressor_model/',
                              ada_boost_regressor_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = AdaBoostRegressor(**ada_boost_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def ada_boost_classifier_model(model_dir: str = 'ada_boost_classifier_model/',
                               ada_boost_classifier_kwargs: dict = None,
                               sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = AdaBoostClassifier(**ada_boost_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def gradient_boosting_regressor_model(model_dir: str = 'gradient_boosting_regressor_model/',
                                      gradient_boosting_regressor_kwargs: dict = None,
                                      sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = GradientBoostingRegressor(**gradient_boosting_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def gradient_boosting_classifier_model(model_dir: str = 'gradient_boosting_classifier_model/',
                                       gradient_boosting_classifier_kwargs: dict = None,
                                       sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = GradientBoostingClassifier(**gradient_boosting_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def hist_gradient_boosting_regressor_model(model_dir: str = 'hist_gradient_boosting_regressor_model/',
                                           hist_gradient_boosting_regressor_kwargs: dict = None,
                                           sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = HistGradientBoostingRegressor(**hist_gradient_boosting_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def hist_gradient_boosting_classifier_model(model_dir: str = 'hist_gradient_boosting_classifier_model/',
                                            hist_gradient_boosting_classifier_kwargs: dict = None,
                                            sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = HistGradientBoostingClassifier(**hist_gradient_boosting_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def voting_regressor_model(model_dir: str = 'voting_regressor_model/',
                           voting_regressor_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = VotingRegressor(**voting_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def voting_classifier_model(model_dir: str = 'voting_classifier_model/',
                            voting_classifier_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = VotingClassifier(**voting_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def stacking_regressor_model(model_dir: str = 'stacking_regressor_model/',
                             stacking_regressor_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = StackingRegressor(**stacking_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def stacking_classifier_model(model_dir: str = 'stacking_classifier_model/',
                              stacking_classifier_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = StackingClassifier(**stacking_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def bagging_regressor_model(model_dir: str = 'bagging_regressor_model/',
                            bagging_regressor_kwargs: dict = None,
                            sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = BaggingRegressor(**bagging_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def bagging_classifier_model(model_dir: str = 'bagging_classifier_model/',
                             bagging_classifier_kwargs: dict = None,
                             sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = BaggingClassifier(**bagging_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


#############################################
### Multiclass and multioutput algorithms ###
#############################################


def one_vs_rest_classifier_model(model_dir: str = 'one_vs_rest_classifier_model/',
                                 one_vs_rest_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = OneVsRestClassifier(**one_vs_rest_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def one_vs_one_classifier_model(model_dir: str = 'one_vs_one_classifier_model/',
                                one_vs_one_classifier_kwargs: dict = None,
                                sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = OneVsOneClassifier(**one_vs_one_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def output_code_classifier_model(model_dir: str = 'output_code_classifier_model/',
                                 output_code_classifier_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = OutputCodeClassifier(**output_code_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multi_output_classifier_model(model_dir: str = 'multi_output_classifier_model/',
                                  multi_output_classifier_kwargs: dict = None,
                                  sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = MultiOutputClassifier(**multi_output_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def classifier_chain_model(model_dir: str = 'classifier_chain_model/',
                           classifier_chain_kwargs: dict = None,
                           sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = ClassifierChain(**classifier_chain_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def multi_output_regressor_model(model_dir: str = 'multi_output_regressor_model/',
                                 multi_output_regressor_kwargs: dict = None,
                                 sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = MultiOutputRegressor(**multi_output_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def regressor_chain_model(model_dir: str = 'regressor_chain_model/',
                          regressor_chain_kwargs: dict = None,
                          sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = RegressorChain(**regressor_chain_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


###########################
### Isotonic regression ###
###########################


def isotonic_regression_model(model_dir: str = 'isotonic_regression_model/',
                              isotonic_regression_kwargs: dict = None,
                              sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = IsotonicRegression(**isotonic_regression_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


#############################
### Neural network models ###
#############################


def mlp_regressor_model(model_dir: str = 'mlp_regressor_model/',
                        mlp_regressor_kwargs: dict = None,
                        sklearn_kwargs: dict = None) -> SklearnModel:
    # Regression model
    model = MLPRegressor(**mlp_regressor_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)


def mlp_classifier_model(model_dir: str = 'mlp_classifier_model/',
                         mlp_classifier_kwargs: dict = None,
                         sklearn_kwargs: dict = None) -> SklearnModel:
    # Classification model
    model = MLPClassifier(**mlp_classifier_kwargs)
    return SklearnModel(model=model, model_dir=model_dir, **sklearn_kwargs)

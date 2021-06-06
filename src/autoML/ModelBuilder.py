from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from models.sklearnModels import SklearnModel
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier


class ModelBuilder(object):
    
    def __init__(self, model_params):
        self.model_params = model_params


    def rf_model_builder(self, params):
        rf_model = RandomForestClassifier(**params)
        return SklearnModel(model=rf_model)


    #def rf_model_optimizer(self, params):
    #    optimizer = GridHyperparamOpt(rf_model_builder)
    #    best_rf, best_hyperparams, all_results = optimizer.hyperparam_search(params, train_dataset,
    #                                                                     valid_dataset, Metric(roc_auc_score))


    def svm_model_builder(self, params):
        svm = SVC(**params)
        return SklearnModel(model=svm)


    def knc_model_builder(self, params):
        knc_model = KNeighborsClassifier(**params)
        return SklearnModel(model=knc_model)

    
    def dt_model_builder(self, params):
        dt_model = tree.DecisionTreeClassifier(**params)
        return SklearnModel(model=dt_model)

    def ridge_model_builder(self, params):
        r_model = RidgeClassifier(**params)
        return SklearnModel(model=r_model)

    def sgd_model_builder(self, params):
        sgd_model = SGDClassifier(**params)
        return SklearnModel(model=sgd_model)

    def ab_model_builder(self, params):
        ab_model = AdaBoostClassifier(**params)
        return SklearnModel(model=ab_model)


    def return_initialized_models(self):

        initialized_models = []

        for model in self.model_params:
            params = {}
            if(model['type'] == 'params'):
                params = model['params']

            if(model['name'] == "RandomForestClassifier"):
                initialized_models.append(self.rf_model_builder(params))
            elif(model['name'] == "SVM"):
                initialized_models.append(self.svm_model_builder(params))
            elif(model['name'] == "KNeighborsClassifier"):
                initialized_models.append(self.knc_model_builder(params))
            elif(model['name'] == "DecisionTreeClassifier"):
                initialized_models.append(self.dt_model_builder(params))
            elif(model['name'] == "RidgeClassifier"):
                initialized_models.append(self.ridge_model_builder(params))
            elif(model['name'] == "SGDClassifier"):
                initialized_models.append(self.sgd_model_builder(params))
            elif(model['name'] == "AdaBoostClassifier"):
                initialized_models.append(self.ab_model_builder(params))
        
        return initialized_models
        
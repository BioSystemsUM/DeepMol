import json
from os import linesep, write

from numpy import info
from loaders.Loaders import CSVLoader
from compoundFeaturization.rdkitFingerprints import MorganFingerprint
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from autoML.ModelBuilder import ModelBuilder
from metrics.Metrics import Metric
from models.sklearnModels import SklearnModel
from splitters.splitters import SingletaskStratifiedSplitter
from metrics.metricsFunctions import roc_auc_score, precision_score, accuracy_score, confusion_matrix, classification_report
from autoML.OptimizationSelector import OptimizationSelector
from autoML.featurizer import Featurizer

class AutoML(object):

    def __init__(self, path):
        self.file_data = self.load_file(path)
        self.info = self.parse_data()
        print(self.info['models_info'])


    def load_file(self, path):
        try:
            with open(path) as json_file:
                data = json.load(json_file)
                return data
        except:
            print("Cant load provided file as JSON.")


    def parse_data(self):

        load_info = self.parse_load()
        featurizer_info = self.parse_featurizer()
        models_info = self.parse_models()
        metrics = self.parse_metrics()

        return {'load_info': load_info,
                'featurizer_info': featurizer_info,
                'models_info': models_info,
                'metrics': metrics,
                }


    def parse_load(self):
        load_params = {
            'id_field': None,
            'labels_fields': None,
            'features_fields': None,
            'features2keep': None,
            'shard_size': None}
        
        for param in self.file_data['load']:
            load_params[param] = self.file_data['load'][param]
        return load_params


    def parse_featurizer(self):
        featurizer_info = []
        for featurizer in self.file_data['featurizers']:
            info = {}
            info['name'] = featurizer['name']
            info['type'] = featurizer['type']
            info['params'] = featurizer['params']
            featurizer_info.append(info)
        return featurizer_info


    def parse_models(self):
        models_info = []
        for model in self.file_data['models']:
            info = {}
            info['name'] = model['name']
            if 'params' not in model.keys():
                info['type'] = 'default'
            elif model['params'] == {}:
                info['type'] = 'default'
            else:
                n_lists = 0
                for hyperparam_list in model['params'].values():
                    print("hyperparam_list:", hyperparam_list)
                    if isinstance(hyperparam_list, list):
                        n_lists += 1
                if n_lists == len(model['params'].values()):
                    info['type'] = 'opt'
                else:
                    info['type'] = 'params'
                info['params'] = model['params']
            models_info.append(info)
        return models_info


    def parse_metrics(self):
        # TODO: dicionario em vez de ifs
        initialized_metrics = []
        for metric in self.file_data['metrics']:
            metric_to_insert = None
            if(metric == "roc_auc_score"):
                metric_to_insert = roc_auc_score
            if(metric == "precision_score"):
                metric_to_insert = precision_score
            if(metric == "accuracy_score"):
                metric_to_insert = accuracy_score
            if(metric == "confusion_matrix"):
                metric_to_insert = confusion_matrix
            initialized_metrics.append(Metric(metric_to_insert))
        return initialized_metrics

    def output2csvFormat(self, models, featurizer, list_scores, fst):
        line = ""
        for i in range(len(models)):
            line = str(models[i].model)
            line += ' ; ' + featurizer
            for score in list_scores[i]:
                line += ' ; ' + str(score)
            line += '\n'
        fst_line = ""
        if(fst):
            fst_line = 'Model ; Featurizer'
            for metric in self.info['metrics']:
                fst_line += ' ; ' + metric.name
            fst_line += ' \n'
        line = fst_line + line
        
        return line

    def execute(self):

        li = self.info['load_info']
        lines = []
        fst = True
        dataset = CSVLoader(
            dataset_path=li['dataset'],
            mols_field=li['mols'], 
            id_field=li['id_field'],
            labels_fields=li['labels_fields'],
            features_fields=li['features_fields'],
            features2keep=li['features2keep'],
            shard_size=li['shard_size'])
        dataset = dataset.create_dataset()
        featurizer = Featurizer(self.info['featurizer_info'])
        allinfo = []
        for featurizer_info in self.info['featurizer_info']:
            featurizer = Featurizer(featurizer_info)
            dataset = featurizer.fingerprint(dataset)
            #print(dataset.get_shape()

            #Data Split
            splitter = SingletaskStratifiedSplitter()
            train_dataset, valid_dataset, test_dataset = \
                splitter.train_valid_test_split(dataset=dataset, 
                                                frac_train=0.6, 
                                                frac_valid=0.2, 
                                                frac_test=0.2)

            normal_models = []
            models_to_optimize = []
            
            for model in self.info['models_info']:
                if model['type'] == 'opt':
                    models_to_optimize.append(model)
                else:
                    normal_models.append(model)

            optimization_selector = OptimizationSelector(models_to_optimize, dataset)
            best_opt_models, results = optimization_selector.select_models()

            model_builder = ModelBuilder(normal_models)
            models = model_builder.return_initialized_models()
            models.extend(best_opt_models)
            
            print("RESULTS \n\n")
            for result in results:
                print(type(result))
            print("RESULTS \n\n")

            #print(models)
            infoline = {"featurizer" : featurizer_info['name'] }
            modelscores = []
            for model in models:
                model.fit(train_dataset)
                scores = model.evaluate(valid_dataset, metrics=self.info['metrics'])
                tuplo = (str(model.model), scores.values())
                modelscores.append(tuplo)
            infoline['scores'] = modelscores

            allinfo.append(infoline)
        
        print(allinfo)
        
        with open("output.csv",'w') as output:
            fst_line = "Model ; Featurizer"
            for metric in self.info['metrics']:
                fst_line += ' ; ' + metric.name
            fst_line += ' \n'
            output.write(fst_line)

            for info in allinfo:
                for tuplo in info['scores']:
                    model,scores = tuplo
                    line = model + "; " + info['featurizer']
                    for score in scores:
                        line += "; " + str(score)
                    line += '\n'
                    output.write(line)
                
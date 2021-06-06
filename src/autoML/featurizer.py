from compoundFeaturization import rdkitFingerprints

class Featurizer(object):
    
    def __init__(self, featurizer_params):
        self.featurizer_params = featurizer_params


    def morganFP(self, params, dataset):
        return rdkitFingerprints.MorganFingerprint(**params).featurize(dataset)


    def maccskeysFP(self, dataset):
        return rdkitFingerprints.MACCSkeysFingerprint().featurize(dataset)


    def layeredFP(self, params, dataset):
        return rdkitFingerprints.LayeredFingerprint(**params).featurize(dataset)


    def rdkFP(self, params, dataset):
        return rdkitFingerprints.RDKFingerprint(**params).featurize(dataset)


    def atomPairFP(self, params, dataset):
        return rdkitFingerprints.AtomPairFingerprint(**params).featurize(dataset)


    def fingerprint(self, dataset):
        if self.featurizer_params['name'] == 'morgan':
            if self.featurizer_params['type'] == 'params':
                return self.morganFP(params = self.featurizer_params['params'], dataset = dataset)
            else:
                return self.morganFP(params = {}, dataset = dataset)

        elif self.featurizer_params['name'] == 'layered':
            if self.featurizer_params['type'] == 'params':
                return self.layeredFP(params = self.featurizer_params['params'], dataset = dataset)
            else:
                return self.layeredFP(params = {}, dataset = dataset)

        elif self.featurizer_params['name'] == 'rdk':
            if self.featurizer_params['type'] == 'params':
                return self.rdkFP(params = self.featurizer_params['params'], dataset = dataset)
            else:
                return self.rdkFP(params = {}, dataset = dataset)        
        
        elif self.featurizer_params['name'] == 'atom':
            if self.featurizer_params['type'] == 'params':
                return self.atomPairFP(params = self.featurizer_params['params'], dataset = dataset)
            else:
                return self.atomPairFP(params = {}, dataset = dataset)

        elif self.featurizer_params['name'] == 'maccs':
                return self.maccskeysFP(dataset = dataset)           




        
        
        
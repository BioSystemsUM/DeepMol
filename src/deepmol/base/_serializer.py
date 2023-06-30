import pickle


class Serializer:

    def to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

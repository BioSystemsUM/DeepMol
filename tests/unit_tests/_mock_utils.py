import copy
from unittest.mock import MagicMock


class SmilesDatasetMagicMock(MagicMock):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __deepcopy__(self, memo):
        # Create a copy of the object using copy.deepcopy
        new_obj = MagicMock()
        memo[id(self)] = new_obj
        for k, v in self.__dict__.items():
            setattr(new_obj, k, copy.deepcopy(v, memo))

        return new_obj
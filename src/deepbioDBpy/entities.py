import ast
import re
import time
from copy import deepcopy

import requests
from pandas import DataFrame

from deepbioDBpy import config


class DeepBioEntity(object):

    def __init__(self):
        self.generic_url = config.API_HOST

    @property
    def generic_url(self):
        return self._generic_url

    @generic_url.setter
    def generic_url(self, value):
        self._generic_url = value


class DeepBioCompound(DeepBioEntity):

    def __init__(self, deepbio_id, get_info_automatically=True):
        super().__init__()

        self.id = deepbio_id
        self.specific_url = self.generic_url + config.API_COMPOUNDS

        if get_info_automatically:
            self.__get_information(deepbio_id)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        self._formula = value

    def __add_prop(instance, prop_name, propr):
        """Attach property proper to instance with name prop_name.

        """
        class_name = instance.__class__.__name__
        child_class = type(class_name, (instance.__class__,), {prop_name: propr})

        instance.__class__ = child_class

    def __get_information(self, deepbio_id):

        res = requests.get(self.specific_url + str(deepbio_id))
        status_code = res.status_code

        if status_code == 200:
            self.id = deepbio_id
            res_dict = res.json()
            structure_representations = res_dict["structure_representation"]
            properties = res_dict["compound_property"]
            self.formula = res_dict["formula"]
            self.__get_info_from_strings(structure_representations)
            self.__get_info_from_strings(properties, True)

    def set_information(self, info):

        structure_representations = info["structure_representation"]
        properties = info["compound_property"]
        self.formula = info["formula"]
        self.__get_info_from_strings(structure_representations)
        self.__get_info_from_strings(properties, True)

    def __get_info_from_strings(self, lst, property=False):

        for l in lst:
            split_string = l.split(":")
            name = split_string[0]

            if " " in name:
                name = name.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "")

            value = split_string[1]

            if property and re.match("[0-9]", value):
                value = ast.literal_eval(value)

            self.__add_prop(name, value)


class DeepBioDataset(DeepBioEntity):

    def __init__(self, properties=[], structure=""):
        super().__init__()
        self.specific_url = self.generic_url + config.DOWNLOAD_DATASET
        self.properties = properties
        self.structure = structure

    @property
    def properties(self):
        return self._properties

    @properties.setter
    def properties(self, value):
        self._properties = value

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        self._structure = value

    # def __download_dataset(self):
    #     properties = str(self.properties)
    #     properties = properties.replace("[","").replace("]","").replace("\\","").replace("'","")
    #     with requests.post(self.specific_url,data={"properties":properties,"structure":self.structure}) as r:
    #         # with open(local_filename, 'wb') as f:
    #         r.raise_for_status()
    #         for line in r.iter_lines(chunk_size=8192):
    #             line_lst = line.split(",")
    #             self.dset.append(line_lst)

    def download_dataset_for_file(self, local_file_name):
        """
        Download dataset in batches
        :param local_file_name: file_name
        :return:
        """

        properties = str(self.properties)
        properties = properties.replace("[", "").replace("]", "").replace("\\", "").replace("'", "")
        data = {"properties": properties, "structure": self.structure}

        with requests.post(self.specific_url, data=data, stream=True) as r:
            r.raise_for_status()
            i = 0
            start = time.time()
            with open(local_file_name, 'wb') as f:
                for line in r.iter_lines(chunk_size=8192):
                    end = time.time()
                    i += 1
                    if end - start > 1:
                        print(i)
                        i = 0
                        start = time.time()

                    f.write(line)
                    f.write(b"\n")

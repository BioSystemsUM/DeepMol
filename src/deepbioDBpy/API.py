import json
import urllib.parse

import requests
from requests.auth import HTTPBasicAuth

from deepbioDBpy import config
from deepbioDBpy.entities import DeepBioCompound



class DeepBioAPI:

    @staticmethod
    def get_compound(deepbio_id):
        return DeepBioCompound(deepbio_id)

    @staticmethod
    def parse_compound_information(info):

        deepbio_id = info["id"]
        compound = DeepBioCompound(deepbio_id, False)
        compound.set_information(info)
        return compound

    @staticmethod
    def get_compounds_with_specific_properties(properties: dict,limit: int):

        query_compounds_url = config.API_HOST + config.API_COMPOUNDS + "?"
        queries = []

        for property in properties:

            query_property_name = "compound_property__property__name="
            query_property_name += property.strip().replace(" ","%20")

            query_compounds_values = "compound_property__value="
            query_compounds_values += str(properties[property])

            queries.append(query_property_name)
            queries.append(query_compounds_values)

        final_query = "&".join(queries)
        query_compounds_url += final_query

        res = requests.get(query_compounds_url,{"limit":limit})
        status_code = res.status_code


        compounds_list = []
        if status_code == 200:

            res_dict = res.json()
            compounds = res_dict["results"]
            for compound in compounds:
                compound = DeepBioAPI.parse_compound_information(compound)

                compounds_list.append(compound)

        return compounds_list


    @staticmethod
    def get_structure_with_specific_properties(structure:str, properties: dict) -> list:

        query_compounds_url = config.API_HOST + config.API_COMPOUNDS + \
                              "structures/?representation=" + structure

        properties_string = json.dumps(properties)
        res = requests.get(query_compounds_url, {"properties": properties_string})
        status_code = res.status_code

        if status_code == 200:
            res_dict = res.json()
            structures = res_dict["results"]

            return structures

        return []

    @staticmethod
    def get_property_nominal_values(name:str):

        query_property_url = config.API_HOST + config.API_PROPERTIES

        params = {"name":name}
        res = requests.get(query_property_url, params)

        status_code = res.status_code

        if status_code == 200:
            res_dict = res.json()
            results = res_dict.get("results")

            if results:

                res = results[0].get("nominal_indexes")
                res_formatted = {}

                if res:
                    for r in res:

                        res_formatted[int(r)] = res[r]

                    return res_formatted

                else:

                    raise Exception("This property assumes real number values")

        return {}


    @staticmethod
    def get_property_nominal_value(name:str,index:int):

        query_property_url = config.API_HOST + config.API_PROPERTIES + "nominal_value/"
        params = {"index" : index, "name":name}
        res = requests.get(query_property_url,params)
        status_code = res.status_code

        if status_code == 200:
            res_dict = res.json()
            value = res_dict.get("nominal_value")

            if value:
                return value

        raise Exception("This property assumes real number values")



    @staticmethod
    def load_dataset(username,password,params,dataset):

        # client = Client()

        # client.login(username=username, password=password)

        params_dict = {}

        for param in params:


            if isinstance(params[param], dict):
                param_string = json.dumps(params[param])

            elif isinstance(params[param], list):
                param_string_lst = [str(param) for param in params[param]]
                param_string = ",".join(param_string_lst)

            else:
                param_string = params[param]

            params_dict[param] = param_string

        with open(dataset, 'rb') as file:

            file = {'file_uploaded' : file}
            # response = requests.post(config.API_HOST +'upload/compounds_dataset/', data=params_dict,auth = HTTPBasicAuth(username, password))
            response = requests.post(config.API_HOST + 'upload/compounds_dataset/', data=params_dict,files=file,auth = HTTPBasicAuth(username, password))

        return response

    @staticmethod
    def get_compound_by_structure(value):

        query_compounds_url = config.API_HOST + config.API_COMPOUNDS + \
                              "?structure_representation__value=" + value

        res = requests.get(query_compounds_url)
        status_code = res.status_code

        if status_code == 200:
            res_dict = res.json()
            value = res_dict.get("results")

            if value:
                compound = DeepBioCompound(value[0].get("id"),False)
                compound.set_information(value[0])

                return compound

        return None
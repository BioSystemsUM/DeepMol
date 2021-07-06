from unittest import TestCase

from deepbioDBpy.API import DeepBioAPI
from deepbioDBpy.entities import DeepBioCompound, DeepBioDataset


class TestWrappers(TestCase):

    # def setUp(self) -> None:
    #     self.factory = RequestFactory()
    #     self.user = User.objects.create_user(
    #         username='test', email='tests@gmail.com', password='top_secret')

    def test_compound_wrapper(self):

        compound = DeepBioCompound(1)
        compound2 = DeepBioCompound(6838)

        assert compound.sweet == "yes"

    def test_dataset(self):

        dataset = DeepBioDataset(["sweet"], "smiles")
        dataset.download_dataset_for_file("test_exporter.csv")

    def test_dataset2(self):
        dataset = DeepBioDataset(["sweet", "molecular weight"], "smiles")
        dataset.download_dataset_for_file("test_exporter.csv")

    def test_query_properties(self):

        properties = {"sweet":1}

        compounds = DeepBioAPI.get_compounds_with_specific_properties(properties,100)

        assert compounds[0].sweet == "yes"
        assert len(compounds) == 100

    def test_get_structures(self):
        properties = {"sweet": 2}

        smiles = DeepBioAPI.get_structure_with_specific_properties("inchi",properties)
        assert smiles

    def test_get_nominal_value(self):

        nominal_value = DeepBioAPI.get_property_nominal_value("sweet",1)
        assert  nominal_value == "yes"

    def test_load_dataset(self):
        name = "Predicted Sweetners third set"
        description = "Third set of predicted sweetners and toxicity probabilities"
        dset_type = "other"
        separator = ";"
        structure_columns = {"smiles": 1}
        properties_columns = [i for i in range(7, 12)]
        properties_columns.append(2)
        predicted_property_columns = [i for i in range(7, 12)]
        predicted_property_columns.append(2)

        # external_references = [7]
        stereoisomerism_columns = {"enantiomer": 4}

        file = "C:/Users/Joao/Desktop/DeepBio/deep_bio_web_server/deep_bio_db_api/datasets/third_sweet_set.csv"

        form = {
            'description': description,
            'name': name,
            'dataset_type': dset_type,
            'separator': separator,
            'structures_columns': structure_columns,
            'properties_columns': properties_columns,
            'predicted_property_columns': predicted_property_columns,
            'external_references_columns': [],
            'external_database_columns': [],
            'name_columns': []
        }

        from deepbioDBpy.API import DeepBioAPI
        DeepBioAPI.load_dataset("joao", "Tuning999", form, file)
from unittest import TestCase

from deepchem.models import MultitaskClassifier
from keras.layers import Dense, Dropout, GaussianNoise, Flatten
from keras.optimizers import Adadelta, Adam
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdmolfiles import MolFromSmiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_score, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Reshape, Conv1D
from tensorflow.python.keras.optimizer_v1 import RMSprop

from deepmol.compound_featurization import MixedFeaturizer
from deepmol.compound_featurization.rdkit_descriptors import AutoCorr3D, All3DDescriptors, RadialDistributionFunction, \
    PlaneOfBestFit, MORSE, WHIM, RadiusOfGyration, InertialShapeFactor, Eccentricity, Asphericity, \
    SpherocityIndex, PrincipalMomentsOfInertia, NormalizedPrincipalMomentsRatios, \
    ThreeDimensionalMoleculeGenerator, generate_conformers_to_sdf_file, get_all_3D_descriptors
from deepmol.compound_featurization import MorganFingerprint
from deepmol.loaders.loaders import SDFLoader, CSVLoader
from deepmol.metrics.metrics import Metric
from deepmol.models.deepchem_models import DeepChemModel
from deepmol.models.keras_models import KerasModel
from deepmol.models.sklearn_models import SklearnModel
from deepmol.splitters.splitters import SingletaskStratifiedSplitter

class Test3DGeneration(TestCase):

    def test_align(self):
        generator = ThreeDimensionalMoleculeGenerator()

        mol = MolFromSmiles("Cc1cc2-c3c(O)cc(cc3OC3(Oc4cc(O)ccc4-c(c1O)c23)c1ccc(O)cc1O)-c1cc2cccc(O)c2o1")

        generator.generate_conformers(mol)


class TestSdfImporter(TestCase):

    def test_sdf_importer(self):
        loader = SDFLoader("tests/data/dataset_sweet_3D_to_test.sdf", "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        assert len(loader.mols_handler) == 100
        assert len(dataset.y) == 100
        assert len(dataset.X) == 100

    def test_2_sdf_importer(self):
        loader = SDFLoader("tests/data/A2780.sdf", "ChEMBL_ID", labels_fields=["pIC50"])
        dataset = loader.create_dataset()

        assert len(dataset.X) == 2255


class TestRdkit3DDescriptors(TestCase):

    def setUp(self) -> None:
        self.test_dataset_sdf = "../data/dataset_sweet_3D_to_test.sdf"
        self.test_dataset_to_fail = "../data/preprocessed_dataset_wfoodb.csv"
        self.mini_dataset_to_generate_conformers = "../data/test_to_convert_to_sdf.csv"

    def test_autocorr3D(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = AutoCorr3D().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)

    def test_autocorr3D_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            AutoCorr3D().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_RDF(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = RadialDistributionFunction().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 210

    def test_RDF_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            RadialDistributionFunction().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_PBF(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = PlaneOfBestFit().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 1

    def test_PDF_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            PlaneOfBestFit().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_MORSE(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = MORSE().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 224

    def test_MORSE_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            MORSE().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_WHIM(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = WHIM().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 114

    def test_WHIM_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            WHIM().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_radius_of_gyration(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = RadiusOfGyration().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 1

    def test_radius_of_gyration_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            RadiusOfGyration().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_isf(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = InertialShapeFactor().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 1

    def test_isf_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            InertialShapeFactor().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_Eccentricity(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = Eccentricity().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 1

    def test_eccentricity_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            Eccentricity().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_Asphericity(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = Asphericity().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 1

    def test_asphericity_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            Asphericity().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_SpherocityIndex(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = SpherocityIndex().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 1

    def test_SpherocityIndex_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            SpherocityIndex().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_PMI(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = PrincipalMomentsOfInertia().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 3

    def test_PMI_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            PrincipalMomentsOfInertia().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_NormalizedPrincipalMomentsRatios(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = NormalizedPrincipalMomentsRatios().featurize(dataset)

        assert len(dataset.X) == 100
        assert isinstance(dataset.X[0][0], float)
        assert len(dataset.X[0]) == 2

    def test_NormalizedPrincipalMomentsRatios_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            NormalizedPrincipalMomentsRatios().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_all_rdkit_descriptors(self):
        loader = SDFLoader(self.test_dataset_sdf, "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        dataset = All3DDescriptors().featurize(dataset)

        assert len(dataset.X) == 100
        assert len(dataset.X[0]) == 639

    def test_all_rdkit_descriptors_to_fail(self):
        loader = CSVLoader(self.test_dataset_to_fail,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        dataset = loader.create_dataset()

        with self.assertRaises(SystemExit) as cm:
            All3DDescriptors().featurize(dataset)

        self.assertEqual(cm.exception.code, 1)

    def test_all_rdkit_descriptors_generating_conformers(self):
        loader = CSVLoader(self.mini_dataset_to_generate_conformers,
                           smiles_field='Smiles',
                           labels_fields='Class')

        dataset = loader.create_dataset()

        All3DDescriptors(generate_conformers=True).featurize(dataset)

        mol = MolFromSmiles("CC(C)(C)C(O)C(O)=O")
        generator = ThreeDimensionalMoleculeGenerator()
        mol = generator.generate_conformers(mol)
        mol = generator.optimize_molecular_geometry(mol)
        descriptors = get_all_3D_descriptors(mol)

        first_instance = dataset.X[0]

        for i in range(len(first_instance)):
            self.assertAlmostEqual(descriptors[i], first_instance[i], delta=1)


class TestMixedDescriptors(TestCase):

    def setUp(self) -> None:
        self.test_dataset_sdf = "../data/dataset_sweet_3D_to_test.sdf"
        self.test_dataset_to_fail = "../preprocessed_dataset_wfoodb.csv"

    def test_mixed_descriptors_fingerprints_rdkit(self):
        loader = SDFLoader("tests/data/dataset_sweet_3D_to_test.sdf", "_SourceID", labels_fields=["_SWEET"])
        dataset = loader.create_dataset()

        descriptors = [All3DDescriptors(), MorganFingerprint()]

        dataset = MixedFeaturizer(featurizers=descriptors).featurize(dataset)

        assert len(dataset.X) == 100
        assert len(dataset.X[0]) == 2687


class TestModels3DDescriptors(TestCase):

    def setUp(self) -> None:
        loader = SDFLoader("tests/data/dataset_sweet_3d_balanced.sdf", "_SourceID", labels_fields=["_SWEET"])
        self.dataset = loader.create_dataset()

        self.dataset = All3DDescriptors().featurize(self.dataset, scale=True)

        splitter = SingletaskStratifiedSplitter()
        self.train_dataset, self.valid_dataset, self.test_dataset = splitter.train_valid_test_split(
            dataset=self.dataset,
            frac_train=0.6,
            frac_valid=0.2,
            frac_test=0.2)

    def test_svm_3d_descriptors(self):
        svm = SVC()
        model = SklearnModel(model=svm)

        res = model.cross_validate(self.dataset, Metric(roc_auc_score), folds=3)
        model = res[0]

        model.fit(self.train_dataset)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        print("#############################")
        # evaluate the model
        print('Training Dataset: ')
        train_score = model.evaluate(self.train_dataset, metrics)
        self.assertAlmostEqual(train_score["accuracy_score"], 0.86, delta=0.05)
        print("#############################")
        print('Validation Dataset: ')
        valid_score = model.evaluate(self.valid_dataset, metrics)
        self.assertAlmostEqual(valid_score["accuracy_score"], 0.80, delta=0.1)
        print("#############################")
        print('Test Dataset: ')
        test_score = model.evaluate(self.test_dataset, metrics)
        self.assertAlmostEqual(test_score["accuracy_score"], 0.80, delta=0.1)
        print("#############################")

    def test_rf_3d_descriptors(self):
        rf = RandomForestClassifier()
        model = SklearnModel(model=rf)

        res = model.cross_validate(self.dataset, Metric(roc_auc_score), folds=3)
        model = res[0]

        model.fit(self.train_dataset)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        print("#############################")
        # evaluate the model
        print('Training Dataset: ')
        train_score = model.evaluate(self.train_dataset, metrics)
        self.assertAlmostEqual(train_score["accuracy_score"], 0.99, delta=0.02)
        print("#############################")
        print('Validation Dataset: ')
        valid_score = model.evaluate(self.valid_dataset, metrics)
        self.assertAlmostEqual(valid_score["accuracy_score"], 0.80, delta=0.1)
        print("#############################")
        print('Test Dataset: ')
        test_score = model.evaluate(self.test_dataset, metrics)
        self.assertAlmostEqual(test_score["accuracy_score"], 0.80, delta=0.1)
        print("#############################")

    def test_dnn_3d_descriptors(self):

        input_dim = self.train_dataset.X.shape[1]

        def create_model(optimizer='adam', dropout=0.5, input_dim=input_dim):
            # create model
            model = Sequential()
            model.add(Dense(12, input_dim=input_dim, activation='relu'))
            model.add(Dropout(dropout))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            return model

        model = KerasModel(create_model, epochs=50, verbose=1, optimizer='adam')

        model.fit(self.train_dataset)

        metrics = [Metric(roc_auc_score), Metric(precision_score),
                   Metric(accuracy_score), Metric(confusion_matrix),
                   Metric(classification_report)]

        print("#############################")
        # evaluate the model
        print('Training Dataset: ')
        train_score = model.evaluate(self.train_dataset, metrics)
        self.assertAlmostEqual(train_score["accuracy_score"], 0.87, delta=0.02)
        print("#############################")
        print('Validation Dataset: ')
        valid_score = model.evaluate(self.valid_dataset, metrics)
        self.assertAlmostEqual(valid_score["accuracy_score"], 0.80, delta=0.1)
        print("#############################")
        print('Test Dataset: ')
        test_score = model.evaluate(self.test_dataset, metrics)
        self.assertAlmostEqual(test_score["accuracy_score"], 0.80, delta=0.1)
        print("#############################")

    def test_cnn_3d_descriptors(self):

        input_dim = self.train_dataset.X.shape[1]

        def make_cnn_model(input_dim=input_dim,
                           g_noise=0.05,
                           DENSE=128,
                           DROPOUT=0.5,
                           C1_K=8,
                           C1_S=32,
                           C2_K=16,
                           C2_S=32,
                           activation='relu',
                           loss='binary_crossentropy',
                           optimizer='adadelta',
                           learning_rate=0.01,
                           metrics='accuracy'):

            model = Sequential()
            # Adding a bit of GaussianNoise also works as regularization
            model.add(GaussianNoise(g_noise, input_shape=(input_dim,)))
            # First two is number of filter + kernel size
            model.add(Reshape((input_dim, 1)))
            model.add(Conv1D(C1_K, C1_S, activation=activation, padding="same"))
            model.add(Conv1D(C2_K, C2_S, padding="same", activation=activation))
            model.add(Flatten())
            model.add(Dropout(DROPOUT))
            model.add(Dense(DENSE, activation=activation))
            model.add(Dense(1, activation='sigmoid'))
            if optimizer == 'adadelta':
                opt = Adadelta(lr=learning_rate)
            elif optimizer == 'adam':
                opt = Adam(lr=learning_rate)
            elif optimizer == 'rsmprop':
                opt = RMSprop(lr=learning_rate)
            else:
                opt = optimizer

            model.compile(loss=loss, optimizer=opt, metrics=metrics, )

            return model

        model = KerasModel(make_cnn_model, epochs=10, verbose=1, optimizer="adam")

        model.fit(self.train_dataset)

        metrics = [Metric(roc_auc_score),
                   Metric(precision_score),
                   Metric(accuracy_score),
                   Metric(confusion_matrix),
                   Metric(classification_report)]

        train_score = model.evaluate(self.train_dataset, metrics)
        print('training set score:', train_score)
        self.assertAlmostEqual(train_score["accuracy_score"], 0.81, delta=0.3)

        validation_score = model.evaluate(self.valid_dataset, metrics)
        print('validation set score:', validation_score)
        self.assertAlmostEqual(validation_score["accuracy_score"], 0.81, delta=0.3)

        test_score = model.evaluate(self.test_dataset, metrics)
        print('test set score:', model.evaluate(self.test_dataset, metrics))
        self.assertAlmostEqual(test_score["accuracy_score"], 0.81, delta=0.3)

    def test_multitaskclass_3d_descriptors(self):

        multitask = MultitaskClassifier(n_tasks=1, n_features=self.train_dataset.X.shape[1], layer_sizes=[1000])
        model_multi = DeepChemModel(multitask)
        # Model training
        model_multi.fit(self.train_dataset)
        # Evaluation
        metrics = [Metric(roc_auc_score), Metric(precision_score), Metric(accuracy_score)]
        print('Training Dataset: ')
        train_score = model_multi.evaluate(self.train_dataset, metrics)
        self.assertAlmostEqual(train_score["accuracy_score"], 0.80, delta=0.1)
        print('Valid Dataset: ')
        valid_score = model_multi.evaluate(self.valid_dataset, metrics)
        self.assertAlmostEqual(valid_score["accuracy_score"], 0.80, delta=0.1)
        print('Test Dataset: ')
        test_score = model_multi.evaluate(self.test_dataset, metrics)
        self.assertAlmostEqual(test_score["accuracy_score"], 0.80, delta=0.1)


class Test3DGenerator(TestCase):

    def setUp(self) -> None:
        self.generator = ThreeDimensionalMoleculeGenerator(n_conformations=20)
        self.test_dataset_to_convert = "../data/test_to_convert_to_sdf.csv"
        loader = CSVLoader(self.test_dataset_to_convert,
                           smiles_field='Smiles',
                           labels_fields='Class',
                           id_field='ID')

        self.test_dataset_to_convert_object = loader.create_dataset()

    def test_generate_20_conformers(self):
        mol = MolFromSmiles("CC(CC(C)(O)C=C)=CC=CC")

        self.assertEquals(mol.GetConformers(), ())

        mol = self.generator.generate_conformers(mol, 1)

        self.assertEquals(len(mol.GetConformers()), 20)

    def test_optimize_geometry(self):
        new_generator = ThreeDimensionalMoleculeGenerator(n_conformations=10)

        mol_raw = MolFromSmiles("CC(CC(C)(O)C=C)=CC=C")
        mol_raw2 = MolFromSmiles("CC(CC(C)(O)C=C)=CC=C")

        self.assertEquals(mol_raw.GetConformers(), ())

        new_mol = new_generator.generate_conformers(mol_raw, 1)

        conformer_1_before = new_mol.GetConformer(1)

        self.assertEquals(len(new_mol.GetConformers()), 10)

        new_mol = new_generator.optimize_molecular_geometry(new_mol, "MMFF94")

        conformer_1_after = new_mol.GetConformer(1)

        mol_raw.AddConformer(conformer_1_before)
        mol_raw2.AddConformer(conformer_1_after)

        rmsd = AlignMol(mol_raw, mol_raw2)

        self.assertNotEqual(rmsd, 0)

    def test_export_to_sdf(self):
        generate_conformers_to_sdf_file(self.test_dataset_to_convert_object, "tests/data/test.sdf",
                                        timeout_per_molecule=40)

        loader = SDFLoader("tests/data/test.sdf", "_ID", "_Class")
        dataset = loader.create_dataset()

        All3DDescriptors().featurize(dataset)

        features_number = dataset.len_X()[1]

        self.assertEqual(features_number, 639)

# DeepMol pipelines

In DeepMol we have implemented a pipeline class that allows to build a ML pipeline with just a few lines of code. The pipeline class is a wrapper around the sklearn pipeline class. The pipeline class allows to build a ML pipeline with the whatever steps you want to build. The steps can be any of the following:
- Feature selection
- Feature extraction
- Data Standardization
- Model

# Predict drug activity against DRD2 receptor using DeepMol

We were able to build a ML pipeline to predict drug activity against DRD2 receptor using DeepMol wit just 9 lines of code.

### Let us define a pipeline to predict drug activity against DRD2 receptor

We will use the following steps:
- Basic standardization
- Morgan fingerprints
- Low variance feature selection
- Random forest model


```python
from deepmol.pipeline import Pipeline

from deepmol.metrics import Metric
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from deepmol.models import SklearnModel
from deepmol.splitters import SingletaskStratifiedSplitter
from deepmol.feature_selection import LowVarianceFS
from deepmol.compound_featurization import MorganFingerprint
from deepmol.standardizer import BasicStandardizer
from deepmol.loaders import CSVLoader

steps = [('basic standardizing', BasicStandardizer()),
         ('morgan fingerprints', MorganFingerprint(radius=2, size=1024)),
         ('low variance feature selection', LowVarianceFS(threshold=0.1)),
         ('random forest', SklearnModel(model=RandomForestClassifier(n_jobs=-1, random_state=42)))
         ]
pipeline = Pipeline(steps=steps, path="DRD2")
```

The steps of the pipeline have to be defined as a list of tuples. The first element of the tuple is the name of the step and the second element is the object that implements the step. The pipeline class will respect the order of the steps you defined. The path parameter is the path where the pipeline will be saved.

### Let us load the data


```python
loader = CSVLoader(dataset_path='../data/CHEMBL217_reduced.csv',
                   smiles_field='SMILES',
                   id_field='Original_Entry_ID',
                   labels_fields=['Activity_Flag'])
data = loader.create_dataset(sep=',', header=0)
train, test = SingletaskStratifiedSplitter().train_test_split(data, fra_train=0.8, seed=42)
```

    2023-06-02 17:54:51,582 — ERROR — Molecule with smiles: ClC1=C(N2CCN(O)(CC2)=C/C=C/CNC(=O)C=3C=CC(=CC3)C4=NC=CC=C4)C=CC=C1Cl removed from dataset.
    2023-06-02 17:54:51,583 — INFO — Assuming classification since there are less than 10 unique y values. If otherwise, explicitly set the mode to 'regression'!


    [17:54:51] Explicit valence for atom # 6 N, 5, is greater than permitted


### Let us fit the pipeline


```python
pipeline.fit(train)
```




    <deepmol.pipeline.pipeline.Pipeline at 0x7efd1c22d250>



### Let us evaluate the pipeline


```python
pipeline.evaluate(test, metrics=[Metric(roc_auc_score), Metric(accuracy_score), Metric(precision_score), Metric(recall_score), Metric(f1_score)])
```




    ({'roc_auc_score': 0.9945097741972742,
      'accuracy_score': 0.9693601682186843,
      'precision_score': 0.9626777251184834,
      'recall_score': 0.9765625,
      'f1_score': 0.9695704057279237},
     {})



### Now we can save it and load it again


```python
pipeline.save()
```


```python
pipeline = Pipeline.load(path="DRD2")
```


```python
pipeline.evaluate(test, metrics=[Metric(roc_auc_score), Metric(accuracy_score), Metric(precision_score), Metric(recall_score), Metric(f1_score)])
```




    ({'roc_auc_score': 0.9945097741972742,
      'accuracy_score': 0.9693601682186843,
      'precision_score': 0.9626777251184834,
      'recall_score': 0.9765625,
      'f1_score': 0.9695704057279237},
     {})



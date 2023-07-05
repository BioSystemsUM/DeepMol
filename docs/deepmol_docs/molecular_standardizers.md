# Compound Standardization with DeepMol

Standardization is the process of converting a chemical structure to a standardized format using a set of rules. The standardized format enables the chemical structure to be easily compared with other chemical structures and used in various computational applications.

It is possible to standardize the loaded molecules using three options. Using a basic standardizer one will only do sanitization (Kekulize, check valencies, set aromaticity, conjugation and hybridization). A more complex standardizer can be customized by choosing or not to perform specific tasks such as removing isotope information, neutralizing charges, removing stereochemistry and removing smaller fragments. Another possibility is to use the ChEMBL Standardizer.

Standardizing molecules is important in machine learning pipelines because it helps to **ensure that the data is consistent and comparable across different samples**.


```python
from deepmol.datasets import SmilesDataset

# list of non-standardized smiles
smiles_non_standardized = ['[H]OC([H])([H])[C@]([H])(O[H])[C@]([H])(O[H])[N+]([H])(C([H])([H])[H])C([H])([H])[H]',
 '[H]C1=C([H])[C@]([H])(C([H])([H])C([H])([H])[H])C([H])([H])[C@@]1([H])C(=O)O[C@@]1([H])O[C@]([H])(C([H])([H])OP(=O)([O-])[O-])C([H])([H])[C@]1([H])N([H])[H]',
 '[H]O[C@@]1([H])[C@]([H])(O[H])[C@@]([H])(O[H])[C@]([H])(OC(=O)[C@@]([H])(N([H])C(=O)[C@@]([H])(N(C([H])([H])[H])C([H])([H])[H])C([H])([H])c2c([H])c([H])c([H])c([H])c2[H])C([H])(C([H])([H])[H])C([H])([H])[H])[C@@]1([H])O[H]',
 '[Cl-].[H]OC(=O)c1c([H])c([H])c([H])c([Fe])c1[H]',
 '[H]N(C(=O)C([H])([H])[H])[C@@]1([H])[C@@]2([H])C([H])([H])C([H])([H])[C@@]([H])(OC(=O)C(F)(F)F)[C@]2([H])C([H])([H])N2C(=O)C([H])([H])C([H])([H])[C@@]21[H]',
 '[H]C([H])([H])[N+]([H])([H])[H].[H]O[C@@]1([H])N(C([H])([H])[H])C([H])([H])C(C(=O)N([H])C([H])([H])c2c([H])c([H])c([H])c([H])c2[H])=C([H])[C@]1([H])C(=O)N([H])[C@@]([H])(C(=O)N([H])[H])C([H])(C([H])([H])[H])C([H])([H])[H]']

# Let's create a small dataset with our non-standardized smiles
df = SmilesDataset(smiles=smiles_non_standardized)
```

#### Let's see how our molecules look like using RDKit


```python
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from IPython.display import SVG, display

IPythonConsole.drawOptions.addAtomIndices = True

# non standard molecules
mols_non_standardized = df.mols
# Draw the molecules to a grid image
img = Draw.MolsToGridImage(mols_non_standardized, molsPerRow=3, subImgSize=(400, 400), useSVG=True)
# Get the SVG image from the grid image
svg = img.data
# Display the SVG image using the IPython.display.SVG object
display(SVG(svg))
```


    
![svg](molecular_standardizers_files/molecular_standardizers_4_0.svg)
    


## BasicStandardizer

The BasicStandardizer only does sanitization (Kekulize, check valencies, set aromaticity, conjugation and hybridization).
To perform the standardization we need to call the `standardize` method with the dataset as input.


```python
from deepmol.standardizer import BasicStandardizer
from copy import deepcopy

# Let's create a copy of our dataset
d1 = deepcopy(df)

# Let's standardize our dataset using the BasicStandardizer
basic_standardizer = BasicStandardizer()
basic_standardizer.standardize(d1, inplace=True)
```

### Let's see how our molecules look like after standardization

With this standardizer you can only notice small changes in the molecules as only sanitization is done.
Some visible changes were mainly due to the conversion of all chiral centers to a consistent configuration, removal of explicit hydrogens, and standardization of atom order.



```python
from rdkit.Chem import Draw
from IPython.display import SVG, display

# Standardized molecules
mols_standardized = d1.mols
# Draw the molecules to a grid image
img = Draw.MolsToGridImage(mols_standardized, molsPerRow=3, subImgSize=(400, 400), useSVG=True)
# Get the SVG image from the grid image
svg = img.data
# Display the SVG image using the IPython.display.SVG object
display(SVG(svg))
```


    
![svg](molecular_standardizers_files/molecular_standardizers_8_0.svg)
    


## CustomStandardizer

In the custom standardizer you can choose which tasks to perform. The default tasks are:
- Remove isotope information (default: False)
- Neutralize charges (default: False)
- Remove stereochemistry (default: True)
- Remove smaller fragments (default: False)
- Add explicit hydrogens (default: False)
- Kekulize (default: False)
- Neutralize charges again (default: True)


```python
from deepmol.standardizer import CustomStandardizer
from copy import deepcopy

# Let's create a copy of our dataset
d2 = deepcopy(df)

# Define the standardization steps
standardization_steps = {'REMOVE_ISOTOPE': True,
                        'NEUTRALISE_CHARGE': True,
                        'REMOVE_STEREO': True,
                        'KEEP_BIGGEST': True,
                        'ADD_HYDROGEN': False,
                        'KEKULIZE': True,
                        'NEUTRALISE_CHARGE_LATE': True}

# Let's standardize our dataset using the CustomStandardizer
custom_standardizer = CustomStandardizer(standardization_steps)
custom_standardizer.standardize(d2, inplace=True)
```

### Let's see how our molecules look like after standardization

As we can see the standadized molecules do not contain any isotopic information, the charges are neutralized, do not contain any stereochemistry information (e.g. chirality centers), and smaller fragments are removed.


```python
from rdkit.Chem import Draw
from IPython.display import SVG, display

# Standardized molecules
mols_standardized = d2.mols
# Draw the molecules to a grid image
img = Draw.MolsToGridImage(mols_standardized, molsPerRow=3, subImgSize=(400, 400), useSVG=True)
# Get the SVG image from the grid image
svg = img.data
# Display the SVG image using the IPython.display.SVG object
display(SVG(svg))
```


    
![svg](molecular_standardizers_files/molecular_standardizers_12_0.svg)
    


## ChEMBLStandardizer

[https://github.com/chembl/ChEMBL_Structure_Pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline)

The ChEMBLStandardizer uses ChEMBL protocols used to standardise and salt strip molecules.


```python
from deepmol.standardizer import ChEMBLStandardizer
from copy import deepcopy

# Let's create a copy of our dataset
d3 = deepcopy(df)

# Let's standardize our dataset using the ChEMBLStandardizer
chembl_standardizer = ChEMBLStandardizer()
chembl_standardizer.standardize(d3, inplace=True)
```

### Let's see how our molecules look like after standardization

As we can see these molecules also underwent a more complex standardization process than the BasicStandardizer but in a less extensive way than the CustomStandardizer.


```python
from rdkit.Chem import Draw
from IPython.display import SVG, display

# Standardized molecules
mols_standardized = d3.mols
# Draw the molecules to a grid image
img = Draw.MolsToGridImage(mols_standardized, molsPerRow=3, subImgSize=(400, 400), useSVG=True)
# Get the SVG image from the grid image
svg = img.data
# Display the SVG image using the IPython.display.SVG object
display(SVG(svg))
```


    
![svg](molecular_standardizers_files/molecular_standardizers_16_0.svg)
    


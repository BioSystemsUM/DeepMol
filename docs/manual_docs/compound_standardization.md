# Compound Standardization

It is possible to standardize the loaded molecules using three option. Using
a basic standardizer that only does sanitization (Kekulize, check valencies, 
set aromaticity, conjugation and hybridization). A more complex standardizer can
be customized by choosing or not to perform specific tasks such as sanitization, 
remove isotope information, neutralize charges, remove stereochemistry and remove
smaller fragments. Another possibility is to use the ChEMBL Standardizer.

```python
from deepmol.standardizer import BasicStandardizer, CustomStandardizer, ChEMBLStandardizer

# Option 1: Basic Standardizer
standardizer = BasicStandardizer().standardize(dataset)

# Option 2: Custom Standardizer
heavy_standardisation = {
    'REMOVE_ISOTOPE': True,
    'NEUTRALISE_CHARGE': True,
    'REMOVE_STEREO': True,
    'KEEP_BIGGEST': True,
    'ADD_HYDROGEN': True,
    'KEKULIZE': False,
    'NEUTRALISE_CHARGE_LATE': True}
standardizer2 = CustomStandardizer(heavy_standardisation).standardize(dataset)

# Option 3: ChEMBL Standardizer
standardizer3 = ChEMBLStandardizer().standardize(dataset)
```
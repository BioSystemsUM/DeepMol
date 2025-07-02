# Introduction

![alt text](../imgs/deepmol_logo.png)

DeepMol is a Python-based machine and deep learning framework for drug discovery. 
It offers a variety of functionalities that enable a smoother approach to many 
drug discovery and chemoinformatics problems. It uses Tensorflow, Keras, 
Scikit-learn and DeepChem to build custom ML and DL models or 
make use of pre-built ones. It uses the RDKit framework to perform 
operations on molecular data.

Here is an image with the overall pipeline of DeepMol and the packages it uses:

![alt text](../imgs/deepmol_pipeline.png)

# Google colabs to run AutoML

- [Binary and multiclass classification](https://colab.research.google.com/drive/1wtiwuuhfWKVo40ywgweWUMavKL0zdwJK?usp=sharing)
- [Regression](https://colab.research.google.com/drive/1vE-Q01orImdD4qFTo20MAT4E4kP2hsYF?usp=sharing)
- [Multi-task/multi-label](https://colab.research.google.com/drive/18z2vN6zLNSVJ3qgskKZTYxA_t9UNS1b8?usp=sharing)

# Available models

In our [publication](https://doi.org/10.1186/s13321-024-00937-7), we present several case studies associated to Absorption, Distribution, Metabolism, Excretion, and Toxicity of molecules. We made them available to make predictions on new data in the following repository: https://github.com/BioSystemsUM/deepmol_case_studies. Moreover, other models from other publications are also made available. Check it out the link to know more.

Alternatively, if you want to use the models directly in a Google Colab, you can access it directly [here](https://colab.research.google.com/drive/1_I-f7jQPx2AR76h431x4AdV5Peybs5LO?usp=sharing).

Models available so far: 

| Model Name                                   | How to Call                     | Prediction Type                                                |
|---------------------------------------------|---------------------------------|----------------------------------------------------------------|
| BBB (Blood-Brain Barrier)                   | `BBB`                  | Penetrates BBB (1) or does not penetrate BBB (0)              |
| AMES Mutagenicity                           | `AMES`                         | Mutagenic (1) or not mutagenic (0)                            |
| Human plasma protein binding rate (PPBR)    | `PPBR`                      | Rate of PPBR expressed in percentage                          |
| Volume of Distribution (VD) at steady state | `VDss`                | Volume of Distribution expressed in liters per kilogram (L/kg)|
| Caco-2 (Cell Effective Permeability)        | `Caco2`                   | Cell Effective Permeability (cm/s)                            |
| HIA (Human Intestinal Absorption)           | `HIA`                      | Absorbed (1) or not absorbed (0)                              |
| Bioavailability                             | `Bioavailability`           | Bioavailable (1) or not bioavailable (0)                      |
| Lipophilicity                               | `Lipophilicity`    | Lipophilicity log-ratio                                       |
| Solubility                                  | `Solubility`           | Solubility (log mol/L)                                        |
| CYP P450 2C9 Inhibition                     | `CYP2C9Inhibition`                 | Inhibit (1) or does not inhibit (0)                           |
| CYP P450 3A4 Inhibition                     | `CYP3A4Inhibition`                 | Inhibit (1) or does not inhibit (0)                           |
| CYP2C9 Substrate                            | `CYP2C9Substrate`| Metabolized (1) or does not metabolize (0)                    |
| CYP2D6 Substrate                            | `CYP2D6Substrate`| Metabolized (1) or does not metabolize (0)                    |
| CYP3A4 Substrate                            | `CYP3A4Substrate`| Metabolized (1) or does not metabolize (0)                    |
| Hepatocyte Clearance                        | `HepatocyteClearance`      | Drug hepatocyte clearance (uL.min-1.(10^6 cells)-1)           |
| NPClassifier                        | `NPClassifier`      | Pathway, Superclass, Class           |
| Plants secondary metabolite precursors predictor                        | `PlantsSMPrecursorPredictor`      | Precursor 1; Precursor 2           |
| Microsome Clearance                 | `MicrosomeClearance`       | Drug microsome clearance (mL.min-1.g-1)          |
| LD50                                | `LD50`        | LD50 (log(1/(mol/kg)))                      |
| hERG Blockers                       | `hERGBlockers`           | hERG blocker (1) or not blocker (0)               |




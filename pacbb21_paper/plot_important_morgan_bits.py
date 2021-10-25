import os

import numpy as np
import pandas as pd

from src.featureImportance.shapValues import ShapValues
from src.utils.utils import draw_morgan_bits


def plot_important_morgan_bits(dataset_name, sample_id, n_top_features=10):
	shap_dir = os.path.join('..', 'pacbb21_paper', 'results', 'shap', dataset_name)
	interpreter = ShapValues(dataset=None,
	                         model=None,
	                         mode=None)
	interpreter.load_explanation_object(os.path.join(shap_dir, 'shap_values_explainer.pkl'))
	id_sorted = np.argsort(interpreter.shap_values.values[sample_id])
	important_bits = id_sorted[:-n_top_features-1:-1].tolist() # select n_top_features with positive SHAP values (postive contribution to the output)
	df = pd.read_csv(os.path.join('..', 'pacbb21_paper', 'data', 'split_datasets', dataset_name, 'test_%s.csv' % dataset_name))
	smiles = df.loc[sample_id, 'mols']
	bits_img = draw_morgan_bits(smiles, bits=important_bits, radius=2, nBits=1024)
	with open(os.path.join('..', 'pacbb21_paper', 'results', 'shap', '1-balance', 'sample%s_important_bits.svg' % sample_id), 'w') as f:
		f.write(bits_img)


if __name__ == '__main__':
	plot_important_morgan_bits('1-balance', 537)
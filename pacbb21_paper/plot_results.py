import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

model_type_dict = {'ECFP4': 'Fingerprint', 'ECFP6': 'Fingerprint', 'MACCS': 'Fingerprint', 'RDKitFP': 'Fingerprint',
                   'AtomPair': 'Fingerprint', 'LayeredFP': 'Fingerprint', 'Mol2vec': 'Embedding',
                   'TextCNN': 'End-to-end DL',
                   'GraphConv': 'End-to-end DL', 'GCN': 'End-to-end DL', 'GAT': 'End-to-end DL',
                   'AttentiveFP': 'End-to-end DL'}
model_order = ['ECFP4', 'ECFP6', 'MACCS', 'RDKitFP', 'AtomPair', 'LayeredFP', 'Mol2vec', 'TextCNN', 'GraphConv', 'GCN',
               'GAT', 'AttentiveFP']

# regression tasks
regression_tasks = [('PC-3', 'results/PC-3_evaluation_results.csv'),
                    ('CCRF-CEM', 'results/CCRF-CEM_evaluation_results.csv'),
                    ('A549/ATCC', 'results/nci60_a549atcc_gi50_evaluation_results.csv')]
regression_metrics = {'RMSE': ('test_root_mean_squared_error', (0.4, 1.6)),  # tuple=ylim values
                      'Pearson correlation': ('test_pearson_score', (-0.2, 1)),
                      'R2': ('test_r2_score', (-1.5, 1))}
for metric in regression_metrics:
	fig, ax = plt.subplots(1, 3)
	for i, task in enumerate(regression_tasks):
		df = pd.read_csv(task[1])
		df = df[df['model'] != 'Weave']
		if metric == 'RMSE':
			df['test_root_mean_squared_error'] = df['test_mean_squared_error'].apply(np.sqrt)
		df['model type'] = df['model'].map(model_type_dict)
		sns.barplot(x=df['model'], y=df[regression_metrics[metric][0]], ax=ax[i], order=model_order,
		            hue=df['model type'], dodge=False).set(xlabel=None, ylabel=metric, title=task[0])
	handles, labels = ax[0].get_legend_handles_labels()
	lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.55, -0.07), fancybox=True, shadow=False,
	                 ncol=3)
	for i, a in enumerate(ax):
		a.set(ylim=regression_metrics[metric][1])
		a.axhline(0, color='black', linewidth=0.7)
		plt.setp(a.get_xticklabels(), rotation=90, ha='center')
		a.get_legend().remove()
	plt.tight_layout()
	# plt.show()
	fig.savefig('results/regression_results_%s.pdf' % metric, bbox_extra_artists=(lgd,), bbox_inches='tight')

# classification tasks
classification_tasks = [('NCI 1', 'results/1-balance_evaluation_results.csv'),
                        ('NCI 109', 'results/109-balance_evaluation_results.csv')]
classification_metrics = {'ROC-AUC': ('test_roc_auc_score', (0.5, 0.85)),  # tuple=ylim values
                          'PRC-AUC': ('test_prc_auc_score', (0.5, 0.90))}
for metric in classification_metrics:
	fig, ax = plt.subplots(1, 2)
	for i, task in enumerate(classification_tasks):
		df = pd.read_csv(task[1])
		df = df[df['model'] != 'Weave']
		df['model type'] = df['model'].map(model_type_dict)
		sns.barplot(x=df['model'], y=df[classification_metrics[metric][0]], ax=ax[i], order=model_order,
		            hue=df['model type'], dodge=False).set(xlabel=None, ylabel=metric, title=task[0])
	handles, labels = ax[0].get_legend_handles_labels()
	lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.55, -0.07), fancybox=True, shadow=False,
	                 ncol=3)
	for a in ax:
		a.set(ylim=classification_metrics[metric][1])
		plt.setp(a.get_xticklabels(), rotation=90, ha='center')
		a.get_legend().remove()
	plt.tight_layout()
	# plt.show()
	fig.savefig('results/classification_results_%s.pdf' % metric, bbox_extra_artists=(lgd,), bbox_inches='tight')

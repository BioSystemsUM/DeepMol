import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

model_type_dict = {'ECFP4': 'Fingerprint', 'ECFP6': 'Fingerprint', 'MACCS': 'Fingerprint', 'RDKitFP': 'Fingerprint',
                   'AtomPair': 'Fingerprint', 'LayeredFP': 'Fingerprint', 'Mol2vec': 'Embedding',
                   'TextCNN': 'End-to-end DL',
                   'GraphConv': 'End-to-end DL', 'GCN': 'End-to-end DL', 'GAT': 'End-to-end DL',
                   'AttentiveFP': 'End-to-end DL'}
pc3_df = pd.read_csv('results/PC-3_evaluation_results.csv')
pc3_df = pc3_df[pc3_df['model'] != 'Weave']
pc3_df['test_root_mean_squared_error'] = pc3_df['test_mean_squared_error'].apply(np.sqrt)
pc3_df['model type'] = pc3_df['model'].map(model_type_dict)
ccrfcem_df = pd.read_csv('results/CCRF-CEM_evaluation_results.csv')
ccrfcem_df = ccrfcem_df[ccrfcem_df['model'] != 'Weave']
ccrfcem_df['test_root_mean_squared_error'] = ccrfcem_df['test_mean_squared_error'].apply(np.sqrt)
ccrfcem_df['model type'] = ccrfcem_df['model'].map(model_type_dict)
nci1_df = pd.read_csv('results/1-balance_evaluation_results.csv')
nci1_df = nci1_df[nci1_df['model'] != 'Weave']
nci1_df['model type'] = nci1_df['model'].map(model_type_dict)
nci109_df = pd.read_csv('results/109-balance_evaluation_results.csv')
nci109_df = nci109_df[nci109_df['model'] != 'Weave']
nci109_df['model type'] = nci109_df['model'].map(model_type_dict)
a549_df = pd.read_csv('results/nci60_a549atcc_gi50_evaluation_results.csv')
a549_df = a549_df[a549_df['model'] != 'Weave']
a549_df['test_root_mean_squared_error'] = a549_df['test_mean_squared_error'].apply(np.sqrt)
a549_df['model type'] = a549_df['model'].map(model_type_dict)

model_order = ['ECFP4', 'ECFP6', 'MACCS', 'RDKitFP', 'AtomPair', 'LayeredFP', 'Mol2vec', 'TextCNN', 'GraphConv', 'GCN',
               'GAT', 'AttentiveFP']

# regression tasks
fig, ax = plt.subplots(1, 3)
sns.barplot(x=pc3_df['model'], y=pc3_df['test_root_mean_squared_error'], ax=ax[0],
            order=model_order, hue=pc3_df['model type'], dodge=False).set(xlabel=None,
                                                                          ylabel='RMSE',
                                                                          title='PC-3')
sns.barplot(x=ccrfcem_df['model'], y=ccrfcem_df['test_root_mean_squared_error'], ax=ax[1],
            order=model_order, hue=ccrfcem_df['model type'], dodge=False).set(
    xlabel=None, ylabel='RMSE', title='CCRF-CEM')
sns.barplot(x=a549_df['model'], y=a549_df['test_root_mean_squared_error'], ax=ax[2],
            order=model_order, hue=a549_df['model type'], dodge=False).set(
    xlabel=None, ylabel='RMSE', title='A549/ATCC')
handles, labels = ax[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.55, -0.07),
                 fancybox=True, shadow=False, ncol=3)
for i, a in enumerate(ax):
    a.set(ylim=(0.4, 1.6))
    plt.setp(a.get_xticklabels(), rotation=90, ha='center')
    a.get_legend().remove()
plt.tight_layout()
# plt.show()
fig.savefig('results/regression_results.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

# classification tasks
fig, ax = plt.subplots(1, 2)
sns.barplot(x=nci1_df['model'], y=nci1_df['test_roc_auc_score'], ax=ax[0], order=model_order, hue=nci1_df['model type'],
            dodge=False).set(
    xlabel=None,
    ylabel='ROC-AUC',
    title='NCI 1')
sns.barplot(x=nci109_df['model'], y=nci109_df['test_roc_auc_score'], ax=ax[1],
            order=model_order, hue=nci109_df['model type'], dodge=False).set(xlabel=None,
                                                                             ylabel='ROC-AUC',
                                                                             title='NCI 109')
handles, labels = ax[0].get_legend_handles_labels()
lgd = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.55, -0.07), fancybox=True, shadow=False, ncol=3)
for a in ax:
    a.set(ylim=(0.5, 0.85))
    plt.setp(a.get_xticklabels(), rotation=90, ha='center')
    a.get_legend().remove()
plt.tight_layout()
# plt.show()
fig.savefig('results/classification_results.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')

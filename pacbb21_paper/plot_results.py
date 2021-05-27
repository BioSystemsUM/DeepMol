import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pc3_df = pd.read_csv('results/PC-3_evaluation_results.csv')
pc3_df = pc3_df[pc3_df['model'] != 'Weave']
pc3_df['test_root_mean_squared_error'] = pc3_df['test_mean_squared_error'].apply(np.sqrt)
ccrfcem_df = pd.read_csv('results/CCRF-CEM_evaluation_results.csv')
ccrfcem_df = ccrfcem_df[ccrfcem_df['model'] != 'Weave']
ccrfcem_df['test_root_mean_squared_error'] = ccrfcem_df['test_mean_squared_error'].apply(np.sqrt)
nci1_df = pd.read_csv('results/1-balance_evaluation_results.csv')
nci1_df = nci1_df[nci1_df['model'] != 'Weave']
nci109_df = pd.read_csv('results/109-balance_evaluation_results.csv')
nci109_df = nci109_df[nci109_df['model'] != 'Weave']
a549_df = pd.read_csv('results/nci60_a549atcc_gi50_evaluation_results.csv')
a549_df = a549_df[a549_df['model'] != 'Weave']
a549_df['test_root_mean_squared_error'] = a549_df['test_mean_squared_error'].apply(np.sqrt)

model_order = ['ECFP4', 'ECFP6', 'MACCS', 'RDKitFP', 'AtomPair', 'LayeredFP', 'Mol2vec', 'TextCNN', 'GraphConv', 'GCN',
               'GAT', 'AttentiveFP']

cmap = sns.husl_palette(len(model_order), l=.7, s=.8)
sns.set_palette(cmap)

# regression tasks
fig, ax = plt.subplots(1, 3)
sns.barplot(x=pc3_df['model'], y=pc3_df['test_root_mean_squared_error'], ax=ax[0],
            order=model_order).set(xlabel='Model',
                                   ylabel='RMSE',
                                   title='PC-3')
sns.barplot(x=ccrfcem_df['model'], y=ccrfcem_df['test_root_mean_squared_error'], ax=ax[1],
            order=model_order).set(
    xlabel='Model', ylabel='RMSE', title='CCRF-CEM')
sns.barplot(x=a549_df['model'], y=a549_df['test_root_mean_squared_error'], ax=ax[2],
            order=model_order).set(
    xlabel='Model', ylabel='RMSE', title='A549/ATCC')
for a in ax:
    a.set(ylim=(0.4, 1.6))
    plt.setp(a.get_xticklabels(), rotation=90, ha='center')
plt.tight_layout()
plt.savefig('results/regression_results.pdf')

# classification tasks
#palette = sns.palplot(sns.husl_palette(len(model_order), s=.4, h=.5, l=.6))
fig, ax = plt.subplots(1, 2)
sns.barplot(x=nci1_df['model'], y=nci1_df['test_roc_auc_score'], ax=ax[0], order=model_order).set(
    xlabel='Model',
    ylabel='ROC-AUC',
    title='NCI 1')
sns.barplot(x=nci109_df['model'], y=nci109_df['test_roc_auc_score'], ax=ax[1],
            order=model_order).set(xlabel='Model',
                                   ylabel='ROC-AUC',
                                   title='NCI 109')
for a in ax:
    a.set(ylim=(0.5, 0.85))
    plt.setp(a.get_xticklabels(), rotation=90, ha='center')
plt.tight_layout()
plt.savefig('results/classification_results.pdf')

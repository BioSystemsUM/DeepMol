import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pc3_df = pd.read_csv('results/PC-3_evaluation_results.csv')
ccrfcem_df = pd.read_csv('results/CCRF-CEM_evaluation_results.csv')
nci1_df = pd.read_csv('results/1-balance_evaluation_results.csv')
nci109_df = pd.read_csv('results/109-balance_evaluation_results.csv')
# TODO: remove Weave

# regression tasks
fig, ax = plt.subplots(1, 2)
sns.barplot(x=pc3_df['model'], y=pc3_df['test_mean_squared_error'], ax=ax[0], palette='colorblind').set(xlabel='Model',
                                                                                                        ylabel='MSE',
                                                                                                        title='PC-3')
plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
sns.barplot(x=ccrfcem_df['model'], y=ccrfcem_df['test_mean_squared_error'], ax=ax[1], palette='colorblind').set(
    xlabel='Model', ylabel='MSE', title='CCRF-CEM')
plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig('results/regression_results.pdf')
# plt.show()

# classification tasks
fig, ax = plt.subplots(1, 2)
sns.barplot(x=nci1_df['model'], y=nci1_df['test_roc_auc_score'], ax=ax[0], palette='colorblind').set(xlabel='Model',
                                                                                                     ylabel='ROC-AUC',
                                                                                                     title='NCI 1')
plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

sns.barplot(x=nci109_df['model'], y=nci109_df['test_roc_auc_score'], ax=ax[1], palette='colorblind').set(xlabel='Model',
                                                                                                         ylabel='ROC-AUC',
                                                                                                         title='NCI 109')
plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
plt.savefig('results/classification_results.pdf')
# plt.show()

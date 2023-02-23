"""
Script containing imports of metrics and new metric functions.
"""

from scipy.stats import pearsonr
from scipy.stats import spearmanr

############################################################
# CLASSIFICATION
############################################################

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def pearson_score(y, y_pred):
    return pearsonr(y, y_pred)[0]


def spearman_score(y, y_pred):
    return spearmanr(y, y_pred)[0]


def prc_auc_score(y, y_pred):
    precision, recall, _ = precision_recall_curve(y, y_pred)
    return auc(recall, precision)

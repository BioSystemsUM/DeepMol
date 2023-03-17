"""
Script containing imports of metrics and new metric functions.
"""

from scipy.stats import pearsonr
from scipy.stats import spearmanr

############################################################
# CLASSIFICATION
############################################################

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

############################################################
# REGRESSION
############################################################

from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance


def pearson_score(y, y_pred):
    return pearsonr(y, y_pred)[0]


def spearman_score(y, y_pred):
    return spearmanr(y, y_pred)[0]


def prc_auc_score(y, y_pred):
    precision, recall, _ = precision_recall_curve(y, y_pred)
    return auc(recall, precision)
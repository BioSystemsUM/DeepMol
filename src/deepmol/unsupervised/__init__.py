try:
    from .umap import UMAP
except Exception as e:
    print(e)
from .base_unsupervised import UnsupervisedLearn, PCA, TSNE, KMeans

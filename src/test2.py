

################################################

# OLD FILE -- ONLY FOR REFERENCE

# REMOVE LATER

#################################################

import deepchem as dc
from deepchem.models.graph_models import GraphConvModel

tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv', reload=False)
train_dataset, valid_dataset, test_dataset = tox21_datasets

n_tasks = len(tox21_tasks)
model = GraphConvModel(n_tasks, batch_size=50, mode='classification')

num_epochs = 10
losses = []
for i in range(num_epochs):
    loss = model.fit(train_dataset, nb_epoch=1)
    print("Epoch %d loss: %f" % (i, loss))
    losses.append(loss)
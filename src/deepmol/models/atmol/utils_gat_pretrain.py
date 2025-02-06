import os
from typing import List, Union
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset
from deepmol.datasets.datasets import Dataset
from .creat_data_DC import smile_to_graph
from torch_geometric import data as tg_data

class AtmolTorchDataset(InMemoryDataset):
    def __init__(self, dataset: Dataset = None):
        """
        Initializes the dataset. Expects a dataset object containing SMILES and optional labels (y).
        
        Args:
            dataset: A Dataset object containing SMILES strings and optional labels.
        """
        super(AtmolTorchDataset, self).__init__()
        self.data_list = []
        self.dataset = dataset
        if dataset is not None:
            self.mode = self.dataset.mode
            self.n_tasks = self.dataset.n_tasks

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def featurize(self):

        compound_iso_smiles = list(set(self.dataset.smiles))

        # Process SMILES strings to create graph data.
        for i, smile in tqdm(enumerate(compound_iso_smiles), total=len(compound_iso_smiles)):
            # Convert SMILES to graph
            x_size, features, edge_index, atoms = smile_to_graph(smile)

            try:
                # Create graph data, with optional label `y`
                if self.dataset.y is not None:
                    if len(self.dataset.y.shape) > 1:
                        y_mol = self.dataset.y[i, :]
                    else:
                        y_mol = self.dataset.y[i]

                    graph_data = tg_data.Data(x=torch.Tensor(features),
                                        edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                        y=torch.Tensor(y_mol))
                else:
                    graph_data = tg_data.Data(x=torch.Tensor(features),
                                        edge_index=torch.LongTensor(edge_index).transpose(1, 0))

                graph_data.__setitem__('x_size', torch.LongTensor([x_size]))
                graph_data.__setitem__('edge_size', torch.LongTensor([len(edge_index)]))
                
                self.data_list.append(graph_data)
            except IndexError:
                pass
        
        data, slices = self.collate(self.data_list)

        self._data = data
        self.slices = slices

        return self
    
    def _infer_mode(self) -> Union[str, None, List[str]]:
        """
        Infers the mode of the dataset.

        Returns
        -------
        str
            The inferred mode.
        """
        y = self.data.y
        if y is None:
            self.mode = None
        if len(y.shape) > 1:
            labels_per_task = []
            for label in range(y.shape[1]):
                label_i = y[:, label]
                classes = np.all(np.isclose(label_i, np.round(label_i), equal_nan=True))
                if classes:
                    labels_per_task.append('classification')
                else:
                    labels_per_task.append('regression')

            self.mode = labels_per_task
            self.n_tasks = len(labels_per_task)

        classes = np.all(np.isclose(y, np.round(y), equal_nan=True))
        if not classes:
            self.mode = 'regression'
            self.n_tasks = 1
        else:
            self.mode = 'classification'
            self.n_tasks = 1
        

    def export(self, output_path):
        # save preprocessed data:
        torch.save((self.data, self.slices), output_path)

    @classmethod
    def from_pt(cls, input_path):
        new_dataset = cls()
        new_dataset.data, new_dataset.slices = torch.load(input_path)
        new_dataset._infer_mode()
        return new_dataset


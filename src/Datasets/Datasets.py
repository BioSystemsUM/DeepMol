import numpy as np
import pandas as pd
from typing import Optional, Sequence, Iterable



class Dataset(object):
    """Abstract base class for datasets
    Subclasses need to implement their own methods based on this class.
    """

    def __init__(self) -> None:
        raise NotImplementedError()

    def __len__(self) -> int:
        """Get the number of elements in the dataset."""
        raise NotImplementedError()

    def get_shape(self):
        """Get the shape of all the elements of the dataset.
        mols, X, y, ids.
        """
        raise NotImplementedError()

    def mols(self) -> np.ndarray:
        """Get the molecules (e.g. SMILES format) vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    def X(self) -> np.ndarray:
        """Get the features array for this dataset as a single numpy array."""
        raise NotImplementedError()

    def y(self) -> np.ndarray:
        """Get the y (tasks) vector for this dataset as a single numpy array."""
        raise NotImplementedError()

    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        raise NotImplementedError()


class NumpyDataset(Dataset):
    """A Dataset defined by in-memory numpy arrays.
      This subclass of 'Dataset' stores arrays mols, X, y, ids in memory as
      numpy arrays.
      """

    def __init__(self,
                 mols: np.ndarray,
                 X: Optional[np.ndarray] = None,
                 y: Optional[np.ndarray] = None,
                 ids: Optional[np.ndarray] = None,
                 features2keep: Optional[np.ndarray] = None,
                 n_tasks: int = 1):
        """Initialize a NumpyDataset object.
        Parameters
        ----------
        mols: np.ndarray
          Input features. A numpy array of shape `(n_samples,)`.
        X: np.ndarray, optional (default None)
          Features. A numpy array of arrays of shape (n_samples, features size)
        y: np.ndarray, optional (default None)
          Labels. A numpy array of shape `(n_samples,)`. Note that each label can
          have an arbitrary shape.
        ids: np.ndarray, optional (default None)
          Identifiers. A numpy array of shape (n_samples,)
        features2keep: np.ndarray, optional (deafult None)
          Indexes of the features of X to keep.
        n_tasks: int, default 1
          Number of learning tasks.
        """
        #super().__init__()

        if not isinstance(mols, np.ndarray):
            mols = np.array(mols)
        if not isinstance(X, np.ndarray) and X is not None:
            X = np.ndarray(X)
        if not isinstance(y, np.ndarray) and y is not None:
            y = np.ndarray(y)
        if not isinstance(ids, np.ndarray) and ids is not None:
            ids = np.ndarray(ids)
        if not isinstance(features2keep, np.ndarray) and features2keep is not None:
            features2keep = np.ndarray(features2keep)

        self.mols = mols
        self.X = X
        self.y = y
        self.ids = ids
        self.features2keep = features2keep
        self.n_tasks = n_tasks

        if features2keep is not None:
            print(self.len_X())
            if len(features2keep) != self.len_X()[1]:
                try:
                    self.selectFeatures(features2keep)
                    print('Defined features extracted!')
                except Exception as e:
                    print('Error while removing defined features!')
                    print(e)


    def len_mols(self):
        return len(self.mols)
    
    def len_X(self):
        if self.X is not None:
            return self.X.shape
        else: return 'X not defined!'

    def len_y(self):
        if self.y is not None:
            return self.y.shape
        else : return 'y not defined!'

    def len_ids(self):
        if self.ids is not None:
            return self.ids.shape
        else: return 'ids not defined!'


    def get_shape(self):
        """Get the shape of the dataset.
        Returns four tuples, giving the shape of the mols, X and y arrays.
        """
        print('Mols_shape: ', self.len_mols())
        print('Features_shape: ', self.len_X())
        print('Labels_shape: ', self.len_y())

    def mols(self) -> np.ndarray:
        """Get the features array for this dataset as a single numpy array."""
        return self.mols
    def X(self) -> np.ndarray:
        """Get the X vector for this dataset as a single numpy array."""
        return self.X

    def y(self) -> np.ndarray:
        """Get the y vector for this dataset as a single numpy array."""
        return self.y

    def ids(self) -> np.ndarray:
        """Get the ids vector for this dataset as a single numpy array."""
        return self.ids

    def removeElements(self, indexes):
        """Remove elements with specific indexes from the dataset
            Very useful when doing feature selection or to remove NAs.
        """
        self.mols = np.delete(self.mols, indexes)
        if self.X is not None:
            self.X = np.delete(self.X, indexes, axis=0)
        if self.y is not None:
            self.y = np.delete(self.y, indexes)
        if self.ids is not None:
            self.ids = np.delete(self.ids, indexes)

    #TODO: test this
    def selectFeatures(self, indexes):
        idx = list(range(len(self.X[0])))
        for i in indexes:
            del idx[i]
        self.X = np.delete(self.X, idx, axis=1)

    def removeNAs(self):
        """Remove samples with NAs from the Dataset"""
        j = 0
        indexes = []
        for i in self.X:
            if np.isnan(i[0]):
                indexes.append(j)
            j+=1
        if len(indexes) > 0:
            print('Elements with indexes: ', indexes, ' were removed due to the presence of NAs!')
            print('The elements in question are: ', self.mols[indexes])
            self.removeElements(indexes)


    #TODO: is this the best way of doing it? maybe directly delete instead of creating new NumpyDataset
    def select(self, indexes: Sequence[int]) -> 'NumpyDataset':
        """Creates a new subdataset of self from a selection of indexes.
        Parameters
        ----------
        indices: List[int]
          List of indices to select.
        Returns
        -------
        Dataset
          A NumpyDataset object containing only the selected indexes.
        """

        mols = self.mols[indexes]

        if self.y is not None:
            y = self.y[indexes]
        else: y = self.y

        if self.X is not None:
            X = self.X[indexes]
        else : X = self.X

        if self.ids is not None:
            ids = self.ids[indexes]
        else : ids = self.ids

        return NumpyDataset(mols, X, y, ids, self.features2keep)   

    def merge(self,
              datasets: Iterable[Dataset]) -> 'NumpyDataset':
        """Merges provided datasets with the self dataset.
        Parameters
        ----------
        datasets: Iterable[Dataset]
            List of datasets to merge.
        Returns
        -------
        NumpyDataset
            A merged NumpyDataset.
        """

        datasets = list(datasets)

        X = self.X
        y = self.y
        ids = self.ids
        mols = self.mols

        for ds in datasets:
            mols = np.append(mols, ds.mols, axis=0)
            y = np.append(y, ds.y, axis=0)
            ids = np.append(ids, ds.ids, axis=0)
            if X is not None:
                if len(X[0])==len(ds.X[0]):
                    X = np.append(X, ds.X, axis=0)
                else:
                    flag2 = False
            else: flag2 = False
        if flag2:
            print('Features are not the same length/type... '
                  '\nRecalculate features for all inputs! '
                  '\nAppending empty array in dataset features!')
            return NumpyDataset(mols, np.empty(), y, ids)
        else:
            return NumpyDataset(mols, X, y, ids, self.features2keep)

    def save_to_csv(self, path):
        df = pd.DataFrame()
        df['ids'] = pd.Series(self.ids)
        df['mols'] = pd.Series(self.mols)
        df['X'] = pd.Series(self.X)
        df['y'] = pd.Series(self.y)
        df.to_csv(path)


    # TODO: test load and save
    def load_features(self, path, sep=',', header=0):
        df = pd.read_csv(path, sep=sep, header=header)
        self.dataset.X = df.to_numpy()

    def save_features(self, path='fingerprints.csv'):
        if self.dataset.X is not None:
            columns_names = ['feat_' + str(i + 1) for i in range(self.dataset.X.shape[1])]
            df = pd.DataFrame(self.dataset.X, columns=columns_names)
            df.to_csv(path, index=False)
        else:
            raise ValueError('No fingerprint was already calculated!')



'''
#TODO: implement a Dataset subclass to use/deal with datasets in disk instead of in-memory
class DiskDataset(Dataset):
    """
    ...
    """

    def __init__(self) -> None:
        raise NotImplementedError()

'''
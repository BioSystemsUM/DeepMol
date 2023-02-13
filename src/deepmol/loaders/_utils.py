import numpy as np
import pandas as pd
from rdkit.Chem import SDMolSupplier


def load_csv_file(input_file: str, fields: list, sep: str = ',', header: int = 0, chunk_size: int = None):
    """
    Load data as pandas dataframe from CSV files.

    Parameters
    ----------
    input_file: str
        data path
    fields: list
        fields to keep
    sep: str
        separator
    header: int
        the row where the header is
    chunk_size: int, default None
        The chunk size to yield at one time.
    Returns
    -------
    pd.DataFrame
        Dataframe with chunk size.
    """

    if chunk_size is None:
        return pd.read_csv(input_file, sep=sep, header=header)[fields]
    else:
        df = pd.read_csv(input_file)
        df = df.replace(np.nan, str(""), regex=True)
        return df[fields].sample(chunk_size)


def load_sdf_file(input_file: str, shard_size: int = None):
    """
    Load data as pandas dataframe from SDF files.

    Parameters
    ----------
    input_file: str
        data path
    shard_size: int
        The chunk size to yield at one time.
    """
    supplier = SDMolSupplier(input_file)
    mols, attempts = [], 0

    while not mols and attempts < 10:
        mols = list(supplier)
        attempts += 1
    # sample from list
    if shard_size is None:
        return mols
    return np.random.choice(mols, shard_size)

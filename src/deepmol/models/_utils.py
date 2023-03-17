import joblib
import numpy as np


# TODO: review this function
def save_to_disk(model: 'Model', filename: str, compress: int = 3):
    """
    Save a model to a file.

    Parameters
    ----------
    model: Model
        The model you want to save.
    filename: str
        Path to save data.
    compress: int, default 3
        The compress option when dumping joblib file.
  """
    if filename.endswith('.joblib'):
        joblib.dump(model, filename, compress=compress)
    elif filename.endswith('.npy'):
        np.save(filename, model)
    else:
        raise ValueError("Filename with unsupported extension: %s" % filename)

import numpy as np
from scipy.sparse import issparse
from anndata import AnnData


def _log1p(x):
    if issparse(x):
        np.log1p(x.data, out=x.data)
    else:
        np.log1p(x, out=x)
    return x


def log1p(data, copy=False):
    if copy:
        data = data.copy()
    if isinstance(data, AnnData):
        _log1p(data.X)
    else:
        _log1p(data)
    return data if copy else None

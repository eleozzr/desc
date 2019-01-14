
from anndata import read_mtx
from anndata.utils import make_index_unique


import numpy as np

def _get_max_prob(prob_mat):
    index = np.argmax(prob_mat, axis=1)
    return prob_mat[np.array([range(prob_mat.shape[0])]), index][0].tolist()



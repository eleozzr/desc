import os
import numpy as np


def write_desc_result(data, save_dir='tmp_result', delimiter=","):

    os.makedirs(save_dir, exist_ok=True)
    assert hasattr(
        data.obsm, 'embedded'), 'The desc result not found, run desc first'

    if not os.path.isfile(save_dir + '/prob_matrix.csv'):
        np.savetxt(save_dir + '/prob_matrix.csv',
                   data.obsm['prob'], delimiter=delimiter)
    else:
        print('The ' + save_dir + '/prob_matrix.csv has already exist!')

    if not os.path.isfile(save_dir + '/cluster_ident.cvs'):
        np.savetxt(save_dir + '/cluster_ident.csv',
                   data.obs.iloc[:, [0, 2]], delimiter=delimiter, fmt="%s")
    else:
        print('The ' + save_dir + '/cluster_ident.csv has already exist!')

    if not os.path.isfile(save_dir + '/embedded.csv'):
        np.savetxt(save_dir + '/embedded.cvs',
                   data.obsm['embedded'], delimiter=delimiter)
    else:
        print('The ' + save_dir + '/embedded.csv has already exist!')

    return None

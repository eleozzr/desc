import os
from anndata import read_h5ad


def pbmc():
    adata = read_h5ad(os.path.dirname(__file__) + '/pbmc.h5ad')
    return adata


def pbmc_processed():
    adata = read_h5ad(os.path.dirname(__file__) + '/pbmc_processed.h5ad')
    return adata


def get_pbmc(save_dir='tmp_data'):
    if not os.path.isfile(save_dir + '/pbmc.h5ad'):
        adata = pbmc()
        adata.write(save_dir + '/pbmc.h5ad')
        print('The pbmc.h5ad has been put into ' + save_dir)
    else:
        print('The pbmc data has already exist!')
    return None
    


	
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scanpy.api as sc

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('DESC requires tensorflow  Please follow instructions'
                      ' at https://www.tensorflow.org/install/ to install'
                      ' it.')
def paul_test(n_top_gene=100):
    adata=sc.read("data/paul15/paul15.h5ad")
    sc.pp.filter_cells(adata,min_genes=10)
    sc.pp.normalize_per_cell(adata,counts_per_cell_after=1e4)
    sc.pp.filter_genes(adata,min_cells=20)
    sc.pp.filter_genes_dispersion(adata,n_top_genes=1000)
    sc.pp.log1p(adata)
    sc.pp.scale(adata,zero_center=True,max_value=False)
    return adata


def test_train():
    #simple filter
    adata=paul_test()
    from train import train
    
    adata=train(adata,louvain_resolution=0.4,use_GPU=True)
     
    #adata=desc(x,dims,louvain_resolution=0.4,use_GPU=False)
    
    #adata=desc(x,dims,louvain_resolution='0.4,0.6,0.8',use_GPU=True)

    #adata=desc(x,dims,louvain_resolution=[0.4,0.5,0.8],use_GPU=True)

test_train()

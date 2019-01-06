import os
from  anndata import AnnData
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import scanpy.api as sc
from scipy.sparse import issparse

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('DESC requires tensorflow  Please follow instructions'
                      'i at https://www.tensorflow.org/install/ to install'
                      ' it.')
#try:
#    from .train import train,train_single
#    from .load_mnist import load_mnist
#except:
from train import train, train_single
from load_mnist import load_mnist

def desc(data,
        dims=None,
        alpha=1.0,
        tol=0.005,
        init='glorot_uniform',
        n_clusters=None,
        louvain_resolution=1.0,
        n_neighbors=15,
        pretrain_epochs=300,
        batch_size=256,
        activation='relu',
        actinlayer1='tanh',
        drop_rate_SAE=0.2,
        is_stacked=True,
        use_earlyStop=True,
        save_dir='result_tmp',
        max_iter=1000,
        epochs_fit=4,
        num_Cores=20,
        use_GPU=True,
        random_seed=201809,
        verbose=True,
):
    """ Deep Embeded single cell clustering(DESC) API
    Conduct clustering for single cell data given in the anndata object or np.ndarray,sp.sparmatrix,or pandas.DataFrame
      
    
    Argument:
    ------------------------------------------------------------------
    data: :class:`~anndata.AnnData`, `np.ndarray`, `sp.spmatrix`,`pandas.DataFrame`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    dims: `list`, the number of node in encoder layer, which include input dim, that is
    [1000,64,32] represents for the input dimension is 1000, the first hidden layer have 64 node, and second hidden layer(or bottle neck layers) have 16 nodes. if not specified, it will be decided automatically according to the sample size.
    
    alpha: `float`, optional. Default: `1.0`, the degree of t-distribution.
    tol: `float`, optional. Default: `0.005`, Stop criterion, clustering procedure will be stoped when the difference ratio betwen the current iteration and last iteration larger than tol.
    init: `str`,optional. Default: `glorot_uniform`.
        Initialization method used to initialize weights.

    n_clusters: `int`, optional. Default:`None`, if we specify it , the algorithm will use K-means initialize the cluster center after autoencoder model trained.
    louvain_resolution: `list  or str or float. like, louvain_resolution=1.2 or louvain_resolution=[0.2,0.4,0.8] or louvain_resolution="0.3,0.4,0.8" sep with ","
    n_neightbors, The size of local neighborhood (in terms of number of neighboring data points) used for connectivity matrix. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. In general values should be in the range 2 to 100. Lo 

    pretrain_epochs:'int',optional. Default:`300`,the number of epochs for autoencoder model. 

    batch_size: `int`, optional. Default:`256`, the batch size for autoencoder model and clustering model. 

    activation; `str`, optional. Default,`relu`. the activation function for autoencoder model,which can be 'elu,selu,softplus,tanh,siogmid et al.', for detail please refer to`https://keras.io/activations/`.

    actinlayer1: `str`, optional. Default,'tanh', the activation function for the last layer in encoder and decoder model.

    drop_rate_SAE=: `float`, optional. Default, `0.2`. The drop rate for Stacked autoencoder, which just for  finetuning. 

    is_stacked:`bool`,optional. Default,`True`.The model wiil be pretrained by stacked autoencoder if is_stacked==True.

    use_earlyStop=:`bool`,optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.

    save_dir: 'str',optional. Default,'result_tmp',some result will be saved in this directory.

    max_iter: `int`, optional. Default,`1000`. The maximum iteration for clustering.

    epochs_fit: `int`,optional. Default,`4`, updateing clustering probability for every epochs_fit.

    num_Cores: `int`, optional. Default,`20`. How many cpus use during tranning. if num_Cores > the max cpus in our computer , num_Cores will use  a half of cpus in your computer. 

    use_GPU=True, `bool`, optional. Default, `True`. it will use GPU to train model if GPU is avaliable 

    random_seed, `int`,optional. Default,`201809`. the random seed for random.seed,,,numpy.random.seed,tensorflow.set_random_seed

    verbose,`bool`, optional. Default, `True`. It will ouput the model summary if verbose==True.
    ------------------------------------------------------------------
    """
    if isinstance(data,AnnData):
        adata=data.copy() 
    elif isinstance(data,pd.DataFrame):
        adata=sc.AnnData(data)
    else:
        x=data.toarray() if issparse(data) else data
        adata=sc.AnnData(x)
    if dims is None:
        dims=getdims(adata.shape) 
    if isinstance(louvain_resolution,float) or isinstance(louvain_resolution,int):
        adata=train_single(data=adata,
            dims=dims,
            alpha=alpha,
            tol=tol,
            init=init,
            n_clusters=n_clusters,
            louvain_resolution=float(louvain_resolution),
            n_neighbors=n_neighbors,
            pretrain_epochs=pretrain_epochs,
            batch_size=batch_size,
            activation=activation,
            actinlayer1=actinlayer1,
            drop_rate_SAE=drop_rate_SAE,
            is_stacked=is_stacked,
            use_earlyStop=use_earlyStop,
            save_dir=save_dir,
            max_iter=max_iter,
            epochs_fit=epochs_fit,
            num_Cores=num_Cores,
            use_GPU=use_GPU,
            random_seed=random_seed,
            verbose=verbose)
    else:
        adata=train(data=adata,
            dims=dims,
            alpha=alpha,
            tol=tol,
            init=init,
            n_clusters=n_clusters,
            louvain_resolution=louvain_resolution,
            n_neighbors=n_neighbors,
            pretrain_epochs=pretrain_epochs,
            batch_size=batch_size,
            activation=activation,
            actinlayer1=actinlayer1,
            drop_rate_SAE=drop_rate_SAE,
            is_stacked=is_stacked,
            use_earlyStop=use_earlyStop,
            save_dir=save_dir,
            max_iter=max_iter,
            epochs_fit=epochs_fit,
            num_Cores=num_Cores,
            use_GPU=use_GPU,
            random_seed=random_seed,
            verbose=verbose)
    return adata

#simple test
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='just for simple test api.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_GPU', default=True, type=bool)
    args = parser.parse_args()
    print(args)
    x,_=load_mnist(sample_size=10000) 
    print('MNiST use',x.shape)
    dims=[x.shape[-1],64,32]
    adata1=desc(x,dims,louvain_resolution=0.4,use_GPU=args.use_GPU)
    adata2=desc(x,dims,louvain_resolution='0.4,0.6,0.8',use_GPU=args.use_GPU)
    print(adata1,adata2)

   

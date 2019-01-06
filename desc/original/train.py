# Copyright 2018 Xiangjie Li
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os,math
from time import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from anndata import AnnData
import scanpy.api as sc
from keras import backend as K
from scipy.sparse import issparse
import keras
try:
    from .network import *
except:
    from network import *
    

#check use gpu
#if len(K.tensorflow_backend._get_available_gpus()):
#if tf.test.is_gpu_available():
#config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,inter_op_parallelism_threads=num_cores,
#    allow_soft_placement=True,device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
#session = tf.Session(config=config)
#K.set_session(session)
#os.environ['CUDA_VISIBLE_DEVICES']=K.tensorflow_backend._get_available_gpus()[0][-1]#use first GPUid


#or 
def getdims(x=(10000,200)):
    """
    return the dims for network
    """
    assert len(x)==2
    n_sample=x[0]
    if n_sample>20000:# may be need complex network
        dims=[x[-1],128,32]
    elif n_sample>10000:#10000
        dims=[x[-1],64,32]
    elif n_sample>5000: #5000
        dims=[x[-1],32,16] #16
    elif n_sample>2000:
        dims=[x[-1],128]
    elif n_sample>500:
        dims=[x[-1],64]
    else:
        dims=[x[-1],16]
    #dims=[x[-1],64,32] if n_sample>10000 else [x[-1],32,16]
    return dims



    
def train_single(data,dims=None,
        alpha=1.0,
        tol=0.005,
        init='glorot_uniform',
        n_clusters=None,
        louvain_resolution=1.0,
        n_neighbors=15,
        pretrain_epochs=300,
        batch_size=256,
        activation='relu',
        actincenter='tanh',
        drop_rate_SAE=0.2,
        is_stacked=True,
        use_earlyStop=True,
        use_ae_weights=False,
	save_encoder_weights=False,
        save_dir='result_tmp',
        max_iter=1000,
        epochs_fit=4, 
        num_Cores=20,
        use_GPU=True,
        random_seed=201809,
        verbose=True,
	do_tsne=False,
):
    if isinstance(data,AnnData):
        adata=data
    else:
        adata=sc.AnnData(data)
    #make sure dims 
    if dims is None:
        dims=getdims(adata.shape)
    assert dims[0]==adata.shape[-1],'the number of columns of x doesnot equal to the first element of dims, we must make sure that dims[0]==x.shape[0]'
    #if use_GPU and tf.test.is_gpu_available():
#set seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    import multiprocessing
    total_cpu=multiprocessing.cpu_count()
    num_Cores=int(num_Cores) if total_cpu>int(num_Cores) else int(math.ceil(total_cpu/2)) 
    print('The number of cpu in your computer is',total_cpu)
    if use_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #os.environ['CUDA_VISIBLE_DEVICES']=K.tensorflow_backend._get_available_gpus()[0][-1]#use first GPUid
    else:
        #set only use cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_Cores, inter_op_parallelism_threads=num_Cores)))
    if not use_ae_weights and os.path.isfile(os.path.join(save_dir,"ae_weights.h5")):
        os.remove(os.path.join(save_dir,"ae_weights.h5"))
  
    tic=time()#recored time         
    desc=DESC_MODEL(dims=dims,
              x=adata.X,
              alpha=alpha,
              tol=tol,
              init=init,
              n_clusters=n_clusters,
              louvain_resolution=louvain_resolution,
              n_neighbors=n_neighbors,
              pretrain_epochs=pretrain_epochs,
              batch_size=batch_size,
              random_seed=random_seed,
              activation=activation,
              actincenter=actincenter,
              drop_rate_SAE=drop_rate_SAE,
              is_stacked=is_stacked,
              use_earlyStop=use_earlyStop,
              use_ae_weights=use_ae_weights,
	      save_encoder_weights=save_encoder_weights,
              save_dir=save_dir
    )
    desc.compile(optimizer=SGD(0.01,0.9),loss='kld')
    Embeded_z,q_pred=desc.fit(maxiter=max_iter,epochs_fit=epochs_fit)
    print("The desc has been trained successfully!!!!!!")
    if verbose:
        print("The summary of desc model is:")
        desc.model.summary()
    print("The runtime is ",time()-tic)
    y_pred=pd.Series(np.argmax(q_pred,axis=1),index=adata.obs.index,dtype='category')
    y_pred.cat.categories=list(range(len(y_pred.unique())))
    adata.obs['desc_'+str(louvain_resolution)]=y_pred
    adata.obsm['X_Embeded_z'+str(louvain_resolution)]=Embeded_z
    if do_tsne:
        sc.tl.tsne(adata,use_rep="X_Embeded_z"+str(louvain_resolution),learning_rate=150,n_jobs=num_Cores)
        adata.obsm["X_tsne"+str(louvain_resolution)]=adata.obsm["X_tsne"]
        sc.logging.msg(' tsne finished', t=True, end=' ', v=4)
        sc.logging.msg('and added\n'
                 '    \'X_tsne\''+str(louvain_resolution),'the tsne coordinates (adata.ob)\n', v=4)
    #prob_matrix
    adata.uns['prob_matrix'+str(louvain_resolution)]=q_pred
    return adata


def train(data,dims=None,
        alpha=1.0,
        tol=0.005,
        init='glorot_uniform',
        n_clusters=None,
        louvain_resolution=[0.8,1.0,1.2],
        n_neighbors=15,
        pretrain_epochs=300,
        batch_size=256,
        activation='relu',
        actincenter='tanh',
        drop_rate_SAE=0.2,
        is_stacked=True,
        use_earlyStop=True,
        use_ae_weights=True,
	save_encoder_weights=False,
        save_dir='result_tmp',
        max_iter=1000,
        epochs_fit=4, 
        num_Cores=20,
        use_GPU=True,
        random_seed=201809,
        verbose=True,
	do_tsne=False
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

    actincenter: `str`, optional. Default,'tanh', the activation function for the last layer in encoder and decoder model.

    drop_rate_SAE=: `float`, optional. Default, `0.2`. The drop rate for Stacked autoencoder, which just for  finetuning. 

    is_stacked:`bool`,optional. Default,`True`.The model wiil be pretrained by stacked autoencoder if is_stacked==True.

    use_earlyStop:`bool`,optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.

    use_ae_weights: `bool`, optional. Default, `True`. Whether use ae_weights that has been pretrained(which must saved in `save_dir/ae_weights.h5`)

    save_encoder_weights: `bool`, optional. Default, `False`, it will save inter_ae_weights for every 20 iterations. )

    save_dir: 'str',optional. Default,'result_tmp',some result will be saved in this directory.

    max_iter: `int`, optional. Default,`1000`. The maximum iteration for clustering.

    epochs_fit: `int`,optional. Default,`4`, updateing clustering probability for every epochs_fit.

    num_Cores: `int`, optional. Default,`20`. How many cpus use during tranning. if num_Cores > the max cpus in our computer , num_Cores will use  a half of cpus in your computer. 

    use_GPU=True, `bool`, optional. Default, `True`. it will use GPU to train model if GPU is avaliable 

    random_seed, `int`,optional. Default,`201809`. the random seed for random.seed,,,numpy.random.seed,tensorflow.set_random_seed

    verbose,`bool`, optional. Default, `True`. It will ouput the model summary if verbose==True.
    do_tsne,`bool`,optional. Default, `False`. Whethter do tsne for representation for each embededing
    ------------------------------------------------------------------
    """

    if isinstance(data,AnnData):
        adata=data
    elif isinstance(data,pd.DataFrame):
        adata=sc.AnnData(data,obs=data.index,var=data.columns)
    else:
        x=data.toarray() if issparse(data) else data
        adata=sc.AnnData(x)

    if dims is None:
        dims=getdims(adata.shape)

    if isinstance(louvain_resolution,float) or isinstance(louvain_resolution,int):
        louvain_resolution=[float(louvain_resolution)]
    elif  isinstance(louvain_resolution,str):
        louvain_resolution=list(map(float,louvain_resolution.split(",")))
    else:
        assert isinstance(louvain_resolution,list),'louvain_resolution must be either a string with spearator "," or a list like [1.0,2.0,3.0] '
        louvain_resolution=list(map(float,louvain_resolution))
    #
    time_start=time()
    for ith,resolution in enumerate(louvain_resolution):
        print("Start to process resolution=",str(resolution))
        use_ae_weights=use_ae_weights if ith==0 else True
        res=train_single(data=data,
            dims=dims,
            alpha=alpha,
            tol=tol,
            init=init,
            n_clusters=n_clusters,
            louvain_resolution=resolution,
            n_neighbors=n_neighbors,
            pretrain_epochs=pretrain_epochs,
            batch_size=batch_size,
            activation=activation,
            actincenter=actincenter,
            drop_rate_SAE=drop_rate_SAE,
            is_stacked=is_stacked,
            use_earlyStop=use_earlyStop,
            use_ae_weights=use_ae_weights,
	    save_encoder_weights=save_encoder_weights,
            save_dir=save_dir,
            max_iter=max_iter,
            epochs_fit=epochs_fit, 
            num_Cores=num_Cores,
            use_GPU=use_GPU,
            random_seed=random_seed,
            verbose=verbose,
	    do_tsne=do_tsne)
        #update adata
        data=res
    print("The run time for all resolution is ",time()-time_start)
    return data


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='just for simple test train.py',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_GPU', default=True, type=bool)
    args = parser.parse_args()
    print(args)
    from load_mnist import load_mnist
    x,y=load_mnist(sample_size=10000,seed=0)
    print ('MNIST use ', x.shape)
    dims=[x.shape[-1],64,32]
    adata=train_single(x,dims,louvain_resolution=0.4,use_GPU=True)
    #adata=train(x,dims,louvain_resolution=[0.2,0.4],use_GPU=False)
    #adata2=train(x,dims,louvain_resolution='0.4,0.6,0.8',use_GPU=args.use_GPU,save_dir='mnist')
    #test for paul15()
    adata=sc.read("data/paul15/paul15.h5ad")
    sc.pp.filter_cells(adata,min_genes=10)
    sc.pp.normalize_per_cell(adata,counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.filter_genes(adata,min_cells=20)
    #adata=train(adata,louvain_resolution=0.4,use_GPU=True,use_ae_weights=False)
    #adata=train(adata,louvain_resolution="0.2,0.4,0.5",use_GPU=True,use_ae_weights=True,save_encoder_weights=True)
    #adata=train(adata,louvain_resolution='0.5,0.4,0.7',use_GPU=True,use_ae_weights=False,save_encoder_weights=True)

         
    
 

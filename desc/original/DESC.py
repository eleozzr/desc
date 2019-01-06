"""
Keras implementation for Deep Embedded Clustering (DEC) algorithm for single cell analysis:

        Xiangjie Li et al 

Usage:
    use `python DEC.py -h` for help.

Author:
    Xiangjie Li. 2018.5.8
"""
from __future__ import division
import matplotlib
matplotlib.use('Agg')# in order to save fig to local disk
import networkx as nx
import matplotlib.pyplot as plt
from time import time
import numpy as np
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import scanpy.api as sc
import pandas as pd
from natsort import natsorted #call natsorted
import os
try:
    from .SAE import SAE  # load Stacked autoencoder
except:
    from SAE import SAE  # load Stacked autoencoder


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DESC(object):
    def __init__(self,
                 dims,
                 x, # input matrix, row sample, col predictors 
                 alpha=1.0,
		 tol=0.005,
                 init='glorot_uniform', #initialization method
                 n_clusters=None,     # Number of Clusters, if provided, the clusters center will be initialized by K-means,
                 louvain_resolution=1.0, # resolution for louvain 
                 n_neighbors=15,    # the 
                 pretrain_epochs=300, # epoch for autoencoder
                 batch_size=256, #batch_size for autoencoder
		 activation='relu',
                 actinlayer1="tanh",# activation for the last layer in encoder, and first layer in the decoder 
                 drop_rate_SAE=0.2,
                 is_stacked=True,
                 use_earlyStop=True,
                 save_dir="result_tmp" # save result to save_dir, the default is "result". if recurvie path, there root dir must be exists, or there will be something wrong: for example : "/result_singlecell/dataset1" will return wrong if "result_singlecell" not exist
                 ):

        if not os.path.exists(save_dir):
            print("Create the directory:"+str(save_dir)+" to save result")
            os.mkdir(save_dir)
        self.dims = dims
        self.x=x #feature n*p, n:number of cells, p: number of genes
        self.alpha = alpha
        self.tol=tol
        self.init=init
        self.input_dim = dims[0]  # for clustering layer 
        self.n_stacks = len(self.dims) - 1
        self.is_stacked=is_stacked
        self.resolution=louvain_resolution
        self.n_neighbors=n_neighbors
        self.pretrain_epochs=pretrain_epochs
        self.batch_size=batch_size
        self.activation=activation
        self.actinlayer1=actinlayer1
        self.drop_rate_SAE=drop_rate_SAE
        self.is_stacked=is_stacked
        self.use_earlyStop=use_earlyStop
        self.save_dir=save_dir
        self.pretrain(n_clusters=n_clusters)

    def pretrain(self,n_clusters=None):
        sae=SAE(dims=self.dims,
				act=self.activation,
                drop_rate=self.drop_rate_SAE,
                batch_size=self.batch_size,
                actinlayer1=self.actinlayer1,
                init=self.init,
                use_earlyStop=self.use_earlyStop
           )
        # begin pretraining
        t0 = time()
        print("Checking whether %s  exists in the directory"%str(os.path.join(self.save_dir,'ae_weights,h5')))
        if not os.path.isfile(self.save_dir+"/ae_weights.h5"):
            if self.is_stacked:
                sae.fit(self.x,epochs=self.pretrain_epochs)
            else:
                sae.fit2(self.x,epochs=self.pretrain_epochs)
            self.autoencoder=sae.autoencoders
            self.encoder=sae.encoder
        else:
            sae.autoencoders.load_weights(os.path.join(self.save_dir,"ae_weights.h5"))
            self.autoencoder=sae.autoencoders
            self.encoder=sae.encoder
        print('Pretraining time is', time() - t0)
        #save ae results into disk
        if not os.path.isfile(os.path.join(self.save_dir,"ae_weights.h5")):
            self.autoencoder.save_weights(os.path.join(self.save_dir,'ae_weights.h5'))
            print('Pretrained weights are saved to %s /ae_weights.h5' % self.save_dir)
        #initialize cluster centers using louvain if n_clusters is not exist
        features=self.extract_features(self.x)
        features=np.asarray(features)
        if isinstance(n_clusters,int):
            print("...number of clusters have been specified, Initializing Cluster centroid  using K-Means")
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            Y_pred_init = kmeans.fit_predict(features)
            self.init_pred= np.copy(Y_pred_init)
            self.n_clusters=n_clusters
            cluster_centers=kmeans.cluster_centers_
            self.init_centroid=[cluster_centers]
        else:
            print("...number of clusters is unknown, Initialize cluster centroid using louvain method")
            #can be replaced by other clustering methods
            #using louvain methods in scanpy
            adata=sc.AnnData(features)
            sc.pp.neighbors(adata, n_neighbors=self.n_neighbors)
            sc.tl.louvain(adata,resolution=self.resolution)
            Y_pred_init=adata.obs['louvain']
            self.init_pred=np.asarray(Y_pred_init,dtype=int)
            features=pd.DataFrame(features,index=np.arange(0,features.shape[0]))
            Group=pd.Series(self.init_pred,index=np.arange(0,features.shape[0]),name="Group")
            Mergefeature=pd.concat([features,Group],axis=1)
            cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
            self.n_clusters=cluster_centers.shape[0]
            self.init_centroid=[cluster_centers]
        #create desc clustering layer
        clustering_layer = ClusteringLayer(self.n_clusters,weights=self.init_centroid,name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        

    
    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def fit(self, maxiter=1e3, epochs_fit=5): # unsupervised
        save_dir=self.save_dir
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights(self.init_centroid)
        # Step 2: deep clustering
        y_pred_last = np.copy(self.init_pred)
        for ite in range(int(maxiter)):
            q = self.model.predict(self.x, verbose=0)
            p = self.target_distribution(q)  # update the auxiliary target distribution p
            # evaluate the clustering performance
            y_pred = q.argmax(1)

             # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < self.tol:
                print('delta_label ', delta_label, '< tol ', self.tol)
                print('Reached tolerance threshold. Stop training.')
                break
            print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",self.tol)
            #train on whole dataset on prespecified batch_size
            if self.use_earlyStop:
                callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=5,verbose=1,mode='auto')]
                self.model.fit(x=self.x,y=p,epochs=epochs_fit,batch_size=self.batch_size,callbacks=callbacks,shuffle=True,verbose=True)
            else:
                self.model.fit(x=self.x,y=p,epochs=epochs_fit,batch_size=self.batch_size,shuffle=True,verbose=True)
        y0=pd.Series(y_pred,dtype='category')
        y0.cat.categories=range(0,len(y0.cat.categories))
        print("The final prediction cluster is:")
        x=y0.value_counts()
        print(x.sort_index(ascending=True))
        Embeded_z=self.extract_features(self.x)
        return Embeded_z,q
         

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='DESC-class test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--tol', default=0.005, type=float)
    parser.add_argument('--save_dir', default='results_tmp')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create mnist data to test 
    import numpy as np
    def load_mnist():
        # the data, shuffled and split between train and test sets
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = x.reshape((x.shape[0], -1))
        x = np.divide(x, 50.)  # normalize as it does in DEC paper
        print ('MNIST samples', x.shape)
        return x, y

    x,y=load_mnist()
    sc.settings.figdir="./figures/mnist"
    init = 'glorot_uniform'
    #dims=[x.shape[-1], 500, 300, 100, 30]
    dims=[x.shape[-1],64,32]
    # prepare sample data to  the DESC model
    id=np.random.choice([True,False],size=x.shape[0],p=[0.3,0.7])
    x=x[id]
    y=y[id] 
    desc = DESC(dims=dims,x=x,louvain_resolution=0.3)
    desc.model.summary()
    t0 = time()
    desc.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    Embeded_z,q_pred= desc.fit(maxiter=30,epochs_fit=4)
    y_pred=q_pred.max(axis=1)
    obs_info=pd.DataFrame()
    obs_info["y_true"]=pd.Series(y.astype("U"),dtype="category")
    obs_info["y_pred"]=pd.Series(y_pred.astype("U"),dtype="category")
    adata=sc.AnnData(x,obs=obs_info)
    adata.obsm["X_Embeded_z"]=Embeded_z
    print('clustering time: ', (time() - t0))
    
    

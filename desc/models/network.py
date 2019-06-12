"""
Keras implement Deep learning enables accurate clustering and batch effect removal in single-cell RNA-seq analysis
"""
from __future__ import division
import os
import matplotlib
havedisplay = "DISPLAY" in os.environ
#if we have a display use a plotting backend
if havedisplay:
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
os.environ['PYTHONHASHSEED'] = '0'
import networkx as nx
import matplotlib.pyplot as plt
from time import time as get_time
import numpy as np
import random
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
from keras.layers import Dense, Input
from keras.models import Model,load_model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import scanpy.api as sc
import pandas as pd
from natsort import natsorted #call natsorted
import os
try:
    from .SAE import SAE  # this is for installing package
except:
    from SAE import SAE  #  this is for testing whether DescModel work or not 
random.seed(201809)
np.random.seed(201809)
tf.set_random_seed(201809) if tf.__version__<="2.0" else tf.random.set_seed(201809)
#tf.set_random_seed(201809)


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
            q: student's t-distribution with degree alpha, or soft labels for each sample. shape=(n_samples, n_clusters)
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

class ClusteringLayerGaussian(ClusteringLayer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super().__init__(n_clusters,weights,alpha,**kwargs)
    
    def call(self,inputs,**kwargs):
        sigma=1.0
        q=K.sum(K.exp(-K.square(K.expand_dims(inputs,axis=1)-self.clusters)/(2.0*sigma*sigma)),axis=2)
        q=K.transpose(K.transpose(q)/K.sum(q,axis=1))
        return q


class DescModel(object):
    def __init__(self,
                 dims,
                 x, # input matrix, row sample, col predictors 
                 alpha=1.0,
		 tol=0.005,
                 init='glorot_uniform', #initialization method
                 n_clusters=None,     # Number of Clusters, if provided, the clusters center will be initialized by K-means,
                 louvain_resolution=1.0, # resolution for louvain 
                 n_neighbors=10,    # the 
                 pretrain_epochs=300, # epoch for autoencoder
                 epochs_fit=4, #epochs for each update,int or float 
                 batch_size=256, #batch_size for autoencoder
                 random_seed=201809,
		 activation='relu',
                 actincenter="tanh",# activation for the last layer in encoder, and first layer in the decoder 
                 drop_rate_SAE=0.2,
                 is_stacked=True,
                 use_earlyStop=True,
                 use_ae_weights=False,
		 save_encoder_weights=False,
                 save_encoder_step=5,
                 save_dir="result_tmp",
                 kernel_clustering="t"
                 # save result to save_dir, the default is "result_tmp". if recurvie path, the root dir must be exists, or there will be something wrong: for example : "/result_singlecell/dataset1" will return wrong if "result_singlecell" not exist
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
        self.epochs_fit=epochs_fit
        self.batch_size=batch_size
        self.random_seed=random_seed
        self.activation=activation
        self.actincenter=actincenter
        self.drop_rate_SAE=drop_rate_SAE
        self.is_stacked=is_stacked
        self.use_earlyStop=use_earlyStop
        self.use_ae_weights=use_ae_weights
        self.save_encoder_weights=save_encoder_weights
        self.save_encoder_step=save_encoder_step
        self.save_dir=save_dir
        self.kernel_clustering=kernel_clustering
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
	#pretrain autoencoder
        self.pretrain(n_clusters=n_clusters)
        

    def pretrain(self,n_clusters=None):
        sae=SAE(dims=self.dims,
		act=self.activation,
                drop_rate=self.drop_rate_SAE,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                actincenter=self.actincenter,
                init=self.init,
                use_earlyStop=self.use_earlyStop,
                save_dir=self.save_dir
           )
        # begin pretraining
        t0 = get_time()
        print("Checking whether %s  exists in the directory"%str(os.path.join(self.save_dir,'ae_weights,h5')))
        if self.use_ae_weights: 
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
        else:
            if self.is_stacked:
                sae.fit(self.x,epochs=self.pretrain_epochs)
            else:
                sae.fit2(self.x,epochs=self.pretrain_epochs)
            self.autoencoder=sae.autoencoders
            self.encoder=sae.encoder
        
        print('Pretraining time is', get_time() - t0)
        #save ae results into disk
        if not os.path.isfile(os.path.join(self.save_dir,"ae_weights.h5")):
            self.autoencoder.save_weights(os.path.join(self.save_dir,'ae_weights.h5'))
            self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights.h5'))
            print('Pretrained weights are saved to %s /ae_weights.h5' % self.save_dir)
        #save autoencoder model
        self.autoencoder.save(os.path.join(self.save_dir,"autoencoder_model.h5"))
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
            adata0=sc.AnnData(features)
            if adata0.shape[0]>200000:
                np.random.seed(adata0.shape[0])#set  seed 
                adata0=adata0[np.random.choice(adata0.shape[0],200000,replace=False)] 
            sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors,use_rep="X")
            sc.tl.louvain(adata0,resolution=self.resolution)
            Y_pred_init=adata0.obs['louvain']
            self.init_pred=np.asarray(Y_pred_init,dtype=int)
            if np.unique(self.init_pred).shape[0]<=1:
                #avoid only a cluster
                #print(np.unique(self.init_pred))
                exit("Error: There is only a cluster detected. The resolution:"+str(self.resolution)+"is too small, please choose a larger resolution!!")
            features=pd.DataFrame(adata0.X,index=np.arange(0,adata0.shape[0]))
            Group=pd.Series(self.init_pred,index=np.arange(0,adata0.shape[0]),name="Group")
            Mergefeature=pd.concat([features,Group],axis=1)
            cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
            self.n_clusters=cluster_centers.shape[0]
            self.init_centroid=[cluster_centers]
        #create desc clustering layer
        if self.kernel_clustering=="gaussian":
            clustering_layer = ClusteringLayerGaussian(self.n_clusters,weights=self.init_centroid,name='clustering')(self.encoder.output)
        else:
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

    def fit_on_batch(self,maxiter=1e4,update_interval=200,save_encoder_step=4):
        save_dir=self.save_dir
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights(self.init_centroid)
        # Step 2: deep clustering
        y_pred_last = np.copy(self.init_pred)
        index_array = np.arange(self.x.shape[0])
        index=0
        for ite in range(int(maxiter)):
            if self.save_encoder_weights and ite%(save_encoder_step*update_interval)==0:
                self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights_resolution_'+str(self.resolution)+"_"+str(ite)+'.h5'))
                print('Fine tuning encoder weights are saved to %s/encoder_weights.h5' % self.save_dir) 
            if ite % update_interval ==0:
                q=self.model.predict(self.x,verbose=0)
                p=self.target_distribution(q)
                y_pred=q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",self.tol)
                if ite > 0 and delta_label < self.tol:
                    print('delta_label ', delta_label, '< tol ', self.tol)
                    print('Reached tolerance threshold. Stop training.')
                    break
            idx=index_array[index * self.batch_size: min((index+1) * self.batch_size, self.x.shape[0])]
            loss = self.model.train_on_batch(x=self.x[idx], y=p[idx])
            index = index + 1 if (index + 1) * self.batch_size <= self.x.shape[0] else 0
        #save encoder model
        self.encoder.save(os.path.join(self.save_dir,"encoder_model.h5"))
        #load model
        #encoder=load_model("encoder.h5")
        #
        y0=pd.Series(y_pred,dtype='category')
        y0.cat.categories=range(0,len(y0.cat.categories))
        print("The final prediction cluster is:")
        x=y0.value_counts()
        print(x.sort_index(ascending=True))
        Embedded_z=self.extract_features(self.x)
        q=self.model.predict(self.x,verbose=0)
        return Embedded_z,q

             
    
    def fit_on_all(self, maxiter=1e3, epochs_fit=5,save_encoder_step=5): # unsupervised
        save_dir=self.save_dir
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights(self.init_centroid)
        # Step 2: deep clustering
        y_pred_last = np.copy(self.init_pred)
        for ite in range(int(maxiter)):
            if self.save_encoder_weights and ite%save_encoder_step==0: #save ae_weights for every 5 iterations
                self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights_resolution_'+str(self.resolution)+"_"+str(ite)+'.h5'))
                print('Fine tuning encoder weights are saved to %s/encoder_weights.h5' % self.save_dir)
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
        #save encoder model
        self.encoder.save(os.path.join(self.save_dir,"encoder_model.h5"))
        #load model
        #encoder=load_model("encoder.h5")
        #
        
        y0=pd.Series(y_pred,dtype='category')
        y0.cat.categories=range(0,len(y0.cat.categories))
        print("The final prediction cluster is:")
        x=y0.value_counts()
        print(x.sort_index(ascending=True))
        Embedded_z=self.extract_features(self.x)
        return Embedded_z,q

    def fit(self,maxiter=1e4):
        if isinstance(self.epochs_fit,int):
            embedded_z,q=self.fit_on_all(maxiter=maxiter,epochs_fit=self.epochs_fit,save_encoder_step=self.save_encoder_step)
        else:
            import math
            update_interval=math.ceil(self.epochs_fit*self.x.shape[0]/self.batch_size)
            embedded_z,q=self.fit_on_batch(maxiter=maxiter,save_encoder_step=self.save_encoder_step,update_interval=update_interval)
        return embedded_z,q
         

if __name__ == "__main__":
    # setting the hyper parameters
    import argparse

    parser = argparse.ArgumentParser(description='DescModel class test',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=2e4, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--tol', default=0.005, type=float)
    parser.add_argument('--save_dir', default='result_tmp')
    args = parser.parse_args()
    print(args)
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # create mnist data to test 
    import numpy as np
    def load_mnist(sample_size=10000):
        # the data, shuffled and split between train and test sets
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        x = x.reshape((x.shape[0], -1))
        print ('MNIST samples', x.shape)
        id0=np.random.choice(x.shape[0],sample_size,replace=False)
        return x[id0], y[id0]
    #from load_mnist import load_mnist
    x,y=load_mnist(sample_size=10000)
    init = 'glorot_uniform'
    #dims=[x.shape[-1], 500, 300, 100, 30]
    dims=[x.shape[-1],64,32]
    # prepare sample data to  the DESC model
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    desc = DescModel(dims=dims,x=x,louvain_resolution=0.8,use_ae_weights=True,epochs_fit=0.4)
    desc.model.summary()
    t0 = get_time()
    desc.compile(optimizer=SGD(0.01, 0.9), loss='kld')
    Embedded_z,q_pred= desc.fit(maxiter=30)
    y_pred=q_pred.max(axis=1)
    obs_info=pd.DataFrame()
    obs_info["y_true"]=pd.Series(y.astype("U"),dtype="category")
    obs_info["y_pred"]=pd.Series(y_pred.astype("U"),dtype="category")
    adata=sc.AnnData(x,obs=obs_info)
    adata.obsm["X_Embeded_z"]=Embedded_z
    print('clustering time: ', (get_time() - t0))
    
    

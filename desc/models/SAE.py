import os
os.environ['PYTHONHASHSEED'] = '0'
import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
import  math
import numpy as np
import random
import tensorflow as tf
#random.seed(201809)
#np.random.seed(201809)
#tf.set_random_seed(201809) if tf.__version__<="2.0" else tf.random.set_seed(201809)

class SAE(object):
    """ 
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input->dropout -> encoder_0->act->dropout -> decoder_0;
        stack_1 model: encoder_0->act->dropout -> encoder_1->dropout -> decoder_1->act;
    
    Usage:
        from SAE import SAE
        sae = SAE(dims=[784, 500, 10])  # define a SAE with 5 layers
        sae.fit(x, epochs=100)
        features = sae.extract_feature(x)
        
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
              The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation (default='relu'), not applied to Input, Hidden and Output layers.
        drop_rate: drop ratio of Dropout for constructing denoising autoencoder 'stack_i' during layer-wise pretraining
        batch_size: `int`, optional. Default:`256`, the batch size for autoencoder model and clustering model.
        random_seed, `int`,optional. Default,`201809`. the random seed for random.seed,,,numpy.random.seed,tensorflow.set_random_seed
        actincenter: the activation function in last layer for encoder and last layer for encoder (avoiding the representation values and reconstruct outputs are all non-negative)
        init: `str`,optional. Default: `glorot_uniform`. Initialization method used to initialize weights.
        use_earlyStop: optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.
        save_dir:'str',optional. Default,'result_tmp',some result will be saved in this directory.
    """
    def __init__(self, dims, act='relu', drop_rate=0.2, batch_size=32,random_seed=201809,actincenter="tanh",init="glorot_uniform",use_earlyStop=True,save_dir='result_tmp'): #act relu
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.n_layers = 2*self.n_stacks  # exclude input layer
        self.activation = act
        self.actincenter=actincenter #linear
        self.drop_rate = drop_rate
        self.init=init
        self.batch_size = batch_size
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        #tf.set_random_seed(random_seed)
        tf.set_random_seed(random_seed) if tf.__version__<="2.0" else tf.random.set_seed(random_seed)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        #
        self.random_seed=random_seed
        self.use_earlyStop=use_earlyStop
        self.stacks = [self.make_stack(i,random_seed=self.random_seed+2*i) for i in range(self.n_stacks)]
        self.autoencoders ,self.encoder= self.make_autoencoders()
        plot_model(self.autoencoders, show_shapes=True, to_file=os.path.join(save_dir,'autoencoders.png'))
    def choose_init(self,init="glorot_uniform",seed=1):
        if init not in {'glorot_uniform','glorot_normal','he_normal','lecun_normal','he_uniform','lecun_uniform','RandomNormal','RandomUniform',"TruncatedNormal"}:
            raise ValueError('Invalid `init` argument: '
                             'expected on of {"glorot_uniform", "glorot_normal", "he_normal","he_uniform","lecun_normal","lecun_uniform","RandomNormal","RandomUniform","TruncatedNormal"} '
                             'but got', mode)
        if init=="glorot_uniform":
            res=keras.initializers.glorot_uniform(seed=seed)
        elif init=="glorot_normal":
            res=keras.initializers.glorot_normal(seed=seed)
        elif init=="he_normal":
            res=keras.initializers.he_normal(seed=seed)
        elif init=='he_uniform':
            res=keras.initializers.he_uniform(seed=seed)
        elif init=="lecun_normal":
            res=keras.initializer.lecun_normal(seed=seed)
        elif init=="lecun_uniform":
            res=keras.initializers.lecun_uniform(seed=seed)
        elif init=="RandomNormal":
            res=keras.initializers.RandomNormal(mean=0.0,stddev=0.04,seed=seed)
        elif init=="RandomUniform":
            res=keras.initializers.RandomUniform(minval=-0.05,maxval=0.05,seed=seed)
        else:
            res=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=seed)
        return res
        
        

    def make_autoencoders(self):
        """ Fully connected autoencoders model, symmetric.
        """
        # input
        x = Input(shape=(self.dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(self.n_stacks-1):
            h = Dense(self.dims[i + 1], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+i),activation=self.activation, name='encoder_%d' % i)(h)

        # hidden layer,default activation is linear
        h = Dense(self.dims[-1],kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+self.n_stacks), name='encoder_%d' % (self.n_stacks - 1),activation=self.actincenter)(h)  # features are extracted from here

        y=h
        # internal layers in decoder       
        for i in range(self.n_stacks-1, 0, -1):
            y = Dense(self.dims[i], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+self.n_stacks+i),activation=self.activation, name='decoder_%d' % i)(y)

        # output
        y = Dense(self.dims[0], kernel_initializer=self.choose_init(init=self.init,seed=self.random_seed+2*self.n_stacks),name='decoder_0',activation=self.actincenter)(y)

        return Model(inputs=x, outputs=y,name="AE"),Model(inputs=x,outputs=h,name="encoder")

    def make_stack(self, ith,random_seed=0):
        """ 
        Make the ith denoising autoencoder for layer-wise pretraining. It has single hidden layer. The input data is 
        corrupted by Dropout(drop_rate)
        
        Arguments:
            ith: int, in [0, self.n_stacks)
        """
        in_out_dim = self.dims[ith]
        hidden_dim = self.dims[ith+1]
        output_act = self.activation
        hidden_act = self.activation
        if ith == 0:
            output_act = self.actincenter# tanh, or linear
        if ith == self.n_stacks-1:
            hidden_act = self.actincenter #tanh, or linear
        model = Sequential()
        model.add(Dropout(self.drop_rate, input_shape=(in_out_dim,),seed=random_seed))
        model.add(Dense(units=hidden_dim, activation=hidden_act, kernel_initializer=self.choose_init(init=self.init,seed=random_seed),name='encoder_%d' % ith))
        model.add(Dropout(self.drop_rate,seed=random_seed+1))
        model.add(Dense(units=in_out_dim, activation=output_act,kernel_initializer=self.choose_init(init=self.init,seed=random_seed+1), name='decoder_%d' % ith))
        return model

    def pretrain_stacks(self, x, epochs=200,decaying_step=3):
        """ 
        Layer-wise pretraining. Each stack is trained for 'epochs' epochs using SGD with learning rate decaying 10
        times every 'epochs/3' epochs.
        
        Arguments:
            x: input data, shape=(n_samples, n_dims)
            epochs: epochs for each stack
            decayiing_step: learning rate multiplies 0.1 every 'epochs/decaying_step' epochs 
        """
        features = x
        for i in range(self.n_stacks):
            print( 'Pretraining the %dth layer...' % (i+1))
            for j in range(int(decaying_step)):  # learning rate multiplies 0.1 every 'epochs/decaying_step' epochs
                print ('learning rate =', pow(10, -1-j))
                self.stacks[i].compile(optimizer=SGD(pow(10, -1-j), momentum=0.9), loss='mse')
                if self.use_earlyStop is True:
                    callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
                    self.stacks[i].fit(features,features,callbacks=callbacks,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step))
                else:
                    self.stacks[i].fit(x=features,y=features,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step))
            print ('The %dth layer has been pretrained.' % (i+1))

            # update features to the inputs of the next layer
            feature_model = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder_%d'%i).output)
            features = feature_model.predict(features)

    def pretrain_autoencoders(self, x, epochs=300):
        """
        Fine tune autoendoers end-to-end after layer-wise pretraining using 'pretrain_stacks()'
        Use SGD with learning rate = 0.1, decayed 10 times every 80 epochs
        
        Arguments:
        x: input data, shape=(n_samples, n_dims)
        epochs: training epochs
        """
        print ('Copying layer-wise pretrained weights to deep autoencoders')
        for i in range(self.n_stacks):
            name = 'encoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
            name = 'decoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())

        print ('Fine-tuning autoencoder end-to-end')
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j)
            print ('learning rate =', lr)
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)

    def fit(self, x, epochs=300,decaying_step=3): # use stacked autoencoder pretrain and fine tuning
        self.pretrain_stacks(x, epochs=int(epochs/2),decaying_step=decaying_step)
        self.pretrain_autoencoders(x, epochs=epochs)

    def fit2(self,x,epochs=300): #no stack directly tran 
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j)
            print ('learning rate =', lr)
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            if self.use_earlyStop:
                callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
                self.autoencoders.fit(x=x,y=x,callbacks=callbacks,batch_size=self.batch_size,epochs=epochs)
            else:
                  self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)

    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders(representation).
        
        Arguments:
        x: data
        """
        return self.encoder.predict(x)


if __name__ == "__main__":
    """
    An example for how to use SAE model on MNIST dataset. You can copy this file, and run `python3 SAE.py` in terminal
    """
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

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" # no use GPU
    x,y=load_mnist(10000)
    db = 'mnist'
    n_clusters = 10
    # define and train SAE model
    sae = SAE(dims=[x.shape[-1], 64,32])
    sae.fit(x=x, epochs=400)
    sae.autoencoders.save_weights('weights_%s.h5' % db)

    # extract features
    print ('Finished training, extracting features using the trained SAE model')
    features = sae.extract_feature(x)

    print ('performing k-means clustering on the extracted features')
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters, n_init=20)
    y_pred = km.fit_predict(features)

    from sklearn.metrics import normalized_mutual_info_score as nmi
    print ('K-means clustering result on extracted features: NMI =', nmi(y, y_pred))

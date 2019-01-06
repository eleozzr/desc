from keras.layers import Input, Dense, Dropout,Activation,BatchNormalization
from keras.models import Model, Sequential 
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,History
import math
advanced_activations=('PReLU','LeakyRelU')

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
        batch_size: batch size
    """
    def __init__(self,
                dims,
                batchnorm=True,
                act='relu',
                actinlayercenter='tanh',
                drop_rate=0.2,
                batch_size=32,
                init="glorot_uniform",
                use_earlyStop=False,
                save_dir='SAE_result_tmp'
            ): 
        assert actinlayercenter in ["tanh","linear"]
        self.dims=dims
        self.n_stacks=len(dims)-1
        self.n_layers=2*self.n_stacks
        self.batchnorm=True
        self.act='relu'
        self.actinlayercenter='tanh'
        self.drop_rate=drop_rate
        self.batch_size=batch_size,
        self.init=init
        self.use_earlyStop=use_earlyStop
        self.stacks=[self.make_stack(i) for i in range(self.n_stacks)]
        self.make_autoencoders()
        plot_model(self.autoencoders,show_shapes=True,to_file="autoendoers.png")

    def make_autoencoders(self):
        """ Fully connected autoencoder model
        """
        x=Input(shape=(self.dims[0],),name='input')
        h=x
        #encoder layer
        for i in range(self.n_stacks-1):
            h=Dense(self.dims[i+1],kernel_initializer=self.init,activation=None,name='encoder_%d'%i)(h)
            if self.batchnorm:
                h=BatchNormalization(center=True, scale=False)(h)
            if self.act in advanced_activations:
                h=keras.layers.__dict__[self.act](name='encoder_%d'%i)(h)
            else:
                h=Activation(self.act,name="act_encoder_%d"%i)(h)
        #hidden layer, recommand to use  a linear or tanh activation 
        h=Dense(self.dims[-1],kernel_initializer=self.init,activation=None,name="encoder_%d"%(self.n_stacks-1))(h)
        if self.batchnorm:
            h=BatchNormalization(center=True, scale=False)(h)
        h=Activation(self.actinlayercenter,name="act_center")(h)
        #decoder layer
        for i in range(self.n_stacks-1,0,-1):
            h=Dense(self.dims[i],kernel_initializer=self.init,activation=None,name='decoder_%d'%i)(h)
            if self.batchnorm:
                h=BatchNormalization(center=True, scale=False)(h)
            if self.act in advanced_activations:
                h=keras.layers.__dict__[self.act](name='decoder_%d'%i)(h)
            else:
                h=Activation(self.act,name="act_decoder_%d"%i)(h)
        #output
        h=Dense(self.dims[0],kernel_initializer=self.init,name="decoder_0",activation=self.actinlayercenter)(h)
        #autoencodes
        self.autoencoders=Model(inputs=x,outputs=h,name="AE")
        #get encoder
        self.encoder=self.get_encoder()
         
    def get_encoder(self):
        name="encoder_"+str(self.n_stacks-1)
        ret= Model(inputs=self.autoencoders.input,
                   outputs=self.autoencoders.get_layer(name).output,name="encoder")
        return ret

    def make_stack(self,ith):
        """
            Make the ith autoencoder for layer-wise pretrainning. It has single hidden layer. The input data is corrupted by Dropout(drop_rate) if self.drop_rate>0
            Arugumets:
            ith: int type, in range(0,self.n_stacks)
        """
        in_out_dim=self.dims[ith]
        hidden_dim=self.dims[ith+1]
        in_act=self.actinlayercenter if ith==self.n_stacks-1 else self.act
        x=Input(shape=(in_out_dim,),name="input")
        h=x
        #encoder stack layer 
        if self.drop_rate>0.0:
            h = Dropout(self.drop_rate, name='encoder_dropout')(h)
        #h=Dense(units=hidden_dim,activation=None,kernel_initializer=self.init,kernel_regularizer=l1_l2(0.1,0.2,name='encoder_%d'%ith)
        h=Dense(units=hidden_dim,activation=None,kernel_initializer=self.init,name='encoder_%d'%ith)(h)
        if in_act in advanced_activations:
            h=keras.layers.__dict__[self.act](name='encoder_%d'%ith)(h)
        else:
            h=Activation(self.act,name='act_encoder_%d'%ith)(h)
        h=Dense(units=in_out_dim,activation=None,kernel_initializer=self.init,name='decoder_%d'%ith)(h)
        #decoder stack layer 
        if self.drop_rate>0.0:
            h = Dropout(self.drop_rate, name='decoder_dropout')(h)
        if in_act in advanced_activations:
            h=keras.layers.__dict__[self.act](name='decoder_%d'%ith)(h)
        else:
            h=Activation(self.act,name="act_decoder_%d"%ith)(h)
        model=Model(inputs=x,outputs=h)
        print("Heere",model,type(model))
        return model

    def pretrain_stacks(self,x,epochs=200,decaying_step=4):
        """ 
        Layer-wise pretraining. Each stack is trained for 'epochs' epochs using SGD with learning rate decaying 4
        times every 'epochs/4' epochs.
        
        Arguments:
            x: input data, shape=(n_samples, n_dims)
            decaying_step, default=4, 
            epochs: epochs for each stack
        """
        features=x
        print(type(features),features.shape)
        decaying_step=int(decaying_step)
        for i in range(self.n_stacks):
            #print('Pretraing the %dth layer...'%(i+1))
            print('Pretraing the %dth layer...'%i)
            for j in range(decaying_step):# learning rate multiple 0.1 every 'epochs/3' epochs
                print('learning rate=',pow(10,-1-j))
                self.stacks[i].compile(optimizer=SGD(pow(10,-1-j),momentum=0.9),loss='mse')
                if self.use_earlyStop is True:
                    print("sdfsd")
                    callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
                    self.stacks[i].fit(features,features,callbacks=callbacks,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step)) 
                else:
                    print("sdfsd222")
                    print(self.stacks[i])
                    self.stacks[i].fit(x=features,y=features,batch_size=self.batch_size,epochs=math.ceil(epochs/4))
            print('The %dth has been pretrained successfully!!!!!'%(i+1))
            #update features to the inputs of the next stacked layer
            feature_model=Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder_%d'%i).output)
            features=feature_model.predict(features)
        print("All stacked autoencoder layers trained successfull!!!!")
    
    def pretrain_autoencoders(self,x,epochs=300):
        """
        Fine tune autoendoers end-to-end after layer-wise pretraining using 'pretrain_stacks()'
        Use SGD with learning rate = 0.1, decayed 10 times every 80 epochs
        
        Arguments: 
            :param x: input data, shape=(n_samples, n_dims)
            :param epochs: training epochs
        """
        print("Copy layer-wise pretrained weights to autoencoders")
        for i in range(self.n_stacks):
            name='encoder_%d'%i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
            name="decoder_%d"%i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
        print("Fine tunning autoendoer end to end!!")
        for j in range(math.ceil(epochs/50)):
            lr=pow(10,-j-1)
            print("Fine tuning autoencoders with learning rate="+str(lr))
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            #early stopping
            if self.use_earlyStop is True:
                callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
                self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size,callbacks=callbacks, epochs=50)
            else:
                self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)

    def fit(self,x,epochs=200):
        self.pretrain_stacks(x,epochs=epochs)
        self.pretrain_autoencoders(x,epochs=epochs)
            

if __name__=='__main__':
    """
    An example for how to use SAE model on MNIST dataset. In terminal run
            python SAE.py
    """
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
    print("Test on x shape is ",x.shape)
    sae=SAE(dims=[x.shape[-1],64,32])
    sae.fit(x=x,epochs=150)
    #extract features
    features=sae.encoder.predict(x)
     



        

    

        
   			





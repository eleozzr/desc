import numpy as np
#just for test desc work right
def load_mnist(sample_size=10000,seed=0):
    from keras.datasets import mnist
    np.random.seed(seed)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 50.)  # normalize as it does in DEC paper
    id0=np.random.choice(x.shape[0],size=sample_size,replace=False) 
    return x[id0], y[id0]

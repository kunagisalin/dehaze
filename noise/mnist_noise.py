import keras
from keras.datasets import mnist
import numpy as np

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

x_train = x_train/255
x_test = x_test/255

#add noise
def mk_noise(x,percent=1):
	size = x.shape
	masking = np.random.binomial(n=1,p=percent,size=size)
	return x*masking

x_train_noise = mk_noise(x_train)
x_test_noise = mk_noise(x_test)

#add gauss
def add_gauss(data,scale=0.8):
	gauss_x = data + np.random.normal(loc=0,scale=scale,)
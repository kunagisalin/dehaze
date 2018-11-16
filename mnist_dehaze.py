#coding=utf-8#
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate,Activation, ZeroPadding2D,UpSampling2D
from keras.layers import add, Flatten
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from keras import optimizers
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
#load data
from keras.preprocessing.image import load_img # load an image from file 
from keras.preprocessing.image import img_to_array # convert the image pixels to a numpy array 
from keras.callbacks import TensorBoard
import math
from keras import backend as K



(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

#x_train = x_train/255
#x_test = x_test/255

#add noise
def mk_noise(x,percent=1):
	size = x.shape
	masking = np.random.binomial(n=1,p=percent,size=size)
	return x*masking

x_train_noise = mk_noise(x_train)
x_test_noise = mk_noise(x_test)

#add gauss
def add_gauss(data,scale=0.8):
	gauss_x = data + np.random.normal(loc=0,scale=scale,size=data.shape)
	gauss_x = np.clip(gauss_x,0,1)
	return gauss_x

x_train_gauss = add_gauss(x_train)
x_test_gauss = add_gauss(x_test)

'''#display
display_png(array_to_img(x_train[0]))
display_png(array_to_img(x_train_noise[0]))
display_png(array_to_img(x_train_gauss[0]))

img = array_to_img(x_train_gauss[0])
plt.imshow(img)
plt.show()
'''
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None

	x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
	x = BatchNormalization(axis=3, name=bn_name)(x)
	return x

def identity2_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
	x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	if with_conv_shortcut:#shortcut的含义是：将输入层x与最后的输出层y进行连接，如上图所示
		shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
		x = add([x, shortcut])
		return x
	else:
		x = add([x, inpt])
		return x
def identity3_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
	x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	if with_conv_shortcut:#shortcut的含义是：将输入层x与最后的输出层y进行连接，如上图所示
		shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
		x = add([x, shortcut])
		return x
	else:
		x = add([x, inpt])
		return x
def identity4_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
	x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
	if with_conv_shortcut:#shortcut的含义是：将输入层x与最后的输出层y进行连接，如上图所示
		shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
		x = add([x, shortcut])
		return x
	else:
		x = add([x, inpt])
		return x

def GAWN(width,height,channel):
	x = inpt = Input(shape=(width,height,channel))
#x = ZeroPadding2D((3, 3))(inpt)
#conv1
	x=Conv2d_BN(x,nb_filter=64,kernel_size=(3,3),padding='same')
	x=Conv2d_BN(x,nb_filter=64,kernel_size=(3,3),padding='same')

#conv2_x
	x=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
	#x=Conv2d_BN(x,nb_filter=128,kernel_size=(3,3),padding='same')
	x=MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

#conv3_x
	x = identity2_Block(x, nb_filter=64, kernel_size=(3, 3))
	x = identity2_Block(x, nb_filter=64, kernel_size=(3, 3))
	x = identity3_Block(x, nb_filter=64, kernel_size=(3, 3))
	x = identity4_Block(x, nb_filter=64, kernel_size=(3, 3))

#conv4_x
	x = UpSampling2D(size=(2, 2), data_format=None)(x)
	x = UpSampling2D(size=(2, 2), data_format=None)(x)

#conv5_x
	x = Conv2d_BN(x,nb_filter=1,kernel_size=(3,3),padding='same')
	x = Activation('relu')(x)

	#x = Flatten()(x)
	#x = Dense(1024, activation='relu')(x)
	model = Model(inputs=inpt, outputs=x)
	return model

#from keras.utils import plot_model
model = GAWN(28,28,1)
model.summary()

model.fit(x_train_gauss,x_train,epochs=10,batch_size=20,shuffle=True)

preds = model.predict(x_test_gauss)

plt.imshow(array_to_img(x_test[0]))
plt.show()
plt.imshow(array_to_img(x_test_gauss[0]))
plt.show()
plt.imshow(array_to_img(preds[0]))
plt.show()
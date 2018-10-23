from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate,Activation, ZeroPadding2D,UpSampling2D
from keras.layers import add, Flatten
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os

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
	x = Conv2d_BN(x,nb_filter=3,kernel_size=(3,3),padding='same')

	model = Model(inputs=inpt, outputs=x)
	return model

from keras.utils import plot_model
model = GAWN(224,224,3)
model.summary()
plot_model(model, to_file='model.png')
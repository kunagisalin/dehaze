import keras
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential 
from IPython.display import display_png
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
#coding=utf-8#
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate,Activation, ZeroPadding2D,UpSampling2D
from keras.layers import add, Flatten
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from keras import optimizers
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
#load data
from keras.preprocessing.image import load_img # load an image from file 
from keras.preprocessing.image import img_to_array,array_to_img# convert the image pixels to a numpy array 
from keras.callbacks import TensorBoard
import math
from keras import backend as K

directory = './outdoor/hazy/'
arr = []
for imgname in os.listdir(directory):
	img = Image.open(directory + imgname)
	img = img.resize((300,300))
	arr.append(np.array(img))
x_train = np.array(arr)
print(x_train.shape)
import matplotlib.pyplot as plt
plt.imshow(x_train[1])
plt.show()

directoryy = './outdoor/gt/'
arry = []
for imgname in os.listdir(directoryy):
	imgy = Image.open(directoryy + imgname)
	imgy = imgy.resize((300,300))
	arry.append(np.array(imgy))
y_train = np.array(arry)
print(y_train.shape)

plt.imshow(y_train[1])
plt.show()
x_train = x_train.astype('float32') / 255.
y_train = y_train.astype('float32') / 255.

model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',padding='same',input_shape=(300,300,3)))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(8,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2),padding='same'))

model.add(Conv2D(8,(3,3),activation='relu',padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(UpSampling2D((2,2)))

model.add(Conv2D(3,(3,3),activation='sigmoid',padding='same'))


model.compile(loss='binary_crossentropy',optimizer='adam')

initial_weights = model.get_weights()

model.summary()

#model.fit(x_train,y_train,epochs=10,batch_size=20,shuffle=True)

#model.save('dn.h5')


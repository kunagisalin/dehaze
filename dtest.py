#coding=utf-8#
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, concatenate,Activation, ZeroPadding2D,UpSampling2D
from keras.layers import add, Flatten
from keras.utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
import cv2
from keras import optimizers
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
#load data
from keras.preprocessing.image import load_img # load an image from file 
from keras.preprocessing.image import img_to_array # convert the image pixels to a numpy array 
import matplotlib.pyplot as plt
from IPython.display import display_png
from keras import backend as K

model = load_model('dn.h5')  
#img = Image.open('./0052_0.9_0.16.jpg')
#img = img.resize((224,224))


#plt.imshow(img)
#plt.show()
#pred = my_model.predict(img) 


arr = []
img = Image.open('./0052_0.9_0.16.jpg')
img = img.resize((300,300))

t = np.array(img)

arr.append(np.array(img))
x_train = np.array(arr)
x_train = x_train.astype('float32') / 255.


pred = model.predict(x_train) 
t1 = pred[0]*255


cv2.imwrite('yuan.jpg', t)
cv2.imwrite('colorful.jpg', t1)

y = t-pred[0]*20
cv2.imwrite('y.jpg', y)

img = Image.open('./colorful.jpg')
img = img.resize((550,550))
img.save('./colorful550.jpg')

#narray=narray.reshape(224,224,3)
#imgtext=Image.fromarray(t1)
#imgtext=imgtext.covert('L')
#imgtext.show()

#plt.imshow(imgtext)
#plt.show()
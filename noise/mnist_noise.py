import keras
from keras.datasets import mnist
import numpy as np
from keras.models import Sequential 
from IPython.display import display_png
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

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
model = Sequential()

model.add(Conv2D(16,(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2),padding='same'))
model.add(Conv2D(8,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D((2,2),padding='same'))

model.add(Conv2D(8,(3,3),activation='relu',padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(16,(3,3),activation='relu',padding='same'))
model.add(UpSampling2D((2,2)))

model.add(Conv2D(1,(3,3),activation='sigmoid',padding='same'))


model.compile(loss='binary_crossentropy',optimizer='adam')

initial_weights = model.get_weights()

model.summary()

model.fit(x_train_gauss,x_train,epochs=10,batch_size=20,shuffle=True)

preds = model.predict(x_test_gauss)

plt.imshow(array_to_img(x_test[0]))
plt.show()
plt.imshow(array_to_img(x_test_gauss[0]))
plt.show()
plt.imshow(array_to_img(preds[0]))
plt.show()
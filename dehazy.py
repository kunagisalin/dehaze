import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D
from keras import backend as K
from keras.regularizers import l2
# 引入Tensorboard
from keras.callbacks import TensorBoard
from keras.utils import plot_model
'''
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

'''
input_shape = (224,224,3)#输入图像大小
#resnet blocks definition
def resnet_block(inputs,num_filters=16,
                  kernel_size=3,strides=1,
                  activation='relu'):
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',
           kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    if(activation):
        x = Activation('relu')(x)
    return x

#
model = Sequential()
model.add(Conv2D(filters = 32,kernel_size=(3,3),activation='relu',input_shape = input_shape,name='conv1'))
model.add(Conv2D(64,(3,3), activation='relu',border_mode='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(128,(3,3), activation='relu',border_mode='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

#4 resnet blocks


model.add(UpSampling2D((2, 2)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3,(3,3), activation='relu',border_mode='same'))

plot_model(model,to_file='model.png')
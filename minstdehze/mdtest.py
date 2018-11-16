#coding=utf-8#
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.datasets import mnist
import cv2
import numpy as np

def main():
    model = load_model('md.h5')

    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_test = x_test.reshape(-1,28,28,1)

    def add_gauss(data,scale=0.8):
        gauss_x = data + np.random.normal(loc=0,scale=scale,size=data.shape)
        gauss_x = np.clip(gauss_x,0,1)
        return gauss_x

    x_test_gauss = add_gauss(x_test)
# Turn the image into an array.
# 根据载入的训练好的模型的配置，将图像统一尺寸
#image_arr = cv2.resize(images, (70, 70))

#image_arr = np.expand_dims(image_arr, axis=0)

# 第一个 model.layers[0],不修改,表示输入数据；
# 第二个model.layers[ ],修改为需要输出的层数的编号[]
    layer_1 = K.function([model.layers[0].input], [model.layers[1].output])

# 只修改inpu_image
    f1 = layer_1([x_test_gauss])[0]

    print(f1.shape)

# 第一层卷积后的特征图展示，输出是（1,66,66,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
    '''
    for _ in range(16):
                show_img = f1[:, :, :, _]
                show_img.shape = [66, 66]
                plt.subplot(4, 4, _ + 1)
                # plt.imshow(show_img, cmap='black')
                plt.imshow(show_img, cmap='gray')
                plt.axis('off')
    plt.show()
    '''
if __name__ == '__main__':
    main()
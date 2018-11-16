#coding=utf-8#
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np


directory = './testdata/'
#arr = []
result = os.listdir(directory)
result.sort()
print(result)
#for imgname in os.listdir(directory):
#	img = Image.open(directory + imgname)
#	img = img.resize((300,300))
#	arr.append(np.array(img))
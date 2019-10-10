# coding: utf-8
from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

def array_from_file(filename):
    img = load_img(filename)
    resized = img.resize((224,224))
    return img_to_array(resized)
    
directory = 'img_align_celeba'

labels = np.load('labels.npy')
data = np.empty((labels.shape[0], 224, 224, 3), dtype=np.float32)

for i, file in enumerate(os.listdir(directory)):
    if i >= data.shape[0]:
        break
    img_file = os.path.join(directory, file)
    data[i] = array_from_file(img_file)
    
np.save('data.npy', data)

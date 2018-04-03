# -*- coding: utf-8 -*-
import os
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 

datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
        )

"""
img = load_img('logos/bcbs_cropped.png')

x = img_to_array(img) # numpy array with shape (16, 32, 3)
x = x.reshape((1,) + x.shape) # numpy array with shape (1, 16, 32, 3)

# the .flow generates batches of randomly transformed images and saves the results to the 'preview/' folder

i = 0
for batch in datagen.flow(x, batch_size = 1, save_to_dir = 'dataset/datagen_preview', save_prefix = 'bcbs', save_format = 'jpeg'  ):
    i += 1
    if i > 50:
        break; # otherwise the generator loops indefinitely
        
"""        
prefixes = ('unhc', 'bcbs', 'ccare')
for prefix in prefixes:    
    path = 'dataset/train_set_card/' + prefix        
    for imgname in os.listdir(path):
        img = load_img(path + '/' + imgname)
        x = img_to_array(img) # numpy array with shape (16, 32, 3)
        x = x.reshape((1,) + x.shape) # numpy array with shape (1, 16, 32, 3)
        
        i = 0
        for batch in datagen.flow(x, batch_size = 1, save_to_dir = path, save_prefix = prefix, save_format = 'jpeg'  ):
            i += 1
            if i > 100:
                break;

    
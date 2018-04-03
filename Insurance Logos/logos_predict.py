# -*- coding: utf-8 -*-
from tensorflow.contrib.keras.api.keras.models import load_model
import numpy as np
from tensorflow.contrib.keras.api.keras.preprocessing import image 
import os

model = load_model('memory.h5')

path = 'sample_cards'
for imgname in os.listdir(path):
    test_image = image.load_img(path = path + '/' + imgname, grayscale = False, target_size = (100, 100))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    print(imgname + ' : ' + str(result))

'''    
# ----------------bcbs----------------------    

# ----------------ccare---------------------    

# ----------------unhc----------------------    

'''
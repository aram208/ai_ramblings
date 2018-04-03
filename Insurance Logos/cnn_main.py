# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 18:19:23 2018

@author: Aram
"""

import matplotlib.pyplot as plt

from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras import optimizers
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.contrib.keras import backend
# import os

#script_dir = os.path.dirname(__file__)
#training_set_path = os.path.join(script_dir, '../dataset/training_set')
#test_set_path = os.path.join(script_dir, '.../dataset/test_set')

training_set_path ='dataset/train_set_card'
test_set_path = 'dataset/test_set_card'
input_size = (100, 100)
train_samples = 1100*3
test_samples = 300*3
epochs = 30
batch_size = 128

# Initializing the CNN
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (*input_size, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2))) # 2x2 is optimal

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Flatten())
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))

# total classes - 1
classifier.add(Dense(units=3, activation='sigmoid'))

# Compiling the CNN
#adam = optimizers.Adam(lr = 0.0001)
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images

train_datagen =  ImageDataGenerator(rescale = 1. / 255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size = input_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = train_datagen.flow_from_directory(test_set_path,
                                             target_size = input_size,
                                             batch_size = batch_size,
                                             class_mode = 'categorical')


history = classifier.fit_generator(training_set,
                         steps_per_epoch = train_samples // batch_size,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = test_samples // batch_size,
                         workers = 12,
                         max_queue_size = 10)

classifier.evaluate_generator(test_set, steps = 300)

# Save model
model_backup_path = 'memory.h5'
classifier.save(model_backup_path)
print("Model saved to ", model_backup_path)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

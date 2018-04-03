# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from tensorflow.contrib.keras.api.keras.layers import Dropout
from tensorflow.contrib.keras.api.keras.layers import Conv2D
from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Flatten
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.callbacks import Callback
from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator

def createModel(class_size = 3):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu', input_shape = input_size))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))    

    model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))    
    model.add(Dropout(0.5))
    model.add(Dense(class_size, activation = 'softmax'))
    
    return model


# =============================================================================
input_size = (100, 100) # 100x100 b&w images
training_set_path ='dataset/train_set'
test_set_path = 'dataset/test_set'
batch_size=50
epochs=100

training_set_path ='dataset/train_set'
test_set_path = 'dataset/test_set'

train_datagen =  ImageDataGenerator(rescale = 1. / 255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size)

test_set = train_datagen.flow_from_directory(test_set_path,
                                             target_size=input_size,
                                             batch_size=batch_size)

model = createModel(class_size = 3)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(training_set,
                              steps_per_epoch=2000//batch_size,
                              epochs=epochs,
                              validation_data=test_set,
                              validation_steps=800//batch_size)
model.save_weights("try_1.h5")

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





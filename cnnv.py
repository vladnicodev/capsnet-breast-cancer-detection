# unused for now, to be used for ROC analysis
from sklearn.metrics import roc_curve, auc
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


# the size of the images in the PCAM dataset
IMAGE_SIZE = 96
def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     TRAIN_PATH = os.path.join(base_dir, 'train+val', 'train')
     VALID_PATH = os.path.join(base_dir, 'train+val', 'valid')
     RESCALING_FACTOR = 1./255
     
     # instantiate data generators
     datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

     train_gen = datagen.flow_from_directory(TRAIN_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(VALID_PATH,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             class_mode='binary',
                                             shuffle=False)
     
     return train_gen, val_gen
def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):
     from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
     from keras.optimizers import SGD

     model = Sequential()

     # First Convolutional Block
     model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(2, 2)))

     # Second Convolutional Block
     model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(2, 2)))

     # Third Convolutional Block
     model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(2, 2)))

     # Fourth Convolutional Block
     model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(2, 2)))

     # Flatten and Dense Layers
     model.add(Flatten())
     model.add(Dense(1024, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(512, activation='relu'))
     model.add(Dropout(0.5))
     model.add(Dense(1, activation='sigmoid'))
          
    
     # compile the model
     model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model
# get the model
model = get_model()

# get the data generators

#train_gen, val_gen = get_pcam_generators('/Users/vlad_/Desktop/10perc/')                  #10% of data
#train_gen, val_gen = get_pcam_generators('/Users/vlad_/Desktop/20perc/')                  #20% of data
train_gen, val_gen = get_pcam_generators('/Users/vlad_/Desktop/vladisacuck/MIA/MIA/pcam/') #30% of data
# save the model and weights
model_name = 'my_first_cnn_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.keras'

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json) 


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model
train_steps = train_gen.n//train_gen.batch_size
val_steps = val_gen.n//val_gen.batch_size

history = model.fit(train_gen, steps_per_epoch=train_steps, 
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=10,
                    callbacks=callbacks_list)
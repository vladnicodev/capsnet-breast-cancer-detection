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
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# the size of the images in the PCAM dataset
IMAGE_SIZE = 96

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):
    # dataset parameters
    TRAIN_PATH = os.path.join(base_dir, 'train+val', 'train')
    VALID_PATH = os.path.join(base_dir, 'train+val', 'valid')
    RESCALING_FACTOR = 1./255

    # instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

    train_gen = datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=train_batch_size,
        class_mode='binary'
    )

    val_gen = datagen.flow_from_directory(
        VALID_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=val_batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_gen, val_gen

def get_model(kernel_size=(3,3), pool_size=(4,4), first_filters=32, second_filters=64):


     # build the model
     model = Sequential()

     model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding = 'same', input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Conv2D(second_filters, kernel_size, activation = 'relu', padding = 'same'))
     model.add(MaxPool2D(pool_size = pool_size))

     model.add(Flatten())
     model.add(Dense(64, activation = 'relu'))
     model.add(Dense(1, activation = 'sigmoid'))


     # compile the model
     model.compile(SGD(learning_rate=0.01, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

     return model

# get the model
model = get_model()

# Compute and output the number of trainable parameters.
# This sums up the parameters for each trainable weight tensor.
trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print("Number of trainable parameters: {:,}".format(trainable_params))

# get the data generators
train_gen, val_gen = get_pcam_generators('/pathto/10perc/')  
# train_gen, val_gen = get_pcam_generators('/pathto/20perc/') # Uncomment this line to use 20% of data
# train_gen, val_gen = get_pcam_generators('/pathto/30perc/')

# save the model and weights
model_name = 'my_cnnx%_model'
model_filepath = model_name + '.json'
weights_filepath = model_name + '_weights.keras'

model_json = model.to_json()  # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)

# define the model checkpoint and TensorBoard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join('logs', model_name))
callbacks_list = [checkpoint, tensorboard]

# train the model
train_steps = train_gen.n // train_gen.batch_size
val_steps = val_gen.n // val_gen.batch_size

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=6,
    callbacks=callbacks_list
)
y_true = val_gen.labels
y_pred_prob = model.predict(val_gen, steps=val_steps)
y_pred_prob = y_pred_prob.flatten()

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)
print("model AUC score:", roc_auc)

# Create the ROC plot
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Save the figure to a PNG file instead of calling plt.show()
plt.savefig("roc_x%.png", dpi=300)
plt.close()
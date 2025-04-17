"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this.

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
"""

import glob
import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from matplotlib.pyplot import imread

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings, batch_size):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Three Keras Models: train_model, eval_model, and manipulate_model.
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size)

    # Layer 1: Conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid',
                          activation='relu', name='conv1')(x)

    # Layer 2: Primary capsule layer
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32,
                             kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer with dynamic routing
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16,
                             routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: Compute capsule lengths (used for prediction)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # For training: mask using the true label.
    masked = Mask()(digitcaps)  # For prediction: mask using the capsule with maximal length.

    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4).
    """
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, 1))


def train(model, data, args):
    """
    Training function for the CapsNet.
    """
    (x_train, y_train), (x_test, y_test) = data

    # Callbacks
    log = callbacks.CSVLogger(os.path.join(args.save_dir, 'log.csv'))
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(args.save_dir, 'weights-{epoch:02d}.weights.h5'),
        monitor='val_capsnet_acc', save_best_only=True,
        save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(
        schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # Compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=args.lr), # IMPORTANT: you might need to change this depending on your machine
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})

    model.fit([x_train, y_train], [y_train, x_train],
              batch_size=args.batch_size, epochs=args.epochs,
              validation_data=([x_test, y_test], [y_test, x_test]),
              callbacks=[log, checkpoint, lr_decay])

    # Save model architecture and weights
    model_json = model.to_json()
    json_path = os.path.join(args.save_dir, 'capsnet_model2.json')
    with open(json_path, 'w') as json_file:
        json_file.write(model_json)

    weights_path = os.path.join(args.save_dir, 'capsnet_model_weights2.weights.h5') # IMPORTANT: you might need to change this depending on your machine
    model.save_weights(weights_path)

    print(f'Model architecture saved to {json_path}')
    print(f'Model weights saved to {weights_path}')

    # Optionally: plot training log
    from utils import plot_log
    plot_log(os.path.join(args.save_dir, 'log.csv'), show=True)

    return model


def load_pcam():
    """
    Loads training and validation data.
    """
    x_train, y_train = load_images_from_folder(
        "/yourpathto/10perctrain+val/train") # Change to percentage of choice
    x_test, y_test = load_images_from_folder(
        "/yourpathto/10perc/train+val/valid")

    x_train = x_train.reshape(-1, 96, 96, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 96, 96, 3).astype('float32') / 255.

    assert x_train.shape[0] == y_train.shape[0], "Mismatch in training data size"
    assert x_test.shape[0] == y_test.shape[0], "Mismatch in validation data size"

    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))

    return (x_train, y_train), (x_test, y_test)


def load_images_from_folder(folder, target_size=(96, 96)):
    """
    Loads images and their labels from folder.
    Assumes subdirectories "0" and "1" for each class.
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    images = []
    labels = []
    for label in ["0", "1"]:
        class_path = os.path.join(folder, label)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(int(label))
    return np.array(images), np.array(labels)


if __name__ == "__main__":
    import argparse

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on PCAM.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="Learning rate decay factor")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="Coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of routing iterations (should > 0)")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift in data augmentation")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="Path of saved weights (for testing)")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load training and validation data.
    (x_train, y_train), (x_test, y_test) = load_pcam()

    # Define the model.
    model, eval_model, _ = CapsNet(
        input_shape=x_train.shape[1:],
        n_class=len(np.unique(np.argmax(y_train, 1))),
        routings=args.routings,
        batch_size=args.batch_size)
    model.summary()

    if args.weights is not None:
        model.load_weights(args.weights)

    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)

    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    y_true = np.argmax(y_test, axis=1)
    preds_val, _ = eval_model.predict(x_test, verbose=0)
    y_pred_prob = preds_val[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    print("Validation ROC AUC:", roc_auc)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Validation Data')
    plt.legend(loc='lower right')
    plt.savefig("roc_capsnet.png", dpi=300)
    plt.close()
    print("ROC curve saved as roc_capsnet.png")
    
    # After training
    test_files = glob.glob(os.path.join(args.test_path, '*.tif'))
    total_files = len(test_files)
    print(f"\nFound {total_files} test images. Generating predictions...")

    submission = pd.DataFrame()
    processed = 0

    for idx in range(0, total_files, args.batch_size):
        batch_files = test_files[idx:idx + args.batch_size]
        current_batch_size = len(batch_files)
        
        # Progress tracking
        processed += current_batch_size
        print(f"Processing {processed}/{total_files} images", end='\r')
        
        try:
            images = [imread(f).astype('float32')/255 for f in batch_files]
            K_test = np.stack(images)
            
            # Predict using BEST weights
            preds, _ = eval_model.predict(K_test, verbose=0)
            
            batch_df = pd.DataFrame({
                'id': [os.path.splitext(os.path.basename(f))[0] for f in batch_files],
                'label': preds[:, 1]  # Assuming class 1 is positive
            })
            submission = pd.concat([submission, batch_df])
            
        except Exception as e:
            print(f"\nError in batch {idx}: {str(e)}")
            continue

    print("\nPrediction complete! Saving results...")
    submission.to_csv('capsnet_submission.csv', index=False)
    print(f"Saved {len(submission)} predictions to capsnet_submission2.csv")
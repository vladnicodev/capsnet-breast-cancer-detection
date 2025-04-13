import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras import initializers

def get_pcam_generators(base_dir, train_batch_size=16, val_batch_size=16, image_size=96):
    TRAIN_PATH = os.path.join(base_dir, 'train+val', 'train')
    VALID_PATH = os.path.join(base_dir, 'train+val', 'valid')
    
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = datagen.flow_from_directory(
        TRAIN_PATH, target_size=(image_size, image_size), batch_size=train_batch_size, class_mode='binary')
    val_gen = datagen.flow_from_directory(
        VALID_PATH, target_size=(image_size, image_size), batch_size=val_batch_size, class_mode='binary', shuffle=False)

    return train_gen, val_gen

def wrap_generator(original_gen, num_classes=2):
    """Convert labels to one-hot and provide proper structure"""
    for x, y in original_gen:
        # Convert scalar labels to one-hot
        y_onehot = tf.one_hot(y.astype(int), depth=num_classes)
        yield [x, y_onehot], [y_onehot, x]  # Inputs: [images, onehot_labels], Outputs: [labels, images]

# Load generators
train_gen, val_gen = get_pcam_generators('/Users/vladbarbulescu/Desktop/8p361-project-imaging/PCAM')

# Wrap generators to provide correct input/target structure
train_wrapped = wrap_generator(train_gen)
val_wrapped = wrap_generator(val_gen)


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        # L2 length which is the square root 
        # of the sum of square of the capsule element
        return K.sqrt(K.sum(K.square(inputs), -1))
    
class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """
    def call(self, inputs, **kwargs):
        # use true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of vectors of capsules
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others < 0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

def CapsNet(input_shape, n_class, num_routing):
    """
    :param input_shape: (None, width, height, channels)
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs (image, label) and 
             2 outputs (capsule output and reconstruct image)
    """
    x = layers.Input(shape=input_shape)
    y = layers.Input(shape=(n_class,))  # Now expects one-hot labels

    # ReLU Conv1
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, 
	             padding='valid', activation='relu', name='conv1')(x)

    # PrimaryCapsules: Conv2D layer with `squash` activation, 
    # reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, 
	                    kernel_size=9, strides=2, padding='valid')

    # DigiCap: Capsule layer. Routing algorithm works here.
    digitcaps = DigiCap(num_capsule=n_class, dim_vector=16, 
	        num_routing=num_routing, name='digitcaps')(primarycaps)

    # The length of the capsule's output vector 
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))

    # Adjusted decoder for 64x64x3 reconstruction
    masked = Mask()([digitcaps, y])  
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(96*96*3, activation='sigmoid')(x_recon)  # For 64x64 RGB
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    return models.Model([x, y], [out_caps, x_recon])

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)



class DigiCap(layers.Layer):
    """
    The capsule layer. 
    
    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 b_initializer='zeros',
                 **kwargs):
        super(DigiCap, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.b_initializer = initializers.get(b_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix W: [input_num, num_caps, in_dim, out_dim]
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, 
                   self.input_dim_vector, self.dim_vector],
            initializer=self.kernel_initializer,
            name='W'
        )

        # Coupling coefficients (batch dimension added later)
        self.b = self.add_weight(
            shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
            initializer=self.b_initializer,
            name='b',
            trainable=False  # Not updated by backprop
        )
        self.built = True

    def call(self, inputs, training=None):
        # inputs shape: [batch, input_num_capsule, input_dim_vector]
        batch_size = tf.shape(inputs)[0]
        
        # Compute inputs_hat using einsum: [batch, input_num, num_caps, dim_vector]
        inputs_hat = tf.einsum('bik,ijkm->bijm', inputs, self.W)
        
        # Expand dimensions for broadcasting: [batch, input_num, num_caps, 1, dim_vector]
        inputs_hat = K.expand_dims(inputs_hat, axis=3)
        
        # Tile b to match batch size: [batch, input_num, num_caps, 1, 1]
        b = K.tile(self.b, [batch_size, 1, 1, 1, 1])
        
        # Dynamic routing iterations
        for i in range(self.num_routing):
            # Softmax coupling coefficients along num_caps dimension
            c = tf.nn.softmax(b, axis=2)  # shape: [batch, input_num, num_caps, 1, 1]
            
            # Compute outputs: [batch, 1, num_caps, 1, dim_vector]
            outputs = squash(K.sum(c * inputs_hat, axis=1, keepdims=True))
            
            # Update agreement if not last iteration
            if i < self.num_routing - 1:
                agreement = K.sum(inputs_hat * outputs, axis=-1, keepdims=True)
                b += agreement

        # Final output shape: [batch, num_capsule, dim_vector]
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])


def margin_loss(y_true, y_pred):
    """
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

caps_model = CapsNet(input_shape=[96, 96, 3], n_class=2, num_routing=3)

# Compile with correct loss weights
caps_model.compile(
    optimizer='adam',
    loss=[margin_loss, 'mse'],
    loss_weights=[1.0, 0.0005],  # Reduce reconstruction weight for 64x64 images
    metrics={'out_caps': 'accuracy'}
)

history = caps_model.fit(
    train_wrapped, steps_per_epoch=200, validation_data=val_wrapped, validation_steps=20, epochs=5)
# -*- coding: utf-8 -*-
import keras.backend as K
import tensorflow as tf
from keras import initializers, layers
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.normalization import BatchNormalization

from keras.layers import Input, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from glob import glob

import pandas as pd
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
import matplotlib.pylab as plt
from keras.optimizers import Adam, SGD
import os
from keras import regularizers

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def save_history(history, result_file,epochs):
    loss = history.history['loss']
    recon_loss = history.history['out_recon_loss']
    acc = history.history['out_caps_acc']
    val_loss = history.history['val_loss']
    val_recon_loss = history.history['val_out_recon_loss']
    val_acc = history.history['val_out_caps_acc']
    nb_epoch = len(acc)

    with open(result_file, "a") as fp:
        if epochs==0:
            fp.write("i\tloss\trecon_loss\tacc\tval_loss\tval_recon_loss\tval_acc\n")
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epochs, loss[i],recon_loss[i], acc[i], val_loss[i],val_recon_loss[i], val_acc[i]))
        else:
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\t%f\t%f\n" % (epochs, loss[i],recon_loss[i], acc[i], val_loss[i],val_recon_loss[i], val_acc[i]))

class Length(layers.Layer):

    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

class Mask(layers.Layer):

    def call(self, inputs, **kwargs):
        if type(inputs) is list:  
            inputs, mask = inputs
        else:  
            x = inputs
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1)  

        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  
            return tuple([None, input_shape[0][-1]])
        else:
            return tuple([None, input_shape[-1]])


def squash(vectors, axis=-1):

    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors


class CapsuleLayer(layers.Layer):

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):

        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))

        for i in range(self.num_routing):
            c = tf.nn.softmax(self.bias, dim=2)
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            if i != self.num_routing - 1:
                self.bias += K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_vector])

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):    
    axis_num = -1
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding, kernel_regularizer=regularizers.l1_l2(0.0))(inputs)
    output = BatchNormalization(axis=axis_num)(output)   #add
    #output = Dropout(0.5)(output) #add
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)


def CapsNet(input_shape, n_class, num_routing):
    axis_num = -1
    x = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='same', activation='relu', name='conv1')(x)
    conv1 = BatchNormalization(axis=axis_num)(conv1)   #add
    #conv1 = Dropout(0.5)(conv1) #add
    conv2 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='same', activation='relu', name='conv2')(conv1)
    conv2 = BatchNormalization(axis=axis_num)(conv2)   #add
    #conv2 = Dropout(0.5)(conv2) #add
    conv3 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv3')(conv2)
    conv3 = BatchNormalization(axis=axis_num)(conv3)   #add
    #conv3 = Dropout(0.5)(conv3) #add
    
    primarycaps = PrimaryCap(conv3, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)


    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])
    """
    x_recon = layers.Dense(512, activation='relu')(masked)
    #x_recon = BatchNormalization(axis=axis_num)(x_recon)   #add
    #x_recon = Dropout(0.5)(x_recon) #add
    
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    #x_recon = BatchNormalization(axis=axis_num)(x_recon)   #add
    #x_recon = Dropout(0.5)(x_recon) #add
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    #x_recon = BatchNormalization(axis=axis_num)(x_recon)   #add
    #x_recon = Dropout(0.2)(x_recon) #add
    """
    x_recon = layers.Dense(512, activation='relu')(masked)
    #x_recon = layers.Dense(8 * 8 * 8, activation='relu')(x_recon)
    x_recon = layers.Reshape((8, 8, 8))(x_recon)

    # upsample to (..., 16, 16)
    x_recon = layers.UpSampling2D(size=(2, 2))(x_recon)
    x_recon = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal')(x_recon)

    # upsample to (..., 32, 32)
    x_recon = layers.UpSampling2D(size=(2, 2))(x_recon)
    x_recon = layers.Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_normal')(x_recon)

    # take a channel axis reduction
    x_recon = layers.Conv2D(3, (2, 2), padding='same', activation='tanh', kernel_initializer='glorot_normal',name='out_recon')(x_recon)
    #x_recon = layers.Flatten()(x_recon)
    
    #x_recon = layers.Dense(np.prod(input_shape), activation='tanh')(x_recon)
    #x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    return models.Model([x, y], [out_caps, x_recon])


def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))  #0.5

    return K.mean(K.sum(L, 1))


def train(model, data, epoch_size=32):

    (x_train, y_train), (x_test, y_test) = data
    lr = 0.0003 #0.00001
    opt = Adam(lr, beta_1=0.6, beta_2=0.999, epsilon=1e-08, decay=1e-6) #1e-6
    
    #loss=[l1_loss, 'mae']
    loss=[margin_loss, 'mse']
    #loss='mae''mse'
    #loss='binary_crossentropy'
    model.compile(optimizer=opt,
                  loss=loss,
                  loss_weights=[1., 1.],
                  metrics={'out_caps': 'accuracy'})

    history=model.fit([x_train, y_train],[y_train, x_train], batch_size=32, epochs=epoch_size,
              validation_data=[[x_test, y_test], [y_test, x_test]])


    return model,history


def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def plot_generated_batch(i, model,data1,data2):
    x_test, y_test = data1
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=32)
    X_gen = x_recon
    X_raw = x_test   
    
    Xs1 = to3d(X_raw[:10])
    Xg1 = to3d(X_gen[:10])
    Xs1 = np.concatenate(Xs1, axis=1)
    Xg1 = np.concatenate(Xg1, axis=1)
    Xs2 = to3d(X_raw[10:20])
    Xg2 = to3d(X_gen[10:20])
    Xs2 = np.concatenate(Xs2, axis=1)
    Xg2 = np.concatenate(Xg2, axis=1)
    
    x_train, y_train = data2
    y_pred, x_recon = model.predict([x_train, y_train], batch_size=32)
    X_gen = x_recon
    X_raw = x_train   
    
    Xs3 = to3d(X_raw[:10])
    Xg3 = to3d(X_gen[:10])
    Xs3 = np.concatenate(Xs3, axis=1)
    Xg3 = np.concatenate(Xg3, axis=1)
    Xs4 = to3d(X_raw[10:20])
    Xg4 = to3d(X_gen[10:20])
    Xs4 = np.concatenate(Xs4, axis=1)
    Xg4 = np.concatenate(Xg4, axis=1)

    XX = np.concatenate((Xs1,Xg1,Xs2,Xg2,Xs3,Xg3,Xs4,Xg4), axis=0)
    plt.imshow(XX)
    plt.axis('off')
    plt.savefig("./caps_figures/real_and_recon{0:03d}.png".format(i))
    plt.clf()
    plt.close()

def test(i,model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=32)
    print('-'*50)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    import matplotlib.pyplot as plt
    from PIL import Image

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save("real_and_recon{0:03d}.png".format(i))
    print()
    print('Reconstructed images are saved to ./real_and_recon.png')
    print('-'*50)
    plt.imshow(plt.imread("real_and_recon{0:03d}.png".format(i), ))
    plt.pause(3)
    plt.close()


def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    
    x_train=x_train[:1000]
    x_test=x_test[:1000]
    y_train=y_train[:1000]
    y_test=y_test[:1000]
    
    return (x_train, y_train), (x_test, y_test)

def load_cifar10():
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    
    x_train=x_train[:50000]
    x_test=x_test[:10000]
    y_train=y_train[:50000]
    y_test=y_test[:10000]
    
    return (x_train, y_train), (x_test, y_test)


#(x_train, y_train), (x_test, y_test) = load_mnist()
(x_train, y_train), (x_test, y_test) = load_cifar10()

model = CapsNet(input_shape=[32, 32, 3], n_class=10, num_routing=3)
model.summary()
#model.load_weights('params_capsnet_epoch_009.hdf5')
for i in range(11):
    model,history=train(model=model, data=((x_train, y_train), (x_test, y_test)), epoch_size=1)
    model.save_weights('./caps_figures/params_capsnet_epoch_{0:03d}.hdf5'.format(i), True)
    plot_generated_batch(i,model=model, data1=(x_test[:1000], y_test[:1000]),data2=(x_train[:1000], y_train[:1000]))
    save_history(history, os.path.join("./caps_figures/", 'history.txt'),i)
    #test(i,model=model, data=(x_test, y_test))
save_history(history, os.path.join("./caps_figures/", 'history_last.txt'),20)

"""
____________________________________________________________________________________________________
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 32, 32, 3)     0
____________________________________________________________________________________________________
conv1 (Conv2D)                   (None, 32, 32, 256)   62464       input_1[0][0]
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 32, 32, 256)   1024        conv1[0][0]
____________________________________________________________________________________________________
conv2 (Conv2D)                   (None, 32, 32, 256)   5308672     batch_normalization_1[0][0]
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 32, 32, 256)   1024        conv2[0][0]
____________________________________________________________________________________________________
conv3 (Conv2D)                   (None, 24, 24, 256)   5308672     batch_normalization_2[0][0]
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 24, 24, 256)   1024        conv3[0][0]
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 8, 8, 256)     5308672     batch_normalization_3[0][0]
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 8, 8, 256)     1024        conv2d_1[0][0]
____________________________________________________________________________________________________
reshape_1 (Reshape)              (None, 2048, 8)       0           batch_normalization_4[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 2048, 8)       0           reshape_1[0][0]
____________________________________________________________________________________________________
digitcaps (CapsuleLayer)         (None, 10, 16)        2641920     lambda_1[0][0]
____________________________________________________________________________________________________
input_2 (InputLayer)             (None, 10)            0
____________________________________________________________________________________________________
mask_1 (Mask)                    (None, 16)            0           digitcaps[0][0]
                                                                   input_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           8704        mask_1[0][0]
____________________________________________________________________________________________________
reshape_2 (Reshape)              (None, 8, 8, 8)       0           dense_1[0][0]
____________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)   (None, 16, 16, 8)     0           reshape_2[0][0]
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 16, 16, 64)    4672        up_sampling2d_1[0][0]
____________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)   (None, 32, 32, 64)    0           conv2d_2[0][0]
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 32, 32, 16)    9232        up_sampling2d_2[0][0]
____________________________________________________________________________________________________
out_caps (Length)                (None, 10)            0           digitcaps[0][0]
____________________________________________________________________________________________________
out_recon (Conv2D)               (None, 32, 32, 3)     195         conv2d_3[0][0]
====================================================================================================
Total params: 18,657,299
Trainable params: 18,634,771
Non-trainable params: 22,528
____________________________________________________________________________________________________

caps_figures_BN50000_0-10C10D05L1
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 32, 32, 3)     0
____________________________________________________________________________________________________
conv1 (Conv2D)                   (None, 24, 24, 256)   62464       input_1[0][0]
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 8, 8, 256)     5308672     conv1[0][0]
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 8, 8, 256)     1024        conv2d_1[0][0]
____________________________________________________________________________________________________
reshape_1 (Reshape)              (None, 2048, 8)       0           batch_normalization_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 2048, 8)       0           reshape_1[0][0]
____________________________________________________________________________________________________
digitcaps (CapsuleLayer)         (None, 10, 16)        2641920     lambda_1[0][0]
____________________________________________________________________________________________________
input_2 (InputLayer)             (None, 10)            0
____________________________________________________________________________________________________
mask_1 (Mask)                    (None, 16)            0           digitcaps[0][0]
                                                                   input_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           8704        mask_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 512)           0           dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1024)          525312      dropout_1[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 1024)          0           dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 3072)          3148800     dropout_2[0][0]
____________________________________________________________________________________________________
out_caps (Length)                (None, 10)            0           digitcaps[0][0]
____________________________________________________________________________________________________
out_recon (Reshape)              (None, 32, 32, 3)     0           dense_3[0][0]
====================================================================================================
Total params: 11,696,896
Trainable params: 11,675,904
Non-trainable params: 20,992
____________________________________________________________________________________________________
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 32, 32, 3)     0
____________________________________________________________________________________________________
conv1 (Conv2D)                   (None, 24, 24, 256)   62464       input_1[0][0]
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 24, 24, 256)   1024        conv1[0][0]
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 8, 8, 256)     5308672     batch_normalization_1[0][0]
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 8, 8, 256)     1024        conv2d_1[0][0]
____________________________________________________________________________________________________
reshape_1 (Reshape)              (None, 2048, 8)       0           batch_normalization_2[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 2048, 8)       0           reshape_1[0][0]
____________________________________________________________________________________________________
digitcaps (CapsuleLayer)         (None, 10, 16)        2641920     lambda_1[0][0]
____________________________________________________________________________________________________
input_2 (InputLayer)             (None, 10)            0
____________________________________________________________________________________________________
mask_1 (Mask)                    (None, 16)            0           digitcaps[0][0]
                                                                   input_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           8704        mask_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1024)          525312      dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 3072)          3148800     dense_2[0][0]
____________________________________________________________________________________________________
out_caps (Length)                (None, 10)            0           digitcaps[0][0]
____________________________________________________________________________________________________
out_recon (Reshape)              (None, 32, 32, 3)     0           dense_3[0][0]
====================================================================================================
Total params: 11,697,920
Trainable params: 11,676,416
Non-trainable params: 21,504
____________________________________________________________________________________________________
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 32, 32, 3)     0
____________________________________________________________________________________________________
conv1 (Conv2D)                   (None, 24, 24, 256)   62464       input_1[0][0]
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 24, 24, 256)   1024        conv1[0][0]
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 8, 8, 256)     5308672     batch_normalization_1[0][0]
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 8, 8, 256)     1024        conv2d_1[0][0]
____________________________________________________________________________________________________
reshape_1 (Reshape)              (None, 2048, 8)       0           batch_normalization_2[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 2048, 8)       0           reshape_1[0][0]
____________________________________________________________________________________________________
digitcaps (CapsuleLayer)         (None, 10, 16)        2641920     lambda_1[0][0]
____________________________________________________________________________________________________
input_2 (InputLayer)             (None, 10)            0
____________________________________________________________________________________________________
mask_1 (Mask)                    (None, 16)            0           digitcaps[0][0]
                                                                   input_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           8704        mask_1[0][0]
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 512)           2048        dense_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1024)          525312      batch_normalization_3[0][0]
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 1024)          4096        dense_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 3072)          3148800     batch_normalization_4[0][0]
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 3072)          12288       dense_3[0][0]
____________________________________________________________________________________________________
out_caps (Length)                (None, 10)            0           digitcaps[0][0]
____________________________________________________________________________________________________
out_recon (Reshape)              (None, 32, 32, 3)     0           batch_normalization_5[0][0]
====================================================================================================
Total params: 11,716,352
Trainable params: 11,685,632
Non-trainable params: 30,720
____________________________________________________________________________________________________
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0
__________________________________________________________________________________________________
conv1 (Conv2D)                  (None, 24, 24, 256)  62464       input_1[0][0]
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 8, 8, 256)    5308672     conv1[0][0]
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 2048, 8)      0           conv2d_1[0][0]
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 2048, 8)      0           reshape_1[0][0]
__________________________________________________________________________________________________
digitcaps (CapsuleLayer)        (None, 10, 16)       2641920     lambda_1[0][0]
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 10)           0
__________________________________________________________________________________________________
mask_1 (Mask)                   (None, 16)           0           digitcaps[0][0]
                                                                 input_2[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 512)          8704        mask_1[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1024)         525312      dense_1[0][0]
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 3072)         3148800     dense_2[0][0]
__________________________________________________________________________________________________
out_caps (Length)               (None, 10)           0           digitcaps[0][0]
__________________________________________________________________________________________________
out_recon (Reshape)             (None, 32, 32, 3)    0           dense_3[0][0]
==================================================================================================
Total params: 11,695,872
Trainable params: 11,675,392
Non-trainable params: 20,480
__________________________________________________________________________________________________

"""

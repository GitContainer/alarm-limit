# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:39:41 2019

@author: Ravi Tiwari
"""

from keras.layers import Input, Dense
from keras.models import Model

###############################################################################
# input: ma_df
###############################################################################
input_data = ma_df.values
encoding_dim = 1
input_img = Input(shape=(5,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(5, activation=None)(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='mse')

autoencoder.fit(input_data, input_data,
                epochs=800,
#                batch_size=5000,
                shuffle=True)

encoded_imgs = encoder.predict(input_data)
decoded_imgs = decoder.predict(encoded_imgs)

###############################################################################
# Checking the accuracy of the decoded values
###############################################################################
for i in range(5):
    plt.plot(input_data[:,i], label = 'original')
    plt.plot(decoded_imgs[:,i], label = 'decoded')
    plt.legend()  
    plt.title(ma_df.columns[i])
    plt.show()



    
###############################################################################
# now test on all other data points
###############################################################################
tp = 5*24*60
col_ind = [0, 1,2,3,4,10]
df_s = abnormal_df.iloc[:,col_ind]
df_s = df_s.sort_values(by = 'Unnamed: 0')
df_s = df_s.set_index('Unnamed: 0')

ma_sdf = df_s.rolling('5d').mean()
ma_sdf = ma_sdf.dropna(axis = 0, how = 'any')

input_sdf = ma_sdf.values
encoded_values = encoder.predict(input_sdf)

y_min = np.min(encoded_imgs)
y_max = np.max(encoded_imgs)

plt.plot(ma_sdf.index, encoded_values[:,0], lw = 0, marker = 'o', ms = 0.03)
plt.axhline( y = y_min)
plt.axhline(y = y_max)


###############################################################################
# running the mnist example
###############################################################################
# 1. Single encoder
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# 2. Multiple layer of encoder
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))





encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()    






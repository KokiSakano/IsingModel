import os, sys
import numpy as np

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras import regularizers
import keras.backend as K

from module import SquareSpinState
from module import TriangularSpinState

class LPTT:
    def __init__(self, N, npy_path, model_path, learning_rate=1e-3, l2_const=1e-4, verbose=1, epochs=100, batch_size=36):
        self.N = N
        self.state_shape = (self.N, self.N, 1) # using to reshape spin state
        self.npy_path = npy_path
        self.model_path = model_path
        # define constant using to learn
        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size

    def build_model(self):
        state_tensor = Input(shape=self.state_shape)

        out = Conv2D(filters = 64,
            kernel_size = (3,3),
            activation='relu',
            kernel_regularizer = regularizers.l2(self.l2_const)
            )(state_tensor)

        out = MaxPooling2D()(out)

        out = Flatten()(out)
        out = Dense(64, activation='relu', kernel_regularizer = regularizers.l2(self.l2_const))(out)
        output_tensor = Dense(1, activation='sigmoid')(out)

        return Model(inputs=state_tensor, outputs=output_tensor)

    def learndata(self):
        # load train and test data
        X_train = np.load(self.npy_path+"x_train.npy")
        X_train = X_train.reshape(X_train.shape+(1,))
        Y_train = np.load(self.npy_path+"y_train.npy")

        model = self.build_model()
        adam = Adam(self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
        model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,  validation_split=0.1)
        model.save(self.model_path)
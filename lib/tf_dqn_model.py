import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


class DQN(keras.Model):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv_layer1 = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4),activation='relu',input_shape=input_shape)
        self.conv_layer2 = layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2),activation='relu')
        self.conv_layer3 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1),activation='relu')
        self.flatten_layer = layers.Flatten()
        self.dense_layer = layers.Dense(512, activation='relu')
        ###self.action_predicter =layers.Dense(n_actions, activation='softmax',kernel_initializer="glorot_uniform")
        self.action_predicter =layers.Dense(n_actions)


    def call(self, input, training=False):

        features = self.conv_layer1(input)
        features = self.conv_layer2(features)
        features = self.conv_layer3(features)
        features = self.flatten_layer(features)
        features = self.dense_layer(features)
        output = self.action_predicter(features)

        return output



import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import collections


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 
                                'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return np.array(states), \
               np.array(actions), \
               np.array(rewards,dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states) 

class ExperienceBuffNStep(ExperienceBuffer):
    def __init__(self, capacity, n_step,gamma):
        super(ExperienceBuffNStep, self).__init__(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer =[]
        self.n_step_reward = 0
    
    def _get_n_step_info(self):

        if len(self.n_step_buffer) > self.n_step:
            n_step_reward = sum([self.n_step_buffer[i][2]*(self.gamma**i) for i in range(self.n_step)])
            n_step_obs, n_step_action, _, n_step_next_obs, n_step_done = self.n_step_buffer.pop(0)
            self.buffer.append(Experience(n_step_obs, n_step_action, n_step_reward, n_step_next_obs, n_step_done))
            self.n_step_reward = n_step_reward
        
    def _done_n_step_info(self):
        while len(self.n_step_buffer) > 0:
            n_step_reward = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(len(self.n_step_buffer))])
            n_step_obs, n_step_action, _, n_step_next_obs, n_step_done = self.n_step_buffer.pop(0)
            self.buffer.append(Experience(n_step_obs, n_step_action, n_step_reward, n_step_next_obs, n_step_done))

    def append(self, experience):
        self.n_step_buffer.append(experience)
        self._get_n_step_info()
               

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



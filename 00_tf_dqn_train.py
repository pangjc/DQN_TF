import wandb
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

import random
import argparse
import numpy as np
import collections
from tensorflow import keras
from keras import layers

from lib.utilities import DQN 
from lib.utilities import ExperienceBuffer
from lib.utilities import Experience
from lib import tf_wrappers

wandb.init(name='DQN', project="deep-rl-atari")

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 17

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01

SEED = 123

DQN_NAME = 'BASIC_DQN'
OUTPUT_PATH = './'

class ActionStateModel:
    def __init__(self, state_dim, action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim
      
        self.model = self.create_model()

    def create_model(self):
        model = DQN(input_shape = self.state_dim, n_actions = self.action_dim)
        model.build(input_shape = (None, *self.state_dim))
        return model

    def predict(self, state):
        return self.model(state)

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()
        self.buffer = ExperienceBuffer(REPLAY_SIZE)

        self.total_rewards = []
        self.best_m_reward = None
        self.frame_idx = 0
        self.epsilon = EPSILON_START 
        self.optimizer = Adam(learning_rate=LEARNING_RATE)

        self._reset()

    def _reset(self):
        self.state = env.reset() # or self.env.reset()??
        self.total_reward = 0.0

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def play_step(self, epsilon=0.01):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample() #random picking.does not matter whether it is env or self.env

        else:
            state_v = np.expand_dims(self.state,axis=0) 
            q_vals_v = self.model.predict(state_v)
            action_v = tf.math.argmax(q_vals_v,axis=1)
            action = action_v[0] 

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.buffer.append(exp)
        self.state = new_state
        
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

    def calc_loss(self, batch):

        states, actions, rewards, dones, next_states = batch
        done_mask = tf.convert_to_tensor(1-dones.astype(float),dtype=tf.float32)
        ###done_mask = dones
        q_vals_v = self.model.predict(states)
    
        state_action_values = tf.gather(q_vals_v,actions, axis=1,batch_dims=1)
        next_state_values = tf.math.reduce_max(self.target_model.predict(next_states),axis=1)

        ### Tensor can not be sliced and assigned, so using the following way to bybass this restrctions
        ### https://github.com/tensorflow/tensorflow/issues/14132#issuecomment-483002522
        
        next_state_values = tf.math.multiply(next_state_values,done_mask)
        ###next_state_values[done_mask] = 0

        expected_state_action_values =  next_state_values*GAMMA+rewards

        mse = tf.keras.losses.MeanSquaredError()
        return mse(state_action_values,expected_state_action_values)

    def train(self,max_frames = 1000000):

        while self.frame_idx<max_frames:
            self.frame_idx +=1
            self.epsilon = max(EPSILON_FINAL, EPSILON_START -
                            self.frame_idx / EPSILON_DECAY_LAST_FRAME)
        
            reward = self.play_step(self.epsilon)

            if reward is not None:
                self.total_rewards.append(reward)
                m_reward = np.mean(self.total_rewards[-100:]) ## Why last 100 episode??
                print("%d done %d games, reward %.3f, " "eps %.2f" %(self.frame_idx, len(self.total_rewards), m_reward, self.epsilon))
            

                if self.best_m_reward is None or self.best_m_reward < m_reward:
                    model_name = OUTPUT_PATH + DEFAULT_ENV_NAME + "-" + DQN_NAME + "-best_%0f" % m_reward
                    if m_reward > MEAN_REWARD_BOUND-0.5:
                        tf.keras.models.save_model(self.model.model,model_name)
                    
                    if self.best_m_reward is not None:
                        self.best_m_reward = m_reward

                if m_reward > MEAN_REWARD_BOUND:
                    break

            if len(self.buffer) < REPLAY_START_SIZE:
                continue

            if self.frame_idx % SYNC_TARGET_FRAMES == 0:       
                self.target_update()

            batch = self.buffer.sample(BATCH_SIZE)

            loss_t = self.calc_loss(batch)

            with tf.GradientTape() as tape:
                loss_t = self.calc_loss(batch)
            gradients = tape.gradient(loss_t,self.model.model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients,self.model.model.trainable_weights))

            if self.frame_idx % 100 == 0:     
                log_dict = {"loss": loss_t, "m_reward": m_reward, "games:": len(self.total_rewards)}
                wandb.log(log_dict,step=self.frame_idx)


if __name__ == "__main__":
    random.seed(SEED)
    env = tf_wrappers.make_env(DEFAULT_ENV_NAME)
    env.seed(SEED)
    agent = Agent(env)
    agent.train()

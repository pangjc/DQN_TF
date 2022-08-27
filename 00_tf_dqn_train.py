from lib.utilities import DQN 
from lib.utilities import ExperienceBuffer
from lib.utilities import Experience
from lib import tf_wrappers

import random
import argparse
import time
import numpy as np
import collections
import tensorflow as tf
from tensorflow import keras
from keras import layers

import datetime
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

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

DQN_NAME = 'BASIC_pc'
OUTPUT_PATH = './'

class Agent:
    def __init__(self, env):
        self.env = env
        self.exp_buffer = ExperienceBuffer(REPLAY_SIZE)

        self.net = DQN(input_shape = env.observation_space.shape, n_actions = env.action_space.n)
        self.tgt_net = DQN(input_shape = env.observation_space.shape,n_actions = env.action_space.n)
        self.net.build(input_shape = (None, *env.observation_space.shape))
        self.tgt_net.build(input_shape = (None, *env.observation_space.shape))

        self.total_rewards = []
        self.best_m_reward = None
        self.frame_idx = 0
        self.epsilon = EPSILON_START 
        self.optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
        
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = OUTPUT_PATH + 'logs/' + self.current_time + '/train_'+ DQN_NAME
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir,flush_millis=1000)
        self.ts_frame = 0
        self.ts = time.time()

        self._reset()

    def _reset(self):
        self.state = env.reset() # or self.env.reset()??
        self.total_reward = 0.0

    def target_update(self):
        self.tgt_net.set_weights(self.net.get_weights()) 

    def play_step(self, epsilon=0.01):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample() #random picking.does not matter whether it is env or self.env

        else:
            state_v = np.expand_dims(self.state,axis=0) 
            q_vals_v = self.net(state_v)
            action_v = tf.math.argmax(q_vals_v,axis=1)
            action = action_v[0] 

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        
        if is_done:
            done_reward = self.total_reward
            self._reset()

        return done_reward

    def training_recording(self,loss_t,m_reward,train_loss,train_reward):
        train_reward(m_reward)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('reward', train_reward.result(), step=self.frame_idx)
        train_reward.reset_states()
        train_loss(loss_t)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=self.frame_idx)
        train_loss.reset_states()
     

    def record_display_record(self,m_reward):
        speed = (self.frame_idx-self.ts_frame)/(time.time()-self.ts)
        self.ts_frame = self.frame_idx
        self.ts = time.time()
        print("%d done %d games, reward %.3f, " "eps %.2f, speed %.2f f/s" %(
                self.frame_idx, len(self.total_rewards), m_reward, self.epsilon, speed)
            )
        if self.best_m_reward is not None and self.best_m_reward < m_reward:
            print("Best reward update %.3f -> %.3f" % (self.best_m_reward, m_reward))
        if m_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % self.frame_idx)


    def calc_loss(self, batch):

        states, actions, rewards, dones, next_states = batch
        done_mask = tf.convert_to_tensor(1-dones.astype(float),dtype=tf.float32)
        ###done_mask = dones
        q_vals_v = self.net(states)
    
        state_action_values = tf.gather(q_vals_v,actions, axis=1,batch_dims=1)
        next_state_values = tf.math.reduce_max(self.tgt_net(next_states),axis=1)

        ### Tensor can not be sliced and assigned, so using the following way to bybass this restrctions
        ### https://github.com/tensorflow/tensorflow/issues/14132#issuecomment-483002522
        
        next_state_values = tf.math.multiply(next_state_values,done_mask)
        ###next_state_values[done_mask] = 0

        expected_state_action_values =  next_state_values*GAMMA+rewards

        mse = tf.keras.losses.MeanSquaredError()
        return mse(state_action_values,expected_state_action_values)

    def train(self,max_frames = 1000000):

        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_reward = tf.keras.metrics.Mean('m_reward', dtype=tf.float32)

        while self.frame_idx<max_frames:
            self.frame_idx +=1
            self.epsilon = max(EPSILON_FINAL, EPSILON_START -
                            self.frame_idx / EPSILON_DECAY_LAST_FRAME)
        
            reward = self.play_step(self.epsilon)

            if reward is not None:
                self.total_rewards.append(reward)
                m_reward = np.mean(self.total_rewards[-100:]) ## Why last 100 episode??
                self.record_display_record(m_reward)

                if self.best_m_reward is None or self.best_m_reward < m_reward:
                    model_name = OUTPUT_PATH + DEFAULT_ENV_NAME + "-" + DQN_NAME + "-best_%0f" % m_reward
                    if m_reward > MEAN_REWARD_BOUND-0.5:
                        tf.keras.models.save_model(self.net,model_name)
                    
                    if self.best_m_reward is not None:
                        self.record_display_record(m_reward)
                        self.best_m_reward = m_reward

                if m_reward > MEAN_REWARD_BOUND:
                    self.record_display_record(m_reward)
                    break

            if len(self.exp_buffer) < REPLAY_START_SIZE:
                continue

            if self.frame_idx % SYNC_TARGET_FRAMES == 0:       
                self.target_update()

            batch = self.exp_buffer.sample(BATCH_SIZE)

            with tf.GradientTape() as tape:
                loss_t = self.calc_loss(batch)
                ###print("frame_idx: %d, loss_t: %.3f " % (frame_idx, loss_t.numpy()))
            gradients = tape.gradient(loss_t,self.net.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients,self.net.trainable_weights))
            if self.frame_idx % 100 == 0:     
                self.training_recording(loss_t, m_reward, train_loss, train_reward)


if __name__ == "__main__":
    random.seed(SEED)
    env = tf_wrappers.make_env(DEFAULT_ENV_NAME)
    env.seed(SEED)
    agent = Agent(env)
    agent.train()

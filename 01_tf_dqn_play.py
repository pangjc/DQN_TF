#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np
import tensorflow as tf

from lib import tf_wrappers
from lib import tf_dqn_model

import collections

from gym.wrappers.monitoring.video_recorder import VideoRecorder

import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

FPS = 25
output_video = './display/record_reward_try_mac.mp4'
model_name = './PongNoFrameskip-v4-best_-4.570000'

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
env = tf_wrappers.make_env(DEFAULT_ENV_NAME)

video_recorder = VideoRecorder(env, output_video, enabled=True)

total_reward = 0.0
total_steps = 0

net = tf.keras.models.load_model(model_name)

total_reward = 0.0
total_steps = 0.0
state = env.reset()

while True:
    start_ts = time.time()

    state_v = np.expand_dims(state,axis=0) 
    q_vals_v = net(state_v)
    action_v = tf.math.argmax(q_vals_v,axis=1)
    action = action_v[0] 

    state, reward, done, _ = env.step(action)
    total_reward += reward
    total_steps += 1
    env.render(mode='rgb_array')
    video_recorder.capture_frame()
    if done:
        break

    delta = 1/FPS - (time.time() - start_ts)
    if delta > 0:
        time.sleep(delta)


print("Episode done in %d steps, total reward %.2f" % (
    total_steps, total_reward))

video_recorder.close()
video_recorder.enabled = False
env.close()

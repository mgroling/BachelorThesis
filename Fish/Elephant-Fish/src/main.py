import pandas as pd
import numpy as np
import gym
import gym_cartpole

import sys
sys.path.insert(0, "SQIL_DQN")
from SQIL_DQN import SQIL_DQN

sys.path.append("I:/Code/BachelorThesis/gym-guppy")
import gym_guppy

env = gym_guppy.envs._configurable_guppy_env.ConfigurableGuppyEnv()

model = SQIL_DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 10000, double_q = False, seed = 37)

"""testing"""
reward_sum = 0.0
obs = env.reset()
for i in range(0, 10):
    timestep = 0
    done = False
    while not done:
        timestep += 1
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        env.render()
        if timestep > 500:
            break
    print(reward_sum)
    reward_sum = 0.0
    obs = env.reset()

env.close()
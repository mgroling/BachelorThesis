import pandas as pd
import numpy as np
import gym
import gym_cartpole

import sys
sys.path.insert(0, "SQIL_DQN")

from SQIL_DQN import SQIL_DQN
from stable_baselines.deepq.policies import FeedForwardPolicy

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 64, 32],
                                           layer_norm=False,
                                           feature_extraction="mlp")

def main():
    env = gym.make('cartpole_custom-v0')

    model = SQIL_DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 100000, double_q = False, seed = 37)
    model.intializeExpertBuffer("Cartpole/data/cartpole_custom_expert.csv", 4)
    model.learn(total_timesteps=100000)
    model.save("Cartpole/data/cartpole_custom-v0_agent")

    # model = SQIL_DQN.load("Cartpole/data/cartpole_custom-v0_agent")

    #test it
    reward_sum = 0.0
    obs = env.reset()
    for i in range(0, 10):
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import gym
import gym_cartpole

from SQIL_DQN import SQIL_DQN
from stable_baselines.deepq.policies import FeedForwardPolicy

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[16, 16],
                                           layer_norm=False,
                                           feature_extraction="mlp")

def main():
    env = gym.make('cartpole_custom-v0')
    model = SQIL_DQN(CustomDQNPolicy, env, verbose=1)
    model.intializeExpertBuffer("Cartpole/data/cartpole_custom_expert.csv", 4)
    model.learn(total_timesteps=10000)



if __name__ == "__main__":
    main()
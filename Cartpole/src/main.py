import pandas as pd
import numpy as np
import gym
import gym_cartpole

import sys
sys.path.insert(0, "SQIL_DQN")

from SQIL_DQN import SQIL_DQN
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.gail import ExpertDataset
from functions import *

def main():
    env = gym.make("cartpole_custom-v0")

    lr_paths = ["Cartpole/data/trajectory_left_right_0.csv", "Cartpole/data/trajectory_left_right_1.csv", "Cartpole/data/trajectory_left_right_2.csv", "Cartpole/data/trajectory_left_right_3.csv"]
    rl_paths = ["Cartpole/data/trajectory_right_left_0.csv", "Cartpole/data/trajectory_right_left_1.csv", "Cartpole/data/trajectory_right_left_2.csv", "Cartpole/data/trajectory_right_left_3.csv"]

    """get trajectories"""
    lr_trajectories, lr_lengths = loadTrajectories(lr_paths, (2, "smaller", 3/4*np.pi), 40)#(2, "smaller", 1/2*np.pi)
    # rl_trajectories, rl_lengths = loadTrajectories(rl_paths, (2, "bigger", 3/2*np.pi), 20)#(2, "bigger", 3/2*np.pi)
    # expert_trajectory = np.append(lr_trajectories, rl_trajectories, axis = 0)

    """create model"""
    # model = SQIL_DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 10000, double_q = False, seed = 37)
    # model = DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 10000, double_q = False, seed = 37)

    """ONLY behavioural cloning"""
    # dic = createDictForBehaviouralCloning(lr_paths, None, 20)
    # expertData = ExpertDataset(traj_data = dic)
    # model.pretrain(expertData, n_epochs = 1000)

    """for SQIL"""
    # model.initializeExpertBuffer(lr_trajectories, 4)
    # model.learn(total_timesteps=100000)

    """for DQN"""
    # model.learn(total_timesteps=1000000)

    """save model"""
    # model.save("Cartpole/models/SQIL_lrALL_3Pi-4_40_100k_random/model")
    # model.save("Cartpole/models/DQN_pretrained_rlALL_None_20_1000_random/model")
    # model.save("Cartpole/models/DQN_100k_random/model")

    """load model"""
    # model = DQN.load("Cartpole/models/DQN_pretrained_rlALL_None_20_1000_random/model")

    """testing"""
    # reward_sum = 0.0
    # obs = env.reset()
    # for i in range(0, 10):
    #     timestep = 0
    #     done = False
    #     while not done:
    #         timestep += 1
    #         action, _ = model.predict(obs)
    #         obs, reward, done, _ = env.step(action)
    #         reward_sum += reward
    #         env.render()
    #         if timestep > 500:
    #             break
    #     print(reward_sum)
    #     reward_sum = 0.0
    #     obs = env.reset()

    # env.close()

    testAll("Cartpole/models/", 100, 500, "Average reward with randomized cart start position with max timesteps per episode = 500")

if __name__ == "__main__":
    main()
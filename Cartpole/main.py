import pandas as pd
import numpy as np
import gym
import gym_cartpole

import sys
sys.path.insert(0, "SQIL_DQN")

from SQIL_DQN import SQIL_DQN
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 64, 32],
                                           layer_norm=False,
                                           feature_extraction="mlp")

def cutTrajectory(trajectory, start_criteria):
    """
    cuts off first steps until start_criteria is met
    start_criteria should be of the form (column, "bigger" or "smaller", value)
    """
    start_index = None
    if start_criteria == None:
        pass
    elif start_criteria[1] == "bigger":
        start_index = np.argmax(trajectory[:, start_criteria[0]] > start_criteria[2])
    elif start_criteria[1] == "smaller":
        start_index = np.argmax(trajectory[:, start_criteria[0]] < start_criteria[2])
    else:
        print("comparison method not supported")

    return trajectory[start_index:]

def loadTrajectories(trajectory_paths, start_criteria, cut_last):
    """
    trajectory_paths should be iterable
    loads csv files from given paths and returns a single concatenated np array
    if start_criteria != None, then all trajectories cut before concatenating them
    cuts of last cut_last elements from each trajectory
    """
    trajectories = []
    for path in trajectory_paths:
        trajectory = cutTrajectory(pd.read_csv(path, sep = ";").to_numpy(), start_criteria)
        trajectories.append(trajectory[:-cut_last])
    
    return np.concatenate(trajectories, axis = 0)

def testModel(model_path, episodes, max_timesteps, sqil = True):
    env = gym.make("cartpole_custom-v0")
    model = None
    if sqil:
        model = SQIL_DQN.load(model_path)
    else:
        model = DQN.load(model_path)

    rewards = np.zeros((episodes))
    for i in range(0, episodes):
        obs = env.reset()
        timestep = 0
        done = False
        while not done:
            timestep += 1
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            rewards[i] += reward
            if timestep > 500:
                break

    return rewards

def main():
    env = gym.make("cartpole_custom-v0")

    lr_paths = ["Cartpole/data/trajectory_left_right_0.csv", "Cartpole/data/trajectory_left_right_1.csv", "Cartpole/data/trajectory_left_right_2.csv", "Cartpole/data/trajectory_left_right_3.csv"]
    rl_paths = ["Cartpole/data/trajectory_right_left_0.csv", "Cartpole/data/trajectory_right_left_1.csv", "Cartpole/data/trajectory_right_left_2.csv", "Cartpole/data/trajectory_right_left_3.csv"]

    lr_trajectories = loadTrajectories(lr_paths, None, 40)#(2, "smaller", 1/2*np.pi)
    rl_trajectories = loadTrajectories(rl_paths, (2, "bigger", 3/2*np.pi), 40)

    # expert_trajectory = np.append(lr_trajectories, rl_trajectories, axis = 0)

    model = SQIL_DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 10000, double_q = False, seed = 37)
    # model = DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 10000, double_q = False, seed = 37)
    model.intializeExpertBuffer(lr_trajectories, 4)
    model.learn(total_timesteps=50000)
    model.save("Cartpole/models/SQIL_lrALL_None_40_50k_center/SQIL_lrALL_None_40_50k_center")

    # model = SQIL_DQN.load("Cartpole/models/SQIL_cartpole_custom_expert_only_latter_v0")

    #test it
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

if __name__ == "__main__":
    main()
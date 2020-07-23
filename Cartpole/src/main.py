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

def loadTrajectories(trajectory_paths, start_criteria, cut_last, concat = False):
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
    
    if concat:
        return np.concatenate(trajectories, axis = 0), [sum([len(trajectories[j]) for j in range(0, i+1)])-2 for i in range(0, len(trajectories))]
    else:
        return trajectories, [(len(trajectory) for trajectory in trajectories)]

def testModel(model_path, episodes, max_timesteps_per_episode, sqil = True):
    """
    runs a model "episodes" iterations and collects rewards for each episode
    each episode stops at max_timesteps_per_episode or earlyier
    """
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
            if timestep > max_timesteps_per_episode:
                break

    return rewards

def createDictForBehaviouralCloning(trajectory_paths, start_criteria, cut_last):
    #get trajectories
    trajectories, len_dfs = loadTrajectories(trajectory_paths, start_criteria, cut_last)

    #create dictionary
    #create episode_starts
    starts = [False for i in range(0, sum(len_dfs)-cut_last*len(trajectory_paths))]
    starts[0] = True
    for i in range(0, len(len_dfs)-1):
        starts[len_dfs[i]] = True
    
    return {"actions": trajectories[:, 4:], "episode_returns": np.array([[elem for elem in len_dfs]]), "rewards": np.ones((len(trajectories), 1)), "obs": trajectories[:, 0:4], "episode_starts": np.array([starts])}

def testAll(directory):
    """
    tests all models in the given diretory, which are in another folder
    """
    print(os.listdir)

def save_obj(obj, name):
    """
    This functions assumes that you have an obj folder in your current working directory, which will be used to store the objects.
    taken from: https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    """
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    This functions assumes that you have an obj folder in your current working directory, which will be used to store the objects.
    taken from: https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file/32216025
    """
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    env = gym.make("cartpole_custom-v0")

    lr_paths = ["Cartpole/data/trajectory_left_right_0.csv", "Cartpole/data/trajectory_left_right_1.csv", "Cartpole/data/trajectory_left_right_2.csv", "Cartpole/data/trajectory_left_right_3.csv"]
    rl_paths = ["Cartpole/data/trajectory_right_left_0.csv", "Cartpole/data/trajectory_right_left_1.csv", "Cartpole/data/trajectory_right_left_2.csv", "Cartpole/data/trajectory_right_left_3.csv"]

    lr_trajectories, lr_lengths = loadTrajectories(lr_paths, None, 20)#(2, "smaller", 1/2*np.pi)
    rl_trajectories, rl_lengths = loadTrajectories(rl_paths, (2, "bigger", 3/2*np.pi), 20)#(2, "bigger", 3/2*np.pi)

    expert_trajectory = np.append(lr_trajectories, rl_trajectories, axis = 0)

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

    """"for DQN"""
    # model.learn(total_timesteps=1000000)

    """save model"""
    # model.save("Cartpole/models/SQIL_lrALL_None_40_100k_random/SQIL_lrALL_None_40_100k_random_updated")
    # model.save("Cartpole/models/DQN_pretrained_rlALL_None_20_1000_random/DQN_pretrained_rlALL_None_20_1000_random")
    # model.save("Cartpole/models/DQN_100k_random/DQN_100k_random")

    """load model"""
    model = SQIL_DQN.load("Cartpole/models/SQIL_lrALL_None_40_100k_random/SQIL_lrALL_None_40_100k_random_updated")

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

if __name__ == "__main__":
    main()
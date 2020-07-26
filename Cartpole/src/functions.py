from stable_baselines.deepq.policies import FeedForwardPolicy
import pandas as pd
import numpy as np
import os
import gym
import matplotlib.pyplot as plt
import pickle

from SQIL_DQN import SQIL_DQN
from stable_baselines import DQN

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
        cut = len(trajectory) - cut_last
        trajectories.append(trajectory[:cut])
    
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

def testAll(directory, episodes, max_timesteps, title):
    """
    tests all models in the given diretory, which are in the direct subfolder, named model
    saves rewards in obj folder, that can be retrieved with load_obj
    """
    paths = os.listdir(directory)
    rewards = []
    for path in paths:
        if path[0:4] == "SQIL":
            rewards.append(testModel(directory + "/" + path + "/model", episodes, max_timesteps))
        else:
            rewards.append(testModel(directory + "/" + path + "/model", episodes, max_timesteps, sqil = False))
    
    os.chdir("Cartpole/")
    save_obj(rewards, "rewards")

    for i in range(0, len(rewards)):
        rewards[i] = np.mean(rewards[i])

    for i in range(0, len(paths)):
        split = paths[i].split("_")
        _apperances = paths[i].count("_")
        _toTake = _apperances -2
        charsToTake = sum([len(split[i]) for i in range(0, len(split)-2)])
        paths[i] = paths[i][:(charsToTake+_toTake)]

    plt.rc("xtick", labelsize = 8)
    fig, ax = plt.subplots(figsize = (18, 11))
    rects = plt.bar(paths, rewards)
    autolabel(rects, ax)
    ax.set_title(title, fontsize = 20)
    ax.set_xlabel("models", fontsize = 20)
    ax.set_ylabel("average rewards over " + str(episodes) + " episodes", fontsize = 20)
    plt.show()

def autolabel(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    taken from https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

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
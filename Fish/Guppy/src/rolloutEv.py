import random
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("Fish")
from convertData import getAll


def distObs(obs_1, obs_2):
    diff_bin = np.abs(np.argmax(obs_1[0]) - np.argmax(obs_2[:, 0], axis=1)) + 1
    diff_bin = np.where(
        diff_bin > len(obs_1[0]) / 2, len(obs_1[0]) - diff_bin, diff_bin
    )
    max_fish = (
        diff_bin
        * (np.abs(obs_1[0].max() - obs_2[:, 0].max(axis=1)))
        * 19.870316067048226
    )
    sum_wall = np.sum(np.abs(obs_1[1] - obs_2[:, 1]), axis=1)
    return max_fish + sum_wall


def closeActions(single_obs, all_obs, all_act, max_dist):
    """
    returns all actions of states that have a distance of less than max_dist between them and single_obs
    """
    temp = distObs(single_obs, all_obs)
    actions = set(np.where(temp < max_dist, all_act.reshape((len(all_act),)), None))
    actions.remove(None)
    return list(actions)


def testExpert(
    paths, model, env, exp_turn_fraction, exp_speed, perc, deterministic=True
):
    turn_bins, speed_bins = len(env.turn_rate_bins), len(env.speed_bins)
    obs, act = getAll(paths, np.pi / exp_turn_fraction, exp_speed, env)
    obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
    reward = []
    random_reward = []
    acceptedActions = loadConfig(
        "Fish/Guppy/rollout/pi_"
        + str(exp_turn_fraction)
        + "_"
        + str(int(exp_speed * 100 // 10))
        + str(int(exp_speed * 100 % 10))
        + "/allowedActions_val_"
        + str(perc)
        + ".json"
    )["allowed actions"]

    for i in range(0, len(obs)):
        action, _ = model.predict(obs[i], deterministic=deterministic)
        if action in acceptedActions[i]:
            reward.append(1)
        else:
            reward.append(0)
        rand = random.randint(0, turn_bins * speed_bins - 1)
        if rand in acceptedActions[i]:
            random_reward.append(1)
        else:
            random_reward.append(0)

    dic = loadConfig(
        "Fish/Guppy/rollout/pi_"
        + str(exp_turn_fraction)
        + "_"
        + str(int(exp_speed * 100 // 10))
        + str(int(exp_speed * 100 % 10))
        + "/perfect_agent_"
        + str(perc)
        + ".json"
    )

    return (
        np.mean(reward),
        np.mean(random_reward),
        dic["perfect agent ratio"],
        dic["closest agent ratio"],
    )


def saveDistributionThreshholds(obs_1, obs_2, save_path):
    distances = []
    for i in range(len(obs_1)):
        if i % 1000 == 0:
            print("timestep", i, "finished")
        distances.extend(distObs(obs_1[i], obs_2))
    distances.sort()
    percentage, threshhold = np.arange(1, 21), []
    percentage = [int(i) for i in percentage]
    for i in range(1, 21):
        threshhold.append(float(distances[int(len(distances) * (i / 100))]))
    dic = {
        "percentage": percentage,
        "threshhold": threshhold,
    }
    saveConfig(save_path + "distribution_threshholds.json", dic)
    plt.hist(distances, bins=100)
    plt.title("distribution of distances")
    plt.xlabel("distance")
    plt.ylabel("# elements")
    plt.savefig(save_path + "distribution.png")


def saveAllowedActions(paths, exp_turn, exp_speed, env, max_dist, save_path):
    obs, act = getAll(paths, exp_turn, exp_speed, env)
    obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
    actions = []
    for i in range(len(obs)):
        if i % 1000 == 0:
            print("timestep", i, "finished")
        actions.append(closeActions(obs[i], obs, act, max_dist))
    dic = {
        "exp_turn": exp_turn,
        "exp_speed": exp_speed,
        "max_dist": max_dist,
        "allowed actions": actions,
    }
    saveConfig(save_path, dic)


def savePerfectAgentActions(
    paths_val, paths_tra, exp_turn_fraction, exp_speed, env, save_path, perc
):
    obs_val, act_val = getAll(paths_val, np.pi / exp_turn_fraction, exp_speed, env)
    obs_val, act_val = np.concatenate(obs_val, axis=0), np.concatenate(act_val, axis=0)
    obs_tra, act_tra = getAll(paths_tra, np.pi / exp_turn_fraction, exp_speed, env)
    obs_tra, act_tra = np.concatenate(obs_tra, axis=0), np.concatenate(act_tra, axis=0)
    correct = []
    perfect = []
    dic = loadConfig(
        "Fish/Guppy/rollout/pi_"
        + str(exp_turn_fraction)
        + "_"
        + str(int(exp_speed * 100 // 10))
        + str(int(exp_speed * 100 % 10))
        + "/allowedActions_val_"
        + str(perc)
        + ".json"
    )
    acceptedActions = dic["allowed actions"]
    max_dist = dic["max_dist"]
    for i in range(len(obs_val)):
        if i % 1000 == 0:
            print("timestep", i, "finished")
        actions = act_tra[distObs(obs_val[i], obs_tra).argmin()]
        if int(actions) in acceptedActions[i]:
            correct.append(1)
        else:
            correct.append(0)

        # actions = closeActions(obs_val[i], obs_tra, act_tra, max_dist)
        # perfect.append(1 if len(set(actions).union(set(acceptedActions[i]))) > 0 else 0)
    dic = {
        "closest agent ratio": np.array(correct).mean(),
        "perfect agent ratio": 1,
    }
    saveConfig(save_path, dic)


def saveConfig(path, dic):
    with open(path, "w+") as f:
        json.dump(dic, f)


def loadConfig(path):
    with open(path, "r") as f:
        return json.load(f)
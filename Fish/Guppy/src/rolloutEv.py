import random
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import jit, njit, prange

sys.path.append("Fish")
from convertData import getAll


def distObs(obs_1, obs_2, env, mode="both"):
    if mode == "fish":
        return distFish(obs_1, obs_2, env)
    elif mode == "wall":
        return distWall(obs_1, obs_2) * 2.123003018359641
    else:
        return distFish(obs_1, obs_2, env) + distWall(obs_1, obs_2) * 2.123003018359641


def distWall(obs_1, obs_2):
    return np.sum(np.abs(obs_1[1] - obs_2[:, 1]), axis=1)


def distFish(obs_1, obs_2, env):
    index_1, index_2 = obs_1[0].argmax(), obs_2[:, 0].argmax(axis=1)
    direction_1, direction_2 = (
        env.sector_bounds[index_1] + env.sector_bounds[index_1 + 1]
    ) / 2, (env.sector_bounds[index_2] + env.sector_bounds[index_2 + 1]) / 2
    pos_1 = np.array(
        [
            (1 - obs_1[0].max()) * np.cos(direction_1) * np.sqrt(2),
            (1 - obs_1[0].max()) * np.sin(direction_1) * np.sqrt(2),
        ]
    )
    pos_2 = np.array(
        [
            (1 - obs_2[:, 0].max(axis=1)) * np.cos(direction_2) * np.sqrt(2),
            (1 - obs_2[:, 0].max(axis=1)) * np.sin(direction_2) * np.sqrt(2),
        ]
    )
    pos_2 = np.swapaxes(pos_2, 0, 1)
    distance = np.linalg.norm(pos_1 - pos_2, axis=1)

    return distance * 100


def closeActions(single_obs, all_obs, all_act, max_dist, env, mode="both"):
    """
    returns all actions of states that have a distance of less than max_dist between them and single_obs
    """
    temp = distObs(single_obs, all_obs, env, mode)
    actions = set(np.where(temp <= max_dist, all_act.reshape((len(all_act),)), None))
    actions.remove(None)
    return list(actions)


def checkAction(action, allowedActions, env):
    maxTurnDiff, maxSpeedDiff = np.radians(2), 0.001

    allowedActions = np.array(allowedActions)
    turn, speed = (
        env.turn_rate_bins[action // len(env.speed_bins)],
        env.speed_bins[action % len(env.speed_bins)],
    )
    allowedTurn, allowedSpeed = (
        env.turn_rate_bins[np.floor(allowedActions / len(env.speed_bins)).astype(int)],
        env.speed_bins[(allowedActions % len(env.speed_bins)).astype(int)],
    )

    return (
        np.logical_and(
            np.abs(turn - allowedTurn) <= maxTurnDiff,
            np.abs(speed - allowedSpeed) <= maxSpeedDiff,
        )
    ).any()


def checkActionVec(action, allowedActions, env):
    maxTurnDiff, maxSpeedDiff = np.radians(2), 0.001
    action, allowedActions = np.array(action), np.array(allowedActions)

    turn, speed = (
        env.turn_rate_bins[(action / len(env.speed_bins)).astype(int)],
        env.speed_bins[(action % len(env.speed_bins)).astype(int)],
    )
    allowedTurn, allowedSpeed = (
        env.turn_rate_bins[(allowedActions / len(env.speed_bins)).astype(int)],
        env.speed_bins[(allowedActions % len(env.speed_bins)).astype(int)],
    )

    return np.logical_and(
        (np.abs(turn - allowedTurn) <= maxTurnDiff),
        (np.abs(speed - allowedSpeed) <= maxSpeedDiff),
    ).any(axis=1)


def testExpert(
    paths,
    model,
    env,
    perc,
    deterministic=True,
    convMat=False,
    mode="both",
):
    turn_bins, speed_bins = len(env.turn_rate_bins), len(env.speed_bins)
    obs, act = getAll(paths, env)
    obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
    reward = np.zeros((len(obs), 1), dtype=int)
    random_reward = np.zeros((len(obs), 1), dtype=int)
    acceptedActions = loadConfig(
        "Fish/Guppy/rollout/tbins"
        + str(turn_bins)
        + "_sbins"
        + str(speed_bins)
        + "/allowedActions_val_"
        + str(perc)
        + "_"
        + mode
        + ".json"
    )["allowed actions"]

    # convert accepted actions to common shape ndarray
    lens = [len(l) for l in acceptedActions]
    maxlen = max(lens)
    arr = np.tile(
        np.array([[elem[0] for elem in acceptedActions]]).transpose(), (1, maxlen)
    )
    mask = np.arange(maxlen) < np.array(lens)[:, None]
    arr[mask] = np.concatenate(acceptedActions)

    agentActions, _ = model.predict(obs, deterministic=deterministic)
    agentActions = np.array(agentActions).transpose()
    if convMat:
        agentActions = np.array(
            [agentActions[:, 0] * speed_bins + agentActions[:, 1]]
        ).transpose()
    randomActions = np.array(
        [np.random.randint(turn_bins * speed_bins, size=len(obs))]
    ).transpose()

    temp = checkActionVec(agentActions, arr, env)
    agentRatio = np.sum(temp) / len(temp)
    temp = checkActionVec(randomActions, arr, env)
    randomRatio = np.sum(temp) / len(temp)

    dic = loadConfig(
        "Fish/Guppy/rollout/tbins"
        + str(turn_bins)
        + "_sbins"
        + str(speed_bins)
        + "/perfect_agent_"
        + str(perc)
        + "_"
        + mode
        + ".json"
    )

    return (
        agentRatio,
        randomRatio,
        dic["perfect agent ratio"],
        dic["closest agent ratio"],
    )


def saveDistributionThreshholds(obs_1, obs_2, save_path, env, mode="both"):
    distances = []
    for i in range(len(obs_1)):
        if i % 1000 == 0:
            print("timestep", i, "finished")
        distances.extend(distObs(obs_1[i], obs_2, env, mode))
    distances.sort()
    percentage, threshhold = np.arange(1, 21), []
    percentage = [int(i) for i in percentage]
    for i in range(1, 21):
        threshhold.append(float(distances[int(len(distances) * (i / 100))]))
    dic = {
        "percentage": percentage,
        "threshhold": threshhold,
    }
    saveConfig(save_path + "distribution_threshholds_" + mode + ".json", dic)
    plt.clf()
    plt.hist(distances, bins=100, density=True)
    plt.title("Distribution of distances (" + mode + " distances)")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.savefig(save_path + "distribution_" + mode + ".png")
    plt.close()


def saveAllowedActions(paths, env, max_dist, save_path, mode="both"):
    obs, act = getAll(paths, env)
    obs, act = np.concatenate(obs, axis=0), np.concatenate(act, axis=0)
    actions = []
    for i in range(len(obs)):
        if i % 1000 == 0:
            print("timestep", i, "finished")
        actions.append(closeActions(obs[i], obs, act, max_dist, env, mode))
    dic = {
        "max_dist": max_dist,
        "allowed actions": actions,
    }
    saveConfig(save_path, dic)


def savePerfectAgentActions(paths_val, paths_tra, env, save_path, perc, mode="both"):
    turn_bins, speed_bins = len(env.turn_rate_bins), len(env.speed_bins)
    obs_val, act_val = getAll(paths_val, env)
    obs_val, act_val = np.concatenate(obs_val, axis=0), np.concatenate(act_val, axis=0)
    obs_tra, act_tra = getAll(paths_tra, env)
    obs_tra, act_tra = np.concatenate(obs_tra, axis=0), np.concatenate(act_tra, axis=0)
    closestAgentActions = np.zeros((len(obs_val), 1))
    dic = loadConfig(
        "Fish/Guppy/rollout/tbins"
        + str(turn_bins)
        + "_sbins"
        + str(speed_bins)
        + "/allowedActions_val_"
        + str(perc)
        + "_"
        + mode
        + ".json"
    )
    acceptedActions = dic["allowed actions"]
    max_dist = dic["max_dist"]

    # convert accepted actions to common shape ndarray
    lens = [len(l) for l in acceptedActions]
    maxlen = max(lens)
    arr = np.tile(
        np.array([[elem[0] for elem in acceptedActions]]).transpose(), (1, maxlen)
    )
    mask = np.arange(maxlen) < np.array(lens)[:, None]
    arr[mask] = np.concatenate(acceptedActions)

    print("Computing perfect agent ratio, mode:", mode, "perc:", perc)
    for i in range(len(obs_val)):
        if i % 1000 == 0:
            print("timestep", i, "finished")
        closestAgentActions[i] = act_tra[
            distObs(obs_val[i], obs_tra, env, mode).argmin()
        ]
    temp = checkActionVec(closestAgentActions, arr, env)

    correct = np.sum(temp) / len(temp)

    dic = {
        "closest agent ratio": correct,
        "perfect agent ratio": 1,
    }
    saveConfig(save_path, dic)


def saveConfig(path, dic):
    with open(path, "w+") as f:
        json.dump(dic, f)


def loadConfig(path):
    with open(path, "r") as f:
        return json.load(f)

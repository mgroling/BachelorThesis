import random
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("SQIL_DQN")
from SQIL_DQN import SQIL_DQN

sys.path.append("Fish")
from convertData import getAll


def distObs(obs_1, obs_2):
    return np.sum(np.abs(obs_1 - obs_2))


def closeActions(single_obs, all_obs, all_act, max_dist):
    """
    returns all actions of states that have a distance of less than max_dist between them and single_obs
    """
    actions = []
    for i in range(0, len(all_obs)):
        if distObs(single_obs, all_obs[i]) < max_dist:
            if not int(all_act[i]) in actions:
                actions.append(int(all_act[i]))
    return actions


def testExpert(paths, model, env, max_dist, deterministic=True):
    obs, act = getAll(paths, np.pi / 4, 0.07, env)
    for i in range(0, len(obs)):
        reward = []
        random_reward = []
        perfect_reward = []
        for j in range(len(obs[i])):
            action, _ = model.predict(obs[i][j], deterministic=deterministic)
            acceptedActions = closeActions(obs[i][j], obs[i], act[i], max_dist)
            if action in acceptedActions:
                reward.append(1)
            else:
                reward.append(0)
            rand = random.randint(0, 199)
            if rand in acceptedActions:
                random_reward.append(1)
            else:
                random_reward.append(0)
            perfect_reward.append(1)

        every_nth = 100
        reward = np.mean(
            np.array(reward[: (len(reward) // every_nth) * every_nth]).reshape(
                -1, every_nth
            ),
            axis=1,
        )
        random_reward = np.mean(
            np.array(
                random_reward[: (len(random_reward) // every_nth) * every_nth]
            ).reshape(-1, every_nth),
            axis=1,
        )
        perfect_reward = np.mean(
            np.array(
                perfect_reward[: (len(perfect_reward) // every_nth) * every_nth]
            ).reshape(-1, every_nth),
            axis=1,
        )

        plt.clf()
        plt.plot(reward, label="SQIL")
        plt.plot(random_reward, label="random agent")
        plt.plot(perfect_reward, label="perfect agent")
        plt.xlabel("batch number (" + str(every_nth) + " elements)")
        plt.ylabel("average reward per sample")
        plt.legend()
        plt.show()
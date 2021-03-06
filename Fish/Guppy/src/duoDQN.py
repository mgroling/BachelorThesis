import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from wrappers import DiscreteActionWrapper, RayCastingWrapper
from convertData import getAll
from rolloutEv import loadConfig, saveConfig
from main import createRolloutFiles

from stable_baselines.deepq.policies import FeedForwardPolicy

sys.path.append("SQIL_DQN")
from SQIL_DQN_manager import SQIL_DQN_MANAGER

sys.path.append("Fish")
from functions import TestEnv


def trainModel(
    dic,
    volatile=False,
    rollout_timesteps=None,
    rollout_determinsitic=True,
    train_plots=None,
    train_plots_path=None,
):
    env = TestEnv(steps_per_robot_action=5)
    env = RayCastingWrapper(env, degrees=dic["degrees"], num_bins=dic["num_bins_rays"])
    env = DiscreteActionWrapper(
        env,
        num_bins_turn_rate=dic["turn_bins"],
        num_bins_speed=dic["speed_bins"],
        max_turn=dic["max_turn"],
        min_speed=dic["min_speed"],
        max_speed=dic["max_speed"],
    )

    class CustomDQNPolicy0(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomDQNPolicy0, self).__init__(
                *args,
                **kwargs,
                layers=dic["nn_layers"][0],
                layer_norm=dic["nn_norm"][0],
                feature_extraction="mlp"
            )

    class CustomDQNPolicy1(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomDQNPolicy1, self).__init__(
                *args,
                **kwargs,
                layers=dic["nn_layers"][1],
                layer_norm=dic["nn_norm"][1],
                feature_extraction="mlp"
            )

    model = SQIL_DQN_MANAGER(
        policy=[CustomDQNPolicy0, CustomDQNPolicy1],
        env=env,
        gamma=dic["gamma"],
        learning_rate=dic["lr"],
        buffer_size=dic["buffer_size"],
        exploration_fraction=dic["explore_fraction"],
        batch_size=dic["n_batch"],
    )

    obs, act = getAll(
        ["Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")],
        0,
        0,
        env,
        False,
    )
    model.initializeExpertBuffer(obs, act)

    model.learn(
        total_timesteps=dic["training_timesteps"],
        sequential_timesteps=dic["sequential_timesteps"],
        rollout_params=dic,
        rollout_timesteps=rollout_timesteps,
        rollout_deterministic=rollout_determinsitic,
        train_plots=train_plots,
        train_plots_path=train_plots_path,
    )

    if volatile:
        reward = []
        for i in range(len(model.rollout_values)):
            for value in model.rollout_values[i]:
                reward.append(value[0])

        return 1 - np.mean(reward)
    else:
        model.save("Fish/Guppy/models/" + dic["model_name"])
        saveConfig("Fish/Guppy/models/" + dic["model_name"] + "/parameters.json")
        createRolloutPlots(dic, model)


def createRolloutPlots(dic, model):
    reward = [[] for i in range(len(model.rollout_values))]
    random_reward = [[] for i in range(len(model.rollout_values))]
    perfect_reward = [[] for i in range(len(model.rollout_values))]
    closest_reward = [[] for i in range(len(model.rollout_values))]
    for i in range(len(model.rollout_values)):
        for value in model.rollout_values[i]:
            reward[i].append(value[0])
            random_reward[i].append(value[1])
            perfect_reward[i].append(value[2])
            closest_reward[i].append(value[3])

    fig, ax = plt.subplots(
        len(model.rollout_values),
        1,
        figsize=(len(model.rollout_values) * 6, 18),
    )
    if len(model.rollout_values) == 1:
        ax = [ax]

    dicThresh = loadConfig(
        "Fish/Guppy/rollout/pi_"
        + str(dic["exp_turn_fraction"])
        + "_"
        + str(int(dic["exp_min_dist"] * 100 // 10))
        + str(int(dic["exp_min_dist"] * 100 % 10))
        + "/distribution_threshholds.json"
    )

    for i in range(len(model.rollout_values)):
        ax[i].plot(reward[i], label="SQIL")
        ax[i].plot(random_reward[i], label="random agent")
        ax[i].plot(perfect_reward[i], label="perfect agent")
        ax[i].plot(closest_reward[i], label="closest state agent")
        ax[i].set_ylabel("average reward")
        ax[i].set_title(
            "max_dist between obs: "
            + str(np.round(dicThresh["threshhold"][dic["perc"]], 2))
            + " ("
            + str(dic["perc"][i] + 1)
            + "% closest states)",
            fontsize=10,
        )
        ax[i].legend(loc="center left")
        for a, b in zip(np.arange(len(reward[i])), reward[i]):
            ax[i].text(a, b, str(np.round(b, 2)), fontsize=6)

    ax[-1].set_xlabel("timestep of training (1000)")
    fig.suptitle("Average reward per sample in Validation Dataset", fontsize=16)
    fig.savefig("Fish/Guppy/models/" + dic["model_name"] + "/rollout.png")
    plt.close()


def testModel(dic):
    env = TestEnv(steps_per_robot_action=5)
    env = RayCastingWrapper(env, degrees=dic["degrees"], num_bins=dic["num_bins_rays"])
    env = DiscreteActionWrapper(
        env,
        num_bins_turn_rate=dic["turn_bins"],
        num_bins_speed=dic["speed_bins"],
        max_turn=dic["max_turn"],
        min_speed=dic["min_speed"],
        max_speed=dic["max_speed"],
    )

    model = SQIL_DQN_MANAGER.load("Fish/Guppy/models/duoDQNtest")

    obs = env.reset()
    for i in range(100):
        action, _ = model.predict(obs)

        obs = env.step(action)[0]

        env.render()  # mode = "video"
    env.close()


if __name__ == "__main__":
    dic = {
        "model_name": "DuoDQN_06_03_2021_01",
        "exp_turn_fraction": 4,
        "exp_min_dist": 0.00,
        "turn_bins": 360,
        "speed_bins": 30,
        "min_speed": 0.00,
        "max_speed": 0.03,
        "max_turn": np.pi / 2,
        "degrees": 360,
        "num_bins_rays": 36,
        "nn_layers": [[256], [256]],
        "nn_norm": [False, False],
        "explore_fraction": [0.3, 0.3],
        "training_timesteps": 10000,
        "sequential_timesteps": 1000,
        "perc": [0],
        "gamma": [0.7, 0.7],
        "lr": [1e-5, 1e-5],
        "n_batch": [32, 32],
        "buffer_size": [1e5, 1e5],
    }
    createRolloutFiles(dic)
    # trainModel(dic)
    # testModel(dic)

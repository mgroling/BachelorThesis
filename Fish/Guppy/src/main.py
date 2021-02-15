import numpy as np
import pandas as pd
import sys
import os
import time
import math
import random

from conversion_scripts.convert_marc import convertTrajectory
from robofish.evaluate import evaluate_all
from wrappers import DiscreteMatrixActionWrapper, RayCastingWrapper, RayCastingObject
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import FeedForwardPolicy

sys.path.append("Fish")
from functions import *
from convertData import *
from rolloutEv import *

sys.path.append("SQIL_DQN")
from SQIL_DQN import SQIL_DQN

import matplotlib.pyplot as plt


def trainModel(dic):
    EXP_TURN_FRACTION = dic["exp_turn_fraction"]
    EXP_TURN, EXP_SPEED = np.pi / EXP_TURN_FRACTION, dic["exp_min_dist"]
    TURN_BINS, SPEED_BINS = dic["turn_bins"], dic["speed_bins"]
    MIN_SPEED, MAX_SPEED, MAX_TURN = dic["min_speed"], dic["max_speed"], dic["max_turn"]
    DEGREES, NUM_RAYS = dic["degrees"], dic["num_bins_rays"]
    NN_LAYERS, NN_NORM, NN_EXPLORE_RATIO = (
        dic["nn_layers"],
        dic["nn_norm"],
        dic["nn_explore_ratio"],
    )
    LEARN_TIMESTEPS = dic["training_timesteps"]
    MODEL_NAME = dic["model_name"]
    PERC = dic["perc"]

    class CustomDQNPolicy(FeedForwardPolicy):
        def __init__(self, *args, **kwargs):
            super(CustomDQNPolicy, self).__init__(
                *args,
                **kwargs,
                layers=NN_LAYERS,
                layer_norm=NN_NORM,
                feature_extraction="mlp"
            )

    env = TestEnv(max_steps_per_action=200)

    env = RayCastingWrapper(env, degrees=DEGREES, num_bins=NUM_RAYS)
    env = DiscreteMatrixActionWrapper(
        env,
        num_bins_turn_rate=TURN_BINS,
        num_bins_speed=SPEED_BINS,
        max_turn=MAX_TURN,
        min_speed=MIN_SPEED,
        max_speed=MAX_SPEED,
    )

    model = SQIL_DQN(
        CustomDQNPolicy,
        env,
        verbose=1,
        buffer_size=100000,
        double_q=False,
        seed=37,
        exploration_initial_eps=NN_EXPLORE_RATIO,
        exploration_final_eps=NN_EXPLORE_RATIO,
    )

    obs, act = getAll(
        ["Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")],
        EXP_TURN,
        EXP_SPEED,
        env,
    )
    print("expert timesteps:", sum([len(elem) for elem in obs]))
    model.initializeExpertBuffer(obs, act)

    model.learn(
        total_timesteps=LEARN_TIMESTEPS,
        rollout_params=dic,
        rollout_timesteps=None,
        train_graph=False,
    )
    # train_plots=3000,
    # train_plots_path="Fish/Guppy/models/" + MODEL_NAME + "/",

    if not os.path.exists("Fish/Guppy/models/" + MODEL_NAME):
        os.makedirs("Fish/Guppy/models/" + MODEL_NAME)
    model.save("Fish/Guppy/models/" + MODEL_NAME + "/model")

    model_params = {
        "degrees": DEGREES,
        "num_bins_rays": NUM_RAYS,
        "turn_bins": TURN_BINS,
        "max_turn": MAX_TURN,
        "speed_bins": SPEED_BINS,
        "min_speed": MIN_SPEED,
        "max_speed": MAX_SPEED,
        "layer_norm": NN_NORM,
        "layers": NN_LAYERS,
        "training_timesteps": LEARN_TIMESTEPS,
        "exp_min_turn": EXP_TURN,
        "exp_min_dist": EXP_SPEED,
        "exp_turn_fraction": EXP_TURN_FRACTION,
        "nn_explore_ratio": NN_EXPLORE_RATIO,
        "nn_layers": NN_LAYERS,
        "nn_norm": NN_NORM,
        "perc": dic["perc"],
    }
    saveConfig("Fish/Guppy/models/" + MODEL_NAME + "/parameters.json", model_params)

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
        figsize=(len(model.rollout_values) * 6, 9),
    )
    if len(model.rollout_values) == 1:
        ax = [ax]

    dic = loadConfig(
        "Fish/Guppy/rollout/pi_"
        + str(EXP_TURN_FRACTION)
        + "_"
        + str(int(EXP_SPEED * 100 // 10))
        + str(int(EXP_SPEED * 100 % 10))
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
            + str(np.round(dic["threshhold"][PERC[i]], 2))
            + " ("
            + str(PERC[i])
            + ")",
            fontsize=10,
        )
        ax[i].legend(loc="center left")
        for a, b in zip(np.arange(len(reward[i])), reward[i]):
            ax[i].text(a, b, str(np.round(b, 2)), fontsize=6)

    ax[-1].set_xlabel("timestep of training (1000)")
    fig.suptitle("Average reward per sample in Validation Dataset", fontsize=16)
    fig.savefig("Fish/Guppy/models/" + MODEL_NAME + "/rollout.png")


def testModel(model_name, save_trajectory=True):
    MODEL_NAME = model_name
    dic = loadConfig("Fish/Guppy/models/" + MODEL_NAME + "/parameters.json")
    DEGREES, NUM_RAYS = dic["degrees"], dic["num_bins_rays"]
    TURN_BINS, SPEED_BINS = dic["turn_bins"], dic["speed_bins"]
    MAX_TURN, MIN_SPEED, MAX_SPEED = dic["max_turn"], dic["min_speed"], dic["max_speed"]

    env = TestEnvM(max_steps_per_action=200)

    # env = RayCastingWrapper(env, degrees=DEGREES, num_bins=NUM_RAYS)
    env = DiscreteMatrixActionWrapper(
        env,
        num_bins_turn_rate=TURN_BINS,
        num_bins_speed=SPEED_BINS,
        max_turn=MAX_TURN,
        min_speed=MIN_SPEED,
        max_speed=MAX_SPEED,
    )

    model = SQIL_DQN.load("Fish/Guppy/models/" + MODEL_NAME + "/model")

    # env.unwrapped.video_path = "Fish/Guppy/video"
    ray = RayCastingObject(degrees=DEGREES, num_bins=NUM_RAYS)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(ray.observation(obs), deterministic=True)
        obs2 = obs.copy()
        obs2[0] = obs[1]
        obs2[1] = obs[0]
        action2, _ = model.predict(ray.observation(obs2), deterministic=True)

        temp = obs.copy()

        obs, reward, done, _ = env.step([action, action2])

        if len(env.state_history) > 50000:
            break

        env.render()  # mode = "video"
    env.close()

    if save_trajectory:
        temp = env.state_history[0::5]
        temp = temp[0:10000]

        trajectory = np.concatenate(temp, axis=0)

        if not os.path.exists("Fish/Guppy/models/" + MODEL_NAME + "/trajectory"):
            os.makedirs("Fish/Guppy/models/" + MODEL_NAME + "/trajectory")

        df = pd.DataFrame(
            data=trajectory,
            columns=[
                "fish0_x",
                "fish0_y",
                "fish0_ori",
                "fish1_x",
                "fish1_y",
                "fish1_ori",
            ],
        )
        df.to_csv(
            "Fish/Guppy/models/" + MODEL_NAME + "/trajectory/trajectory.csv",
            index=False,
            sep=";",
        )

        convertTrajectory(
            "Fish/Guppy/models/" + MODEL_NAME + "/trajectory/trajectory.csv",
            "Fish/Guppy/models/" + MODEL_NAME + "/trajectory/trajectory_io.hdf5",
            ["robot", "robot"],
        )

        evaluate_all(
            [
                ["Fish/Guppy/models/" + MODEL_NAME + "/trajectory/trajectory_io.hdf5"],
                [
                    "Fish/Guppy/rollout/validationData/Q19A_Fri_Dec__6_14_57_14_2019_Robotracker.hdf5",
                    "Fish/Guppy/rollout/validationData/Q20I_Fri_Dec__6_15_13_09_2019_Robotracker.hdf5",
                ],
            ],
            names=["model", "validationData"],
            save_folder="Fish/Guppy/models/" + MODEL_NAME + "/trajectory/",
            consider_categories=[None, "fish"],
        )


def createRolloutFiles(dic):
    DEGREES, NUM_RAYS = dic["degrees"], dic["num_bins_rays"]
    TURN_BINS, SPEED_BINS = dic["turn_bins"], dic["speed_bins"]
    MAX_TURN, MIN_SPEED, MAX_SPEED = dic["max_turn"], dic["min_speed"], dic["max_speed"]
    EXP_TURN_FRACTION, EXP_SPEED = dic["exp_turn_fraction"], dic["exp_min_dist"]
    EXP_TURN = np.pi / EXP_TURN_FRACTION
    PERC = dic["perc"]

    env = TestEnv(max_steps_per_action=200)

    env = RayCastingWrapper(env, degrees=DEGREES, num_bins=NUM_RAYS)
    env = DiscreteMatrixActionWrapper(
        env,
        num_bins_turn_rate=TURN_BINS,
        num_bins_speed=SPEED_BINS,
        max_turn=MAX_TURN,
        min_speed=MIN_SPEED,
        max_speed=MAX_SPEED,
    )

    folder = (
        "Fish/Guppy/rollout/pi_"
        + str(EXP_TURN_FRACTION)
        + "_"
        + str(int(EXP_SPEED * 100 // 10))
        + str(int(EXP_SPEED * 100 % 10))
        + "/"
    )

    if not os.path.exists(folder[:-1]):
        os.makedirs(folder[:-1])

    """ Distribution Threshholds"""
    if not os.path.isfile(folder + "distribution_threshholds.json"):
        obs, act = getAll(
            [
                "Fish/Guppy/validationData/" + elem
                for elem in os.listdir("Fish/Guppy/validationData")
            ],
            EXP_TURN,
            EXP_SPEED,
            env,
        )
        obs = np.concatenate(obs, axis=0)
        saveDistributionThreshholds(obs, obs, folder)

    """ Allowed Actions """
    for perc in PERC:
        if not os.path.isfile(folder + "allowedActions_val_" + str(perc) + ".json"):
            max_dist = loadConfig(folder + "distribution_threshholds.json")[
                "threshhold"
            ][perc]
            saveAllowedActions(
                paths=[
                    "Fish/Guppy/validationData/" + elem
                    for elem in os.listdir("Fish/Guppy/validationData")
                ],
                exp_turn=EXP_TURN,
                exp_speed=EXP_SPEED,
                env=env,
                max_dist=max_dist,
                save_path=folder + "allowedActions_val_" + str(perc) + ".json",
            )

    """ Perfect Agent Actions """
    for perc in PERC:
        if not os.path.isfile(folder + "perfect_agent_" + str(perc) + ".json"):
            savePerfectAgentActions(
                paths_val=[
                    "Fish/Guppy/validationData/" + elem
                    for elem in os.listdir("Fish/Guppy/validationData")
                ],
                paths_tra=[
                    "Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")
                ],
                exp_turn_fraction=EXP_TURN_FRACTION,
                exp_speed=EXP_SPEED,
                env=env,
                save_path=folder + "perfect_agent_" + str(perc) + ".json",
                perc=perc,
            )


def main():
    dic = {
        "model_name": "DQN_15_02_2021_01",
        "exp_turn_fraction": 4,
        "exp_min_dist": 0.01,
        "turn_bins": 20,
        "speed_bins": 10,
        "min_speed": 0.01,
        "max_speed": 0.07,
        "max_turn": np.pi,
        "degrees": 360,
        "num_bins_rays": 36,
        "nn_layers": [256, 128],
        "nn_norm": True,
        "nn_explore_ratio": 0.5,
        "training_timesteps": 25000,
        "perc": [0],
    }
    createRolloutFiles(dic)
    trainModel(dic)
    testModel(dic["model_name"], save_trajectory=True)


if __name__ == "__main__":
    # env = TestEnv(max_steps_per_action=200)
    # env = RayCastingWrapper(env, degrees=360, num_bins=36)
    # env = DiscreteMatrixActionWrapper(
    #     env,
    #     num_bins_turn_rate=1,
    #     num_bins_speed=1,
    #     max_turn=np.pi / 50,
    #     min_speed=0.01,
    #     max_speed=0.01,
    # )
    # obs, act = getAll(
    #     ["Fish/Guppy/Data/" + elem for elem in os.listdir("Fish/Guppy/Data")],
    #     np.pi / 4,
    #     0.01,
    #     env,
    # )
    # print("expert timesteps:", sum([len(elem) for elem in obs]))
    main()
    # env = TestEnv(max_steps_per_action=200)

    # env = RayCastingWrapper(env, degrees=360, num_bins=36)
    # env = DiscreteMatrixActionWrapper(
    #     env,
    #     num_bins_turn_rate=20,
    #     num_bins_speed=10,
    #     max_turn=np.pi,
    #     min_speed=0.03,
    #     max_speed=0.1,
    # )
    # obs, act = getAll(
    #     [
    #         "Fish/Guppy/validationData/" + elem
    #         for elem in os.listdir("Fish/Guppy/validationData")
    #     ],
    #     np.pi / 4,
    #     0.07,
    #     env,
    # )
    # obs = np.concatenate(obs, axis=0)
    # fish, wall = [], []
    # for i in range(len(obs)):
    #     f, w = distObs(obs[i], obs)
    #     fish.extend(f)
    #     wall.extend(w)
    # plt.hist([fish, wall], bins=100, label=["fish_dist", "wall_dist"], density=True)
    # plt.legend()
    # plt.xlabel("distance")
    # plt.ylabel("frequency")
    # plt.title("Distance of validationData obs: fish vs wall impact")
    # plt.show()
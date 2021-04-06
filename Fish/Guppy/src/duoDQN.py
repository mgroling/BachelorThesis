import sys
import optuna
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wrappers import DiscreteActionWrapper, RayCastingWrapper
from convertData import getAll
from rolloutEv import loadConfig, saveConfig, distObs, checkAction
from hyperparameter_tuning import saveStudy, loadStudy
from _marc_guppy import MarcGuppyDuo
from main import createRolloutFiles

from conversion_scripts.convert_marc import convertTrajectory
from robofish.evaluate import evaluate_all
import robofish.trackviewer
from stable_baselines.deepq.policies import FeedForwardPolicy

sys.path.append("SQIL_DQN")
from SQIL_DQN_manager import SQIL_DQN_MANAGER

sys.path.append("Fish")
from functions import TestEnv, ApproachLeadGuppy, distanceToClosestWall

sys.path.append("gym-guppy")
from gym_guppy import GuppyEnv, TurnSpeedRobot, BoostCouzinGuppy, GlobalTargetRobot


def trainModel(
    dic,
    volatile=False,
    rollout_timesteps=None,
    rollout_determinsitic=True,
    train_plots=None,
    train_plots_path=None,
):
    env = TestEnv(time_step_s=0.04)
    env = RayCastingWrapper(
        env,
        degrees=dic["degrees"],
        num_bins=dic["num_bins_rays"],
        last_act=dic["last_act"],
    )
    env = DiscreteActionWrapper(
        env,
        num_bins_turn_rate=dic["turn_bins"],
        num_bins_speed=dic["speed_bins"],
        max_turn=dic["max_turn"],
        min_speed=dic["min_speed"],
        max_speed=dic["max_speed"],
        last_act=dic["last_act"],
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
        env,
        False,
        dic["last_act"],
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
        # reward = []
        # for i in range(len(model.rollout_values)):
        #     for value in model.rollout_values[i]:
        #         reward.append(value[0])

        # return 1 - np.mean(reward)

        trajectory = testModel(None, save_trajectory=False, model=model, dic=dic)

        mean, std = distanceToClosestWall(trajectory)

        val_mean, val_std = 0.14953857969150395, 0.09866383291652214

        return np.abs(mean - val_mean) + np.abs(std - val_std)

    else:
        model.save("Fish/Guppy/models/" + dic["model_name"])
        saveConfig("Fish/Guppy/models/" + dic["model_name"] + "/parameters.json", dic)
        # createRolloutPlots(dic, model)


def createRolloutPlots(dic, model):
    for mode in dic["mode"]:
        reward = [[] for i in range(len(model.rollout_values[mode]))]
        random_reward = [[] for i in range(len(model.rollout_values[mode]))]
        perfect_reward = [[] for i in range(len(model.rollout_values[mode]))]
        closest_reward = [[] for i in range(len(model.rollout_values[mode]))]
        for i in range(len(model.rollout_values[mode])):
            for value in model.rollout_values[mode][i]:
                reward[i].append(value[0])
                random_reward[i].append(value[1])
                perfect_reward[i].append(value[2])
                closest_reward[i].append(value[3])

        fig, ax = plt.subplots(
            len(model.rollout_values[mode]),
            1,
            figsize=(int(len(reward[0]) / 5 + 3), len(model.rollout_values[mode]) * 6),
        )
        if len(model.rollout_values[mode]) == 1:
            ax = [ax]

        dicThresh = loadConfig(
            "Fish/Guppy/rollout/tbins"
            + str(dic["turn_bins"])
            + "_sbins"
            + str(dic["speed_bins"])
            + "/distribution_threshholds_"
            + mode
            + ".json"
        )

        for i in range(len(model.rollout_values[mode])):
            x = np.arange(len(reward[i])) * 1000
            ax[i].plot(x, reward[i], label="SQIL")
            ax[i].plot(x, random_reward[i], label="random agent")
            ax[i].plot(x, perfect_reward[i], label="perfect agent")
            ax[i].plot(x, closest_reward[i], label="closest state agent")
            ax[i].set_ylim(0, 1)
            ax[i].set_ylabel("average reward")
            ax[i].set_title(
                "max_dist between obs: "
                + str(np.round(dicThresh["threshhold"][dic["perc"][i]], 2))
                + " ("
                + str(dic["perc"][i] + 1)
                + "% closest states, "
                + mode
                + " distances)",
                fontsize=10,
            )
            ax[i].legend(loc="center left")
            for a, b in zip(np.arange(len(reward[i])), reward[i]):
                ax[i].text(a * 1000, b, str(np.round(b, 2)), fontsize=6)

            ax[-1].set_xlabel("timestep of training")
            fig.suptitle("Average reward per sample in Validation Dataset", fontsize=16)
            fig.savefig(
                "Fish/Guppy/models/" + dic["model_name"] + "/rollout_" + mode + ".png"
            )
            plt.close()


def testModel(
    model_name,
    save_trajectory=True,
    partner="self",
    determinsitic=True,
    model=None,
    dic=None,
):
    if not model_name is None:
        model = SQIL_DQN_MANAGER.load("Fish/Guppy/models/" + model_name)
        dic = loadConfig("Fish/Guppy/models/" + model_name + "/parameters.json")

    class TestEnvM(GuppyEnv):
        def _reset(self):
            # set frequency of guppies to 25Hz
            self._guppy_steps_per_action = 4

            num_guppies = 2
            positions = np.random.normal(size=(num_guppies, 2), scale=0.02) + (
                0.05,
                0.05,
            )
            orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
            i = 0
            for p, o in zip(positions, orientations):
                if i == 0:
                    self._add_guppy(
                        MarcGuppyDuo(
                            model=model,
                            dic=dic,
                            deterministic=determinsitic,
                            world=self.world,
                            world_bounds=self.world_bounds,
                            position=(
                                np.random.uniform(low=-0.45, high=0.45),
                                np.random.uniform(low=-0.45, high=0.45),
                            ),
                            orientation=np.random.uniform() * 2 * np.pi,
                        )
                    )
                elif i == 1:
                    self._add_guppy(
                        ApproachLeadGuppy(
                            world=self.world,
                            world_bounds=self.world_bounds,
                            position=(
                                np.random.uniform(low=-0.45, high=0.45),
                                np.random.uniform(low=-0.45, high=0.45),
                            ),
                            orientation=np.random.uniform() * 2 * np.pi,
                        )
                    )
                if not partner == "self":
                    i += 1

    env = TestEnvM(time_step_s=0.04)

    env.unwrapped.video_path = "Fish/Guppy/video"

    obs = env.reset()
    trajectory = np.empty((10000, 6))
    for i in range(10000):
        obs = env.step(action=None)[0]
        obs[:, 2] = obs[:, 2] % (2 * np.pi)
        trajectory[i] = obs.reshape(1, 6)

        # print(trajectory[i])
        # time.sleep(0.05)

        env.render()  # mode = "video"
    env.close()

    if save_trajectory:
        if not os.path.exists("Fish/Guppy/models/" + model_name + "/trajectory"):
            os.makedirs("Fish/Guppy/models/" + model_name + "/trajectory")

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
            "Fish/Guppy/models/" + model_name + "/trajectory/trajectory.csv",
            index=False,
            sep=";",
        )

        entities = ["fish", "fish" if partner == "self" else "robot"]
        convertTrajectory(
            "Fish/Guppy/models/" + model_name + "/trajectory/trajectory.csv",
            "Fish/Guppy/models/" + model_name + "/trajectory/trajectory_io.hdf5",
            entities,
        )

        evaluate_all(
            [
                ["Fish/Guppy/models/" + model_name + "/trajectory/trajectory_io.hdf5"],
                [
                    "Fish/Guppy/validationData/CameraCapture2019-05-03T14_58_30_8108-sub_0.hdf5",
                    "Fish/Guppy/validationData/CameraCapture2019-06-20T15_35_23_672-sub_2.hdf5",
                ],
            ],
            labels=["model", "validationData"],
            save_folder="Fish/Guppy/models/" + model_name + "/trajectory/",
            predicate=[lambda e: e.category == "fish", None],
        )
    else:
        return trajectory


def objective(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int("n_layers0", 1, 4)
    layer_structure = []
    for i in range(n_layers):
        layer_structure.append(
            int(trial.suggest_loguniform("n_units_l" + str(i), 16, 512))
        )
    norm = trial.suggest_categorical("layer_norm", [True, False])
    explore_fraction = trial.suggest_uniform("explore_fraction", 0.01, 0.5)
    gamma = trial.suggest_uniform("gamma", 0.5, 0.999)
    lr = trial.suggest_loguniform("lr", 1e-6, 1e-3)
    n_batch = trial.suggest_int("n_batch", 1, 128)
    buffer_size = trial.suggest_int("buffer_size", 1e4, 1e5, 1e3)

    dic = {
        "turn_bins": 721,
        "speed_bins": 201,
        "min_speed": 0.00,
        "max_speed": 0.02,
        "max_turn": np.pi,
        "degrees": 360,
        "num_bins_rays": 36,
        "nn_layers": [layer_structure, layer_structure],
        "nn_norm": [norm, norm],
        "explore_fraction": [explore_fraction, explore_fraction],
        "training_timesteps": trial.suggest_int("learn_timesteps", 5000, 3e5, 1000),
        "sequential_timesteps": 1000,
        "perc": [0],
        "mode": ["both"],
        "gamma": [gamma, gamma],
        "lr": [lr, lr],
        "n_batch": [n_batch, n_batch],
        "buffer_size": [25000, 25000],
        "learn_partner": "self",
        "clipping_during_training": trial.suggest_categorical(
            "clipping", [True, False]
        ),
        "last_act": False,
    }
    print("Learn timesteps", dic["training_timesteps"])

    return trainModel(dic, volatile=True, rollout_timesteps=-1)


def study():
    # study = optuna.create_study()
    study = loadStudy("Fish/Guppy/studies/study_duoDQN.pkl")

    # while True:
    #     study.optimize(objective, n_trials=1)
    #     saveStudy(study, "Fish/Guppy/studies/study_duoDQN.pkl")

    print(study.best_params)


if __name__ == "__main__":
    dic = {
        "model_name": "DuoDQN_05_04_2021_03",
        "turn_bins": 721,
        "speed_bins": 201,
        "min_speed": 0.00,
        "max_speed": 0.02,
        "max_turn": np.pi,
        "degrees": 360,
        "num_bins_rays": 36,
        "nn_layers": [[256], [256]],
        "nn_norm": [False, False],
        "explore_fraction": [0.1, 0.1],
        "training_timesteps": 2e5,
        "sequential_timesteps": 1000,
        "perc": [0, 1],
        "mode": ["both", "fish", "wall"],
        "gamma": [0.99, 0.99],
        "lr": [1e-6, 1e-6],
        "n_batch": [64, 64],
        "buffer_size": [25e3, 25e3],
        "learn_partner": "self",
        "clipping_during_training": False,
        "last_act": False,
    }
    createRolloutFiles(dic)
    # trainModel(dic, volatile=False, rollout_timesteps=-1)
    # testModel(
    #     dic["model_name"], save_trajectory=True, partner="self", determinsitic=True
    # )
    # study()
    # env = TestEnv(time_step_s=0.04)
    # env = RayCastingWrapper(
    #     env,
    #     degrees=dic["degrees"],
    #     num_bins=dic["num_bins_rays"],
    #     last_act=dic["last_act"],
    # )
    # env = DiscreteActionWrapper(
    #     env,
    #     num_bins_turn_rate=dic["turn_bins"],
    #     num_bins_speed=dic["speed_bins"],
    #     max_turn=dic["max_turn"],
    #     min_speed=dic["min_speed"],
    #     max_speed=dic["max_speed"],
    #     last_act=dic["last_act"],
    # )
    # obs_, act_ = getAll(
    #     ["Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")],
    #     env,
    #     False,
    #     dic["last_act"],
    # )
    # obs_, act_ = obs_[0], act_[0]#np.concatenate(obs_), np.concatenate(act_)
    # obs = env.reset()
    # for i in range(10000):
    #     closest_obs_index = distObs(obs, obs_, env).argmin()
    #     act = act_[closest_obs_index]
    #     obs = env.step(act)[0]
    #     env.render()
    # poses1 = np.array([[0, 0, 0], [-0.3, -0.45, 1.5]])
    # poses2 = np.array([[0, 0, 0], [0.4, 0.4, 1.5]])
    # poses3 = np.array([[0, 0, 0], [0.3, 0.4, 1.5]])
    # obs1 = env.observation(poses1).copy()
    # obs2 = env.observation(poses2).copy()
    # obs3 = env.observation(poses3).copy()
    # obs2 = np.array([obs2, obs3])
    # print(distObs(obs1, obs2, env))
    # allowedActions = loadConfig(
    #     "Fish/Guppy/rollout/tbins361_sbins51/allowedActions_val_0.json"
    # )["allowed actions"][0]
    # print(checkAction(721, allowedActions, env))
    # obs, act = getAll(
    #     ["Fish/Guppy/validationData/CameraCapture2019-05-03T14_58_30_8108-sub_0.hdf5"],
    #     env,
    # )
    # obs = obs[0]
    # f, w = [], []
    # for elem in obs:
    #     f_t, w_t = distObs(elem, obs, env)
    #     f.extend(f_t)
    #     w.extend(w_t)
    # print("x should be", np.mean(f) / np.mean(w))
    # plt.hist(
    #     [f, w],
    #     bins=100,
    #     range=[0, max(max(f), max(w))],
    #     label=["fish raycasts", "wall raycasts"],
    #     density=True,
    # )
    # plt.title("Fish vs wall impact on distance between observations")
    # plt.xlabel("distance")
    # plt.ylabel("density")
    # plt.legend()
    # plt.show()
    # evaluate_all(
    #     [
    #         [
    #             "Fish/Guppy/rollout/validationData/Q12I_Fri_Dec__6_11_59_34_2019_Robotracker.hdf5",
    #             "Fish/Guppy/rollout/validationData/Q15A_Fri_Dec__6_13_47_05_2019_Robotracker.hdf5",
    #         ],
    #     ],
    #     names=["validationData"],
    #     save_folder="Fish/Guppy/random_plots/robot_valData/",
    #     predicate=[lambda e: e.category == "robot"],
    # )

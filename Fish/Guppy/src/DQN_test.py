import numpy as np
import robofish.io
import os
import sys

from wrappers import DiscreteActionWrapper, RayCastingWrapper
from convertData import getAll
from duoDQN import testModel
from rolloutEv import loadConfig, saveConfig

from stable_baselines import DQN
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.gail import ExpertDataset

sys.path.append("gym-guppy")
from gym_guppy import GuppyEnv, TurnSpeedRobot, BoostCouzinGuppy, GlobalTargetRobot

if __name__ == "__main__":
    dic = {
        "model_name": "simpleDQN_01_04_2021_01",
        "turn_bins": 721,
        "speed_bins": 201,
        "min_speed": 0.00,
        "max_speed": 0.02,
        "max_turn": np.pi,
        "degrees": 360,
        "num_bins_rays": 36,
        "nn_layers": [[256], [256]],
        "nn_norm": [False, False],
        "explore_fraction": [0.2, 0.2],
        "training_timesteps": 5e4,
        "sequential_timesteps": 1000,
        "perc": [0, 1],
        "mode": ["both", "fish", "wall"],
        "gamma": [0.99, 0.99],
        "lr": [1e-5, 1e-5],
        "n_batch": [32, 32],
        "buffer_size": [100000, 100000],
        "learn_partner": "self",
        "clipping_during_training": False,
        "last_act": True,
    }

    class TestEnvM(GuppyEnv):
        def _reset(self):
            self._add_robot(
                TurnSpeedRobot(
                    world=self.world,
                    world_bounds=self.world_bounds,
                    position=(
                        np.random.uniform(low=-0.45, high=0.45),
                        np.random.uniform(low=-0.45, high=0.45),
                    ),
                    orientation=np.random.uniform() * 2 * np.pi,
                )
            )

    env = TestEnvM(time_step_s=0.04)
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

    obs, act = getAll(
        ["Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")],
        env,
        False,
        dic["last_act"],
    )
    obs, act = np.concatenate(obs), np.concatenate(act)

    model_turn = DQN(
        MlpPolicy,
        env,
        verbose=1,
        _init_setup_model=False,
    )
    model_speed = DQN(
        MlpPolicy,
        env,
        verbose=1,
        _init_setup_model=False,
    )
    model_turn.action_space = env.action_space.space[0]
    model_speed.action_space = env.action_space.space[1]
    model_turn.setup_model()
    model_speed.setup_model()

    numpy_dic_turn = {
        "actions": act[:, 0].reshape((len(act), 1)),
        "obs": obs,
        "rewards": np.ones((len(act), 1)),
        "episode_returns": np.array([len(act)]),
        "episode_starts": np.zeros((len(act), 1)),
    }
    numpy_dic_speed = {
        "actions": act[:, 1].reshape((len(act), 1)),
        "obs": obs,
        "rewards": np.ones((len(act), 1)),
        "episode_returns": np.array([len(act)]),
        "episode_starts": np.zeros((len(act), 1)),
    }

    dataset_turn = ExpertDataset(
        traj_data=numpy_dic_turn, batch_size=128, traj_limitation=-1
    )
    dataset_speed = ExpertDataset(
        traj_data=numpy_dic_speed, batch_size=128, traj_limitation=-1
    )

    model_turn.pretrain(dataset_turn, n_epochs=100)
    model_speed.pretrain(dataset_speed, n_epochs=100)

    if not os.path.exists("Fish/Guppy/models/" + dic["model_name"]):
        os.makedirs("Fish/Guppy/models/" + dic["model_name"])
    model_turn.save("Fish/Guppy/models/" + dic["model_name"] + "/model_turn")
    model_speed.save("Fish/Guppy/models/" + dic["model_name"] + "/model_speed")
    saveConfig("Fish/Guppy/models/" + dic["model_name"] + "/parameters.json", dic)

    testModel(dic["model_name"])
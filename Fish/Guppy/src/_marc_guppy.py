import numpy as np
import json
import math
import sys

from stable_baselines import DQN

from scipy.spatial import cKDTree

sys.path.append("gym-guppy")
from gym_guppy import (
    Guppy,
    TurnBoostAgent,
    TurnSpeedAgent,
    VelocityControlledAgent,
    GlobalTargetRobot,
)
from gym_guppy.tools.math import ray_casting_walls, compute_dist_bins


class MarcGuppy(Guppy, TurnSpeedAgent):
    def __init__(self, model_path, **kwargs):
        super().__init__(**kwargs)
        self._model = DQN.load(model_path + "model")
        dic = loadConfig(model_path + "parameters.json")

        degrees = dic["degrees"]
        num_bins = dic["num_bins_rays"]
        num_bins_turn_rate = dic["turn_bins"]
        num_bins_speed = dic["speed_bins"]
        min_turn_rate = dic["min_turn"]
        max_turn_rate = dic["max_turn"]
        min_speed = dic["min_speed"]
        max_speed = dic["max_speed"]

        self._turn_rate_bins = np.linspace(
            min_turn_rate, max_turn_rate, num_bins_turn_rate
        )
        self._speed_bins = np.linspace(min_speed, max_speed, num_bins_speed)

        self.world_bounds = [np.array([-0.5, -0.5]), np.array([0.5, 0.5])]
        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.sector_bounds = np.linspace(-self.cutoff, self.cutoff, num_bins + 1)
        self.ray_directions = np.linspace(-self.cutoff, self.cutoff, num_bins)
        self.obs_placeholder = np.empty((2, num_bins))

    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        self.obs_placeholder[0] = compute_dist_bins(
            state[0], state[1:], self.sector_bounds, self.diagonal * 1.1
        )
        self.obs_placeholder[1] = ray_casting_walls(
            state[0], self.world_bounds, self.ray_directions, self.diagonal * 1.1
        )

        action, _ = self._model.predict(self.obs_placeholder, deterministic=True)

        turn_rate = math.floor(action / len(self._speed_bins))
        speed = action % len(self._speed_bins)
        turn, speed = self._turn_rate_bins[turn_rate], self._speed_bins[speed]

        self.turn = turn * 0.1
        self.speed = speed * 1.5


def saveConfig(path, dic):
    with open(path, "w+") as f:
        json.dump(dic, f)


def loadConfig(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    dic = {
        "degrees": 360,
        "num_bins_rays": 72,
        "turn_bins": 20,
        "min_turn": -np.pi / 2,
        "max_turn": np.pi / 2,
        "speed_bins": 10,
        "min_speed": 0.03,
        "max_speed": 0.1,
        "layer_norm": True,
        "layers": [256, 128],
        "training_timesteps": 75000,
        "exp_min_turn": np.pi / 4,
        "exp_min_dist": 0.07,
        "explore_ratio": 0.5,
    }
    saveConfig("Fish/Guppy/models/DQN_22_12_2020_03.json", dic)


if __name__ == "__main__":
    main()
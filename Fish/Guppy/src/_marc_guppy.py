import numpy as np
import json
import math
import sys
import logging

from stable_baselines import DQN

from scipy.spatial import cKDTree

sys.path.append("gym-guppy")
from gym_guppy import (
    TurnSpeedGuppy,
    TurnSpeedRobot,
)
from gym_guppy.tools.math import ray_casting_walls, compute_dist_bins

from stable_baselines import DQN


class MarcGuppy(TurnSpeedGuppy, TurnSpeedRobot):
    _frequency = 20

    def __init__(self, model_path=None, model=None, dic=None, **kwargs):
        super(MarcGuppy, self).__init__(**kwargs)
        if model_path is None and (model is None or dic is None):
            logging.exception(
                "Either model_path or model and dic have to be specified."
            )
        elif not model_path is None and not model is None:
            logging.exception("Specify either model_path or model, not both.")

        if model is None:
            self._model = DQN.load(model_path + "model")
        else:
            self._model = model
        if dic is None:
            dic = loadConfig(model_path + "parameters.json")

        degrees = dic["degrees"]
        num_bins = dic["num_bins_rays"]
        num_bins_turn_rate = dic["turn_bins"]
        num_bins_speed = dic["speed_bins"]
        min_turn_rate = -dic["max_turn"]
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
        if not (len(state) == 1 or self.id == 0):
            temp = state.copy()
            state[0] = state[self.id]
            state[self.id] = temp[0]
        self.obs_placeholder[1] = ray_casting_walls(
            state[0], self.world_bounds, self.ray_directions, self.diagonal
        )
        if len(state) == 1:
            self.obs_placeholder[0] = np.zeros((len(self.obs_placeholder[1])))
        else:
            self.obs_placeholder[0] = compute_dist_bins(
                state[0], state[1:], self.sector_bounds, self.diagonal
            )

        action, _ = self._model.predict(self.obs_placeholder, deterministic=True)

        turn_rate = math.floor(action / len(self._speed_bins))
        speed = action % len(self._speed_bins)
        turn, speed = self._turn_rate_bins[turn_rate], self._speed_bins[speed]

        self.turn = turn
        self.speed = speed * self._frequency


class MarcGuppyDuo(TurnSpeedGuppy, TurnSpeedRobot):
    _frequency = 25

    def __init__(
        self, model_path=None, model=None, dic=None, deterministic=True, **kwargs
    ):
        super(MarcGuppyDuo, self).__init__(**kwargs)
        if model_path is None and model is None and dic is None:
            logging.exception("Either model_path or model and dic have to be given.")
        elif not model_path is None and not model is None:
            logging.exception("Specify either model_path or model, not both.")

        if model is None:
            self._model = Model(model_path + "model")
        else:
            self._model = model
        if dic is None:
            dic = loadConfig(model_path + "parameters.json")

        degrees = dic["degrees"]
        num_bins = dic["num_bins_rays"]
        num_bins_turn_rate = dic["turn_bins"]
        num_bins_speed = dic["speed_bins"]
        min_turn_rate = -dic["max_turn"]
        max_turn_rate = dic["max_turn"]
        min_speed = dic["min_speed"]
        max_speed = dic["max_speed"]

        self.deterministic = deterministic
        self.stuck_counter = 0
        self.last_act = dic["last_act"]

        self._turn_rate_bins = np.linspace(
            min_turn_rate, max_turn_rate, num_bins_turn_rate
        )
        self._speed_bins = np.linspace(min_speed, max_speed, num_bins_speed)

        self.world_bounds = [np.array([-0.5, -0.5]), np.array([0.5, 0.5])]
        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.sector_bounds = np.linspace(-self.cutoff, self.cutoff, num_bins + 1)
        self.ray_directions = np.linspace(-self.cutoff, self.cutoff, num_bins)
        if self.last_act:
            self.last_action = [0, 0]
            self.obs_placeholder = np.empty((num_bins * 2 + 2,))
            self.num_bins = num_bins
        else:
            self.obs_placeholder = np.empty((2, num_bins))

    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        if not (len(state) == 1 or self.id == 0):
            temp = state.copy()
            state[0] = state[self.id]
            state[self.id] = temp[0]
        if self.last_act:
            if len(state) == 1:
                self.obs_placeholder[: self.num_bins] = 0
            else:
                self.obs_placeholder[: self.num_bins] = compute_dist_bins(
                    state[0], state[1:], self.sector_bounds, self.diagonal
                )
            self.obs_placeholder[self.num_bins : -2] = ray_casting_walls(
                state[0], self.world_bounds, self.ray_directions, self.diagonal
            )
            self.obs_placeholder[-2:] = np.array(self.last_action)
        else:
            self.obs_placeholder[1] = ray_casting_walls(
                state[0], self.world_bounds, self.ray_directions, self.diagonal
            )
            if len(state) == 1:
                self.obs_placeholder[0] = np.zeros((len(self.obs_placeholder[1])))
            else:
                self.obs_placeholder[0] = compute_dist_bins(
                    state[0], state[1:], self.sector_bounds, self.diagonal
                )

        action, _ = self._model.predict(
            self.obs_placeholder, deterministic=self.deterministic
        )

        turn, speed = self._turn_rate_bins[action[0]], self._speed_bins[action[1]]
        self.last_action = [
            turn / np.pi,
            speed / self._speed_bins[-1],
        ]

        self.turn = turn
        if speed == 0:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter == 20:
            self.speed = 0.01 * self._frequency
            self.stuck_counter = 0
        else:
            self.speed = speed * self._frequency

    def step(self, time_step):
        self.set_angular_velocity(0)
        if self.turn:
            self._body.angle += self.turn
            self.turn = 0
            self.__turn = None
        if self.speed:
            self.set_linear_velocity([self.speed, 0.0], local=True)
            self.__speed = None


class Model:
    def __init__(self, path):
        self._models = [None, None]
        self._models[0] = DQN.load(path + "/model_turn")
        self._models[1] = DQN.load(path + "/model_speed")

    def predict(self, observation, deterministic=True):
        turn, _ = self._models[0].predict(observation, deterministic=deterministic)
        speed, _ = self._models[1].predict(observation, deterministic=deterministic)

        return [turn, speed], None


def saveConfig(path, dic):
    with open(path, "w+") as f:
        json.dump(dic, f)


def loadConfig(path):
    with open(path, "r") as f:
        return json.load(f)
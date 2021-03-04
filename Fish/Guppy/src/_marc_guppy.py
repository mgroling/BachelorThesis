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


class MarcGuppy(TurnSpeedGuppy, TurnSpeedRobot):
    _frequency = 20

    def __init__(self, model_path=None, model=None, dic=None, **kwargs):
        super(MarcGuppy, self).__init__(**kwargs)
        if model_path is None and model is None and dic is None:
            logging.exception("Either model_path or model and dic have to be given.")
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
            state[0], self.world_bounds, self.ray_directions, self.diagonal * 1.1
        )
        if len(state) == 1:
            self.obs_placeholder[0] = np.zeros((len(self.obs_placeholder[1])))
        else:
            self.obs_placeholder[0] = compute_dist_bins(
                state[0], state[1:], self.sector_bounds, self.diagonal * 1.1
            )

        action, _ = self._model.predict(self.obs_placeholder, deterministic=True)

        turn_rate = math.floor(action / len(self._speed_bins))
        speed = action % len(self._speed_bins)
        turn, speed = self._turn_rate_bins[turn_rate], self._speed_bins[speed]

        # # check if action would lead fish outside of tank
        # global_turn = state[0][2] + turn
        # coords = [
        #     state[0][0] + speed * np.cos(global_turn),
        #     state[0][1] + speed * np.sin(global_turn),
        # ]
        # changed = False
        # if coords[0] < -0.49:
        #     coords[0] = -0.47
        #     changed = True
        # elif coords[0] > 0.49:
        #     coords[0] = 0.47
        #     changed = True

        # if coords[1] < -0.49:
        #     coords[1] = -0.47
        #     changed = True
        # elif coords[1] > 0.49:
        #     coords[1] = 0.47
        #     changed = True

        # if changed:
        #     speed = np.sqrt(
        #         np.power(coords[0] - state[0][0], 2)
        #         + np.power(coords[1] - state[0][1], 2)
        #     )
        #     temp_x, temp_y = (
        #         state[0][0] + 0.1 * np.cos(state[0][2]),
        #         state[0][1] + 0.1 * np.sin(state[0][2]),
        #     )
        #     look_vec = temp_x - state[0][0], temp_y - state[0][1]
        #     move_vec = coords[0] - state[0][0], coords[1] - state[0][1]
        #     turn = getAngle(look_vec, move_vec, mode="radians")
        #     if turn > np.pi:
        #         turn = turn - 2 * np.pi

        self.turn = turn
        self.speed = speed * self._frequency


def saveConfig(path, dic):
    with open(path, "w+") as f:
        json.dump(dic, f)


def loadConfig(path):
    with open(path, "r") as f:
        return json.load(f)


def getAngle(vector1, vector2, mode="degrees"):
    """
    Given 2 vectors, in the form of tuples (x1, y1) this will return an angle in degrees, if not specfified further.
    If mode is anything else than "degrees", it will return angle in radians
    30° on the right are actually 30° and 30° on the left are 330° (relative to vector1).
    """
    # Initialize an orthogonal vector, that points to the right of your first vector.
    orth_vector1 = (vector1[1], -vector1[0])

    # Calculate angle between vector1 and vector2 (however this will only yield angles between 0° and 180° (the shorter one))
    temp = np.dot(vector1, vector2) / np.linalg.norm(vector1) / np.linalg.norm(vector2)
    angle = np.degrees(np.arccos(np.clip(temp, -1, 1)))

    # Calculate angle between orth_vector1 and vector2 (so we can get a degree between 0° and 360°)
    temp_orth = (
        np.dot(orth_vector1, vector2)
        / np.linalg.norm(orth_vector1)
        / np.linalg.norm(vector2)
    )
    angle_orth = np.degrees(np.arccos(np.clip(temp_orth, -1, 1)))

    # It is on the left side of our vector
    if angle_orth < 90:
        angle = 360 - angle

    return angle if mode == "degrees" else math.radians(angle)
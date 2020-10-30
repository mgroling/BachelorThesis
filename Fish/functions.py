import numpy as np
import matplotlib.pyplot as plt
import math
import sys

from stable_baselines.deepq.policies import FeedForwardPolicy

sys.path.append("gym-guppy")
from gym_guppy import VariableStepGuppyEnv, PolarCoordinateTargetRobot, BoostCouzinGuppy, GlobalTargetRobot

from _marc_guppy import MarcGuppy

class TestEnv(VariableStepGuppyEnv):
    def __init__(self, *, min_steps_per_action=0, max_steps_per_action=None, **kwargs):
        super().__init__(**kwargs)

        self._min_steps_per_action = min_steps_per_action
        self._max_steps_per_action = max_steps_per_action
        self._step_logger = []
        self.enable_step_logging = True
        self._reset_step_logger = False
        self.too_many_steps = 0
        self.state_history = []

    def _reset(self):
        controller_params = {
            "ori_ctrl_params": {
                "p": 1.2,
                "i": 0.0,
                "d": 0.,
                "speed": 0.2,
                "slope": 0.75,
            },
            "fwd_ctrl_params": {
                "p": 1.0,
                "i": 0.0,
                "d": 1.0,
                "speed": 0.2,
                "slope": 0.5,
                "ori_gate_slope": 3.0,
            },
        }

        self._add_robot(GlobalTargetRobot(world=self.world,
                                          world_bounds=self.world_bounds,
                                          position=(0, 0),
                                          orientation=0,
                                          ctrl_params=controller_params))

        num_guppies = 1
        positions = np.random.normal(size=(num_guppies, 2), scale=.02) + (.05, .05)
        orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
        for p, o in zip(positions, orientations):
            self._add_guppy(BoostCouzinGuppy(
                world=self.world,
                world_bounds=self.world_bounds,
                position=p, orientation=o
            ))

    @property
    def _max_steps_per_action_reached(self):
        if self._max_steps_per_action is None:
            return False
        if self._action_steps >= self._max_steps_per_action:
            self.too_many_steps += 1
            return True
        else:
            return False

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[256, 128],
                                           layer_norm=True,
                                           feature_extraction="mlp")

def getAngle(vector1, vector2, mode = "degrees"):
    """
    Given 2 vectors, in the form of tuples (x1, y1) this will return an angle in degrees, if not specfified further.
    If mode is anything else than "degrees", it will return angle in radians
    30° on the right are actually 30° and 30° on the left are 330° (relative to vector1).
    """
    #Initialize an orthogonal vector, that points to the right of your first vector.
    orth_vector1 = (vector1[1], -vector1[0])

    #Calculate angle between vector1 and vector2 (however this will only yield angles between 0° and 180° (the shorter one))
    temp = np.dot(vector1, vector2)/np.linalg.norm(vector1)/np.linalg.norm(vector2)
    angle = np.degrees(np.arccos(np.clip(temp, -1, 1)))

    #Calculate angle between orth_vector1 and vector2 (so we can get a degree between 0° and 360°)
    temp_orth = np.dot(orth_vector1, vector2)/np.linalg.norm(orth_vector1)/np.linalg.norm(vector2)
    angle_orth = np.degrees(np.arccos(np.clip(temp_orth, -1, 1)))

    #It is on the left side of our vector
    if angle_orth < 90:
        angle = 360 - angle

    return angle if mode == "degrees" else math.radians(angle)

class TestEnvM(VariableStepGuppyEnv):
    def __init__(self, *, min_steps_per_action=0, max_steps_per_action=None, **kwargs):
        super().__init__(**kwargs)

        self._min_steps_per_action = min_steps_per_action
        self._max_steps_per_action = max_steps_per_action
        self._step_logger = []
        self.enable_step_logging = True
        self._reset_step_logger = False
        self.too_many_steps = 0
        self.state_history = []

    def _reset(self):
        controller_params = {
            "ori_ctrl_params": {
                "p": 1.2,
                "i": 0.0,
                "d": 0.,
                "speed": 0.2,
                "slope": 0.75,
            },
            "fwd_ctrl_params": {
                "p": 1.0,
                "i": 0.0,
                "d": 1.0,
                "speed": 0.2,
                "slope": 0.5,
                "ori_gate_slope": 3.0,
            },
        }

        self._add_robot(GlobalTargetRobot(world=self.world,
                                          world_bounds=self.world_bounds,
                                          position=(0, 0),
                                          orientation=0,
                                          ctrl_params=controller_params))

        num_guppies = 1
        positions = np.random.normal(size=(num_guppies, 2), scale=.02) + (.05, .05)
        orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
        for p, o in zip(positions, orientations):
            self._add_guppy(MarcGuppy("Fish/Guppy/models/DQN_256_128_25k_pi-4_7-100_03_10_20t_10s_norm",
                world=self.world,
                world_bounds=self.world_bounds,
                position=p, orientation=o
            ))

    @property
    def _max_steps_per_action_reached(self):
        if self._max_steps_per_action is None:
            return False
        if self._action_steps >= self._max_steps_per_action:
            self.too_many_steps += 1
            return True
        else:
            return False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import os

from robofish.evaluate import evaluate_all
from conversion_scripts.convert_marc import convertTrajectory

sys.path.append("Fish")
from wrappers import RayCastingObject, DiscreteMatrixActionWrapper

sys.path.append("gym-guppy")
from gym_guppy import (
    VariableStepGuppyEnv,
    GuppyEnv,
    PolarCoordinateTargetRobot,
    BoostCouzinGuppy,
    GlobalTargetRobot,
    TurnBoostRobot,
)

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
                "d": 0.0,
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

        self._add_robot(
            GlobalTargetRobot(
                world=self.world,
                world_bounds=self.world_bounds,
                position=(0, 0),
                orientation=0,
                ctrl_params=controller_params,
            )
        )

        num_guppies = 1
        positions = np.random.normal(size=(num_guppies, 2), scale=0.02) + (0.05, 0.05)
        orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
        for p, o in zip(positions, orientations):
            self._add_guppy(
                BoostCouzinGuppy(
                    world=self.world,
                    world_bounds=self.world_bounds,
                    position=p,
                    orientation=o,
                )
            )

    @property
    def _max_steps_per_action_reached(self):
        if self._max_steps_per_action is None:
            return False
        if self._action_steps >= self._max_steps_per_action:
            self.too_many_steps += 1
            return True
        else:
            return False


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


class TestEnvM(VariableStepGuppyEnv):
    def __init__(
        self,
        *,
        model_path="Fish/Guppy/models/DQN_29_01_2021_01/",
        min_steps_per_action=0,
        max_steps_per_action=None,
        **kwargs
    ):
        self.model_path = model_path
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
                "d": 0.0,
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

        self._add_robot(
            GlobalTargetRobot(
                world=self.world,
                world_bounds=self.world_bounds,
                position=(0, 0),  # np.random.rand() - 0.5, np.random.rand() - 0.5
                orientation=0,
                ctrl_params=controller_params,
            )
        )

        self._add_robot(
            GlobalTargetRobot(
                world=self.world,
                world_bounds=self.world_bounds,
                position=(0.1, 0.1),
                orientation=0,
                ctrl_params=controller_params,
            )
        )

        # num_guppies = 1
        # positions = np.random.normal(size=(num_guppies, 2), scale=0.3)
        # orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
        # for p, o in zip(positions, orientations):
        #     self._add_guppy(
        #         MarcGuppy(
        #             self.model_path,
        #             world=self.world,
        #             world_bounds=self.world_bounds,
        #             position=p,
        #             orientation=o,
        #         )
        #     )

    @property
    def _max_steps_per_action_reached(self):
        if self._max_steps_per_action is None:
            return False
        if self._action_steps >= self._max_steps_per_action:
            self.too_many_steps += 1
            return True
        else:
            return False


def testModel_(model, path, dic, timestep):
    if path is None:
        print("path is missing")
        return

    TURN_BINS, SPEED_BINS = dic["turn_bins"], dic["speed_bins"]
    MIN_SPEED, MAX_SPEED, MAX_TURN = dic["min_speed"], dic["max_speed"], dic["max_turn"]
    DEGREES, NUM_RAYS = dic["degrees"], dic["num_bins_rays"]

    env = TestEnvM(max_steps_per_action=200)

    env = DiscreteMatrixActionWrapper(
        env,
        num_bins_turn_rate=TURN_BINS,
        num_bins_speed=SPEED_BINS,
        max_turn=MAX_TURN,
        min_speed=MIN_SPEED,
        max_speed=MAX_SPEED,
    )

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
    env.close()

    temp = env.state_history[0::5]
    temp = temp[0:10000]

    trajectory = np.concatenate(temp, axis=0)

    if not os.path.exists(path + "training_plots/timestep_" + str(timestep)):
        os.makedirs(path + "training_plots/timestep_" + str(timestep))

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
        path + "training_plots/timestep_" + str(timestep) + "/trajectory.csv",
        index=False,
        sep=";",
    )

    convertTrajectory(
        path + "training_plots/timestep_" + str(timestep) + "/trajectory.csv",
        path + "training_plots/timestep_" + str(timestep) + "/trajectory_io.hdf5",
        ["robot", "robot"],
    )

    evaluate_all(
        [
            [path + "training_plots/timestep_" + str(timestep) + "/trajectory_io.hdf5"],
            [
                "Fish/Guppy/rollout/validationData/Q19A_Fri_Dec__6_14_57_14_2019_Robotracker.hdf5",
                "Fish/Guppy/rollout/validationData/Q20I_Fri_Dec__6_15_13_09_2019_Robotracker.hdf5",
            ],
        ],
        names=["model", "validationData"],
        save_folder=path + "training_plots/timestep_" + str(timestep) + "/",
        consider_categories=[None, "fish"],
    )


def distance(x_1, y_1, x_2, y_2):
    return math.sqrt(math.pow(x_1 - x_2, 2) + math.pow(y_1 - y_2, 2))

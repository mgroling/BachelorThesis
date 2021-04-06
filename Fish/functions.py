import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import os
import gym

from robofish.evaluate import *
from conversion_scripts.convert_marc import convertTrajectory
from scipy.spatial import cKDTree

sys.path.append("gym-guppy")
from gym_guppy import (
    GuppyEnv,
    TurnSpeedRobot,
    BoostCouzinGuppy,
    TurnSpeedGuppy,
    GlobalTargetRobot,
)

# from _marc_guppy import MarcGuppy


class ApproachLeadGuppy(TurnSpeedGuppy, TurnSpeedRobot):
    """ Approach Lead Guppy for one partner fish, simplified version of https://arxiv.org/abs/2009.06633 """

    _frequency = 20

    def __init__(self, **kwargs):
        super(ApproachLeadGuppy, self).__init__(**kwargs)
        self.mode = "approach"
        self.target_points = np.array(
            [[0.4, -0.4], [0.4, 0.4], [-0.4, 0.4], [-0.4, -0.4]]
        )
        self.current_target = 0
        # speed is 1 cm/timestep
        self.constant_speed = 0.005
        # max turn is 5°/timestep
        self.max_turn = 0.0174533

    def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
        if not (len(state) == 1 or self.id == 0):
            temp = state.copy()
            state[0] = state[self.id]
            state[self.id] = temp[0]
        state[0][2] = state[0][2] % (2 * np.pi)
        diff = np.diff(state, axis=0)
        iid = np.linalg.norm(diff[0, :2])
        # close enough to partner fish, go into lead mode
        if iid < 0.12 and self.mode == "approach":
            self.mode = "lead"
            pos = np.array(state[0][:2])
            distancesToTargetPoints = np.linalg.norm(
                pos[np.newaxis] - self.target_points, axis=1
            )
            if (self.current_target - 1) % 4 != distancesToTargetPoints.argmin():
                self.current_target = distancesToTargetPoints.argmin()
        # too far away from partner fish, go into approach mode
        elif iid > 0.28 and self.mode == "lead":
            self.mode = "approach"
        if self.mode == "approach":
            angle = np.arctan2(diff[:, 1], diff[:, 0])
            turn = (angle[0] - state[0][2]) % (2 * np.pi)
            turn = np.where(turn > np.pi, turn - 2 * np.pi, turn)
            turn = np.where(turn < -np.pi, turn + 2 * np.pi, turn)
            self.turn = np.clip(turn, -self.max_turn, self.max_turn)
            self.speed = self.constant_speed * self._frequency
        elif self.mode == "lead":
            pos = np.array(state[0][:2])
            # check if we are close enough to current target, if yes select next target point, else keep going to current target
            if np.linalg.norm(pos - self.target_points[self.current_target]) < 0.015:
                self.current_target = (self.current_target + 1) % 4
            diff = self.target_points[self.current_target] - pos
            angle = np.arctan2(diff[1], diff[0])
            turn = (angle - state[0][2]) % (2 * np.pi)
            turn = np.where(turn > np.pi, turn - 2 * np.pi, turn)
            turn = np.where(turn < -np.pi, turn + 2 * np.pi, turn)
            self.turn = np.clip(turn, -self.max_turn, self.max_turn)
            self.speed = self.constant_speed * self._frequency

    def step(self, time_step):
        self.set_angular_velocity(0)
        if self.turn:
            self._body.angle += self.turn
            self.__turn = None
        if self.speed:
            self.set_linear_velocity([self.speed, 0.0], local=True)
            self.__speed = None


class TestEnv(GuppyEnv):
    def _reset(self):
        # set frequency to 20Hz
        self._guppy_steps_per_action = 5

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

        num_guppies = 1
        positions = np.random.normal(size=(num_guppies, 2), scale=0.02) + (0.05, 0.05)
        orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
        for p, o in zip(positions, orientations):
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


def testModel_(model, path, dic, timestep):
    if path is None:
        print("path is missing")
        return

    TURN_BINS, SPEED_BINS = dic["turn_bins"], dic["speed_bins"]
    MIN_SPEED, MAX_SPEED, MAX_TURN = dic["min_speed"], dic["max_speed"], dic["max_turn"]
    DEGREES, NUM_RAYS = dic["degrees"], dic["num_bins_rays"]

    class TestEnvM(GuppyEnv):
        def _reset(self):
            # set frequency to 20Hz
            self._guppy_steps_per_action = 5

            num_guppies = 2
            positions = np.random.normal(size=(num_guppies, 2), scale=0.02) + (
                0.05,
                0.05,
            )
            orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
            for p, o in zip(positions, orientations):
                self._add_guppy(
                    MarcGuppy(
                        model=model,
                        dic=dic,
                        world=self.world,
                        world_bounds=self.world_bounds,
                        position=p,
                        orientation=o,
                    )
                )

    env = TestEnvM(steps_per_robot_action=5)

    NUM_STEPS = 2500
    trajectory = np.stack([env.step(action=None)[0] for _ in range(NUM_STEPS)]).reshape(
        (NUM_STEPS, 6)
    )

    if not os.path.exists(path + "training_plots/trajec"):
        os.makedirs(path + "training_plots/trajec")

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
        path + "training_plots/trajec/timestep_" + str(timestep) + ".csv",
        index=False,
        sep=";",
    )

    convertTrajectory(
        path + "training_plots/trajec/timestep_" + str(timestep) + ".csv",
        path + "training_plots/trajec/timestep_" + str(timestep) + "_io.hdf5",
        ["robot", "robot"],
    )

    # create folders for plots
    plot_kinds = [
        "distanceToWall",
        "follow_iid",
        "orientation",
        "posVec",
        "relOrientation",
        "speed",
        "tankpositions",
        "trajectories",
        "turn",
    ]
    plot_functions = [
        evaluate_distanceToWall,
        evaluate_follow_iid,
        evaluate_orientation,
        evaluate_positionVec,
        evaluate_relativeOrientation,
        evaluate_speed,
        evaluate_tankpositions,
        evaluate_trajectories,
        evaluate_turn,
    ]
    for kind in plot_kinds:
        if not os.path.exists(path + "training_plots/" + kind):
            os.makedirs(path + "training_plots/" + kind)

    in_path = [
        [path + "training_plots/trajec/timestep_" + str(timestep) + "_io.hdf5"],
        [
            "Fish/Guppy/rollout/validationData/Q19A_Fri_Dec__6_14_57_14_2019_Robotracker.hdf5",
            "Fish/Guppy/rollout/validationData/Q20I_Fri_Dec__6_15_13_09_2019_Robotracker.hdf5",
        ],
    ]
    in_names = ["model", "validationData"]
    in_save_folder = path + "training_plots/"
    in_consider_cats = [None, "fish"]

    for i, plot_function in enumerate(plot_functions):
        plot_function(
            paths=in_path,
            names=in_names,
            save_path=in_save_folder
            + plot_kinds[i]
            + "/timestep_"
            + str(timestep)
            + ".png",
            consider_categories=in_consider_cats,
        )


def distance(x_1, y_1, x_2, y_2):
    return math.sqrt(math.pow(x_1 - x_2, 2) + math.pow(y_1 - y_2, 2))


def distanceToClosestWall(trajectory):
    wall_lines = [
        (-0.5, -0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5, 0.5),
    ]
    distances = np.empty((trajectory.shape[1] // 3 * trajectory.shape[0]))

    for i in range(trajectory.shape[1] // 3):
        dist = []
        for wall in wall_lines:
            dist.append(
                calculate_distLinePoint(
                    wall[0],
                    wall[1],
                    wall[2],
                    wall[3],
                    trajectory[:, [i * 3, i * 3 + 1]],
                )
            )
        distances[i * trajectory.shape[0] : (i + 1) * trajectory.shape[0]] = np.stack(
            dist
        ).min(axis=0)

    return np.mean(distances), np.std(distances)
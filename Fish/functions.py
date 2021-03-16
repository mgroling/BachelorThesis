import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import os
import gym

from robofish.evaluate import *
from conversion_scripts.convert_marc import convertTrajectory

sys.path.append("Fish")
from wrappers import RayCastingObject, DiscreteMatrixActionWrapper

sys.path.append("gym-guppy")
from gym_guppy import (
    GuppyEnv,
    TurnSpeedRobot,
    BoostCouzinGuppy,
)

from _marc_guppy import MarcGuppy


class TestEnv(GuppyEnv):
    def _reset(self):
        # set frequency to 20Hz
        self._guppy_steps_per_action = 5

        self._add_robot(
            TurnSpeedRobot(
                world=self.world,
                world_bounds=self.world_bounds,
                position=(0.3, 0.3),
                orientation=1.57,
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
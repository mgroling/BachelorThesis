import sys
import pandas as pd
import numpy as np
import os

from wrappers import DiscreteMatrixActionWrapper, RayCastingWrapper

sys.path.append("I:/Code/io/src")
import robofish.io

sys.path.append("Fish")
from functions import *
from evaluation import normalize_series


def convertTrajectory(path):
    ar = pd.read_csv(path, sep=";").to_numpy()
    new = robofish.io.File(world_size=[100, 100])

    # convert orientation from radians to normalized orientation vector
    ori_vec = normalize_series(np.array([np.cos(ar[:, 2]), np.sin(ar[:, 2])]).T)
    # get new trajectory with x, y, ori_x, ori_y
    temp = np.append(ar[:, [0, 1]], ori_vec, axis=1)
    # convert x,y from m to cm
    temp[:, [0, 1]] = temp[:, [0, 1]] * 100

    new.create_single_entity(type_="fish", name="fish_1", poses=temp, monotonic_step=20)

    # convert orientation from radians to normalized orientation vector
    ori_vec = normalize_series(np.array([np.cos(ar[:, 5]), np.sin(ar[:, 5])]).T)
    # get new trajectory with x, y, ori_x, ori_y
    temp = np.append(ar[:, [3, 4]], ori_vec, axis=1)
    # convert x,y from m to cm
    temp[:, [0, 1]] = temp[:, [0, 1]] * 100

    new.create_single_entity(type_="fish", name="fish_2", poses=temp, monotonic_step=20)
    new.validate()
    new.save("Fish/Guppy/io/DQN_22_12_2020_03_2models_det.hdf5")


convertTrajectory(
    "Fish/Guppy/trajectories/DQN_22_12_2020_03/DQN_22_12_2020_03_2models_det.csv"
)

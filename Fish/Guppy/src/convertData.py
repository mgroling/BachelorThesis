import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from conversion_scripts.convert_marc import convertTrajectory

sys.path.append("gym-guppy")
from gym_guppy import GuppyEnv
from gym_guppy.wrappers.observation_wrapper import RayCastingWrapper
from wrappers import DiscreteMatrixActionWrapper

sys.path.append("Fish")
from functions import TestEnv


def getExpert(path, env, matrixMode=True):
    # we only need fish_x, fish_y, fish_orientation_radians, robo_x, robo_y, robo_orientation_radians
    ar = pd.read_csv(path).to_numpy()[:, [11, 12, 14, 5, 6, 8]].astype(np.float64)
    # convert x,y from cm to m
    ar[:, [0, 1, 3, 4]] = ar[:, [0, 1, 3, 4]] / 100
    # convert x,y from (0,1) to (-0.5,0.5)
    ar[:, [0, 1, 3, 4]] = ar[:, [0, 1, 3, 4]] - 0.5
    # invert y positions
    ar[:, [1, 4]] = ar[:, [1, 4]] * (-1)
    # convert orientation from 0, 2pi to -pi,pi
    ar[:, [2, 5]] = np.where(
        ar[:, [2, 5]] > np.pi, ar[:, [2, 5]] - 2 * np.pi, ar[:, [2, 5]]
    )

    poses = ar[:, [0, 1]]

    n = poses.shape[0]
    diff = np.diff(poses, axis=0)
    speed = np.linalg.norm(diff, axis=1)
    angles = np.arctan2(diff[:, 1], diff[:, 0])

    turn = np.zeros_like(angles)
    turn[0] = angles[0]
    turn[1:] = np.diff(angles)
    # Convert turn values from [-2pi, 2pi] to [-pi, pi]
    turn = np.where(turn > np.pi, turn - 2 * np.pi, turn)
    turn = np.where(turn < -np.pi, turn + 2 * np.pi, turn)

    # change orientations of the fish to the vector from the last position, except the first position which we set to 0
    ar[0, 2] = 0
    ar[1:, 2] = angles

    # remove last row, cause we dont have turn/speed for it
    ar = ar[:-1]

    # remove all rows, that exceed a speed value of 5cm (probably wrong tracking and we cannot represent them in gym-guppy (with 20Hz))
    rows_to_keep = speed <= 0.05
    turn = turn[rows_to_keep]
    speed = speed[rows_to_keep]
    ar = ar[rows_to_keep]

    # Convert raw turn/speed values to bin format
    # get distance from each turn/speed to each bin of the corresponding type
    dist_turn = np.abs(turn[:, np.newaxis] - env.turn_rate_bins)
    dist_speed = np.abs(speed[:, np.newaxis] - env.speed_bins)

    # get indice with minimal distance (chosen action)
    bin_turn = np.argmin(dist_turn, axis=1)
    bin_speed = np.argmin(dist_speed, axis=1)

    chosen_action = None
    if matrixMode:
        chosen_action = bin_turn * len(env.speed_bins) + bin_speed
    else:
        chosen_action = np.append(
            bin_turn.reshape(len(bin_turn), 1),
            bin_speed.reshape(len(bin_speed), 1),
            axis=1,
        )

    # Get Raycasts
    # reshape data in form of guppy-env output
    ar = ar.reshape(len(ar), 2, 3)

    rays = np.empty((len(ar), 2, len(env.ray_directions)))

    for i in range(0, len(ar)):
        rays[i] = env.observation(ar[i]).copy()

    return (
        rays,
        chosen_action.reshape((len(chosen_action), 1))
        if matrixMode
        else chosen_action.reshape((len(chosen_action), 2)),
    )


def getAll(paths, env, matrixMode=True):
    obs, act = [], []
    for path in paths:
        temp_obs, temp_act = getExpert(path, env, matrixMode)
        obs.append(temp_obs)
        act.append(temp_act)

    return obs, act


def reduceData(path, target_path):
    """
    deletes columns from data, such that we can easily use it in evaluation
    """
    # we only need time, robo_x, robo_y, robo_orientation_radians, fish_x, fish_y, fish_orientation_radians
    ar = pd.read_csv(path).to_numpy()[:, [2, 5, 6, 8, 11, 12, 14]].astype(np.float64)
    # convert x,y from cm to m
    ar[:, [1, 2, 4, 5]] = ar[:, [1, 2, 4, 5]] / 100
    # convert x,y from (0,1) to (-0.5,0.5)
    ar[:, [1, 2, 4, 5]] = ar[:, [1, 2, 4, 5]] - 0.5
    # invert y axis
    ar[:, [2, 5]] = ar[:, [2, 5]] * (-1)
    # convert orientation from 0, 2pi to -pi,pi
    ar[:, [3, 6]] = np.where(
        ar[:, [3, 6]] > np.pi, ar[:, [3, 6]] - 2 * np.pi, ar[:, [3, 6]]
    )
    start_timestep = ar[0, 0]
    ar[:, 0] = ar[:, 0] - start_timestep

    poses = ar[:, [4, 5]]
    diff = np.diff(poses, axis=0)
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    ar[0, 6] = 0
    ar[1:, 6] = angles

    df = pd.DataFrame(
        data=ar[:, [1, 2, 3, 4, 5, 6, 0]],
        columns=[
            "fish0_x",
            "fish0_y",
            "fish0_ori",
            "fish1_x",
            "fish1_y",
            "fish1_ori",
            "time in ms",
        ],
    )
    df.to_csv(target_path, index=False, sep=";")


def main():
    pass
    # env = TestEnv()
    # env = DiscreteMatrixActionWrapper(env)
    # env = RayCastingWrapper(env)

    # data_paths = ["Fish/Guppy/Data/" + elem for elem in os.listdir("Fish/Guppy/Data")]

    # turn = []
    # dist = []
    # for path in data_paths:
    #     print("current file", path)
    #     turn.append(getExpert(path, env))
    # zipped_lists = zip(turn, data_paths)
    # sorted_pairs = sorted(zipped_lists)

    # tuples = zip(*sorted_pairs)
    # list1, list2 = [list(t) for t in tuples]

    # # print(list1)
    # print(list2)

    # n, bins, patches = plt.hist(turn, bins=360, range=[-np.pi, np.pi])
    # plt.show()
    # print(n, bins)

    # data_paths = [
    #     "Fish/Guppy/validationData/" + elem
    #     for elem in os.listdir("Fish/Guppy/validationData")
    # ]
    # names = [elem for elem in os.listdir("Fish/Guppy/validationData")]
    # for i in range(len(data_paths)):
    #     reduceData(data_paths[i], "Fish/Guppy/rollout/validationData/" + names[i])
    #     convertTrajectory(
    #         "Fish/Guppy/rollout/validationData/" + names[i],
    #         "Fish/Guppy/rollout/validationData/" + names[i][:-3] + "hdf5",
    #         ["robot", "fish"],
    #     )


if __name__ == "__main__":
    main()
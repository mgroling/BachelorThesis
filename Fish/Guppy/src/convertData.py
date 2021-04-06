import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import robofish.io

from conversion_scripts.convert_marc import convertTrajectory

sys.path.append("gym-guppy")
from gym_guppy import GuppyEnv

# from gym_guppy.wrappers.observation_wrapper import RayCastingWrapper
from wrappers import DiscreteMatrixActionWrapper, RayCastingWrapper

sys.path.append("Fish")
from functions import TestEnv


def getExpert(path, env, matrixMode=True, last_act=False):
    f = robofish.io.File(path)
    # extract x and y only, convert it from cm to m
    poses = f.entity_poses[:, :, :2] / 100

    # calculate turn/speed values
    n = poses.shape[1]
    diff = np.diff(poses, axis=1)
    speed = np.linalg.norm(diff, axis=2)
    angles = np.arctan2(diff[:, :, 1], diff[:, :, 0])

    turn = np.zeros_like(angles)
    turn[:, 0] = angles[:, 0]
    turn[:, 1:] = np.diff(angles)
    # convert turn values from [-2pi, 2pi] to [-pi, pi]
    turn = np.where(turn > np.pi, turn - 2 * np.pi, turn)
    turn = np.where(turn < -np.pi, turn + 2 * np.pi, turn)

    poses = f.entity_poses_rad
    # convert x,y from cm to m
    poses[:, :, :2] = poses[:, :, :2] / 100
    # change orientations of the fish to the vector from the last position, except the first position which we set to 0
    poses[:, 0, 2] = 0
    poses[:, 1:, 2] = angles

    # remove last row of poses, since we do not have a turn/speed value for it
    poses = poses[:, :-1]

    # convert raw turn/speed values to bin format
    # get distance from each turn/speed to each bin of the corresponding type
    dist_turn = np.abs(turn[:, :, np.newaxis] - env.turn_rate_bins)
    dist_speed = np.abs(speed[:, :, np.newaxis] - env.speed_bins)

    # get indice with minimal distance (chosen action)
    bin_turn = np.argmin(dist_turn, axis=2)
    bin_speed = np.argmin(dist_speed, axis=2)

    if matrixMode:
        chosen_action = bin_turn * len(env.speed_bins) + bin_speed
    else:
        chosen_action = np.append(
            bin_turn.reshape(bin_turn.shape[0], bin_turn.shape[1], 1),
            bin_speed.reshape(bin_speed.shape[0], bin_speed.shape[1], 1),
            axis=2,
        )

    # get raycasts
    # reshape data in form of guppy-env output
    poses = np.swapaxes(poses, 0, 1)

    if last_act:
        rays = np.empty(
            (poses.shape[1], poses.shape[0], len(env.ray_directions) * 2 + 2)
        )
    else:
        rays = np.empty((poses.shape[1], poses.shape[0], 2, len(env.ray_directions)))

    for i in range(len(poses)):
        for j in range(poses.shape[1]):
            poses[i, [0, j]] = poses[i, [j, 0]]
            rays[j, i] = env.observation(poses[i])

    if last_act:
        # reset last action values (wrapper does not know them)
        rays[:, 0, -2] = 0
        rays[:, 0, -1] = 0
        rays[:, 1:, -2] = turn[:, :-1] / np.pi
        rays[:, 1:, -2] = speed[:, :-1] / env.speed_bins[-1]

    return rays, chosen_action


def getAll(paths, env, matrixMode=True, last_act=False):
    obs, act = [], []
    for path in paths:
        temp_obs, temp_act = getExpert(path, env, matrixMode, last_act)
        for elem in temp_obs:
            obs.append(elem)
        for elem in temp_act:
            act.append(elem)

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
    # ar[:, [1, 2, 4, 5]] = np.clip(ar[:, [1, 2, 4, 5]], -0.499, 0.499)
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
    env = TestEnv()
    env = DiscreteMatrixActionWrapper(
        env,
        num_bins_turn_rate=10,
        num_bins_speed=10,
        max_turn=np.pi,
        min_speed=0,
        max_speed=0.05,
    )
    env = RayCastingWrapper(env, last_act=False)

    data_paths = ["Fish/Guppy/Data/" + elem for elem in os.listdir("Fish/Guppy/Data")]

    # turn = []
    # dist = []
    for path in data_paths:
        print("current file", path)
        getExpert(path, env, False, False)

    # getAll(data_paths, env, False, False)
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
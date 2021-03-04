import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from conversion_scripts.convert_marc import convertTrajectory

sys.path.append("gym-guppy")
from gym_guppy.envs._configurable_guppy_env import ConfigurableGuppyEnv
from gym_guppy.wrappers.observation_wrapper import RayCastingWrapper
from wrappers import DiscreteMatrixActionWrapper


def getExpert(path, min_turn, min_dist, env, matrixMode = True):
    # we only need fish_x, fish_y, fish_orientation_radians, robo_x, robo_y, robo_orientation_radians
    ar = pd.read_csv(path).to_numpy()[:, [11, 12, 14, 5, 6, 8]].astype(np.float64)
    # convert x,y from cm to m
    ar[:, [0, 1, 3, 4]] = ar[:, [0, 1, 3, 4]] / 100
    # convert x,y from (0,1) to (-0.5,0.5)
    ar[:, [0, 1, 3, 4]] = ar[:, [0, 1, 3, 4]] - 0.5
    # convert orientation from 0, 2pi to -pi,pi
    ar[:, [2, 5]] = np.where(
        ar[:, [2, 5]] > np.pi, ar[:, [2, 5]] - 2 * np.pi, ar[:, [2, 5]]
    )

    # calculate turn and dist for each timestep
    turn_dist = np.empty((len(ar) - 1, 2))
    # (orientation{t} - orientation{t-1}) = turn, also make it take the "shorter" turn (the shorter angle)
    turn_dist[:, 0] = ar[1:, 2] - ar[:-1, 2]
    turn_dist[:, 0] = np.where(
        turn_dist[:, 0] < -np.pi, turn_dist[:, 0] + 2 * np.pi, turn_dist[:, 0]
    )
    turn_dist[:, 0] = np.where(
        turn_dist[:, 0] > np.pi, turn_dist[:, 0] - 2 * np.pi, turn_dist[:, 0]
    )
    # sqrt((x{t}-x{t-1})**2 + (y{t}-y{t-1})**2) = dist
    turn_dist[:, 1] = np.sqrt(
        np.array(
            np.power(ar[1:, 0] - ar[:-1, 0], 2) + np.power(ar[1:, 1] - ar[:-1, 1], 2),
            dtype=np.float64,
        )
    )

    # plt.hist(turn_dist[:, 1], bins=20, range = [0, 0.1])
    # plt.show()

    # Convert raw turn/dist values to bin format
    # get distance from each turn/speed to each bin of the corresponding type
    dist_turn = np.abs(turn_dist[:, 0, np.newaxis] - env.turn_rate_bins)
    dist_dist = np.abs(turn_dist[:, 1, np.newaxis] - env.speed_bins)

    # get indice with minimal distance (chosen action)
    bin_turn = np.argmin(dist_turn, axis=1)
    bin_dist = np.argmin(dist_dist, axis=1)

    chosen_action = None
    if matrixMode:
        chosen_action = bin_turn * len(env.speed_bins) + bin_dist
    else:
        chosen_action = np.append(bin_turn.reshape(len(bin_turn), 1), bin_dist.reshape(len(bin_dist), 1), axis = 1)

    # turn_rate = np.floor(chosen_action / len(env.speed_bins)).astype(int)
    # speed = (chosen_action % len(env.speed_bins)).astype(int)
    # turn, speed = env.turn_rate_bins[turn_rate], env.speed_bins[speed]

    # fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    # ax[0][0].hist(turn_dist[:, 0], bins=20, range=[-np.pi, np.pi])
    # ax[0][0].set_title("original turn")
    # ax[0][1].hist(turn, bins=20, range=[-np.pi, np.pi])
    # ax[0][1].set_title("converted turn")
    # ax[1][0].hist(turn_dist[:, 1], bins=10, range=[0.01, 0.07])
    # ax[1][0].set_title("original speed")
    # ax[1][1].hist(speed, bins=10, range=[0.01, 0.07])
    # ax[1][1].set_title("converted speed")
    # plt.show()

    # Get Raycasts
    # remove last row, cause we dont have turn/dist for it
    ar = ar[:-1]

    # reshape data in form of guppy-env output
    ar = ar.reshape(len(ar), 2, 3)

    rays = np.empty((len(ar), 2, len(env.ray_directions)))

    for i in range(0, len(ar)):
        rays[i] = env.observation(ar[i])

    return rays, chosen_action.reshape((len(chosen_action), 1)) if matrixMode else chosen_action.reshape((len(chosen_action), 2))


def getAll(paths, min_turn, min_dist, env, matrixMode = True):
    obs, act = [], []
    cc = 0
    for path in paths:
        temporary = pd.read_csv(path, sep=";")
        cc += len(temporary)
        temp_obs, temp_act = getExpert(path, min_turn, min_dist, env, matrixMode)
        obs.append(temp_obs)
        act.append(temp_act)
    print("timesteps count", cc)

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
    # convert orientation from 0, 2pi to -pi,pi
    ar[:, [3, 6]] = np.where(
        ar[:, [3, 6]] > np.pi, ar[:, [3, 6]] - 2 * np.pi, ar[:, [3, 6]]
    )
    start_timestep = ar[0, 0]
    ar[:, 0] = ar[:, 0] - start_timestep

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
    # env = ConfigurableGuppyEnv()
    # env = DiscreteMatrixActionWrapper(env)
    # env = RayCastingWrapper(env)
    # # convertFile("Fish/Guppy/data/test_robotracker.csv", env)

    # getExpert("Fish/Guppy/data/test_robotracker.csv", np.pi/4, 0.1)

    data_paths = [
        "Fish/Guppy/validationData/" + elem
        for elem in os.listdir("Fish/Guppy/validationData")
    ]
    names = [elem for elem in os.listdir("Fish/Guppy/validationData")]
    for i in range(len(data_paths)):
        reduceData(data_paths[i], "Fish/Guppy/rollout/validationData/" + names[i])
        convertTrajectory(
            "Fish/Guppy/rollout/validationData/" + names[i],
            "Fish/Guppy/rollout/validationData/" + names[i][:-3] + "hdf5",
            ["robot", "fish"],
        )


if __name__ == "__main__":
    main()
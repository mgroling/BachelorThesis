import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import time

from convertData import getAll

sys.path.append("gym-guppy")
from gym_guppy import (
    GuppyEnv,
    TurnSpeedRobot,
)

if __name__ == "__main__":
    ar = pd.read_csv(
        "Fish/Guppy/rollout/validationData/Q20I_Fri_Dec__6_15_13_09_2019_Robotracker.csv",
        sep=";",
    ).to_numpy()[:, [3, 4, 5]]

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

    # for i, elem in enumerate(turn_dist[:, 1]):
    #     if elem > 0.1:
    #         print(ar[i + 1, [0, 1]], ar[i, [0, 1]], i)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Spread of turn values in linear bins")
    ax[0][0].hist(turn_dist[:, 0], bins=20, range=[-np.pi / 2, np.pi / 2])
    ax[0][0].set_title("20 turn bins")
    ax[0][1].hist(turn_dist[:, 0], bins=40, range=[-np.pi / 2, np.pi / 2])
    ax[0][1].set_title("40 turn bins")
    ax[1][0].hist(turn_dist[:, 0], bins=80, range=[-np.pi / 2, np.pi / 2])
    ax[1][0].set_title("80 turn bins")
    ax[1][1].hist(turn_dist[:, 0], bins=160, range=[-np.pi / 2, np.pi / 2])
    ax[1][1].set_title("160 turn bins")
    plt.show()
    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].hist(turn_dist[:, 1], bins=10, range=[0, 0.03])
    # ax[0][1].hist(turn_dist[:, 1], bins=20, range=[0, 0.03])
    # ax[1][0].hist(turn_dist[:, 1], bins=40, range=[0, 0.03])
    # ax[1][1].hist(turn_dist[:, 1], bins=80, range=[0, 0.03])
    # plt.show()

    # print(ar[5000], ar[5001], turn_dist[5000])

    # class TestEnv(GuppyEnv):
    #     def _reset(self):
    #         # set frequency to 20Hz
    #         self._guppy_steps_per_action = 5

    #         self._add_robot(
    #             TurnSpeedRobot(
    #                 world=self.world,
    #                 world_bounds=self.world_bounds,
    #                 position=(0, 0),
    #                 orientation=0,
    #             )
    #         )

    # env = TestEnv(steps_per_robot_action=100)

    # obs = env.reset()
    # last_obs = None
    # for i in range(1000):
    #     turn, dist = 0.01, 0.001
    #     last_obs = obs
    #     obs = env.step([turn, dist])[0]

    #     print(obs[0, 2] - last_obs[0, 2])

    #     env.render()

    # trajectory = np.stack(
    #     [
    #         env.step(action=[turn_dist[i, 0], turn_dist[i, 1] * 2])[0]
    #         for i in range(len(turn_dist))
    #     ]
    # ).reshape((len(turn_dist), 3))

    # trajectory = []
    # obs = env.reset()
    # for i in range(len(turn_dist)):
    #     obs = env.step(action=[turn_dist[i, 0], turn_dist[i, 1] * 2])[0]
    #     print("turn, dist", turn_dist[i, 0], turn_dist[i, 1])
    #     print(obs, ar[i + 1])

    #     if np.sum(np.abs(obs - ar[i + 1])) > 0.001:
    #         print("error at timestep", i)
    #         break

    #     env.render()

    # plt.plot(trajectory[:, 0], trajectory[:, 1] * (-1))
    # plt.show()
    # plt.plot(ar[:, 0], ar[:, 1] * (-1))
    # plt.show()
    # print(np.sum(trajectory[:10000] - ar[1:10001]))

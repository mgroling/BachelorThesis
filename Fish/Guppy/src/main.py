import numpy as np
import pandas as pd
import sys
import os
import time
import math
import random

from wrappers import DiscreteMatrixActionWrapper, RayCastingWrapper, VectorActionWrapper
from stable_baselines.common.policies import MlpPolicy

sys.path.append("Fish")
from functions import *
from convertData import *
from rolloutEv import *

sys.path.append("SQIL_DQN")
from SQIL_DQN import SQIL_DQN

sys.path.append("SQIL_PPO")
from SQIL_PPO import SQIL_PPO

sys.path.append("gym-guppy")
from gym_guppy.guppies._robot import (
    GlobalTargetRobot,
    TurnBoostRobot,
    PolarCoordinateTargetRobot,
)

import matplotlib.pyplot as plt


def main():
    env = TestEnvM(max_steps_per_action=200)
    # env = TestEnvNV()

    env = RayCastingWrapper(env)
    # env = VectorActionWrapper(env)
    env = DiscreteMatrixActionWrapper(env, num_bins_turn_rate=20, num_bins_speed=10)

    # model = SQIL_DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 20000, double_q = False, seed = 37, exploration_initial_eps = 0.5, exploration_final_eps = 0.5)
    # # # model = SQIL_PPO(MlpPolicy, env)

    # obs, act = getAll(["Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")], np.pi/4, 0.07, env)
    # print("expert timesteps:", sum([len(elem) for elem in obs]))
    # model.initializeExpertBuffer(obs, act)

    # model.learn(total_timesteps = 25000, train_graph = True)

    model_name = "DQN_22_12_2020_03"

    # model.save("Fish/Guppy/models/" + model_name + "/model")

    model = SQIL_DQN.load("Fish/Guppy/models/" + model_name + "/model")

    testExpert(
        [
            "Fish/Guppy/validationData/" + elem
            for elem in os.listdir("Fish/Guppy/validationData")
        ],
        model,
        env,
        2,
        deterministic=True,
    )

    env.unwrapped.state_history = []

    # # print("too many steps", env.too_many_steps)

    # env.unwrapped.video_path = "Fish/Guppy/video"

    """testing"""
    obs = env.reset()
    done = False
    timestep = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)

        temp = obs.copy()

        obs, reward, done, _ = env.step(action)

        # print("distance:")
        # print(distance(temp[0,0], temp[0,1], obs[0,0], obs[0,1]))
        # print(distance(temp[1,0], temp[1,1], obs[1,0], obs[1,1]))
        # print("turn:")
        # print(temp[0,2] - obs[0,2])
        # print(temp[1,2] - obs[1,2])

        if len(env.state_history) > 25000:
            break

        env.render()
        time.sleep(0.1)
        timestep += 1

        # if timestep >= 100:
        #     break

    temp = env.state_history[0::5]
    temp = temp[0:5000]

    trajectory = np.concatenate(temp, axis=0)

    # df = pd.DataFrame(
    #     data=trajectory,
    #     columns=["fish0_x", "fish0_y", "fish0_ori", "fish1_x", "fish1_y", "fish1_ori"],
    # )
    # df.to_csv(
    #     "Fish/Guppy/trajectories/" + model_name + "_2models_det.csv",
    #     index=False,
    #     sep=";",
    # )

    # # print("too many steps", env.too_many_steps)

    # env.close()


if __name__ == "__main__":
    main()
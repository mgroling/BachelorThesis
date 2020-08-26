import numpy as np
import pandas as pd
import sys
import os
import time

from wrappers import DiscreteMatrixActionWrapper, DiscreteMatrixActionWrapperCus, RayCastingWrapper

sys.path.append("Fish")
from functions import *
from convertData import convertFile, getTurnrateSpeed

sys.path.append("SQIL_DQN")
from SQIL_DQN import SQIL_DQN

sys.path.append("gym-guppy")
from gym_guppy.envs._configurable_guppy_env import ConfigurableGuppyEnv
from gym_guppy.guppies._robot import GlobalTargetRobot, TurnBoostRobot

def main():
    env = ConfigurableGuppyEnv(robot_type = GlobalTargetRobot)

    temp = getTurnrateSpeed("Fish/Guppy/data/test_robotracker.csv")
    max_turn = temp[:, 0].max()
    max_speed = 0.15

    env = RayCastingWrapper(env)
    env = DiscreteMatrixActionWrapperCus(env, max_turn, max_speed)

    # model = SQIL_DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 20000, double_q = False, seed = 37, exploration_initial_eps = 0.5, exploration_final_eps = 0.5)
    # obs, act = convertFile("Fish/Guppy/data/test_robotracker.csv", env)
    # model.initializeExpertBufferSep([obs], [act])

    # model.learn(total_timesteps = 100000)

    # model.save("Fish/Guppy/models/first_model")

    # model.load("Fish/Guppy/models/first_model")

    """testing"""
    obs = env.reset()
    done = False
    timestep = 0
    while not done:
        # action, _ = model.predict(obs)
        action = 55
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.5)
        timestep += 1

        if timestep > 100:
            break

    env.close()

if __name__ == "__main__":
    main()
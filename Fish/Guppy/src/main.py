import numpy as np
import pandas as pd
import sys
import os
import time

from wrappers import DiscreteMatrixActionWrapper, RayCastingWrapper, VectorActionWrapper
from stable_baselines.common.policies import MlpPolicy

sys.path.append("Fish")
from functions import *
from convertData import *

sys.path.append("SQIL_DQN")
from SQIL_DQN import SQIL_DQN

sys.path.append("SQIL_PPO")
from SQIL_PPO import SQIL_PPO

sys.path.append("gym-guppy")
from gym_guppy.guppies._robot import GlobalTargetRobot, TurnBoostRobot, PolarCoordinateTargetRobot

import matplotlib.pyplot as plt

def main():
    env = TestEnv(max_steps_per_action = 200)

    env = RayCastingWrapper(env)
    # env = VectorActionWrapper(env)
    env = DiscreteMatrixActionWrapper(env, num_bins_turn_rate=20, num_bins_speed=10)

    # model = SQIL_DQN(CustomDQNPolicy, env, verbose=1, buffer_size = 20000, double_q = False, seed = 37, exploration_initial_eps = 0.5, exploration_final_eps = 0.5)
    # # # model = SQIL_PPO(MlpPolicy, env)

    # obs, act = getAll(["Fish/Guppy/data/" + elem for elem in os.listdir("Fish/Guppy/data")], np.pi/4, 0.07, env)
    # print("expert timesteps:", sum([len(elem) for elem in obs]))
    # model.initializeExpertBuffer(obs, act)

    # model.learn(total_timesteps = 25000)

    # model.save("Fish/Guppy/models/DQN_256_128_25k_pi-4_7-100_DQNCLIP")

    model = SQIL_DQN.load("Fish/Guppy/models/DQN_256_128_25k_pi-4_7-100_DQNCLIP")

    print("too many steps", env.too_many_steps)

    env.unwrapped.video_path = "Fish/Guppy/video"

    #create trajectory with 10k timesteps
    # trajectory = np.empty((10000, 6))

    """testing"""
    obs = env.reset()
    done = False
    timestep = 0
    while not done:
        # trajectory[timestep] = np.array(env.get_state()).reshape((1, 6))

        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)
        timestep += 1

        if timestep >= 100:
            break

    # df = pd.DataFrame(data = trajectory, columns = ["fish0_x", "fish0_y", "fish0_ori", "fish1_x", "fish1_y", "fish1_ori"])
    # df.to_csv("Fish/Guppy/trajectories/trajectory_0.csv", index = False, sep = ";")

    print("too many steps", env.too_many_steps)
    
    env.close()

if __name__ == "__main__":
    main()
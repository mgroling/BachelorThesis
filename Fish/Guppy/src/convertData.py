import pandas as pd
import numpy as np
import sys
sys.path.append("gym-guppy")
from gym_guppy.envs._configurable_guppy_env import ConfigurableGuppyEnv
from gym_guppy.wrappers.observation_wrapper import RayCastingWrapper
from wrappers import DiscreteMatrixActionWrapper

def convertFile(path, env):
    return getRays(path, env), convertTurnrateSpeed(getTurnrateSpeed(path), env)

def getTurnrateSpeed(path):
    #only need timestep, x, y, orientation_radians
    ar = pd.read_csv(path).to_numpy()[:, [2,5,6,8]].astype(np.float64)
    #convert x,y from cm to m
    ar[:, [1,2]] = ar[:, [1,2]] / 100
    #range in tank is from 0 to 1 in given data, however in the guppy-gym it is -0.5 to 0.5, so we need to convert that
    ar[:, [1,2]] = ar[:, [1,2]] - 0.5
    #convert orientation from 0, 2pi to -pi,pi
    ar[:, 3] = ar[:, 3] - np.pi

    out = np.empty((len(ar)-1, 2))
    #length of timesteps, convert from ms to s
    timestep_len = (ar[1:, 0] - ar[:-1, 0]) / 1000
    # (orientation{t} - orientation{t-1}) / timestep_len = turnrate, also make it take the "shorter" turn (the shorter angle)
    out[:, 0] = (ar[1:, 3] - ar[:-1, 3])
    out[:, 0] = np.where(out[:, 0] < -np.pi, out[:, 0] + 2*np.pi, out[:, 0])
    out[:, 0] = np.where(out[:, 0] > np.pi, out[:, 0] - 2*np.pi, out[:, 0])
    # sqrt((x{t}-x{t-1})**2 + (y{t}-y{t-1})**2) / timestep_len = speed
    out[:, 1] = np.sqrt( np.array(np.power(ar[1:, 1]-ar[:-1, 1], 2) + np.power(ar[1:, 2]-ar[:-1, 2], 2), dtype = np.float64) )

    return out

def convertTurnrateSpeed(ar, env):
    #get distance from each turn/speed to each bin of the corresponding type
    dist_turn = np.abs(ar[:, 0, np.newaxis] - env.turn_rate_bins)
    dist_speed = np.abs(ar[:, 1, np.newaxis] - env.speed_bins)

    #get indice with minimal distance (chosen action)
    bin_turn = np.argmin(dist_turn, axis = 1)
    bin_speed = np.argmin(dist_speed, axis = 1)

    chosen_action = bin_turn * len(env.speed_bins) + bin_speed

    return chosen_action.reshape((len(chosen_action), 1))

def getRays(path, env):
    #only need robo_x, robo_y, robo_orientation_radians, fish_x, fish_y, fish_orientation_radians
    ar = pd.read_csv(path).to_numpy()[:, [5,6,8,11,12,14]].astype(np.float64)
    #convert x,y from cm to m
    ar[:, [0,1,3,4]] = ar[:, [0,1,3,4]] / 100
    #range in tank is from 0 to 1 in given data, however in the guppy-gym it is -0.5 to 0.5, so we need to convert that
    ar[:, [0,1,3,4]] = ar[:, [0,1,3,4]] - 0.5
    #reshape it in form of guppy-env output
    ar = ar.reshape(len(ar), 2, 3)

    out = np.empty((len(ar), 2, len(env.ray_directions)))

    for i in range(0, len(ar)):
        out[i] = env.observation(ar[i])

    return out

def main():
    env = ConfigurableGuppyEnv()
    env = DiscreteMatrixActionWrapper(env)
    env = RayCastingWrapper(env)
    convertFile("Fish/Guppy/data/test_robotracker.csv", env)
    
if __name__ == "__main__":
    main()
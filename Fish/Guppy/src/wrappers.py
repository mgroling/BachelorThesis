import gym
import numpy as np
import math

import sys
sys.path.append("gym-guppy")
from gym_guppy.tools.math import ray_casting_walls, compute_dist_bins

class DiscreteMatrixActionWrapper(gym.ActionWrapper):
    def __init__(self, env, num_bins_turn_rate=10, num_bins_speed=10):
        super(DiscreteMatrixActionWrapper, self).__init__(env)
        assert isinstance(self.action_space, gym.spaces.Box)
        max_turn_rate = self.action_space.high[0, 0]
        self.turn_rate_bins = np.linspace(-max_turn_rate, max_turn_rate, num_bins_turn_rate)
        max_speed = self.action_space.high[0, 1]
        self.speed_bins = np.linspace(0, max_speed, num_bins_speed)

        self.action_space = gym.spaces.Discrete(num_bins_turn_rate*num_bins_speed)

    def action(self, action):
        turn_rate = math.floor(action/len(self.speed_bins))
        speed = action%len(self.speed_bins)
        return [self.turn_rate_bins[turn_rate], self.speed_bins[speed]]

    def reverse_action(self, action):
            raise NotImplementedError

class DiscreteMatrixActionWrapperCus(gym.ActionWrapper):
    def __init__(self, env, max_turn_rate, max_speed, num_bins_turn_rate=10, num_bins_speed=10):
        super(DiscreteMatrixActionWrapperCus, self).__init__(env)
        assert isinstance(self.action_space, gym.spaces.Box)
        max_turn_rate = max_turn_rate
        self.turn_rate_bins = np.linspace(-max_turn_rate, max_turn_rate, num_bins_turn_rate)
        max_speed = max_speed
        self.speed_bins = np.linspace(0, max_speed, num_bins_speed)

        self.action_space = gym.spaces.Discrete(num_bins_turn_rate*num_bins_speed)

    def action(self, action):
        turn_rate = math.floor(action/len(self.speed_bins))
        speed = action%len(self.speed_bins)
        turn_rate, speed = self.turn_rate_bins[turn_rate], self.speed_bins[speed]

        global_turn = self.state[0][2] + turn_rate
        global_x, global_y = self.state[0][0] + speed*np.cos(global_turn), self.state[0][1] + speed*np.sin(global_turn)

        print("state:",self.state)
        print("action:",turn_rate,speed)
        print("next state should be:", global_x, global_y, global_turn)

        # self.state = np.array([[global_x, global_y, global_turn], [self.state[0][0]], self.state[0][1], self.state[0][2]])

        return [global_x, global_y]

    def reverse_action(self, action):
        raise NotImplementedError

class RayCastingWrapper(gym.ObservationWrapper):
    def __init__(self, env, degrees=360, num_bins=36 * 2):
        super(RayCastingWrapper, self).__init__(env)
        # redefine observation space
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, num_bins), dtype=np.float64)

        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.sector_bounds = np.linspace(-self.cutoff, self.cutoff, num_bins + 1)
        self.ray_directions = np.linspace(-self.cutoff, self.cutoff, num_bins)
        # TODO: is this faster than just recreating the array?
        self.obs_placeholder = np.empty(self.observation_space.shape)

    def observation(self, state):
        self.state = state
        self.obs_placeholder[0] = compute_dist_bins(state[0], state[1:], self.sector_bounds, self.diagonal * 1.1)
        self.obs_placeholder[1] = ray_casting_walls(state[0], self.world_bounds, self.ray_directions, self.diagonal * 1.1)
        return self.obs_placeholder
import gym
import numpy as np
import math

import sys

sys.path.append("gym-guppy")
from gym_guppy.tools.math import ray_casting_walls, compute_dist_bins


class DiscreteMatrixActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env,
        num_bins_turn_rate=10,
        num_bins_speed=10,
        max_turn=np.pi / 4,
        min_speed=0.03,
        max_speed=0.1,
    ):
        super(DiscreteMatrixActionWrapper, self).__init__(env)
        assert isinstance(self.action_space, gym.spaces.Box)
        self.turn_rate_bins = np.linspace(-max_turn, max_turn, num_bins_turn_rate)
        self.speed_bins = np.linspace(min_speed, max_speed, num_bins_speed)

        self.action_space = gym.spaces.Discrete(num_bins_turn_rate * num_bins_speed)

    def action(self, action):
        turn_rate = math.floor(action / len(self.speed_bins))
        speed = action % len(self.speed_bins)
        turn, speed = self.turn_rate_bins[turn_rate], self.speed_bins[speed]

        return [turn, speed]

    def reverse_action(self, action):
        raise NotImplementedError


class DiscreteMatrixActionWrapperCustomBins(gym.ActionWrapper):
    def __init__(
        self,
        env,
        num_bins_turn_rate=10,
        num_bins_speed=10,
        max_turn=np.pi / 4,
        min_speed=0.03,
        max_speed=0.1,
    ):
        super(DiscreteMatrixActionWrapperCustomBins, self).__init__(env)
        assert isinstance(self.action_space, gym.spaces.Box)
        self.turn_rate_bins = np.linspace(-max_turn, max_turn, num_bins_turn_rate)
        self.speed_bins = np.linspace(min_speed, max_speed, num_bins_speed)

        self.action_space = gym.spaces.Discrete(num_bins_turn_rate * num_bins_speed)

    def action(self, action):
        turn_rate = math.floor(action / len(self.speed_bins))
        speed = action % len(self.speed_bins)
        turn, speed = self.turn_rate_bins[turn_rate], self.speed_bins[speed]

        return [turn, speed]

    def reverse_action(self, action):
        raise NotImplementedError


class ActionSpaceHolder:
    def __init__(self, one, two):
        self.space = [
            gym.spaces.Discrete(one),
            gym.spaces.Discrete(two),
        ]

    def seed(self, seed):
        for space in self.space:
            space.seed(seed)


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env,
        num_bins_turn_rate=10,
        num_bins_speed=10,
        max_turn=np.pi / 4,
        min_speed=0.03,
        max_speed=0.1,
        frequency=25,
        last_act=False,
    ):
        super(DiscreteActionWrapper, self).__init__(env)
        self.last_act = last_act
        assert isinstance(self.action_space, gym.spaces.Box)
        self.turn_rate_bins = np.linspace(-max_turn, max_turn, num_bins_turn_rate)
        self.speed_bins = np.linspace(min_speed, max_speed, num_bins_speed)
        self.frequency = frequency

        self.action_space = ActionSpaceHolder(num_bins_turn_rate, num_bins_speed)

    def action(self, action):
        turn, speed = self.turn_rate_bins[action[0]], self.speed_bins[action[1]]

        if self.last_act:
            self.last_action = [
                turn / np.pi,
                speed / self.speed_bins[-1],
            ]

        return [turn, speed * self.frequency]

    def reverse_action(self, action):
        raise NotImplementedError


class VectorActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(VectorActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(
            low=np.array([0, -np.pi], dtype=np.float64),
            high=np.array([0.15, np.pi], dtype=np.float64),
            dtype=np.float64,
        )

    def action(self, action):
        return action


class RayCastingWrapper(gym.ObservationWrapper):
    def __init__(self, env, degrees=360, num_bins=36, last_act=False):
        super(RayCastingWrapper, self).__init__(env)
        # redefine observation space
        self.last_act = last_act
        if last_act:
            self.last_action = [0, 0]
            self.num_bins = num_bins
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(num_bins * 2 + 2,), dtype=np.float64
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(2, num_bins), dtype=np.float64
            )

        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.sector_bounds = np.linspace(-self.cutoff, self.cutoff, num_bins + 1)
        self.ray_directions = np.linspace(-self.cutoff, self.cutoff, num_bins)
        # TODO: is this faster than just recreating the array?
        self.obs_placeholder = np.empty(self.observation_space.shape)

    def observation(self, state):
        if self.last_act:
            self.obs_placeholder[: self.num_bins] = compute_dist_bins(
                state[0], state[1:], self.sector_bounds, self.diagonal
            )
            self.obs_placeholder[self.num_bins : -2] = ray_casting_walls(
                state[0], self.world_bounds, self.ray_directions, self.diagonal
            )
            self.obs_placeholder[-2:] = self.last_action
        else:
            self.obs_placeholder[0] = compute_dist_bins(
                state[0], state[1:], self.sector_bounds, self.diagonal
            )
            self.obs_placeholder[1] = ray_casting_walls(
                state[0], self.world_bounds, self.ray_directions, self.diagonal
            )
        return self.obs_placeholder


class RayCastingObject:
    def __init__(self, degrees=360, num_bins=36 * 2, last_act = False):
        self.last_act = last_act
        self.world_bounds = [np.array([-0.5, -0.5]), np.array([0.5, 0.5])]
        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.sector_bounds = np.linspace(-self.cutoff, self.cutoff, num_bins + 1)
        self.ray_directions = np.linspace(-self.cutoff, self.cutoff, num_bins)
        if last_act:
            self.obs_placeholder = np.empty((num_bins * 2 + 2,))
        else:
            self.obs_placeholder = np.empty((2, num_bins))

    def observation(self, state):
        if self.last_act:
            self.obs_placeholder[: self.num_bins] = compute_dist_bins(
                state[0], state[1:], self.sector_bounds, self.diagonal
            )
            self.obs_placeholder[self.num_bins : -2] = ray_casting_walls(
                state[0], self.world_bounds, self.ray_directions, self.diagonal
            )
            self.obs_placeholder[-2:] = self.last_action
        else:
            self.obs_placeholder[0] = compute_dist_bins(
                state[0], state[1:], self.sector_bounds, self.diagonal
            )
            self.obs_placeholder[1] = ray_casting_walls(
                state[0], self.world_bounds, self.ray_directions, self.diagonal
            )
        return self.obs_placeholder
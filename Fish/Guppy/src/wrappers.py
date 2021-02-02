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
        if not type(action) == list:
            action = [action]
        actions = []
        state = self.get_state()

        for i in range(len(action)):
            pose = state[i]

            turn_rate = math.floor(action[i] / len(self.speed_bins))
            speed = action[i] % len(self.speed_bins)
            turn, speed = self.turn_rate_bins[turn_rate], self.speed_bins[speed]

            global_turn = pose[2] + turn
            global_x, global_y = (
                pose[0] + speed * np.cos(global_turn),
                pose[1] + speed * np.sin(global_turn),
            )

            actions.append([global_x, global_y])

        return actions[0] if len(actions) == 1 else actions

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
    def __init__(self, env, degrees=360, num_bins=36 * 2):
        super(RayCastingWrapper, self).__init__(env)
        # redefine observation space
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
        self.obs_placeholder[0] = compute_dist_bins(
            state[0], state[1:], self.sector_bounds, self.diagonal * 1.1
        )
        self.obs_placeholder[1] = ray_casting_walls(
            state[0], self.world_bounds, self.ray_directions, self.diagonal * 1.1
        )
        return self.obs_placeholder


class RayCastingObject:
    def __init__(self, degrees=360, num_bins=36 * 2):
        self.world_bounds = [np.array([-0.5, -0.5]), np.array([0.5, 0.5])]
        self.diagonal = np.linalg.norm(self.world_bounds[0] - self.world_bounds[1])
        self.cutoff = np.radians(degrees) / 2.0
        self.sector_bounds = np.linspace(-self.cutoff, self.cutoff, num_bins + 1)
        self.ray_directions = np.linspace(-self.cutoff, self.cutoff, num_bins)
        self.obs_placeholder = np.empty((2, num_bins))

    def observation(self, state):
        self.obs_placeholder[0] = compute_dist_bins(
            state[0], state[1:], self.sector_bounds, self.diagonal * 1.1
        )
        self.obs_placeholder[1] = ray_casting_walls(
            state[0], self.world_bounds, self.ray_directions, self.diagonal * 1.1
        )
        return self.obs_placeholder
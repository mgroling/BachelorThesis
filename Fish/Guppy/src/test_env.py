import numpy as np
import sys
import time
import gym

sys.path.append("gym-guppy")
from gym_guppy import (
    GuppyEnv,
    BoostCouzinGuppy,
    SimpleRobot,
)


# class SimpleRobot(Robot):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._max_velocity = 0.1
#         self._max_turn = np.pi

#     def action_completed(self):
#         return True

#     @property
#     def action_space(self):
#         return gym.spaces.Box(
#             low=np.float32([-self._max_turn, 0.0]),
#             high=np.float32([self._max_turn, self._max_velocity]),
#         )

#     def set_action(self, action):
#         self._turn = action[0]
#         self._velocity = action[1] * 2

#     def step(self, time_step):
#         self.set_orientation(self.get_orientation() + self._turn)
#         self._turn = 0.0
#         self.set_angular_velocity(0.0)
#         self.set_linear_velocity((self._velocity, 0.0), local=True)


class TestEnv(GuppyEnv):
    def _reset(self):
        self._add_robot(
            SimpleRobot(
                world=self.world,
                world_bounds=self.world_bounds,
                position=(0, 0),
                orientation=0,
            )
        )

        num_guppies = 1
        positions = np.random.normal(size=(num_guppies, 2), scale=0.02) + (0.05, 0.05)
        orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
        for p, o in zip(positions, orientations):
            self._add_guppy(
                BoostCouzinGuppy(
                    world=self.world,
                    world_bounds=self.world_bounds,
                    position=p,
                    orientation=o,
                )
            )


env = TestEnv()

obs = env.reset()
turn = 0.5
dist = 0.1
for i in range(1000):
    old_obs = obs
    obs, reward, done, _ = env.step([turn, dist])
    distance_travelled = np.round(
        ((obs[0][0] - old_obs[0][0]) ** 2 + (obs[0][1] - old_obs[0][1]) ** 2)
        ** (1 / 2),
        5,
    )
    print(distance_travelled)
    conversion_rate = distance_travelled / dist
    # print(conversion_rate)
    time.sleep(0.1)
    env.render()

env.close()
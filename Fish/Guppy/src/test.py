from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("gym-guppy")
from gym_guppy import VariableStepGuppyEnv, PolarCoordinateTargetRobot, BoostCouzinGuppy, GlobalTargetRobot
# from gym_guppy.wrappers.evaluation_wrapper import FeedbackInspectionWrapper


class TestEnv(VariableStepGuppyEnv):
    def _reset(self):
        controller_params = {
            'ori_ctrl_params': {
                'p':     1.,
                'i':     0.,
                'd':     0.,
                'speed': .2,
                'slope': 1.
            },
            'fwd_ctrl_params': {
                'p':              1.,
                'i':              0.,
                'd':              0.,
                'speed':          .2,
                'slope':          100.,
                'ori_gate_slope': 1.
            }
        }

        self._add_robot(GlobalTargetRobot(world=self.world,
                                          world_bounds=self.world_bounds,
                                          position=(0, 0),
                                          orientation=0,
                                          ctrl_params=controller_params))

        num_guppies = 1
        positions = np.random.normal(size=(num_guppies, 2), scale=.02) + (.05, .05)
        orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
        for p, o in zip(positions, orientations):
            self._add_guppy(BoostCouzinGuppy(
                world=self.world,
                world_bounds=self.world_bounds,
                position=p, orientation=o
            ))


target_points = np.array([[-.3, .3],
                          [.3, .3],
                          [.3, -.3],
                          [-.3, -.3],
                          [-.3, .3]])

if __name__ == '__main__':
    # env = LocalObservationsWrapper(TestEnv())
    env = TestEnv()
    # env = FeedbackInspectionWrapper(TestEnv())
    env.reset()
    # env.video_path = 'video_out'

    obs_listx = []
    obs_listy = []

    for t, a in zip(range(20), cycle(target_points)):
        env.render(mode='human')

        # state_t, reward_t, done, info = env.step(np.array([1.366212, 0.859359]))
        # state_t, reward_t, done, info = env.step(env.action_space.sample())
        obs, reward_t, done, info = env.step(a)
        obs_listx.append(obs[0][0])
        obs_listy.append(obs[0][1])

    plt.plot(obs_listx, obs_listy)
    plt.show()
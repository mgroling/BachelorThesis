import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.append("gym-guppy")
from gym_guppy import GuppyEnv, TurnSpeedRobot, BoostCouzinGuppy, TurnSpeedGuppy

if __name__ == "__main__":

    class SimpleGuppy(TurnSpeedGuppy, TurnSpeedRobot):
        _frequency = 20

        def __init__(self, **kwargs):
            super(SimpleGuppy, self).__init__(**kwargs)

        def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
            print("computing action")
            self.speed = 0.01 * self._frequency
            self.turn = np.pi / 5

        def step(self, time_step):
            print("stepping")
            self.set_angular_velocity(0)
            if self.turn:
                self._body.angle += self.turn
                self.__turn = None
            if self.speed:
                self.set_linear_velocity([self.speed, 0.0], local=True)
                self.__speed = None

    class TestEnv(GuppyEnv):
        def _reset(self):
            # set frequency of guppies to 20Hz
            self._guppy_steps_per_action = 5

            num_guppies = 1
            positions = np.random.normal(size=(num_guppies, 2), scale=0.02) + (
                0.05,
                0.05,
            )
            orientations = np.random.random_sample(num_guppies) * 2 * np.pi - np.pi
            for p, o in zip(positions, orientations):
                self._add_guppy(
                    SimpleGuppy(
                        world=self.world,
                        world_bounds=self.world_bounds,
                        position=p,
                        orientation=o,
                    )
                )

    env = TestEnv(time_step_s=0.05)

    track = np.stack([env.step(action=None)[0] for _ in range(100)])

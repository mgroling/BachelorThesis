import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import robofish.io
from scipy.spatial import cKDTree

sys.path.append("gym-guppy")
from gym_guppy import GuppyEnv, TurnSpeedRobot, BoostCouzinGuppy, TurnSpeedGuppy


def testRobot():
    # Real poses
    f = robofish.io.File("Fish/Guppy/src/turn_speed_pipeline/live.hdf5")
    poses = f.entity_poses[0, :, :2]

    n = poses.shape[0]
    diff = np.diff(poses, axis=0)
    norm = np.linalg.norm(diff, axis=1)
    angles = np.arctan2(diff[:, 1], diff[:, 0])

    turn = np.zeros_like(angles)
    turn[0] = angles[0]
    turn[1:] = np.diff(angles)
    turn = np.where(turn > np.pi, turn - 2 * np.pi, turn)
    turn = np.where(turn < -np.pi, turn + 2 * np.pi, turn)

    class TestEnv(GuppyEnv):
        def _reset(self):
            # set frequency to 20Hz
            self._guppy_steps_per_action = 5

            self._add_robot(
                TurnSpeedRobot(
                    world=self.world,
                    world_bounds=self.world_bounds,
                    position=(poses[0, 0] / 100, poses[0, 1] / 100),
                    orientation=0,
                )
            )

    env = TestEnv(steps_per_robot_action=5)

    new_poses = np.zeros((n, 3))
    new_poses[0] = env.reset()[0]

    for i in range(0, poses.shape[0] - 1):
        obs = env.step([turn[i], norm[i] / 100 * 20])[0]
        new_poses[i + 1] = obs[0]
        print(new_poses[i + 1] % (2 * np.pi), angles[i])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(poses[:, 0], poses[:, 1])
    ax[0].set_title("Original trajectory")
    ax[1].plot(new_poses[:, 0] * 100, new_poses[:, 1] * 100)
    ax[1].set_title("Generated trajectory")
    plt.show()


def testGuppy():
    # Real poses
    f = robofish.io.File("Fish/Guppy/src/turn_speed_pipeline/live.hdf5")
    poses = f.entity_poses[0, :, :2]

    n = poses.shape[0]
    diff = np.diff(poses, axis=0)
    norm = np.linalg.norm(diff, axis=1)
    angles = np.arctan2(diff[:, 1], diff[:, 0])

    turn = np.zeros_like(angles)
    turn[0] = angles[0]
    turn[1:] = np.diff(angles)
    turn = np.where(turn > np.pi, turn - 2 * np.pi, turn)
    turn = np.where(turn < -np.pi, turn + 2 * np.pi, turn)

    class TestGuppy(TurnSpeedGuppy, TurnSpeedRobot):
        _frequency = 20

        def __init__(self, turn, speed, **kwargs):
            super(TestGuppy, self).__init__(**kwargs)
            self.turn_tra = turn
            self.speed_tra = speed

        def compute_next_action(self, state: np.ndarray, kd_tree: cKDTree = None):
            self.turn = self.turn_tra[timestep]
            self.speed = self.speed_tra[timestep] / 100 * self._frequency

    class TestEnv(GuppyEnv):
        def _reset(self):
            # set frequency to 20Hz
            self._guppy_steps_per_action = 5

            self._add_guppy(
                TestGuppy(
                    turn=turn,
                    speed=norm,
                    world=self.world,
                    world_bounds=self.world_bounds,
                    position=(poses[0, 0] / 100, poses[0, 1] / 100),
                    orientation=0,
                )
            )

    env = TestEnv(steps_per_robot_action=5)

    new_poses = np.zeros((n, 3))
    new_poses[0] = env.reset()[0]

    for timestep in range(0, poses.shape[0] - 1):
        obs = env.step(action=None)[0]
        new_poses[timestep + 1] = obs[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(poses[:, 0], poses[:, 1])
    ax[0].set_title("Original trajectory")
    ax[1].plot(new_poses[:, 0] * 100, new_poses[:, 1] * 100)
    ax[1].set_title("Generated trajectory")
    plt.show()


if __name__ == "__main__":
    testRobot()
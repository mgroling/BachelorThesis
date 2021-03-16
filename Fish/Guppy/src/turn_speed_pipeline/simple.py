import numpy as np
import robofish.io

# Real poses
f = robofish.io.File("Fish/Guppy/src/turn_speed_pipeline/live.hdf5")
poses = f.entity_poses[0, :, :2]

# Test poses
# poses = np.array([[0, 0], [1, 0], [1, 1], [2, 1]])

n = poses.shape[0]
print("poses")
print(poses)

diff = np.diff(poses, axis=0)
print("diff")
print(diff)

norm = np.linalg.norm(diff, axis=1)
print("norm")
print(norm)

angles = np.arctan2(diff[:, 1], diff[:, 0])
print("angles")
print((angles / (np.pi) * 180))

turn = np.zeros_like(angles)
turn[0] = angles[0]
turn[1:] = np.diff(angles)

print("turn")
print(turn * 180 / np.pi)

# execute

new_poses = np.zeros((n, 3))
new_poses[0, :2] = poses[0]

for i in range(0, poses.shape[0] - 1):
    new_orientation = new_poses[i, 2] + turn[i]

    new_poses[i + 1, 0] = new_poses[i, 0] + np.cos(new_orientation) * norm[i]
    new_poses[i + 1, 1] = new_poses[i, 1] + np.sin(new_orientation) * norm[i]
    new_poses[i + 1, 2] = new_orientation

print(new_poses)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(poses[:, 0], poses[:, 1])
ax[0].set_title("Original trajectory")
ax[1].plot(new_poses[:, 0], new_poses[:, 1])
ax[1].set_title("Created trajectory")
plt.show()

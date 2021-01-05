import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("I:/Code/io/src")
import robofish.io

def evaluate_turn(paths):
    files_per_path = [robofish.io.read_multiple_files(p) for p in paths]
    turns = []
    for files in files_per_path:
        path_turns = []
        for p, file in files.items():
            poses = file.get_poses_array()
            for e_poses in poses:
                #https://math.stackexchange.com/questions/180874/convert-angle-radians-to-a-heading-vector
                #let's assume it look like this: x, y, ori_radians
                e_turns = e_poses[1:, 2] - e_poses[:-1, 2]
                path_turns.extend(e_turns)
        turns.append(path_turns)

    plt.hist(turns, bins = 20, label = paths, density = True, range = [-np.pi, np.pi])
    plt.title("Agent turns")
    plt.xlabel("Change in orientation (radians)")
    plt.ylabel("Frequency")
    plt.ticklabel_format(useOffset = False)
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_distanceToWall(paths):
    files_per_path = [robofish.io.read_multiple_files(p) for p in paths]
    distances = []
    world_bounds = [-0.5, -0.5, 0.5, 0.5]
    wall_lines = [(world_bounds[0], world_bounds[2]), (world_bounds[0], world_bounds[3]), (world_bounds[1], world_bounds[2]), (world_bounds[1], world_bounds[3])]
    for files in files_per_path:
        path_distances = []
        for p, file in files.items():
            poses = file.get_poses_array()
            for e_poses in poses:
                for wall in wall_lines:
                    pass

def dist_line_point(x1, y1, x2, y2, x3, y3):
    """
    returns distance between line (x1, y1, x2, y2) and point (x3, y3)
    from: https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    theory: http://paulbourke.net/geometry/pointlineplane/
    """
    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = (dx*dx + dy*dy)**.5

    return dist

# print(robofish.io.File("Fish/Guppy/io/DQN_19_12_2020_02_1models_det.hdf5"))
# print(list(robofish.io.read_multiple_files(["Fish/Guppy/io/DQN_19_12_2020_02_1models_det.hdf5"]).values())[0])
evaluate_turn([["Fish/Guppy/io/DQN_19_12_2020_02_1models_det.hdf5"], ["Fish/Guppy/io/DQN_19_12_2020_02_1models_det_01.hdf5"]])

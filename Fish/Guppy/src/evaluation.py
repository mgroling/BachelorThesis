from itertools import combinations

import numpy as np
import pandas as pd
from numpy import float32
from numpy.linalg import norm
from pandas import DataFrame

def normalize_series(x):
    """
    Given a series of vectors, return a series of normalized vectors.
    Null vectors are mapped to `NaN` vectors.
    """
    return (x.T / norm(x, axis=-1)).T

def calc_iid(a, b):
    """
    Given two series of poses - with X and Y coordinates of their positions as the first two elements -
    return the inter-individual distance (between the positions).
    """
    return norm(b[:, :2] - a[:, :2], axis=-1)

def calc_tlvc(a, b, tau_min, tau_max):
    """
    Given two velocity series and both minimum and maximum time lag return the
    time lagged velocity correlation from the first to the second series.
    """
    length = tau_max - tau_min
    return float32(
        [
            (a[t] @ b[t + tau_min :][:length].T).mean()
            for t in range(min(len(a), len(b) - tau_max + 1))
        ]
    )

def calc_follow(a, b):
    """
    Given two series of poses - with X and Y coordinates of their positions as the first two elements -
    return the follow metric from the first to the second series.
    """
    a_v = a[1:, :2] - a[:-1, :2]
    b_p = normalize_series(b[:-1, :2] - a[:-1, :2])
    return (a_v * b_p).sum(axis=-1)

def calc_follow_iid(tracksets):
    follow = []
    iid = []

    for trackset in tracksets:
        tracks = list(x[:] for x in trackset.values())
        for a, b in combinations(tracks, 2):
            iid.append(calc_iid(a[:-1], b[:-1]))
            iid.append(iid[-1])

            follow.append(calc_follow(a, b))
            follow.append(calc_follow(b, a))
    return DataFrame(
        {
            "IID [cm]": np.concatenate(iid, axis=0),
            "Follow": np.concatenate(follow, axis=0),
        }
    )

def calc_tlvc_iid(tracksets, time_step, tau_seconds=(0.3, 1.3)):
    tau_min_seconds, tau_max_seconds = tau_seconds

    tlvc = []
    iid = []

    tau_min_frames = int(tau_min_seconds * 1000.0 / time_step)
    tau_max_frames = int(tau_max_seconds * 1000.0 / time_step)

    for trackset in tracksets:
        tracks = list(x[:] for x in trackset.values())
        for a, b in combinations(tracks, 2):
            iid.append(calc_iid(a[1 : -tau_max_frames + 1], b[1 : -tau_max_frames + 1]))
            iid.append(iid[-1])

            a_v = a[1:, :2] - a[:-1, :2]
            b_v = b[1:, :2] - b[:-1, :2]
            tlvc.append(calc_tlvc(a_v, b_v, tau_min_frames, tau_max_frames))
            tlvc.append(calc_tlvc(b_v, a_v, tau_min_frames, tau_max_frames))
    return DataFrame(
        {"IID [cm]": np.concatenate(iid, axis=0), "TLVC": np.concatenate(tlvc, axis=0)}
    )

def collect_tracksets(trajectory_paths):
    tracksets = []
    for path in trajectory_paths:
        ar = pd.read_csv(path, sep = ";").to_numpy()

        dic = {}

        for i in range(0, ar.shape[1]//3):
            trajectory = ar[:, i*3:i*3+3]
            #convert orientation from radians to normalized orientation vector
            ori_vec = normalize_series(np.array([trajectory[:, 0] + np.cos(trajectory[:, 2]), trajectory[:, 1] + np.cos(trajectory[:, 2])]).T)
            #get new trajectory with x, y, ori_x, ori_y
            temp = np.append(trajectory[:, [0, 1]], ori_vec, axis = 1)

            dic[i] = temp

        tracksets.append(dic)

    return tracksets

print(calc_follow_iid(collect_tracksets(["Fish/Guppy/trajectories/trajectory_0.csv"])))
# print(calc_tlvc_iid(collect_tracksets(["Fish/Guppy/trajectories/trajectory_0.csv"]), 9999))
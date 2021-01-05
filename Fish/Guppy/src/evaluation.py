from itertools import combinations

import seaborn
import numpy as np
import pandas as pd
import os
from numpy import float32
from numpy.linalg import norm
from pandas import DataFrame
import matplotlib.pyplot as plt

### DISCLAIMER: these functions are not my own!


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
        ar = pd.read_csv(path, sep=";").to_numpy()

        dic = {}

        for i in range(0, ar.shape[1] // 3):
            trajectory = ar[:, i * 3 : i * 3 + 3]
            # convert orientation from radians to normalized orientation vector
            ori_vec = normalize_series(
                np.array(
                    [
                        trajectory[:, 0] + np.cos(trajectory[:, 2]),
                        trajectory[:, 1] + np.cos(trajectory[:, 2]),
                    ]
                ).T
            )
            # get new trajectory with x, y, ori_x, ori_y
            temp = np.append(trajectory[:, [0, 1]], ori_vec, axis=1)

            dic[i] = temp

        tracksets.append(dic)

    return tracksets


def plot_tankpositions(trackset, multipletracksets=False, size=(25, 25), path=None):
    """
    Heatmap of fishpositions
    By Moritz Maxeiner
    """
    x_pos = []
    y_pos = []

    for track in trackset:
        for value in track.values():
            x_pos.append(value[:, 0])
            y_pos.append(value[:, 1])

    x_pos = np.concatenate(x_pos, axis=0)
    y_pos = np.concatenate(y_pos, axis=0)
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)

    # print(x_pos.shape, y_pos.shape)

    seaborn.kdeplot(x_pos, y_pos * (-1), n_levels=25, shade=True, ax=ax)

    if path is None:
        return fig
    else:
        fig.savefig(path)


def plot_trajectories(tracks, size=(25, 25), path=None):
    """
    Plots tank trajectory of fishes
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    assert tracks.shape[-1] % 2 == 0
    nfish = int(tracks.shape[-1] / 2)

    data = {
        fish: pd.DataFrame(
            {
                "x": tracks[:, fish * 2],
                "y": tracks[:, fish * 2 + 1],
            }
        )
        for fish in range(nfish)
    }
    combined_data = pd.concat(
        [data[fish].assign(Agent=f"Agent {fish}") for fish in data.keys()]
    )

    fig, ax = plt.subplots(figsize=size)

    seaborn.set_style("white", {"axes.linewidth": 2, "axes.edgecolor": "black"})

    seaborn.scatterplot(
        x="x", y="y", hue="Agent", linewidth=0, s=16, data=combined_data, ax=ax
    )
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.invert_yaxis()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_position("left")

    ax.scatter(
        [frame["x"][0] for frame in data.values()],
        [frame["y"][0] for frame in data.values()],
        marker="h",
        c="black",
        s=64,
        label="Start",
    )
    ax.scatter(
        [frame["x"][len(frame["x"]) - 1] for frame in data.values()],
        [frame["y"][len(frame["y"]) - 1] for frame in data.values()],
        marker="x",
        c="black",
        s=64,
        label="End",
    )
    ax.legend()

    if path is None:
        return fig
    else:
        fig.savefig(path)


def get_indices(i):
    """
    returns right indices for fishpositions
    """
    return (2 * i, 2 * i + 1)


def plot_tlvc_iid(
    tracks,
    time_step=(1000 / 30),
    tau_seconds=(0.3, 1.3),
    multipletracksets=False,
    path=None,
):
    """
    TLVC_IDD by Moritz Maxeiner
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    """
    tau_min_seconds, tau_max_seconds = tau_seconds

    tau_min_frames = int(tau_min_seconds * 1000.0 / time_step)
    tau_max_frames = int(tau_max_seconds * 1000.0 / time_step)

    tlvc = []
    iid = []

    if not multipletracksets:
        assert tracks.shape[-1] % 2 == 0
        nfish = int(tracks.shape[-1] / 2)
        # for every fish combination calculate the follow
        for i1 in range(nfish):
            for i2 in range(i1 + 1, nfish):
                f1_x, f1_y = get_indices(i1)
                f2_x, f2_y = get_indices(i2)
                iid.append(
                    calc_iid(
                        tracks[1 : -tau_max_frames + 1, f1_x : f1_y + 1],
                        tracks[1 : -tau_max_frames + 1, f2_x : f2_y + 1],
                    )
                )
                iid.append(iid[-1])

                a_v = tracks[1:, f1_x : f1_y + 1] - tracks[:-1, f1_x : f1_y + 1]
                b_v = tracks[1:, f2_x : f2_y + 1] - tracks[:-1, f2_x : f2_y + 1]
                tlvc.append(calc_tlvc(a_v, b_v, tau_min_frames, tau_max_frames))
                tlvc.append(calc_tlvc(b_v, a_v, tau_min_frames, tau_max_frames))
    else:
        for trackset in tracks:
            assert trackset.shape[-1] % 2 == 0
            nfish = int(trackset.shape[-1] / 2)
            # for every fish combination calculate the follow
            for i1 in range(nfish):
                for i2 in range(i1 + 1, nfish):
                    f1_x, f1_y = get_indices(i1)
                    f2_x, f2_y = get_indices(i2)
                    iid.append(
                        calc_iid(
                            trackset[1 : -tau_max_frames + 1, f1_x : f1_y + 1],
                            trackset[1 : -tau_max_frames + 1, f2_x : f2_y + 1],
                        )
                    )
                    iid.append(iid[-1])

                    a_v = trackset[1:, f1_x : f1_y + 1] - trackset[:-1, f1_x : f1_y + 1]
                    b_v = trackset[1:, f2_x : f2_y + 1] - trackset[:-1, f2_x : f2_y + 1]
                    tlvc.append(calc_tlvc(a_v, b_v, tau_min_frames, tau_max_frames))
                    tlvc.append(calc_tlvc(b_v, a_v, tau_min_frames, tau_max_frames))

    tlvc_iid_data = pd.DataFrame(
        {"IID [m]": np.concatenate(iid, axis=0), "TLVC": np.concatenate(tlvc, axis=0)}
    )

    grid = seaborn.jointplot(
        x="IID [m]", y="TLVC", data=tlvc_iid_data, linewidth=0, s=1, kind="scatter"
    )
    # grid.ax_joint.set_xlim(0, 700)
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.subplots_adjust(top=0.9)

    if path is None:
        return grid.fig
    else:
        grid.fig.savefig(path)


def plot_follow_iid(tracks, multipletracksets=False, path=None):
    """
    plots fancy graph with follow and iid, only use with center values
    Expects one node per fish at max: tracks: [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              [fish1_x, fish1_y, fish2_x, fish2_y,..]
                                              ...
    copied from Moritz Maxeiner
    """
    follow = []
    iid = []

    if not multipletracksets:
        assert tracks.shape[-1] % 2 == 0
        nfish = int(tracks.shape[-1] / 2)
        # for every fish combination calculate the follow
        for i1 in range(nfish):
            for i2 in range(i1 + 1, nfish):
                f1_x, f1_y = get_indices(i1)
                f2_x, f2_y = get_indices(i2)
                print(tracks[:-1, f1_x : f1_y + 1].shape)
                iid.append(
                    calc_iid(tracks[:-1, f1_x : f1_y + 1], tracks[:-1, f2_x : f2_y + 1])
                )
                iid.append(iid[-1])

                follow.append(
                    calc_follow(tracks[:, f1_x : f1_y + 1], tracks[:, f2_x : f2_y + 1])
                )
                follow.append(
                    calc_follow(tracks[:, f2_x : f2_y + 1], tracks[:, f1_x : f1_y + 1])
                )
    else:
        for trackset in tracks:
            assert trackset.shape[-1] % 2 == 0
            nfish = int(trackset.shape[-1] / 2)
            for i1 in range(nfish):
                for i2 in range(i1 + 1, nfish):
                    f1_x, f1_y = get_indices(i1)
                    f2_x, f2_y = get_indices(i2)
                    iid.append(
                        calc_iid(
                            trackset[:-1, f1_x : f1_y + 1],
                            trackset[:-1, f2_x : f2_y + 1],
                        )
                    )
                    # iid.append(iid[-1])

                    follow.append(
                        calc_follow(
                            trackset[:, f1_x : f1_y + 1], trackset[:, f2_x : f2_y + 1]
                        )
                    )
                    # follow.append(calc_follow(trackset[:, f2_x:f2_y + 1], trackset[:, f1_x:f1_y + 1]))

    follow_iid_data = pd.DataFrame(
        {
            "IID [m]": np.concatenate(iid, axis=0),
            "Follow": np.concatenate(follow, axis=0),
        }
    )

    grid = seaborn.jointplot(
        x="IID [m]", y="Follow", data=follow_iid_data, linewidth=0, s=1, kind="scatter"
    )
    grid.fig.set_figwidth(9)
    grid.fig.set_figheight(6)
    grid.fig.subplots_adjust(top=0.9)

    if path is None:
        return grid.fig
    else:
        grid.fig.savefig(path)


def plot_dist_ori(paths, size=(18, 18), path=None):
    ar = pd.read_csv(paths[0], sep=";").to_numpy()
    distances = np.sqrt(
        np.power(ar[1:, 0] - ar[:-1, 0], 2) + np.power(ar[1:, 1] - ar[:-1, 1], 2)
    )
    orientations = ar[1:, 2] - ar[:-1, 2]

    fig_angular, ax = plt.subplots(figsize=size)
    fig_angular.subplots_adjust(top=0.93)
    ax.set_xlim(-np.pi, np.pi)
    seaborn.distplot(
        pd.Series(orientations, name="Angular movement"),
        ax=ax,
        hist_kws={"rwidth": 0.9, "color": "y"},
    )

    fig_linear, ax = plt.subplots(figsize=size)
    fig_linear.subplots_adjust(top=0.93)
    # ax.set_xlim(-20, 20)
    seaborn.distplot(
        pd.Series(distances, name="Linear movement"),
        ax=ax,
        hist_kws={"rwidth": 0.9, "color": "y"},
    )

    if path is None:
        return fig_angular, fig_linear
    else:
        fig_angular.savefig(path + "fig_angular.png")
        fig_linear.savefig(path + "fig_linear.png")


def createPlots(paths, folder_path, multipletracksets=False):
    if not multipletracksets:
        trackset = collect_tracksets(paths)
        plot_tankpositions(
            trackset, size=(18, 18), path=folder_path + "tankpositions_all.png"
        )

        trajectories = np.empty((len(trackset[0][0]), len(trackset[0]) * 2))
        for track in trackset:
            i = 0
            for value in track.values():
                trajectories[:, [i * 2, i * 2 + 1]] = value[:, [0, 1]]
                i += 1
        plot_trajectories(
            trajectories, size=(18, 18), path=folder_path + "trajectories_all.png"
        )
        plot_trajectories(
            trajectories[:, 0:2],
            size=(18, 18),
            path=folder_path + "trajectories_agent_0.png",
        )
        plot_trajectories(
            trajectories[:, 2:4],
            size=(18, 18),
            path=folder_path + "trajectories_agent_1.png",
        )
        plot_follow_iid(trajectories, path=folder_path + "follow_iid.png")
        plot_tlvc_iid(
            trajectories, time_step=(1000 / 20), path=folder_path + "tlvc_iid.png"
        )

        plot_dist_ori(paths, path=folder_path)

    else:
        pass


if __name__ == "__main__":
    # data_names = [elem for elem in os.listdir("Fish/Guppy/reducedData")]

    # for elem in data_names:
    #     print(elem)
    #     os.mkdir("Fish/Guppy/plots/liveData/" + elem[:-4])
    #     createPlots(["Fish/Guppy/reducedData/" + elem], "Fish/Guppy/plots/liveData/" + elem[:-4] + "/")

    createPlots(
        ["Fish/Guppy/trajectories/DQN_22_12_2020_03_1models_det.csv"],
        "Fish/Guppy/plots/DQN_22_12_2020_03_1models_det/",
    )

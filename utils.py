"""
Utility functions
"""
import numpy as np
from dataclasses import dataclass
import pandas as pd
from typing import List
from os import path
import os


@dataclass
class ChamberEstimate:
    """Class for storing the estimated boundaries of the behavior chamber"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def get_data_weights(data: np.ndarray, norm_bins: np.ndarray, norm_values: np.ndarray) -> np.ndarray:
    """
    Uses a binned normalization curve to compute weights for a set of data points
    :param data: The datapoints to be weighted
    :param norm_bins: The bins of the normalization curve
    :param norm_values: The values of the normalization curve
    :return: data.size vector of weights - the inverse of the normalization values of the bin to which datum belongs
    """
    bin_ix = np.digitize(data, norm_bins, right=True) - 1
    bin_ix[bin_ix >= norm_values.size] = norm_values.size - 1
    bin_ix[bin_ix < 0] = 0
    return 1 / norm_values[bin_ix].astype(float)


def bootstrap(data: np.ndarray, nboot: int, bootfun: callable) -> np.ndarray:
    """
    For a a n_samples x m_features array creates nboot bootstrap variates of bootfun
    :param data: The data to be bootstrapped
    :param nboot: The number of boostrap variates to create
    :param bootfun: The function to apply, must take axis parameter
    :return: nboot x m_features array of bootstrap variates
    """
    indices = np.arange(data.shape[0]).astype(int)
    variates = np.full((nboot, data.shape[1]), np.nan)
    for i in range(nboot):
        chosen = np.random.choice(indices, data.shape[0], True)
        variates[i, :] = bootfun(data[chosen], axis=0)
    return variates


def bootstrap_weighted_histogram_avg(bin_data: np.ndarray, bins: np.ndarray, weights: np.ndarray, n_boot:int) -> np.ndarray:
    """
    Computes bootstrap variates of a weighted histogram average (i.e. average value of quantity in "weights" binned by
    "bin_data" according to "bins"
    :param bin_data: The data by which to bin
    :param bins: The bin boundaries
    :param weights: The weights, i.e. the data that will be averaged in each bin
    :param n_boot: The number of bootstrap variates to create
    :return: n_boot x n_bins-1 matrix of bootstrap variates
    """
    if bin_data.size != weights.size:
        raise ValueError("bin_data and weights must have same size (i.e., must be paired)")
    indices = np.arange(bin_data.size).astype(int)
    variates = np.full((n_boot, bins.size-1), np.nan)
    for i in range(n_boot):
        chosen = np.random.choice(indices, indices.size, True)
        these_data = bin_data[chosen]
        these_weights = weights[chosen]
        wh = np.histogram(these_data, bins=bins, weights=these_weights)[0].astype(float)
        ch = np.histogram(these_data, bins=bins)[0].astype(float)
        variates[i, :] = wh / ch
    return variates


def edge_filter_bouts(chamber: ChamberEstimate, edge_distance: float, bout_data: pd.DataFrame,
                      fish_data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all bouts that are close to the chamber edge
    :param chamber: Definition of the likely chamber boundaries
    :param edge_distance: Minimal distance in mm from the edge for bouts to be valid
    :param bout_data: The original bout data to filter
    :param fish_data: The experimental data to find the respective positions of the bouts
    :return: New dataframe with only the bouts that are outside the edge area
    """
    bout_x = np.array(fish_data["X Position"][bout_data["Start"]])
    bout_y = np.array(fish_data["Y Position"][bout_data["Start"]])
    val_x = np.logical_and(bout_x - chamber.x_min > edge_distance, chamber.x_max - bout_x > edge_distance)
    val_y = np.logical_and(bout_y - chamber.y_min > edge_distance, chamber.y_max - bout_y > edge_distance)
    valid = np.logical_and(val_x, val_y)
    df_filtered = bout_data.iloc[valid].copy(deep=True)
    print(f"{np.round(df_filtered.shape[0]/bout_data.shape[0]*100, 0)} % of bouts passed filtering.")
    return df_filtered


def collect_trajectories(df: pd.DataFrame, frame_rate=100, n_seconds=2):
    """
    From one bout structure generates a list of bout structures with each structure representing
    one continuous trajectory
    :param df: A bout dataframe
    :param frame_rate: To convert bout distances into time for trajectory breaking
    :param n_seconds: Maximal delta-t between consecutive bout starts within a trajectory
    :return: List of trajectory bout dataframes
    """
    traj_breaks = np.r_[1, np.diff(df['Original index'])] > 1
    # add additional trajectory breaks for bouts that are more than n_seconds apart
    delta_time = np.r_[0, np.diff(df['Start'])] / frame_rate
    traj_breaks = np.logical_or(traj_breaks, delta_time > n_seconds)
    traj_starts = np.r_[0, np.where(traj_breaks)[0], traj_breaks.size]
    traj_list = []
    for i, ts in enumerate(traj_starts[:-1]):
        traj_list.append(df.iloc[ts:traj_starts[i+1]])
    return traj_list


def find_all_pickle_paths(root_f: str) -> List[str]:
    """
    Recursively searches for pandas pickle files identified by their .pk extension
    :param root_f: The folder from which to start the current search
    :return: List of all .pk files at and below root_f with their full path
    """
    try:
        objects = os.listdir(root_f)
    except PermissionError:
        return []
    exp_local = [path.join(root_f, o) for o in objects if ".pk" in o and "fish" not in o]
    dir_local = [path.join(root_f, o) for o in objects if path.isdir(path.join(root_f, o))]
    exp_deep = []
    for dl in dir_local:
        e_deep = find_all_pickle_paths(dl)
        exp_deep += e_deep
    return exp_local + exp_deep


def load_all_trajectories(folder: str, len_thresh=3) -> List[pd.DataFrame]:
    """
    Load all trajectory information from pandas pickle files in the given folder
    :param folder: The folder to search for files
    :param len_thresh: Remove all trajectories with a length less than this value
    :return: Aggregated list of all trajectories
    """
    pickle_files = find_all_pickle_paths(folder)
    pickle_files = sorted(pickle_files)
    trajectories = []
    for fishid, pf in enumerate(pickle_files):
        temp_npy_file = path.splitext(pf)[0] + "_activity.npy"
        if path.exists(temp_npy_file):
            bout_activity = np.load(temp_npy_file)
        else:
            bout_activity = None
        df = pd.read_pickle(pf)
        if bout_activity is not None:
            assert bout_activity.shape[0] == df.shape[0]
            assert bout_activity.shape[1] % 3 == 0  # we assume that each cluster is represented by three timepoints
        # insert fish ids
        df.insert(0, f"Fish ID", fishid, False)
        # insert average cluster activity for each bout if present
        if bout_activity is not None:
            for clix in range(bout_activity.shape[1] // 3):
                df.insert(0, f"Cluster activity_{clix}", np.mean(bout_activity[:, clix*3:(clix+1)*3], axis=1), False)
        possibles = collect_trajectories(df)
        trajectories += [p for p in possibles if p.shape[0] >= len_thresh]
    return trajectories


def split_reversal_trajectories(trajectory: pd.DataFrame, aln_thresh: float) -> List:
    """
    Takes a continuous trajectory of bouts and identifies all subtrajectories (if any) that contain
    reversals in gradient direction as defined by transitioning from one aligned state to the opposite
    :param trajectory: The trajectory to scan for reversals
    :param aln_thresh: The cosine threshold for considering a bout positively or negative aligned
    :return: List of reversal trajectories from the first aligned bout to the first bout in the opposite alignment
    """
    gdir = np.array(trajectory["Gradient direction"])
    ix_neg = np.where(gdir < -aln_thresh)[0]
    ix_pos = np.where(gdir > aln_thresh)[0]
    # to have at least one reversal, both indices need to have at least one element
    if ix_neg.size == 0 or ix_pos.size == 0:
        return []
    if ix_neg[0] < ix_pos[0]:
        return [trajectory[ix_neg[0]:ix_pos[0]+1]] + split_reversal_trajectories(trajectory[ix_pos[0]:], aln_thresh)
    else:
        return [trajectory[ix_pos[0]:ix_neg[0]+1]] + split_reversal_trajectories(trajectory[ix_neg[0]:], aln_thresh)


def split_persistent_trajectories(trajectory: pd.DataFrame, aln_thresh: float) -> List:
    """
    Takes a continuous trajectory of bouts and identifies subtrajectories for which all bouts are fully aligned
    to the same gradient direction
    :param trajectory: The trajectory to scan for persistence
    :param aln_thresh: The cosine threshold for considering a bout positively or negative aligned
    :return: List of trajectories that are fully persistent in gradient direction
    """
    gdir = np.array(trajectory["Gradient direction"])
    g_alignment = np.zeros(gdir.size)
    g_alignment[gdir > aln_thresh] = 1
    g_alignment[gdir < -aln_thresh] = -1
    pos_aligned = []
    neg_aligned = []
    if np.any(g_alignment > 0):
        pos_aligned = collect_trajectories(trajectory[g_alignment > 0])
    if np.any(g_alignment < 0):
        neg_aligned = collect_trajectories(trajectory[g_alignment < 0])
    return pos_aligned + neg_aligned


def split_reorient_aborts(trajectory: pd.DataFrame, aln_thresh: float, rit: float, mrl: int) -> List:
    """
    Takes a continuous trajectory of bouts and identifies subtrajectories that indicate the start of reversal but
    for which the reversal is subsequently not completed
    :param trajectory: The input trajectory
    :param aln_thresh: The cosine threshold for considering a bout positively or negative aligned
    :param rit: The threshold over the first two turns to call a beginning of a reversal
    :param mrl: The maximal allowed length of true reversals
    :return: List of trajectories that are reversal aborts
    """
    if trajectory.shape[0] == 0:
        return []
    gdir = np.array(trajectory["Gradient direction"])
    g_alignment = np.zeros(gdir.size)
    g_alignment[gdir > aln_thresh] = 1
    g_alignment[gdir < -aln_thresh] = -1
    if np.sum(g_alignment != 0) == 0:
        return []
    ix_aligned = np.nonzero(g_alignment)[0]
    abt_trajs = []
    for ix in ix_aligned:
        if ix+3 >= trajectory.shape[0]:
            # we cannot judge whether this trajectory was an abort
            break
        if g_alignment[ix+1] != 0:
            # we are in a persistent stretch go to its end
            continue
        start_dir = g_alignment[ix]
        # test if the next three moves cross our potential reversal threshold
        if np.abs(gdir[ix+2] - gdir[ix]) >= rit:
            # this is a potential abort, the start of a realignment
            if np.any(g_alignment[ix+1:ix+mrl] == -1*start_dir):
                # this is actually a reversal
                abt_trajs += split_reorient_aborts(trajectory.iloc[ix+1:], aln_thresh, rit, mrl)
                break
            else:
                # this is an abort
                abt_trajs += [trajectory.iloc[ix:ix+mrl]]
                abt_trajs += split_reorient_aborts(trajectory.iloc[ix+mrl:], aln_thresh, rit, mrl)
                break
    return abt_trajs


def augment_state_info(bouts: pd.DataFrame, aln_thresh: float, max_rl: int) -> pd.DataFrame:
    """
    Assigns states (-1 = reversal; 0 = general; 1 = persistent) and gradient directions to each bout
    :param bouts: Bout dataframe to augment
    :param aln_thresh: The cosine threshold for considering a bout positively or negative aligned
    :param max_rl: The maximal allowed length of true reversals
    :return: Augmented bout dataframe
    """

    def label_states(gradient_directions: np.ndarray) -> np.ndarray:
        """
        Based on collection of gradient directions labels bout states
        :param gradient_directions: Cosine bout directions
        :return: Array of states
        """
        nonlocal aln_thresh
        nonlocal max_rl
        alignment = np.zeros(gradient_directions.size)
        alignment[gradient_directions >= aln_thresh] = 1
        alignment[gradient_directions <= -aln_thresh] = -1
        states = np.zeros(gradient_directions.size, dtype=int)
        # first label possible persistent states - a bout is persistent if it is aligned and the
        # previous bout was aligned as well
        for i, a in enumerate(alignment):
            if i > 0:
                if a != 0 and a == alignment[i-1]:
                    states[i] = 1
        # label reversals - these exist if the alignment becomes inverted within max_rl bouts - since we assume
        # that fish turn before they displace, the last aligned bout before the reversal is not part of the reversal
        # but the first bout aligned in the opposite direction is part of the reversal (the turn within that bout
        # effectively completed the reversal)
        ix_aligned = np.nonzero(alignment)[0]
        for i, ix in enumerate(ix_aligned[:-1]):
            if alignment[ix] != alignment[ix_aligned[i+1]] and ix_aligned[i+1] - ix < max_rl:
                # the alignment inverted from this aligned bout to the next and this happened within our length
                # threshold
                states[ix+1:ix_aligned[i+1]+1] = -1
        return states

    trajectories = collect_trajectories(bouts)
    b_in_traj = sum([t.shape[0] for t in trajectories])
    assert bouts.shape[0] == b_in_traj, f"DF had {bouts.shape[0]} bouts but trajectories only cover {b_in_traj}"
    # Loop over consecutive trajectories and find persistent and reversal stretches within and label them
    # Note: Since these are references the insertion should happen in-place
    for t in trajectories:
        grad_angles = np.array(np.arctan2(t["Delta X"], t["Delta Y"]))
        grad_direction = np.cos(grad_angles)
        bout_states = label_states(grad_direction)
        t.insert(0, "Gradient direction", grad_direction, False)
        t.insert(0, "State", bout_states, False)
    # Re-assemble bout dataframe and return
    return pd.concat(trajectories, axis=0, ignore_index=True)


def occupancy(data, axis=0):
    """
    Function to calculate average maximum-normalized occupancy from densities/proportions
    :param data: The densities/proportions n_samples x n_points
    :param axis: The axis across which to average
    :return: Occupancy
    """
    max_val = np.nanmax(np.mean(data, axis=axis))
    return np.mean(data, axis=axis)/max_val


if __name__ == '__main__':
    pass

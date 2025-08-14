"""
Analysis script for KAB gradient experiments
To be run in context of behavioral_fever repo
"""

import matplotlib as mpl
import matplotlib.pyplot as pl
import argparse
import os
from os import path
from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import loading
import pre_processing as preproc
import seaborn as sns
import plot_funs as pf
import utils
from utils import occupancy
from plot_funs import set_journal_style, remove_spines


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'cold' or self.dest == 'hot' or self.dest == 'constant':
            if not path.exists(values):
                raise argparse.ArgumentError(self, "Specified directory does not exist")
            if not path.isdir(values):
                raise argparse.ArgumentError(self, "The destination is a file but should be a directory")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def load_all(exp_list: List[str]) -> Tuple[List, List, List]:
    """
    For list of experiments returns pre-processed fish data and bout data frames
    :param exp_list: List of experimental info file paths
    :return:
        [0]: List of pre-processed fish dataframes
        [1]: List of extracted bout dataframes
    """
    all_fish_data = []
    all_bout_data = []
    all_expname_data = []
    # load
    for e in exp_list:
        fish = loading.load_exp_data_from_nwb(e)
        for k in fish:
            all_fish_data += [fs[0] for fs in fish[k]]
            all_expname_data += [f"{e}&&{fs[1]}" for fs in fish[k]]
    # pre-process
    for f in all_fish_data:
        preproc.pre_process_fish_data(f)
        all_bout_data.append(preproc.identify_bouts(f, 100))
    return all_fish_data, all_bout_data, all_expname_data


max_reversal_length = 10
align_thresh = 0.71  # within a cone of cos(alpha)>=align_thresh we consider bouts to be aligned to the gradient


if __name__ == '__main__':
    burn_in = 5*60*100  # 5-minute burn-in period

    mpl.rcParams['pdf.fonttype'] = 42
    set_journal_style(23, 23)
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="gradient_analysis",
                                       description="Runs analysis for hot and cold gradient experiments")
    a_parser.add_argument("-hf", "--hot", help="Path to folder with hot gradient experiments", type=str,
                          default="", action=CheckArgs)
    a_parser.add_argument("-cf", "--cold", help="Path to folder with cold gradient experiments", type=str,
                          default="", action=CheckArgs)

    args = a_parser.parse_args()

    hot_folder = args.hot
    cold_folder = args.cold

    hot_exp = loading.find_all_exp_paths(hot_folder)[0]
    cold_exp = loading.find_all_exp_paths(cold_folder)[0]

    all_exp = {"hot": hot_exp, "cold": cold_exp}

    plot_dir = "REVISION_KAB_Gradient"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    # load and process data
    all_fish = {}
    all_bouts = {}
    all_expnames = {}
    for k in all_exp:
        all_fish[k], all_bouts[k], all_expnames[k] = load_all(all_exp[k])

    # create dictionaries of valid experiments - experiments with at least one bout per second
    val_fish, val_bouts, val_expnames = {}, {}, {}
    for k in all_fish:
        for f, b, e in zip(all_fish[k], all_bouts[k], all_expnames[k]):
            if b.shape[0] >= f.shape[0]/100:
                if k not in val_fish:
                    val_fish[k] = []
                    val_bouts[k] = []
                    val_expnames[k] = []
                val_fish[k].append(f)
                val_bouts[k].append(b)
                val_expnames[k].append(e)

    # estimate confines of chamber
    x_min = min([np.nanmin(f["X Position"]) for k in val_fish for f in val_fish[k]])
    x_max = max([np.nanmax(f["X Position"]) for k in val_fish for f in val_fish[k]])
    y_min = min([np.nanmin(f["Y Position"]) for k in val_fish for f in val_fish[k]])
    y_max = max([np.nanmax(f["Y Position"]) for k in val_fish for f in val_fish[k]])
    chamber = utils.ChamberEstimate(x_min, x_max, y_min, y_max)

    border_size = 4  # Border size in 4mm - all data within will be excluded

    # filter bouts
    filtered_bouts = {}
    for k in val_bouts:
        filtered_bouts[k] = []
        for bout, fish in zip(val_bouts[k], val_fish[k]):
            filtered_bouts[k].append(utils.edge_filter_bouts(chamber, border_size, bout, fish))
    val_bouts = filtered_bouts

    temp_pref = {"Treatment": [], "Avg. temperature [C]": [], "Median temperature [C]": []}
    temp_distribution = {"Treatment": [], "Density": [], "Cumulative density": []}
    temp_oc_bins = np.linspace(18, 32, 29)  # 0.25 C bin width
    temp_oc_bc = temp_oc_bins[:-1] + np.diff(temp_oc_bins)/2

    for k in val_fish:
        for df_fish, df_bout in zip(val_fish[k], val_bouts[k]):
            temp_pref["Treatment"].append(k)
            temp_distribution["Treatment"].append(k)
            temps = df_fish['Temperature'][burn_in:]
            temps = temps[np.isfinite(temps)]
            temp_pref["Avg. temperature [C]"].append(np.nanmean(temps))
            temp_pref["Median temperature [C]"].append(np.nanmedian(temps))
            temp_distribution["Density"].append(np.histogram(temps, bins=temp_oc_bins, density=True)[0])
            h = np.histogram(temps, bins=temp_oc_bins, density=False)[0].astype(float)
            h /= h.sum()
            temp_distribution["Cumulative density"].append(np.cumsum(h))
    df_temp_pref = pd.DataFrame(temp_pref)
    df_temp_distribution = pd.DataFrame(temp_distribution)

    order = ["hot", "cold"]

    # plot temperature distribution densities with bootstrap statistics
    df_temp_dist_hot = df_temp_distribution[df_temp_distribution["Treatment"] == "hot"]
    df_temp_dist_cold = df_temp_distribution[df_temp_distribution["Treatment"] == "cold"]

    fig = pf.lineplot(df_temp_dist_hot, "Density", "Treatment", temp_oc_bc, "Temperature [C]", occupancy, add_marker=True)
    pl.xlim(24, 32)
    fig.savefig(path.join(plot_dir, "REVISION_1F_Occupancy_Hot.pdf"))

    fig = pf.lineplot(df_temp_dist_cold, "Density", "Treatment", temp_oc_bc, "Temperature [C]", occupancy, add_marker=True)
    pl.xlim(18, 26)
    fig.savefig(path.join(plot_dir, "REVISION_1G_Occupancy_Cold.pdf"))

    # analyze bout features by temperature
    temp_bins = np.linspace(18, 32, 29)
    temp_bincents = temp_bins[:-1] + np.diff(temp_bins) / 2
    temp_aln_distribution = {"Treatment": [], "Gradient alignment": [], "Gradient direction": []}

    for k in val_fish:
        for fish, bouts in zip(val_fish[k], val_bouts[k]):
            b_displace = bouts["Displacement"]
            b_mag = np.abs(np.rad2deg(bouts["Angle change"]))
            ibi = bouts["IBI"]
            # b_temps = fish["Temperature"][bouts["Start"]]
            # all_temps = fish["Temperature"]
            b_temps = bouts["Temperature"]
            grad_angles = np.array(np.arctan2(bouts["Delta X"], bouts["Delta Y"]))
            grad_direction = np.cos(grad_angles)
            temp_aln_distribution["Treatment"].append(k)
            a_hist = np.histogram(b_temps, bins=temp_bins, weights=np.abs(grad_direction))[0].astype(float)
            gd_hist = np.histogram(b_temps, bins=temp_bins, weights=grad_direction)[0].astype(float)
            n_hist = np.histogram(b_temps, bins=temp_bins)[0].astype(float)
            temp_aln_distribution["Gradient alignment"].append(a_hist / n_hist)
            temp_aln_distribution["Gradient direction"].append(gd_hist / n_hist)

    df_temp_aln_distribution = pd.DataFrame(temp_aln_distribution)

    fig = pf.lineplot(df_temp_aln_distribution, "Gradient alignment", "Treatment", temp_bincents, "Temperature [C]",
                      np.nanmean, add_marker=True)
    pl.plot([temp_bincents[0], temp_bincents[-1]], [0.64, 0.64], 'k--')  # expected alignment
    fig.savefig(path.join(plot_dir, "REVISION_2E_Gradient_alignment_by_temperature.pdf"))

    trajectories = {}
    for k in val_bouts:
        if k not in trajectories:
            trajectories[k] = []
        for bouts in val_bouts[k]:
            traj = utils.collect_trajectories(bouts, 100, 2)
            for t in traj:
                grad_angles = np.array(np.arctan2(t["Delta X"], t["Delta Y"]))
                grad_direction = np.cos(grad_angles)
                t.insert(0, "Gradient direction", grad_direction, False)
                trajectories[k].append(t)

    ###################################################################################################################
    # Length until reversal is completed
    ###################################################################################################################

    # go through trajectories - whenever a bout with significant alignment (<-0.7 > 0.7) is found count the number
    # of bouts until significant alignment in the opposite direction is identified. Then for such a trajectory store
    # starting temperature, starting alignment and number of bouts
    reversal_trajectories = []
    for k in trajectories:
        for t in trajectories[k]:
            reversal_trajectories += utils.split_reversal_trajectories(t, align_thresh)

    reversals = []
    for rt in reversal_trajectories:
        t = np.array(rt["Temperature"])[0]
        gd = np.array(rt["Gradient direction"])[0]
        ln = rt.shape[0]
        reversals.append(np.r_[t, gd, ln])
    reversals = np.vstack(reversals)
    # reversals that start with a positive direction (twds. inc. y), i.e. decreasing temperature
    pos_reversals = reversals[reversals[:, 1] > 0]
    # reversals that start with a negative direction (twds. dec. y), i.e. increasing temperature
    neg_reversals = reversals[reversals[:, 1] < 0]

    # reduce number of bins
    temp_bins = np.linspace(18, 32, 21)
    temp_bincents = temp_bins[:-1] + np.diff(temp_bins) / 2

    pos_res_bs = utils.bootstrap_weighted_histogram_avg(pos_reversals[:, 0], temp_bins, pos_reversals[:, 2], 1000)
    m_pos_res = np.nanmean(pos_res_bs, axis=0)
    e_pos_res = np.nanstd(pos_res_bs, axis=0)
    neg_res_bs = utils.bootstrap_weighted_histogram_avg(neg_reversals[:, 0], temp_bins, neg_reversals[:, 2], 1000)
    m_neg_res = np.nanmean(neg_res_bs, axis=0)
    e_neg_res = np.nanstd(neg_res_bs, axis=0)

    fig, axes = pl.subplots(ncols=2, sharey=True, figsize=(6.4*2, 4.8*2))
    cmap = pl.colormaps["viridis_r"]
    time_colors = cmap(np.linspace(0, 1, 40))
    time_colors[:, -1] = 0.7  # alpha
    for rt in reversal_trajectories:
        t = np.array(rt["Temperature"])[0]
        if t < 19 or t > 22:
            continue
        gd = np.array(rt["Gradient direction"])[0]
        tlen = rt.shape[0]
        if tlen >= time_colors.shape[0]:
            tlen = -1
        if gd > 0:
            ax = axes[0]
        else:
            ax = axes[1]
        ax.plot(np.cumsum(rt["Delta X"]), np.cumsum(rt["Delta Y"]), c=time_colors[tlen])
    axes[0].set_title("Cold going trajectories")
    axes[1].set_title("Hot going trajectories")
    fig.suptitle('19<=T>=22')
    fig.savefig(path.join(plot_dir, "REVISION_2I_Trajectories_from_cold.pdf"))

    fig, axes = pl.subplots(ncols=2, sharey=True, figsize=(6.4*2, 4.8*2))
    cmap = pl.colormaps["viridis_r"]
    time_colors = cmap(np.linspace(0, 1, 40))
    time_colors[:, -1] = 0.7  # alpha
    for rt in reversal_trajectories:
        t = np.array(rt["Temperature"])[0]
        if t < 28 or t > 31:
            continue
        gd = np.array(rt["Gradient direction"])[0]
        tlen = rt.shape[0]
        if tlen >= time_colors.shape[0]:
            tlen = -1
        if gd > 0:
            ax = axes[0]
        else:
            ax = axes[1]
        ax.plot(np.cumsum(rt["Delta X"]), np.cumsum(rt["Delta Y"]), c=time_colors[tlen])
    axes[0].set_title("Cold going trajectories")
    axes[1].set_title("Hot going trajectories")
    fig.suptitle('28<=T>=31')
    fig.savefig(path.join(plot_dir, "REVISION_2I_Trajectories_from_hot.pdf"))

    fig, axes = pl.subplots(ncols=2, sharey=True, figsize=(6.4*2, 4.8*2))
    cmap = pl.colormaps["viridis_r"]
    time_colors = cmap(np.linspace(0, 1, 40))
    time_colors[:, -1] = 0.7  # alpha
    for rt in reversal_trajectories:
        t = np.array(rt["Temperature"])[0]
        if t < 24 or t > 28:
            continue
        gd = np.array(rt["Gradient direction"])[0]
        tlen = rt.shape[0]
        if tlen >= time_colors.shape[0]:
            tlen = -1
        if gd > 0:
            ax = axes[0]
        else:
            ax = axes[1]
        ax.plot(np.cumsum(rt["Delta X"]), np.cumsum(rt["Delta Y"]), c=time_colors[tlen])
    axes[0].set_title("Cold going trajectories")
    axes[1].set_title("Hot going trajectories")
    fig.suptitle('24<=T>=28')
    fig.savefig(path.join(plot_dir, "REVISION_2I_Trajectories_from_pref.pdf"))

    fig, ax = pl.subplots(figsize=(1, 6), layout='constrained')
    norm = mpl.colors.Normalize(vmin=0, vmax=25)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='vertical', label="Trajectory length [bouts]")
    fig.savefig(path.join(plot_dir, "REVISION_2I_Trajectories_color_scale.pdf"))

    ###################################################################################################################
    # Length of actual reversals
    ###################################################################################################################
    # instead of "length until reversal" look at the "reversal time" in other words from the end of a reversal
    # trajectory go back until the first bout that heads in the opposite direction to determine how long the
    # turning took
    reversal_times = []
    for rt in reversal_trajectories:
        t = np.array(rt["Temperature"])[-1]
        all_gd = np.array(rt["Gradient direction"])
        alignment = np.zeros(all_gd.size)
        alignment[all_gd < -align_thresh] = -1
        alignment[all_gd > align_thresh] = 1
        final_alignment = alignment[-1]
        last_opposite = np.where(alignment == -1*final_alignment)[0][-1]
        rtime = alignment.size - last_opposite
        reversal_angles = np.diff(all_gd[last_opposite:])
        # for reversals that last at least 4 bouts, determine the cumulative effect
        # of the first two reorientations
        if rtime < 4:
            c_after_2 = np.nan
        else:
            c_after_2 = np.sum(reversal_angles[:2])
        avg_mag = np.mean(np.abs(reversal_angles))
        dirs = np.sign(reversal_angles)
        dominant_dir = np.sign(dirs.sum())
        handedness = np.sum(dirs == dominant_dir) / dirs.size
        # Temperature - Initial alignment - NBouts - Avg. magnitude of cosine change across swims
        # - Fraction of bouts in major reversal direction
        reversal_times.append(np.r_[t, -1*final_alignment, rtime, avg_mag, handedness, c_after_2])
    reversal_times = np.vstack(reversal_times)
    # reversals that start with a positive direction (twds. inc. y), i.e. face to decreasing temperature before reversal
    face_colder_reversals = reversal_times[reversal_times[:, 1] > 0]
    # reversals that start with a negative direction (twds. dec. y), i.e. face to increasing temperature before reversal
    face_hotter_reversals = reversal_times[reversal_times[:, 1] < 0]

    cold_bs = utils.bootstrap_weighted_histogram_avg(face_colder_reversals[:, 0], temp_bins, face_colder_reversals[:, 2], 1000)
    m_cold = np.nanmean(cold_bs, axis=0)
    e_cold = np.nanstd(cold_bs, axis=0)
    hot_bs = utils.bootstrap_weighted_histogram_avg(face_hotter_reversals[:, 0], temp_bins, face_hotter_reversals[:, 2], 1000)
    m_hot = np.nanmean(hot_bs, axis=0)
    e_hot = np.nanstd(hot_bs, axis=0)

    p_vals = utils.compute_weighted_histogram_p([face_colder_reversals[:, 0], face_hotter_reversals[:, 0]], temp_bins,
                                                [face_colder_reversals[:, 2], face_hotter_reversals[:, 2]],
                                                10_000)
    max_y = max(np.max(m_cold), np.max(m_hot))
    fig = pl.figure()
    pl.fill_between(temp_bincents, m_cold-e_cold, m_cold+e_cold, alpha=0.3, color="C0")
    pl.plot(temp_bincents, m_cold, label="Facing cold", color="C0", marker='.')
    pl.fill_between(temp_bincents, m_hot - e_hot, m_hot + e_hot, alpha=0.3, color="C3")
    pl.plot(temp_bincents, m_hot, label="Facing hot", color="C3", marker='.')
    # plot p-value indicators
    for pv, xv in zip(p_vals, temp_bincents):
        if pv >= 0.05:
            marker = '.'
        elif pv >= 0.01:
            marker = '+'
        elif pv >= 0.001:
            marker = 'x'
        elif np.isfinite(pv):
            marker = '*'
        else:
            marker = "None"
        pl.scatter(xv, max_y, marker=marker, color='k')
    pl.xlabel("Temperature [C]")
    pl.ylabel("Avg. reversal length")
    pl.legend()
    sns.despine()
    fig.savefig(path.join(plot_dir, "REVISION_S2C_Bouts_per_reversal_BootTest.pdf"))

    cold_bs = utils.bootstrap_weighted_histogram_avg(face_colder_reversals[:, 0], temp_bins, face_colder_reversals[:, 4], 1000)
    m_cold = np.nanmean(cold_bs, axis=0)
    e_cold = np.nanstd(cold_bs, axis=0)
    hot_bs = utils.bootstrap_weighted_histogram_avg(face_hotter_reversals[:, 0], temp_bins, face_hotter_reversals[:, 4], 1000)
    m_hot = np.nanmean(hot_bs, axis=0)
    e_hot = np.nanstd(hot_bs, axis=0)

    fig, ax = pl.subplots()  # no p-values calculated since curves overlap
    pl.fill_between(temp_bincents, m_cold - e_cold, m_cold + e_cold, alpha=0.3, color="C0")
    pl.plot(temp_bincents, m_cold, label="Facing cold", color="C0", marker='.')
    pl.fill_between(temp_bincents, m_hot - e_hot, m_hot + e_hot, alpha=0.3, color="C3")
    pl.plot(temp_bincents, m_hot, label="Facing hot", color="C3", marker='.')
    pl.xlabel("Temperature [C]")
    pl.ylabel("Avg. handedness")
    pl.legend()
    remove_spines(ax)
    fig.savefig(path.join(plot_dir, "REVISION_S2D_Average_handedness_in_reversal.pdf"))

    # augment bouts with reversal information and save fish and bout dataframes
    for k in val_bouts:
        for bout, fish, ename in zip(val_bouts[k], val_fish[k], val_expnames[k]):
            bout = utils.augment_state_info(bout, align_thresh, max_reversal_length)
            e_base = path.split(ename)[1]
            save_base = path.join(plot_dir, e_base)
            bout.to_pickle(f"{save_base}_bout.pkl")
            fish.to_pickle(f"{save_base}_fish.pkl")

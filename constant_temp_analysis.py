"""
Analysis script for KAB gradient experiments
To be run in context of behavioral_fever repo
"""

import matplotlib as mpl
import matplotlib.pyplot as pl
import argparse
import os
from os import path
import numpy as np
import pandas as pd
import loading
import plot_funs as pf
import utils
from gradient_analysis import load_all, CheckArgs
from gradient_analysis import align_thresh, max_reversal_length
from utils import occupancy
from plot_funs import set_journal_style


if __name__ == '__main__':
    burn_in = 5*60*100  # 5-minute burn-in period

    mpl.rcParams['pdf.fonttype'] = 42
    set_journal_style(23, 23)
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="constant_temp_analysis",
                                       description="Runs analysis for constant temperature experiments")
    a_parser.add_argument("-cf", "--constant", help="Path to root-folder with constant temperature subfolders", type=str,
                          default="", action=CheckArgs)

    args = a_parser.parse_args()

    const_root = args.constant

    c_folders = [d for d in os.listdir(const_root) if path.isdir(path.join(const_root, d))]

    all_exp = {}

    for cf in c_folders:
        all_exp[cf] = loading.find_all_exp_paths(path.join(const_root, cf))[0]

    plot_dir = "REVISION_KAB_Constant"
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
            if b.shape[0] >= f.shape[0]/150:  # one bout every 1.5 s
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

    # save fish and bout dataframes
    for k in val_bouts:
        for bout, fish, ename in zip(val_bouts[k], val_fish[k], val_expnames[k]):
            bout = utils.augment_state_info(bout, align_thresh, max_reversal_length)
            e_base = path.split(ename)[1]
            save_base = path.join(plot_dir, e_base)
            bout.to_pickle(f"{save_base}_bout.pkl")
            fish.to_pickle(f"{save_base}_fish.pkl")

    align_pref = {"Treatment": [], "Gradient alignment": [], "Gradient direction": []}

    y_distribution = {"Treatment": [], "Density": [], "Cumulative density": []}
    y_bins = np.linspace(y_min, y_max, 15)
    y_bc = y_bins[:-1] + np.diff(y_bins)/2

    disp_distribution = {"Treatment": [], "Density": []}
    disp_bins = np.linspace(0, 10, 100)
    disp_bc = disp_bins[:-1] + np.diff(disp_bins)/2

    angle_distribution = {"Treatment": [], "Density": []}
    angle_bins = np.linspace(-180, 180, 50)
    angle_bc = angle_bins[:-1] + np.diff(angle_bins)/2

    ibi_distribution = {"Treatment": [], "Density": []}
    # NOTE: The wonky bincount is because this is a discrete measure (resolution = 10ms). This means that if some
    # bin boundaries exactly line up with discrete steps we can get bins that are enriched/starved of data
    ibi_bins = np.linspace(0, 1500, 51)
    ibi_bc = ibi_bins[:-1] + np.diff(ibi_bins)/2

    for k in val_fish:
        for df_fish, df_bout in zip(val_fish[k], val_bouts[k]):
            grad_angles = np.array(np.arctan2(df_bout["Delta X"], df_bout["Delta Y"]))
            grad_direction = np.cos(grad_angles)
            align_pref["Treatment"].append(k)
            align_pref["Gradient alignment"].append(np.nanmean(np.abs(grad_direction)))
            align_pref["Gradient direction"].append(np.nanmean(grad_direction))
            y_distribution["Treatment"].append(k)
            disp_distribution["Treatment"].append(k)
            angle_distribution["Treatment"].append(k)
            ibi_distribution["Treatment"].append(k)
            y_positions = df_fish['Y Position'][burn_in:]
            y_positions = y_positions[np.isfinite(y_positions)]
            y_distribution["Density"].append(np.histogram(y_positions, bins=y_bins, density=True)[0])
            h = np.histogram(y_positions, bins=y_bins, density=False)[0].astype(float)
            h /= h.sum()
            y_distribution["Cumulative density"].append(np.cumsum(h))
            disp_distribution["Density"].append(np.histogram(df_bout['Displacement'], bins=disp_bins, density=True)[0])
            angle_distribution["Density"].append(np.histogram(np.rad2deg(df_bout['Angle change']), bins=angle_bins,
                                                              density=True)[0])
            ibi_distribution["Density"].append(np.histogram(df_bout["IBI"], bins=ibi_bins, density=True)[0])

    order = sorted(list(val_fish.keys()))

    df_y_distribution = pd.DataFrame(y_distribution)
    df_disp_distribution = pd.DataFrame(disp_distribution)
    df_angle_distribution = pd.DataFrame(angle_distribution)
    df_ibi_distribution = pd.DataFrame(ibi_distribution)
    df_align_pref = pd.DataFrame(align_pref)

    # plot y-coordinate distribution densities with bootstrap statistics
    fig = pf.lineplot(df_y_distribution, "Density", None, y_bc, "Y Position [mm]", occupancy, add_marker=True)
    pl.ylim(0)
    fig.savefig(path.join(plot_dir, "REVISION_S1A_ConstExp_Occupancy.pdf"))

    # plot bout feature distribution densities with bootstrap statistics
    fig = pf.lineplot(df_disp_distribution, "Density", "Treatment", disp_bc, "Displacement [mm]", np.mean)
    fig.savefig(path.join(plot_dir, "REVISION_1C_DisplacementDistribution.pdf"))

    # plot bout turn angle distribution densities with bootstrap statistics
    fig = pf.lineplot(df_angle_distribution, "Density", "Treatment", angle_bc, "Turn angle [deg]", np.mean)
    pl.plot([-10, -10], [0, 0.05], 'k--')
    pl.plot([10, 10], [0, 0.05], 'k--')
    fig.savefig(path.join(plot_dir, "REVISION_1E_TurnAngleDistribution.pdf"))

    # plot interbout interval distribution densities with bootstrap statistics
    fig = pf.lineplot(df_ibi_distribution, "Density", "Treatment", ibi_bc, "Interbout interval [ms]", np.mean)
    fig.savefig(path.join(plot_dir, "REVISION_1D_InterboutIntervalDistribution.pdf"))

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

    reversal_trajectories = {}
    for k in trajectories:
        if k not in reversal_trajectories:
            reversal_trajectories[k] = []
        for t in trajectories[k]:
            reversal_trajectories[k] += utils.split_reversal_trajectories(t, align_thresh)

    reversals = {}
    for k in reversal_trajectories:
        reversals[k] = []
        for rt in reversal_trajectories[k]:
            y = np.array(rt["Y Position"])[0]
            gd = np.array(rt["Gradient direction"])[0]
            ln = rt.shape[0]
            reversals[k].append(np.r_[y, gd, ln])
        reversals[k] = np.vstack(reversals[k])

    fig, axes = pl.subplots(ncols=2, sharey=True, figsize=(6.4*2, 4.8*2))
    cmap = pl.colormaps["viridis_r"]
    time_colors = cmap(np.linspace(0, 1, 40))
    time_colors[:, -1] = 0.4  # alpha
    for k in reversal_trajectories:
        for rt in reversal_trajectories[k]:
            t = np.array(rt["Y Position"])[0]
            if t < 25 or t > 100:
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
    fig.suptitle('25 mm<=Y>=100 mm')
    fig.savefig(path.join(plot_dir, "REVISION_S2F_Trajectories_from_front.pdf"))

    fig, axes = pl.subplots(ncols=2, sharey=True, figsize=(6.4*2, 4.8*2))
    cmap = pl.colormaps["viridis_r"]
    time_colors = cmap(np.linspace(0, 1, 40))
    time_colors[:, -1] = 0.4  # alpha
    for k in reversal_trajectories:
        for rt in reversal_trajectories[k]:
            t = np.array(rt["Y Position"])[0]
            if t < 100 or t > 175:
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
    fig.suptitle('100 mm<=Y>=175 mm')
    fig.savefig(path.join(plot_dir, "REVISION_S2F_Trajectories_from_back.pdf"))

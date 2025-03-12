"""
Script for additional analysis of swim kinematics and comparison between gradient and constant temperature experiments
"""

import matplotlib as mpl
import matplotlib.pyplot as pl
import argparse
import os
from os import path
import numpy as np
import utils
from typing import Dict, List, Optional
from gradient_analysis import align_thresh, max_reversal_length
import scipy.stats as sts
from plot_funs import set_journal_style, remove_spines, format_legend


def bootstrap_nan_data(data: np.ndarray, n_boot: int, bootfun) -> np.ndarray:
    """
    Bootstraps 2d data along the first axis with irregular nan-values
    :param data:
    :param n_boot:
    :param bootfun:
    :return:
    """
    boot_ret = np.full((n_boot, data.shape[1]), np.nan)
    for i in range(data.shape[1]):
        bootvals = data[:, i][np.isfinite(data[:, i])]
        indices = np.arange(bootvals.size).astype(int)
        for bix in range(n_boot):
            sel = np.random.choice(indices, indices.size, replace=True)
            boot_ret[bix, i] = bootfun(bootvals[sel])
    return boot_ret

def make_quant_dict(name: str, grad: List, const: Optional[List]) -> Dict:
    d = {"Constant": np.hstack([traj[name] for traj in const]) if const is not None else None,
         "Gradient": np.hstack([traj[name] for traj in grad])}
    return d


def bin_by_temp(values: Dict, name: str) -> Dict:
    by_temp = {k: {"Temperature": None, name: []} for k in values}
    for k in values:
        for fish_id in np.unique(all_fish_id[k]):
            sel = all_fish_id[k] == fish_id
            sel = np.logical_and(sel, np.isfinite(values[k]))
            weighted = np.histogram(all_temperatures[k][sel], bins=temp_bins[k], weights=values[k][sel])[0]
            counts = np.histogram(all_temperatures[k][sel], bins=temp_bins[k])[0]
            by_temp[k][name].append(weighted / counts)
        by_temp[k]["Temperature"] = temp_bc[k]
        by_temp[k][name] = np.vstack(by_temp[k][name])
    return by_temp


def bin_by_y(values: Dict, name: str) -> Dict:
    by_y = {k: {"Y Position": None, name: []} for k in values}
    for k in values:
        for fish_id in np.unique(all_fish_id[k]):
            sel = all_fish_id[k] == fish_id
            sel = np.logical_and(sel, np.isfinite(values[k]))
            weighted = np.histogram(all_ypos[k][sel], bins=y_bins, weights=values[k][sel])[0]
            counts = np.histogram(all_ypos[k][sel], bins=y_bins)[0]
            by_y[k][name].append(weighted / counts)
        by_y[k]["Y Position"] = y_bc
        by_y[k][name] = np.vstack(by_y[k][name])
    return by_y


def bin_by_dtemp(values: Dict, name: str) -> Dict:
    by_dtemp = {k: {"Temperature change": None, name: []} for k in values}
    k = "Gradient"
    for fish_id in np.unique(all_fish_id[k]):
        sel = all_fish_id[k] == fish_id
        sel = np.logical_and(sel, np.isfinite(values[k]))
        weighted = np.histogram(all_prev_delta_t[k][sel], bins=dt_bins, weights=values[k][sel])[0]
        counts = np.histogram(all_prev_delta_t[k][sel], bins=dt_bins)[0]
        by_dtemp[k][name].append(weighted / counts)
    by_dtemp[k]["Temperature"] = dt_bc
    by_dtemp[k][name] = np.vstack(by_dtemp[k][name])
    return by_dtemp


def plot_bin_by_temp(d_bbt: Dict, name: str, shaded=False):
    fig, ax = pl.subplots()
    for p_ix, k in enumerate(d_bbt):  # fix keys to ensure fixed plot order
        x = d_bbt[k]["Temperature"]  # shared bins
        bs_vars = bootstrap_nan_data(d_bbt[k][name], 1000, np.median)
        y = np.mean(bs_vars, axis=0)
        e = np.std(bs_vars, axis=0)
        if not shaded:
            ax.errorbar(x, y, yerr=e, label=k)
        else:
            ax.fill_between(x, y - e, y + e, color=f'C{p_ix}', alpha=0.4)
            ax.plot(x, y, color=f'C{p_ix}', label=k)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel(name)
    remove_spines(ax)
    format_legend(ax)
    return fig


def plot_bin_by_y(d_bby: Dict, name: str, shaded=False):
    fig, ax = pl.subplots()
    for k in ["Constant"]:  # fix keys to ensure fixed plot order
        x = d_bby[k]["Y Position"]  # shared bins
        bs_vars = bootstrap_nan_data(d_bby[k][name], 1000, np.median)
        y = np.mean(bs_vars, axis=0)
        e = np.std(bs_vars, axis=0)
        if not shaded:
            ax.errorbar(x, y, yerr=e, label=k)
        else:
            ax.fill_between(x, y-e, y+e, color='k', alpha=0.4)
            ax.plot(x, y, color='k')
    ax.legend()
    ax.set_xlabel("Y Position [mm]")
    ax.set_ylabel(name)
    remove_spines(ax)
    format_legend(ax)
    return fig


def plot_bin_by_dtemp(d_bbt: Dict, name: str):
    fig, ax = pl.subplots()
    k = "Gradient"  # fix keys to ensure fixed plot order
    x = d_bbt[k]["Temperature"]  # shared bins
    bs_vars = bootstrap_nan_data(d_bbt[k][name], 1000, np.median)
    y = np.mean(bs_vars, axis=0)
    e = np.std(bs_vars, axis=0)
    ax.errorbar(x, y, yerr=e, label=k)
    ax.legend()
    ax.set_xlabel("Temperature change [C/bout]")
    ax.set_ylabel(name)
    remove_spines(ax)
    format_legend(ax)
    return fig


def bin_dir_by_temp(values: Dict, name: str) -> Dict:
    by_dir = {k: {"Temperature": None, name: []} for k in ["Heating", "Cooling"]}
    k = "Gradient"
    for fish_id in np.unique(all_fish_id[k]):
        sel = all_fish_id[k] == fish_id
        sel = np.logical_and(sel, np.isfinite(values[k]))
        cooling = np.logical_and(sel, all_prev_delta_t[k] < -delta_cut)
        heating = np.logical_and(sel, all_prev_delta_t[k] > delta_cut)
        weighted = np.histogram(all_temperatures[k][cooling], bins=temp_bins[k], weights=values[k][cooling])[0]
        counts = np.histogram(all_temperatures[k][cooling], bins=temp_bins[k])[0]
        by_dir["Cooling"][name].append(weighted / counts)
        by_dir["Cooling"]["Temperature"] = temp_bc[k]
        weighted = np.histogram(all_temperatures[k][heating], bins=temp_bins[k], weights=values[k][heating])[0]
        counts = np.histogram(all_temperatures[k][heating], bins=temp_bins[k])[0]
        by_dir["Heating"][name].append(weighted / counts)
        by_dir["Heating"]["Temperature"] = temp_bc[k]
    by_dir["Cooling"][name] = np.vstack(by_dir["Cooling"][name])
    by_dir["Heating"][name] = np.vstack(by_dir["Heating"][name])
    return by_dir


def plot_corr_straps(pair_dict, temp_centers, n_boot=1000, plot_keys=None):

    if plot_keys is None:
        plot_keys = ["Cooling", "Heating"]

    f_ids = np.unique(pair_dict["Fish ID"])
    corrs = {k: np.zeros((f_ids.size, temp_centers.size)) for k in plot_keys}

    for diri, gdir in enumerate(plot_keys):
        for it, t in enumerate(temp_centers):
            for i, f in enumerate(f_ids):
                val = np.logical_and(pair_dict["Direction"] == gdir, pair_dict["Temperature"] == t)
                val = np.logical_and(val, pair_dict["Fish ID"] == f)
                if np.sum(val) < 2:
                    corrs[gdir][i, it] = np.nan
                else:
                    selected = np.c_[pair_dict["Previous"][val][:, None], pair_dict["Current"][val][:, None]]
                    corrs[gdir][i, it] = np.corrcoef(selected[:, 0], selected[:, 1])[0, 1]

    fig, ax = pl.subplots()
    for gdir in plot_keys:
        bsams = bootstrap_nan_data(corrs[gdir], n_boot, np.median)
        m = np.mean(bsams, 0)
        e = np.std(bsams, 0)
        ax.errorbar(temp_centers, m, e, label=gdir)
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Lag 1 correlation")
    remove_spines(ax)
    format_legend(ax)
    return fig

turn_cut = 5  # ~ the standard deviation of the straight swim gaussian


if __name__ == '__main__':

    set_journal_style(23, 23)
    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="kinematics_compare.py",
                                       description="Comparative bout kinematic analysis")
    a_parser.add_argument("-gf", "--gradient", help="Path to folder with gradient trajectory pickles",
                          type=str)
    a_parser.add_argument("-cf", "--constant", help="Path to folder with constant trajectory pickles",
                          type=str)

    args = a_parser.parse_args()

    gradient_folder = args.gradient
    constant_folder = args.constant

    plot_dir = "kinematics_compare"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    gradient_trajectories = utils.load_all_trajectories(gradient_folder, len_thresh=3)
    constant_trajectories = utils.load_all_trajectories(constant_folder, len_thresh=3)

    all_displacements = make_quant_dict("Displacement", gradient_trajectories, constant_trajectories)
    all_angle_change = make_quant_dict("Angle change", gradient_trajectories, constant_trajectories)
    all_angles_degrees = {k: np.rad2deg(all_angle_change[k]) for k in all_angle_change}
    all_mag_degrees = {k: np.abs(all_angles_degrees[k]) for k in all_angle_change}
    # limit to turns
    for k in all_mag_degrees:
        mag = all_mag_degrees[k]
        mag[mag < turn_cut] = np.nan
        all_mag_degrees[k] = mag
    all_temperatures = make_quant_dict("Temperature", gradient_trajectories, constant_trajectories)
    all_prev_delta_t = make_quant_dict("Prev Delta T", gradient_trajectories, constant_trajectories)
    all_ibi = make_quant_dict("IBI", gradient_trajectories, constant_trajectories)
    all_fish_id = make_quant_dict("Fish ID", gradient_trajectories, constant_trajectories)

    delta_cut = np.percentile(np.abs(all_prev_delta_t["Gradient"]), 50)  # ~ 0.052

    temp_bins = {"Constant": np.array([15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]),
                 "Gradient": np.array([19, 21, 23, 25, 27, 29, 31])}
    temp_bc = {k: temp_bins[k][:-1] + 1 for k in temp_bins}

    # analysis of temperature effect
    disp_by_temp = bin_by_temp(all_displacements, "Displacement [mm]")
    fig = plot_bin_by_temp(disp_by_temp, "Displacement [mm]")
    fig.savefig(path.join(plot_dir, "F1_displacement_compare.pdf"))

    mag_by_temp = bin_by_temp(all_mag_degrees, "Magnitude [deg]")
    fig = plot_bin_by_temp(mag_by_temp, "Magnitude [deg]")
    fig.savefig(path.join(plot_dir, "F1_magnitude_compare.pdf"))

    ibi_by_temp = bin_by_temp(all_ibi, "IBI [ms]")
    fig = plot_bin_by_temp(ibi_by_temp, "IBI [ms]")
    fig.savefig(path.join(plot_dir, "S1_ibi_compare.pdf"))

    # analysis by raw delta-T
    dt_bins = np.linspace(-0.2, 0.2, 7)
    dt_bc = dt_bins[:-1] + np.diff(dt_bins)/2

    disp_by_dtemp = bin_by_dtemp(all_displacements, "Displacement [mm]")
    fig = plot_bin_by_dtemp(disp_by_dtemp, "Displacement [mm]")
    fig.savefig(path.join(plot_dir, "S1_displacement_by_deltaT.pdf"))

    mag_by_dtemp = bin_by_dtemp(all_mag_degrees, "Magnitude [deg]")
    fig = plot_bin_by_dtemp(mag_by_dtemp, "Magnitude [deg]")
    fig.savefig(path.join(plot_dir, "S1_magnitude_by_deltaT.pdf"))

    ibi_by_dtemp = bin_by_dtemp(all_ibi, "IBI [ms]")
    fig = plot_bin_by_dtemp(ibi_by_dtemp, "IBI [ms]")
    fig.savefig(path.join(plot_dir, "S1_ibi_by_deltaT.pdf"))

    # analysis of heating/cooling direction by temperature
    disp_by_dir = bin_dir_by_temp(all_displacements, "Displacement [mm]")
    fig = plot_bin_by_temp(disp_by_dir, "Displacement [mm]")
    fig.savefig(path.join(plot_dir, "F1_displacement_by_direction_and_temperature.pdf"))

    mag_by_dir = bin_dir_by_temp(all_mag_degrees, "Magnitude [deg]")
    fig = plot_bin_by_temp(mag_by_dir, "Magnitude [deg]")
    fig.savefig(path.join(plot_dir, "F1_magnitude_by_direction_and_temperature.pdf"))

    ibi_by_dir = bin_dir_by_temp(all_ibi, "IBI [ms]")
    fig = plot_bin_by_temp(ibi_by_dir, "IBI [ms]")
    fig.savefig(path.join(plot_dir, "S1_ibi_by_direction_and_temperature.pdf"))

    # gradient direction analysis for average cluster activity
    clus_names = ["Cold adapting", "Hot", "Hot and Cooling", "Cold", "Cold and Cooling", "Hot and Heating",
                  "Cold and Heating"]
    for clix in range(7):
        all_clust_act = make_quant_dict(f"Cluster activity_{clix}", gradient_trajectories, None)
        act_by_dir = bin_dir_by_temp(all_clust_act, f"{clus_names[clix]} [AU]")
        fig = plot_bin_by_temp(act_by_dir, f"{clus_names[clix]} [AU]")
        fig.savefig(path.join(plot_dir, f"S7_{clus_names[clix]}_by_direction_and_temperature.pdf"))

    # analyze correlations of successive turns/displacement by direction
    # successive turn and displacement correlations
    temp_centers = temp_bc["Gradient"]
    # all_dtemp = np.hstack([np.array(tj["Prev Delta T"]) for tj in gradient_trajectories])
    # all_temp = np.hstack([np.array(tj["Temperature"]) for tj in gradient_trajectories])
    # all_dangles = np.hstack([np.array(tj["Angle change"]) for tj in gradient_trajectories])
    turn_cut = np.deg2rad(5)  # ~ the standard deviation of the straight swim gaussian
    displacement_pairs = {"Direction": [], "Previous": [], "Current": [], "Temperature": [], "Fish ID": []}
    angle_pairs = {"Direction": [], "Previous": [], "Current": [], "Temperature": [], "Fish ID": []}
    for traj in gradient_trajectories:
        for i in range(1, traj.shape[0]):
            prev = traj.iloc[i-1]
            current = traj.iloc[i]

            if current["Prev Delta T"] < -delta_cut:
                direction = "Cooling"
            elif current["Prev Delta T"] > delta_cut:
                direction = "Heating"
            else:
                continue
            try:
                ix_temp = np.where(np.abs(current["Temperature"] - temp_centers) <= 1)[0][0]
            except IndexError:
                continue
            temperature = temp_centers[ix_temp]
            displacement_pairs["Direction"].append(direction)
            displacement_pairs["Previous"].append(prev["Displacement"])
            displacement_pairs["Current"].append(current["Displacement"])
            displacement_pairs["Temperature"].append(temperature)
            displacement_pairs["Fish ID"].append(current["Fish ID"])

            # limit to actual turns
            if np.abs(current["Angle change"]) < turn_cut or np.abs(prev["Angle change"]) < turn_cut:
                continue

            angle_pairs["Direction"].append(direction)
            angle_pairs["Previous"].append(prev["Angle change"])
            angle_pairs["Current"].append(current["Angle change"])
            angle_pairs["Temperature"].append(temperature)
            angle_pairs["Fish ID"].append(current["Fish ID"])

    for k in angle_pairs.keys():
        angle_pairs[k] = np.hstack(angle_pairs[k])
        displacement_pairs[k] = np.hstack(displacement_pairs[k])

    fig = plot_corr_straps(angle_pairs, temp_centers, 1000)
    pl.xticks([20.0, 22.5, 25.0, 27.5, 30.0])
    fig.savefig(path.join(plot_dir, "F3_angle_correlations.pdf"))

    fig = plot_corr_straps(displacement_pairs, temp_centers, 1000)
    pl.xticks([20.0, 22.5, 25.0, 27.5, 30.0])
    fig.savefig(path.join(plot_dir, "F3_displacement_correlations.pdf"))

    # constant temperature kinematics by y-position
    y_bins = np.array([25, 75, 125, 150, 175])  # same starting distance as the temperature bins from edge
    y_bc = y_bins[:-1] + np.diff(y_bins/2)
    all_ypos = make_quant_dict("Y Position", gradient_trajectories, constant_trajectories)

    disp_by_y = bin_by_y(all_displacements, "Displacement [mm]")
    fig = plot_bin_by_y(disp_by_y, "Displacement [mm]")
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "S1_constant_displacement_by_YPosition.pdf"))

    mag_by_y = bin_by_y(all_mag_degrees, "Magnitude [deg]")
    fig = plot_bin_by_y(mag_by_y, "Magnitude [deg]")
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "S1_constant_magnitude_by_YPosition.pdf"))

    ibi_by_y = bin_by_y(all_ibi, "IBI [ms]")
    fig = plot_bin_by_y(ibi_by_y, "IBI [ms]")
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "S1_constant_ibi_by_YPosition.pdf"))

    # plot delta-T experienced by zebrafish per bout and as rates of change
    fig, ax = pl.subplots()
    pl.hist(all_prev_delta_t["Gradient"], np.linspace(-0.25, 0.25, 31))
    pl.xlabel("Temperature change [C/bout]")
    pl.xticks([-0.2, -0.1, 0, 0.1, 0.2])
    remove_spines(ax)
    fig.savefig(path.join(plot_dir, "S1_gradient_perBoutDeltaT.pdf"))

    all_starts = make_quant_dict("Start", gradient_trajectories, constant_trajectories)
    all_ends = make_quant_dict("Stop", gradient_trajectories, constant_trajectories)
    grad_bout_length_s = (all_ends["Gradient"] - all_starts["Gradient"] + 1) / 100

    fig, ax = pl.subplots()
    pl.hist(all_prev_delta_t["Gradient"] / grad_bout_length_s, np.linspace(-1, 1, 31))
    pl.xlabel("Temperature change [C/s]")
    pl.xticks([-1, -0.5, 0, 0.5, 1])
    remove_spines(ax)
    fig.savefig(path.join(plot_dir, "S1_gradient_perSecondDeltaT.pdf"))

    # compare angle and displacement correlations for small delta-T in gradient with constant temperature data
    temp_centers = temp_bc["Constant"]
    displacement_pairs = {"Direction": [], "Previous": [], "Current": [], "Temperature": [], "Fish ID": []}
    angle_pairs = {"Direction": [], "Previous": [], "Current": [], "Temperature": [], "Fish ID": []}
    for traj in gradient_trajectories:
        for i in range(1, traj.shape[0]):
            prev = traj.iloc[i-1]
            current = traj.iloc[i]

            if current["Temperature"] < 19 or current["Temperature"] > 31:
                # as in other gradient/constant comparisons we exlude the first and last degree as those bins
                # are in fact only half sized, i.e. their center is not aligned with the center value
                continue

            if current["Prev Delta T"] < -delta_cut:
                continue
            elif current["Prev Delta T"] > delta_cut:
                continue
            else:
                direction = "Gradient steady"
            try:
                ix_temp = np.where(np.abs(current["Temperature"] - temp_centers) <= 1)[0][0]
            except IndexError:
                continue
            temperature = temp_centers[ix_temp]
            displacement_pairs["Direction"].append(direction)
            displacement_pairs["Previous"].append(prev["Displacement"])
            displacement_pairs["Current"].append(current["Displacement"])
            displacement_pairs["Temperature"].append(temperature)
            displacement_pairs["Fish ID"].append(current["Fish ID"])

            # limit to actual turns
            if np.abs(current["Angle change"]) < turn_cut or np.abs(prev["Angle change"]) < turn_cut:
                continue

            angle_pairs["Direction"].append(direction)
            angle_pairs["Previous"].append(prev["Angle change"])
            angle_pairs["Current"].append(current["Angle change"])
            angle_pairs["Temperature"].append(temperature)
            angle_pairs["Fish ID"].append(current["Fish ID"])

    # collect binned information for constant temperature experiments
    const_aln_by_temp = {"Temperature": [], "Fish ID": [], "Alignment": []}
    for traj in constant_trajectories:
        for i in range(1, traj.shape[0]):
            prev = traj.iloc[i-1]
            current = traj.iloc[i]
            direction = "Constant"

            try:
                ix_temp = np.where(np.abs(current["Temperature"] - temp_centers) <= 1)[0][0]
            except IndexError:
                continue
            temperature = temp_centers[ix_temp]
            displacement_pairs["Direction"].append(direction)
            displacement_pairs["Previous"].append(prev["Displacement"])
            displacement_pairs["Current"].append(current["Displacement"])
            displacement_pairs["Temperature"].append(temperature)
            displacement_pairs["Fish ID"].append(current["Fish ID"])

            const_aln_by_temp["Temperature"].append(temperature)
            const_aln_by_temp["Fish ID"].append(current["Fish ID"])
            const_aln_by_temp["Alignment"].append(np.abs(current["Gradient direction"]))

            # limit to actual turns
            if np.abs(current["Angle change"]) < turn_cut or np.abs(prev["Angle change"]) < turn_cut:
                continue

            angle_pairs["Direction"].append(direction)
            angle_pairs["Previous"].append(prev["Angle change"])
            angle_pairs["Current"].append(current["Angle change"])
            angle_pairs["Temperature"].append(temperature)
            angle_pairs["Fish ID"].append(current["Fish ID"])

    for k in angle_pairs.keys():
        angle_pairs[k] = np.hstack(angle_pairs[k])
        displacement_pairs[k] = np.hstack(displacement_pairs[k])

    fig = plot_corr_straps(angle_pairs, temp_centers, 1000, ["Constant", "Gradient steady"])
    pl.xticks([17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5])
    fig.savefig(path.join(plot_dir, "S3_angle_correlations.pdf"))

    fig = plot_corr_straps(displacement_pairs, temp_centers, 1000, ["Constant", "Gradient steady"])
    pl.xticks([17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5])
    fig.savefig(path.join(plot_dir, "S3_displacement_correlations.pdf"))

    all_grad_dir = make_quant_dict("Gradient direction", gradient_trajectories, constant_trajectories)
    all_grad_align = {k: np.abs(all_grad_dir[k]) for k in all_grad_dir}

    # Gradient alignment across constant temperatures by chamber position
    align_by_y = bin_by_y(all_grad_align, "Gradient alignment")
    fig = plot_bin_by_y(align_by_y, "Gradient alignment")
    pl.plot([y_bc[0], y_bc[-1]], [0.64, 0.64], 'k--')
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "S3_constant_gradient_alignment_by_YPosition.pdf"))

    # Gradient alignment across all y-positions according to temperature
    align_by_temp = bin_by_temp(all_grad_align, "Gradient alignment")
    fig = plot_bin_by_temp(align_by_temp, "Gradient alignment")
    pl.plot([16, 34], [0.64, 0.64], 'k--')
    pl.xticks([17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5])
    fig.savefig(path.join(plot_dir, "S3_constant_gradient_alignment_by_Temperature.pdf"))

    # reversals - note the following code is largely copied from gradient_analysis.py and some features are unused
    # for this analysis on constant temperatures!!
    reversal_trajectories = []
    for t in constant_trajectories:
        reversal_trajectories += utils.split_reversal_trajectories(t, align_thresh)

    reversal_times = []
    for rt in reversal_trajectories:
        t = np.array(rt["Temperature"])[-1]
        y = np.array(rt["Y Position"])[-1]
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
        # Temperature - Initial alignment - NBouts - y-coordinate
        reversal_times.append(np.r_[t, -1*final_alignment, rtime, y])
    reversal_times = np.vstack(reversal_times)
    # reversals that start with a positive direction (twds. inc. y), i.e. face to decreasing temperature before reversal
    face_colder_reversals = reversal_times[reversal_times[:, 1] > 0]
    # reversals that start with a negative direction (twds. dec. y), i.e. face to increasing temperature before reversal
    face_hotter_reversals = reversal_times[reversal_times[:, 1] < 0]

    true_hot_reversals = face_hotter_reversals[face_hotter_reversals[:, 2] <= max_reversal_length, :]
    true_cold_reversals = face_colder_reversals[face_colder_reversals[:, 2] <= max_reversal_length, :]

    all_aligned = []  # characteristics of bouts that are considered aligned to the gradient
    for t in constant_trajectories:
        temperatures = np.array(t["Temperature"])
        grad_dirs = np.array(t["Gradient direction"])
        y_positions = np.array(t["Y Position"])
        aligned = np.logical_or(grad_dirs < -align_thresh, grad_dirs > align_thresh)
        all_aligned.append(np.c_[temperatures[aligned][:, None], grad_dirs[aligned][:, None], y_positions[aligned][:, None]])
    all_aligned = np.vstack(all_aligned)

    # for each constant temperature, plot the fraction of reversal trajectories when cold- or hot-facing in relation
    # to the total number of in-trajectory bouts that are cold- or hot-facing
    h_rev_cold = np.histogram(true_cold_reversals[:, 0], bins=temp_bins["Constant"])[0]
    h_aln_cold = np.histogram(all_aligned[all_aligned[:, 1] > 0][:, 0], bins=temp_bins["Constant"])[0]

    h_rev_hot = np.histogram(true_hot_reversals[:, 0], bins=temp_bins["Constant"])[0]
    h_aln_hot = np.histogram(all_aligned[all_aligned[:, 1] < 0][:, 0], bins=temp_bins["Constant"])[0]

    temp_bincents = temp_bc["Constant"]

    fig, ax = pl.subplots()
    ci = [sts.binomtest(h_rev_cold[i], h_aln_cold[i]).proportion_ci(confidence_level=0.68) for i in
          range(temp_bincents.size)]
    upper = np.hstack([ci[i].high for i in range(temp_bincents.size)])
    lower = np.hstack([ci[i].low for i in range(temp_bincents.size)])
    pl.fill_between(temp_bincents, upper, lower, alpha=0.2, color="C0")
    pl.plot(temp_bincents, h_rev_cold / h_aln_cold, label="Y increasing [cold]", color="C0")
    ci = [sts.binomtest(h_rev_hot[i], h_aln_hot[i]).proportion_ci(confidence_level=0.68) for i in
          range(temp_bincents.size)]
    upper = np.hstack([ci[i].high for i in range(temp_bincents.size)])
    lower = np.hstack([ci[i].low for i in range(temp_bincents.size)])
    pl.fill_between(temp_bincents, upper, lower, alpha=0.2, color="C1")
    pl.plot(temp_bincents, h_rev_hot / h_aln_hot, label="Y decreasing [hot]", color="C1")
    remove_spines(ax)
    pl.xlabel("Temperature [C]")
    pl.ylabel("Reversal probability")
    pl.legend()
    pl.xticks([17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5])
    fig.savefig(path.join(plot_dir, "S3_Constant_Temp_Reversal_probabilities.pdf"))

    # the same but this time binned according to y-position
    h_rev_cold = np.histogram(true_cold_reversals[:, 3], bins=y_bins)[0]
    h_aln_cold = np.histogram(all_aligned[all_aligned[:, 1] > 0][:, 2], bins=y_bins)[0]

    h_rev_hot = np.histogram(true_hot_reversals[:, 3], bins=y_bins)[0]
    h_aln_hot = np.histogram(all_aligned[all_aligned[:, 1] < 0][:, 2], bins=y_bins)[0]

    fig, ax = pl.subplots()
    ci = [sts.binomtest(h_rev_cold[i], h_aln_cold[i]).proportion_ci(confidence_level=0.68) for i in
          range(y_bc.size)]
    upper = np.hstack([ci[i].high for i in range(y_bc.size)])
    lower = np.hstack([ci[i].low for i in range(y_bc.size)])
    pl.fill_between(y_bc, upper, lower, alpha=0.2, color="C0")
    pl.plot(y_bc, h_rev_cold / h_aln_cold, label="Y increasing [cold]", color="C0")
    ci = [sts.binomtest(h_rev_hot[i], h_aln_hot[i]).proportion_ci(confidence_level=0.68) for i in
          range(y_bc.size)]
    upper = np.hstack([ci[i].high for i in range(y_bc.size)])
    lower = np.hstack([ci[i].low for i in range(y_bc.size)])
    pl.fill_between(y_bc, upper, lower, alpha=0.2, color="C1")
    pl.plot(y_bc, h_rev_hot / h_aln_hot, label="Y decreasing [hot]", color="C1")
    remove_spines(ax)
    pl.xlabel("Y Position [mm]")
    pl.ylabel("Reversal probability")
    pl.legend()
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "S3_Constant_Temp_Reversal_probabilities_by_YCoordinate.pdf"))

    # persistence length by temperature and y coordinate

    persistent_trajectories = []
    for t in constant_trajectories:
            persistent_trajectories += utils.split_persistent_trajectories(t, align_thresh)

    pers_info = []
    for pt in persistent_trajectories:
        t = np.array(pt["Temperature"])[0]
        y = np.array(pt["Y Position"])[0]
        gd = np.array(pt["Gradient direction"])[0]
        ln = pt.shape[0]
        pers_info.append(np.r_[t, gd, ln, y])
    pers_info = np.vstack(pers_info)
    # persistence in the positive direction (twds. inc. y), i.e. decreasing temperature
    face_colder_pers = pers_info[pers_info[:, 1] > 0]
    # persistence in the negative direction (twds. dec. y), i.e. increasing temperature
    face_hotter_pers = pers_info[pers_info[:, 1] < 0]

    f_cold_p_bs = utils.bootstrap_weighted_histogram_avg(face_colder_pers[:, 0], temp_bins["Constant"], face_colder_pers[:, 2], 1000)
    m_cold_p = np.nanmean(f_cold_p_bs, axis=0)
    e_cold_p = np.nanstd(f_cold_p_bs, axis=0)
    f_hot_p_bs = utils.bootstrap_weighted_histogram_avg(face_hotter_pers[:, 0], temp_bins["Constant"], face_hotter_pers[:, 2], 1000)
    m_hot_p = np.nanmean(f_hot_p_bs, axis=0)
    e_hot_p = np.nanstd(f_hot_p_bs, axis=0)

    fig, ax = pl.subplots()
    pl.fill_between(temp_bincents, m_cold_p-e_cold_p, m_cold_p+e_cold_p, alpha=0.3, color="C0")
    pl.plot(temp_bincents, m_cold_p, label="Y increasing [cold]", color="C0")
    pl.fill_between(temp_bincents, m_hot_p-e_hot_p, m_hot_p+e_hot_p, alpha=0.3, color="C1")
    pl.plot(temp_bincents, m_hot_p, label="Y decreasing [hot]", color="C1")
    pl.xlabel("Temperature [C]")
    pl.ylabel("Avg. persistence length")
    pl.legend()
    pl.xticks([17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5])
    remove_spines(ax)
    fig.savefig(path.join(plot_dir, "S3_Constant_Temp_Persistence_trajectory_length.pdf"))

    f_cold_p_bs = utils.bootstrap_weighted_histogram_avg(face_colder_pers[:, 3], y_bins, face_colder_pers[:, 2], 1000)
    m_cold_p = np.nanmean(f_cold_p_bs, axis=0)
    e_cold_p = np.nanstd(f_cold_p_bs, axis=0)
    f_hot_p_bs = utils.bootstrap_weighted_histogram_avg(face_hotter_pers[:, 3], y_bins, face_hotter_pers[:, 2], 1000)
    m_hot_p = np.nanmean(f_hot_p_bs, axis=0)
    e_hot_p = np.nanstd(f_hot_p_bs, axis=0)

    fig, ax = pl.subplots()
    pl.fill_between(y_bc, m_cold_p-e_cold_p, m_cold_p+e_cold_p, alpha=0.3, color="C0")
    pl.plot(y_bc, m_cold_p, label="Y increasing [cold]", color="C0")
    pl.fill_between(y_bc, m_hot_p-e_hot_p, m_hot_p+e_hot_p, alpha=0.3, color="C1")
    pl.plot(y_bc, m_hot_p, label="Y decreasing [hot]", color="C1")
    pl.xlabel("Y Position [mm]")
    pl.ylabel("Avg. persistence length")
    pl.legend()
    remove_spines(ax)
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "S3_Constant_Temp_Persistence_trajectory_length_by_ycoordinate.pdf"))

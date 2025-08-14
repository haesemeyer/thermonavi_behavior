"""
Script for additional analysis of swim kinematics and comparison between gradient and constant temperature experiments
"""

import matplotlib as mpl
import matplotlib.pyplot as pl
import argparse
import os
from os import path
import numpy as np
import pandas as pd
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


def plot_bin_by_temp(d_bbt: Dict, name: str, shaded=True):
    # compute p-values across two conditions
    data_keys = list(d_bbt.keys())
    if len(data_keys) == 2:
        group_a_data = d_bbt[data_keys[0]][name]
        group_a_x = d_bbt[data_keys[0]]["Temperature"]
        group_b_data = d_bbt[data_keys[1]][name]
        group_b_x = d_bbt[data_keys[1]]["Temperature"]
        shared_x = group_a_x if group_b_x.size > group_a_x.size else group_b_x

        p_vals = np.full(shared_x.size, np.nan)
        for dix, xval in enumerate(shared_x):
            da = group_a_data[:, group_a_x == xval]
            db = group_b_data[:, group_b_x == xval]
            p_vals[dix] = sts.ranksums(da[np.isfinite(da)], db[np.isfinite(db)]).pvalue
        p_vals *= p_vals.size  # Bonferroni correction
    else:
        shared_x = np.full(1, np.nan)
        p_vals = shared_x
    max_y = - np.inf


    fig, ax = pl.subplots()
    for p_ix, k in enumerate(d_bbt):  # fix keys to ensure fixed plot order
        x = d_bbt[k]["Temperature"]  # shared bins
        bs_vars = bootstrap_nan_data(d_bbt[k][name], 1000, np.median)
        y = np.mean(bs_vars, axis=0)
        max_y = max(max_y, np.nanmax(y))
        e = np.std(bs_vars, axis=0)
        if not shaded:
            ax.errorbar(x, y, yerr=e, label=k)
        else:
            ax.fill_between(x, y - e, y + e, color=f'C{p_ix}', alpha=0.4)
            ax.plot(x, y, color=f'C{p_ix}', label=k, marker='.')
    for pv, xv in zip(p_vals, shared_x):
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
        ax.scatter(xv, max_y, marker=marker, color='k')
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    if np.min(x) < 20:
        ax.set_xticks(np.arange(16, 34, 4))
    else:
        ax.set_xticks([20, 22.5, 25, 27.5, 30])
    ax.set_ylabel(name)
    remove_spines(ax)
    format_legend(ax)
    return fig


def plot_bin_by_y(d_bby: Dict, name: str, shaded=True):
    fig, ax = pl.subplots()
    for p_ix, k in enumerate(["Constant"]):  # fix keys to ensure fixed plot order
        x = d_bby[k]["Y Position"]  # shared bins
        bs_vars = bootstrap_nan_data(d_bby[k][name], 1000, np.median)
        y = np.mean(bs_vars, axis=0)
        e = np.std(bs_vars, axis=0)
        if not shaded:
            ax.errorbar(x, y, yerr=e, label=k)
        else:
            ax.fill_between(x, y-e, y+e, color=f'C{p_ix}', alpha=0.4)
            ax.plot(x, y, color=f'C{p_ix}', marker='.')
    ax.legend()
    ax.set_xlabel("Y Position [mm]")
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

    # compute p-values across two conditions
    if len(plot_keys) == 2:
        group_a_data = corrs[plot_keys[0]]
        group_b_data = corrs[plot_keys[1]]

        p_vals = np.full(temp_centers.size, np.nan)
        for dix in range(temp_centers.size):
            da = group_a_data[:, dix]
            db = group_b_data[:, dix]
            p_vals[dix] = sts.ranksums(da[np.isfinite(da)], db[np.isfinite(db)]).pvalue
        p_vals *= p_vals.size  # Bonferroni correction
    else:
        p_vals = temp_centers
    max_y = - np.inf

    fig, ax = pl.subplots()
    for p_ix, gdir in enumerate(plot_keys):
        bsams = bootstrap_nan_data(corrs[gdir], n_boot, np.median)
        m = np.mean(bsams, 0)
        max_y = max(max_y, np.nanmax(m))
        e = np.std(bsams, 0)
        ax.fill_between(temp_centers, m - e, m + e, color=f'C{p_ix}', alpha=0.4)
        ax.plot(temp_centers, m, color=f'C{p_ix}', marker='.', label=gdir)
    for pv, xv in zip(p_vals, temp_centers):
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
        ax.scatter(xv, max_y, marker=marker, color='k')
    ax.legend()
    ax.set_xlabel("Temperature [C]")
    ax.set_ylabel("Lag 1 correlation")
    remove_spines(ax)
    format_legend(ax)
    return fig


def persistence_analysis(trajectories: List[pd.DataFrame]) -> pd.DataFrame:
    out_data = {"Fish ID": [], "Direction": [], "Length": [], "Temperature": [], "Y Position": []}
    for the_traj in trajectories:
        is_persistent = np.array(the_traj["State"] == 1).astype(int)
        if np.sum(is_persistent) == 0:
            continue
        se = np.r_[0, np.diff(is_persistent)]
        starts = np.where(se == 1)[0]
        ends = np.where(se == -1)[0]
        if is_persistent[0]:
            starts = np.r_[0, starts]
        if is_persistent[-1]:
            ends = np.r_[ends, is_persistent.size-1]
        assert starts.size == ends.size
        gdir = np.array(the_traj["Gradient direction"])
        fid = np.array(the_traj["Fish ID"])
        temp = np.array(the_traj["Temperature"])
        ypos = np.array(the_traj["Y Position"])
        for s, e in zip(starts, ends):
            out_data["Fish ID"].append(fid[s])
            out_data["Direction"].append("increasing" if gdir[s] < 0 else "decreasing")
            out_data["Length"].append(e-s+1)
            out_data["Temperature"].append(temp[s])
            out_data["Y Position"].append(ypos[s])
    return pd.DataFrame(out_data)


def reversal_analysis(trajectories: List[pd.DataFrame]) -> pd.DataFrame:
    out_data = {"Fish ID": [], "Direction": [], "Reversal start": [], "Temperature": [], "Y Position": []}
    for the_traj in trajectories:
        is_reversal = np.array(the_traj["State"] == -1).astype(int)  # marks all reversal bouts
        rev_start_ix = np.where(np.r_[0, np.diff(is_reversal)] > 0)[0] -1 # subtract one since reversal mode starts after the last aligned bout
        rev_start = np.zeros(is_reversal.size, dtype=bool)
        rev_start[rev_start_ix] = True
        is_aligned = np.array(np.abs(the_traj["Gradient direction"]) >= align_thresh)
        if not np.all(is_aligned[rev_start]):
            print("Ooops")
        if np.sum(is_aligned) == 0:
            continue
        gdir = np.array(the_traj["Gradient direction"])
        fid = np.array(the_traj["Fish ID"])
        temp = np.array(the_traj["Temperature"])
        ypos = np.array(the_traj["Y Position"])
        for bix, a in enumerate(is_aligned):
            if not a:
                continue
            out_data["Fish ID"].append(fid[bix])
            out_data["Direction"].append("increasing" if gdir[bix] < 0 else "decreasing")
            out_data["Reversal start"].append(rev_start[bix])
            out_data["Temperature"].append(temp[bix])
            out_data["Y Position"].append(ypos[bix])
    return pd.DataFrame(out_data)

turn_cut = 5  # ~ the standard deviation of the straight swim gaussian in degrees


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

    plot_dir = "REVISION_kinematics_compare"
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
    all_1s_delta_t = make_quant_dict("1s Delta T", gradient_trajectories, constant_trajectories)

    all_ibi = make_quant_dict("IBI", gradient_trajectories, constant_trajectories)
    all_fish_id = make_quant_dict("Fish ID", gradient_trajectories, constant_trajectories)

    delta_cut = np.percentile(np.abs(all_prev_delta_t["Gradient"]), 50)  # ~ 0.052

    temp_bins = {"Constant": np.array([15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]),
                 "Gradient": np.array([19, 21, 23, 25, 27, 29, 31])}
    temp_bc = {k: temp_bins[k][:-1] + 1 for k in temp_bins}

    # analysis of temperature effect
    disp_by_temp = bin_by_temp(all_displacements, "Displacement [mm]")
    fig = plot_bin_by_temp(disp_by_temp, "Displacement [mm]")
    fig.savefig(path.join(plot_dir, "REVISION_S1C_displacement_compare.pdf"))

    ibi_by_temp = bin_by_temp(all_ibi, "IBI [ms]")
    fig = plot_bin_by_temp(ibi_by_temp, "IBI [ms]")
    fig.savefig(path.join(plot_dir, "REVISION_S1B_ibi_compare.pdf"))

    mag_by_dir = bin_dir_by_temp(all_mag_degrees, "Magnitude [deg]")
    fig = plot_bin_by_temp(mag_by_dir, "Magnitude [deg]")
    fig.savefig(path.join(plot_dir, "REVISION_1H_magnitude_by_direction_and_temperature.pdf"))

    # gradient direction analysis for average cluster activity
    clus_names = ["Cold adapting", "Hot", "Hot and Cooling", "Cold", "Cold and Cooling", "Hot and Heating",
                  "Cold and Heating"]
    for clix in range(7):
        all_clust_act = make_quant_dict(f"Cluster activity_{clix}", gradient_trajectories, None)
        act_by_dir = bin_dir_by_temp(all_clust_act, f"{clus_names[clix]} [AU]")
        fig = plot_bin_by_temp(act_by_dir, f"{clus_names[clix]} [AU]")
        fig.savefig(path.join(plot_dir, f"REVISION_S4G-M_{clus_names[clix]}_by_direction_and_temperature.pdf"))

    # analyze correlations of successive turns/displacement by direction
    # successive turn and displacement correlations
    temp_centers = temp_bc["Gradient"]
    turn_cut = np.deg2rad(turn_cut)  # ~ the standard deviation of the straight swim gaussian in radians
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
    pl.xticks([20, 22.5, 25, 27.5, 30])
    fig.savefig(path.join(plot_dir, "REVISION_2B_angle_correlations.pdf"))

    # constant temperature kinematics by y-position
    y_bins = np.array([25, 75, 125, 150, 175])  # same starting distance as the temperature bins from edge
    y_bc = y_bins[:-1] + np.diff(y_bins/2)
    all_ypos = make_quant_dict("Y Position", gradient_trajectories, constant_trajectories)

    # plot delta-T experienced by zebrafish per bout and as rates of change
    fig, ax = pl.subplots()
    pl.hist(all_prev_delta_t["Gradient"], np.linspace(-0.25, 0.25, 31))
    pl.xlabel("Temperature change [C/bout]")
    pl.xticks([-0.2, -0.1, 0, 0.1, 0.2])
    remove_spines(ax)
    fig.savefig(path.join(plot_dir, "REVISION_S1D_gradient_perBoutDeltaT.pdf"))

    all_starts = make_quant_dict("Start", gradient_trajectories, constant_trajectories)
    all_ends = make_quant_dict("Stop", gradient_trajectories, constant_trajectories)
    grad_bout_length_s = (all_ends["Gradient"] - all_starts["Gradient"] + 1) / 100

    fig, ax = pl.subplots()
    pl.hist(all_prev_delta_t["Gradient"] / grad_bout_length_s, np.linspace(-1, 1, 31))
    pl.xlabel("Temperature change [C/s]")
    pl.xticks([-1, -0.5, 0, 0.5, 1])
    remove_spines(ax)
    fig.savefig(path.join(plot_dir, "REVISION_S1E_gradient_perSecondDeltaT.pdf"))

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

    fig = plot_corr_straps(displacement_pairs, temp_centers, 1000, ["Constant", "Gradient steady"])
    pl.xticks(np.arange(16, 34, 4))
    fig.savefig(path.join(plot_dir, "REVISION_2A_displacement_correlations.pdf"))

    all_grad_dir = make_quant_dict("Gradient direction", gradient_trajectories, constant_trajectories)
    all_grad_align = {k: np.abs(all_grad_dir[k]) for k in all_grad_dir}

    # Gradient alignment across constant temperatures by chamber position
    align_by_y = bin_by_y(all_grad_align, "Gradient alignment")
    fig = plot_bin_by_y(align_by_y, "Gradient alignment")
    pl.plot([y_bc[0], y_bc[-1]], [0.64, 0.64], 'k--')
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "REVISION_S2A_constant_gradient_alignment_by_YPosition.pdf"))

    # Analyze reversals for constant experiments by y-position
    const_rev_info = reversal_analysis(constant_trajectories)
    rev_cold_y = np.array(const_rev_info[np.logical_and(const_rev_info["Direction"] == "decreasing", const_rev_info["Reversal start"])]["Y Position"])
    aln_cold_y = np.array(const_rev_info[const_rev_info["Direction"] == "decreasing"]["Y Position"])
    rev_hot_y = np.array(const_rev_info[np.logical_and(const_rev_info["Direction"] == "increasing", const_rev_info["Reversal start"])]["Y Position"])
    aln_hot_y = np.array(const_rev_info[const_rev_info["Direction"] == "increasing"]["Y Position"])

    h_rev_cold = np.histogram(rev_cold_y, bins=y_bins)[0]
    h_aln_cold = np.histogram(aln_cold_y, bins=y_bins)[0]

    h_rev_hot = np.histogram(rev_hot_y, bins=y_bins)[0]
    h_aln_hot = np.histogram(aln_hot_y, bins=y_bins)[0]

    fig, ax = pl.subplots()
    ci = [sts.binomtest(h_rev_cold[i], h_aln_cold[i]).proportion_ci(confidence_level=0.68) for i in
          range(y_bc.size)]
    upper = np.hstack([ci[i].high for i in range(y_bc.size)])
    lower = np.hstack([ci[i].low for i in range(y_bc.size)])
    pl.fill_between(y_bc, upper, lower, alpha=0.2, color="C0")
    pl.plot(y_bc, h_rev_cold / h_aln_cold, label="Y increasing [cold]", color="C0", marker='.')
    ci = [sts.binomtest(h_rev_hot[i], h_aln_hot[i]).proportion_ci(confidence_level=0.68) for i in
          range(y_bc.size)]
    upper = np.hstack([ci[i].high for i in range(y_bc.size)])
    lower = np.hstack([ci[i].low for i in range(y_bc.size)])
    pl.fill_between(y_bc, upper, lower, alpha=0.2, color="C1")
    pl.plot(y_bc, h_rev_hot / h_aln_hot, label="Y decreasing [hot]", color="C1", marker='.')
    remove_spines(ax)
    pl.xlabel("Y Position [mm]")
    pl.ylim(0.00, 0.17)
    pl.yticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
    pl.ylabel("Reversal probability")
    pl.legend()
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "REVISION_S2E_Constant_Temp_Reversal_probabilities_by_YCoordinate.pdf"))

    # Analyze reversals for gradient experiments by temperature including worsening context preference
    temp_bins = np.linspace(18, 32, 21)
    temp_bincents = temp_bins[:-1] + np.diff(temp_bins) / 2
    grad_rev_info = reversal_analysis(gradient_trajectories)

    rev_cold_t = np.array(grad_rev_info[np.logical_and(grad_rev_info["Direction"] == "decreasing", grad_rev_info["Reversal start"])]["Temperature"])
    aln_cold_t = np.array(grad_rev_info[grad_rev_info["Direction"] == "decreasing"]["Temperature"])
    rev_hot_t = np.array(grad_rev_info[np.logical_and(grad_rev_info["Direction"] == "increasing", grad_rev_info["Reversal start"])]["Temperature"])
    aln_hot_t = np.array(grad_rev_info[grad_rev_info["Direction"] == "increasing"]["Temperature"])

    h_rev_cold = np.histogram(rev_cold_t, bins=temp_bins)[0]
    h_aln_cold = np.histogram(aln_cold_t, bins=temp_bins)[0]

    h_rev_hot = np.histogram(rev_hot_t, bins=temp_bins)[0]
    h_aln_hot = np.histogram(aln_hot_t, bins=temp_bins)[0]

    fig, ax = pl.subplots()
    ci = [sts.binomtest(h_rev_cold[i], h_aln_cold[i]).proportion_ci(confidence_level=0.68) for i in
          range(temp_bincents.size)]
    proportion_cold = h_rev_cold / h_aln_cold
    upper = np.hstack([ci[i].high for i in range(temp_bincents.size)])
    lower = np.hstack([ci[i].low for i in range(temp_bincents.size)])
    pl.fill_between(temp_bincents, upper, lower, alpha=0.2, color="C0")
    pl.plot(temp_bincents, proportion_cold, label="Facing cold", color="C0", marker='.')
    ci = [sts.binomtest(h_rev_hot[i], h_aln_hot[i]).proportion_ci(confidence_level=0.68) for i in
          range(temp_bincents.size)]
    proportion_hot = h_rev_hot / h_aln_hot
    p_vals = [sts.binomtest(h_rev_hot[i], h_aln_hot[i], proportion_cold[i]).pvalue for i in range(temp_bincents.size)]
    upper = np.hstack([ci[i].high for i in range(temp_bincents.size)])
    lower = np.hstack([ci[i].low for i in range(temp_bincents.size)])
    pl.fill_between(temp_bincents, upper, lower, alpha=0.2, color="C1")
    pl.plot(temp_bincents, proportion_hot, label="Facing hot", color="C1", marker='.')
    max_y = max(np.max(proportion_cold), np.max(proportion_hot))
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
    remove_spines(ax)
    pl.ylim(0.00, 0.17)
    pl.yticks([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
    pl.xlabel("Temperature [C]")
    pl.ylabel("Reversal probability")
    pl.legend()
    fig.savefig(path.join(plot_dir, "REVISION_2G_Reversal_probabilities_Binomtest.pdf"))

    # Analyze worsening context reversal preference in gradient
    t_center = 25
    n_boot = 10_000
    worsening_cold_bc = t_center - temp_bincents[temp_bincents < t_center]
    worsening_hot_bc = temp_bincents[temp_bincents > t_center] - t_center
    assert worsening_cold_bc.size == worsening_hot_bc.size

    def get_hot_cold_pi(data: pd.DataFrame, indexer=None, shuffle_dir=False):
        if indexer is None:
            indexer = np.arange(data.shape[0]).astype(int)

        directions = np.array(data["Direction"])[indexer]
        if shuffle_dir:
            np.random.shuffle(directions)
        reversal_starts = np.array(data["Reversal start"])[indexer]
        temperatures = np.array(data["Temperature"])[indexer]
        rev_cold_t = temperatures[np.logical_and(directions=="decreasing", reversal_starts)]
        aln_cold_t = temperatures[directions=="decreasing"]
        rev_hot_t = temperatures[np.logical_and(directions == "increasing", reversal_starts)]
        aln_hot_t = temperatures[directions == "increasing"]
        hr_cold = np.histogram(rev_cold_t, bins=temp_bins)[0]
        ha_cold = np.histogram(aln_cold_t, bins=temp_bins)[0]
        hr_hot = np.histogram(rev_hot_t, bins=temp_bins)[0]
        ha_hot = np.histogram(aln_hot_t, bins=temp_bins)[0]
        cold_pi = ((hr_cold / ha_cold - hr_hot / ha_hot) / (hr_cold / ha_cold + hr_hot / ha_hot))[
            temp_bincents < t_center]
        hot_pi = ((hr_hot / ha_hot - hr_cold / ha_cold) / (hr_cold / ha_cold + hr_hot / ha_hot))[
            temp_bincents > t_center]
        return hot_pi, cold_pi


    # compute true difference - NOTE: The plot below inverts the two relative to each other to align by the distance
    # to the preference (through the values of the bin-centers calculated above). In our comparisons however, we have
    # to invert one of the arrays before calculating differences!!
    hpi, cpi = get_hot_cold_pi(grad_rev_info)
    true_diff = cpi[::-1] - hpi  # one-sided: Hypothesis "Is cold larger than hot"
    avg_true_diff = np.mean(true_diff)

    bsam_rev_cold_pi = np.full((n_boot, worsening_cold_bc.size), np.nan)
    bsam_rev_hot_pi = np.full((n_boot, worsening_hot_bc.size), np.nan)
    bsam_diffs = bsam_rev_hot_pi.copy()
    bsam_avg_diffs = np.full(n_boot, np.nan)
    indices = np.arange(grad_rev_info.shape[0]).astype(int)
    for i in range(n_boot):
        ix_chosen = np.random.choice(indices, indices.size, replace=True)
        # for bootstrapping, vary the samples chosen
        hpi, cpi = get_hot_cold_pi(grad_rev_info, indexer=ix_chosen)
        bsam_rev_cold_pi[i] = cpi
        bsam_rev_hot_pi[i] = hpi
        # for bootstrap testing, shuffle assignment to gradient direction
        hpi, cpi = get_hot_cold_pi(grad_rev_info, shuffle_dir=True)
        bsam_avg_diffs[i] = np.mean(cpi[::-1] - hpi)

    m_cold = np.mean(bsam_rev_cold_pi, 0)
    e_cold = np.std(bsam_rev_cold_pi, 0)
    m_hot = np.mean(bsam_rev_hot_pi, 0)
    e_hot = np.std(bsam_rev_hot_pi, 0)

    print(f"P-value cold PI larger than hot PI = {np.sum(bsam_avg_diffs>=avg_true_diff)/n_boot}")

    fig, ax = pl.subplots()
    pl.fill_between(worsening_cold_bc, m_cold-e_cold, m_cold+e_cold, color="C0", alpha=0.4)
    pl.plot(worsening_cold_bc, m_cold, label="Cold regime", color="C0", marker='.')
    pl.fill_between(worsening_hot_bc, m_hot - e_hot, m_hot + e_hot, color="C1", alpha=0.4)
    pl.plot(worsening_hot_bc, m_hot, label="Hot regime", color="C1", marker='.')
    pl.legend()
    pl.xlabel(f"Distance from {t_center}C [C]")
    pl.ylabel("W Ctx Reversal preference")
    remove_spines(ax)
    fig.savefig(path.join(plot_dir, "REVISION_2H_Reversal_WorseningContext_InitPreference_OverallDiffBootTest.pdf"))

    # persistence length by y coordinate in constant experiments
    const_pers_info = persistence_analysis(constant_trajectories)


    f_cold_p_bs = utils.bootstrap_weighted_histogram_avg(np.array(const_pers_info["Y Position"][const_pers_info["Direction"] == "decreasing"]), y_bins, np.array(const_pers_info["Length"][const_pers_info["Direction"] == "decreasing"]), 1000)
    m_cold_p = np.nanmean(f_cold_p_bs, axis=0)
    e_cold_p = np.nanstd(f_cold_p_bs, axis=0)
    f_hot_p_bs = utils.bootstrap_weighted_histogram_avg(np.array(const_pers_info["Y Position"][const_pers_info["Direction"] == "increasing"]), y_bins, np.array(const_pers_info["Length"][const_pers_info["Direction"] == "increasing"]), 1000)
    m_hot_p = np.nanmean(f_hot_p_bs, axis=0)
    e_hot_p = np.nanstd(f_hot_p_bs, axis=0)

    fig, ax = pl.subplots()
    pl.fill_between(y_bc, m_cold_p-e_cold_p, m_cold_p+e_cold_p, alpha=0.3, color="C0")
    pl.plot(y_bc, m_cold_p, label="Y increasing [cold]", color="C0", marker='.')
    pl.fill_between(y_bc, m_hot_p-e_hot_p, m_hot_p+e_hot_p, alpha=0.3, color="C1")
    pl.plot(y_bc, m_hot_p, label="Y decreasing [hot]", color="C1", marker='.')
    pl.xlabel("Y Position [mm]")
    pl.ylabel("Avg. persistence length")
    pl.ylim(2, 8)
    pl.yticks([2, 3, 4, 5, 6, 7, 8])
    pl.legend()
    remove_spines(ax)
    pl.gca().invert_xaxis()
    fig.savefig(path.join(plot_dir, "REVISION_S2B_Constant_Temp_Persistence_trajectory_length_by_ycoordinate.pdf"))

    # persistence length by temperature in gradient experiments
    pers_info = persistence_analysis(gradient_trajectories)

    f_cold_p_bs = utils.bootstrap_weighted_histogram_avg(np.array(pers_info["Temperature"][pers_info["Direction"] == "decreasing"]),
                                                         temp_bins,
                                                         np.array(pers_info["Length"][pers_info["Direction"] == "decreasing"]), 1000)
    m_cold_p = np.nanmean(f_cold_p_bs, axis=0)
    e_cold_p = np.nanstd(f_cold_p_bs, axis=0)
    f_hot_p_bs = utils.bootstrap_weighted_histogram_avg(np.array(pers_info["Temperature"][pers_info["Direction"] == "increasing"]),
                                                        temp_bins,
                                                        np.array(pers_info["Length"][pers_info["Direction"] == "increasing"]), 1000)
    m_hot_p = np.nanmean(f_hot_p_bs, axis=0)
    e_hot_p = np.nanstd(f_hot_p_bs, axis=0)

    p_vals = utils.compute_weighted_histogram_p([np.array(pers_info["Temperature"][pers_info["Direction"] == "decreasing"]), np.array(pers_info["Temperature"][pers_info["Direction"] == "increasing"])], temp_bins,
                                                [np.array(pers_info["Length"][pers_info["Direction"] == "decreasing"]), np.array(pers_info["Length"][pers_info["Direction"] == "increasing"])],
                                                10_0000)
    max_y = max(np.max(m_cold_p), np.max(m_hot_p))

    fig, ax = pl.subplots()
    pl.fill_between(temp_bincents, m_cold_p-e_cold_p, m_cold_p+e_cold_p, alpha=0.3, color="C0")
    pl.plot(temp_bincents, m_cold_p, label="Facing cold", color="C0", marker='.')
    pl.fill_between(temp_bincents, m_hot_p-e_hot_p, m_hot_p+e_hot_p, alpha=0.3, color="C1")
    pl.plot(temp_bincents, m_hot_p, label="Facing hot", color="C1", marker='.')
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
    pl.ylabel("Avg. persistence length")
    pl.ylim(2, 8)
    pl.yticks([2, 3, 4, 5, 6, 7, 8])
    pl.legend()
    remove_spines(ax)
    fig.savefig(path.join(plot_dir, "REVISION_2F_Persistence_trajectory_length_BootTest.pdf"))

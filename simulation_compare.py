"""
Script for behavioral analysis in 3rd order Markov Model simulations
"""


import pickle
import numpy as np
import plot_funs as pf
from os import path
import os
import matplotlib.pyplot as pl
import matplotlib as mpl
import pandas as pd
import utils
from kinematics_compare import turn_cut, plot_corr_straps
from typing import List, Tuple
import seaborn as sns
import argparse

delta_temps = np.linspace(0, 0.15, 4)
delta_temps_centers = delta_temps[:-1] + np.diff(delta_temps) / 2

temperature_bins = np.arange(14)+18.5
temp_bc = temperature_bins[:-1] + np.diff(temperature_bins)/2


delta_cut = 0.052

def norm_hist(data: np.ndarray, bins: np.ndarray, weights: np.ndarray) -> np.ndarray:
    hw = np.histogram(data, bins, weights=weights)[0].astype(float)
    hc = np.histogram(data, bins)[0].astype(float)
    return hw/hc


def filter_bouts(simbouts: pd.DataFrame, x_max: int, y_max: int, border: int) -> []:
    bout_x = np.array(simbouts["X Position"])
    bout_y = np.array(simbouts["Y Position"])
    val_x = np.logical_and(bout_x > border, x_max - bout_x > border)
    val_y = np.logical_and(bout_y > border, y_max - bout_y > border)
    valid = np.logical_and(val_x, val_y)
    df_filtered = simbouts.iloc[valid].copy(deep=True)
    return df_filtered


if __name__ == '__main__':

    mpl.rcParams['pdf.fonttype'] = 42
    pf.set_journal_style(23, 23)

    a_parser = argparse.ArgumentParser(prog="simulation_compare.py",
                                       description="Comparative bout kinematic analysis for simulation data")
    a_parser.add_argument("-gf", "--gradient", help="Path to folder with gradient trajectory pickles",
                          type=str)
    a_parser.add_argument("-sf", "--simulation", help="Path to folder with simulation data",
                          type=str)

    args = a_parser.parse_args()

    gradient_folder = args.gradient
    simulation_folder = args.simulation

    plot_dir = "simulation_compare"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    sim_kinematics_by_temp = {
        "Displacement [mm]": [],
        "IBI [ms]": [],
        "Turn magnitude [degrees]": []
    }

    # for bout edge filtering as in regular experiments
    border_size = 4 * 7  # since simulations run in pixels

    with open(path.join(simulation_folder, "3rdorderFish_model_3_bouts.pkl"), 'rb') as bout_file:
        mm_bouts = pickle.load(bout_file)
        for bout_key in mm_bouts["fishmodel_F32_B18"]:  # sim indices
            bouts = filter_bouts(mm_bouts["fishmodel_F32_B18"][bout_key], 320, 2562, border_size)
            b_displace = bouts["Displacement"] / 7
            b_angle = np.rad2deg(bouts["Angle change"])
            b_ibi = bouts["IBI"] * 10
            b_temp = bouts["Temperature"]
            sim_kinematics_by_temp["Displacement [mm]"].append(norm_hist(b_temp, temperature_bins, b_displace))
            sim_kinematics_by_temp["IBI [ms]"].append(norm_hist(b_temp, temperature_bins, b_ibi))
            # for turns magnitude limit to actual turns
            is_turn = np.abs(b_angle) > turn_cut
            sim_kinematics_by_temp["Turn magnitude [degrees]"].append(norm_hist(b_temp[is_turn], temperature_bins, np.abs(b_angle[is_turn])))

    for k in sim_kinematics_by_temp:
        fig, ax = pl.subplots()
        bvars = utils.bootstrap(np.vstack(sim_kinematics_by_temp[k]), 1000, np.nanmean)
        pl.errorbar(temp_bc, np.nanmean(bvars, axis=0), np.nanstd(bvars, axis=0))
        pl.ylabel(k)
        pl.xlabel("Temperature [C]")
        pf.remove_spines(ax)
        fig.savefig(path.join(plot_dir, "S5_simulation_"+k[:k.find('[')]+".pdf"))

    heating_kinematics_by_temp = {k: [] for k in sim_kinematics_by_temp}
    cooling_kinematics_by_temp = {k: [] for k in sim_kinematics_by_temp}
    for bout_key in mm_bouts["fishmodel_F32_B18"]:  # sim indices
        bouts = filter_bouts(mm_bouts["fishmodel_F32_B18"][bout_key], 320, 2562, border_size)

        b_displace = bouts["Displacement"] / 7
        b_angle = np.rad2deg(bouts["Angle change"])
        b_ibi = bouts["IBI"] * 10
        b_dt = bouts["Prev Delta T"]
        b_temp = bouts["Temperature"]

        cooling = b_dt < -delta_cut
        heating = b_dt > delta_cut

        heating_kinematics_by_temp["Displacement [mm]"].append(
            norm_hist(b_temp[heating], temperature_bins, b_displace[heating]))
        cooling_kinematics_by_temp["Displacement [mm]"].append(
            norm_hist(b_temp[cooling], temperature_bins, b_displace[cooling]))

        heating_kinematics_by_temp["IBI [ms]"].append(
            norm_hist(b_temp[heating], temperature_bins, b_ibi[heating]))
        cooling_kinematics_by_temp["IBI [ms]"].append(
            norm_hist(b_temp[cooling], temperature_bins, b_ibi[cooling]))

        # for turns magnitude limit to actual turns
        is_turn = np.abs(b_angle) > turn_cut
        t_c = np.logical_and(is_turn, cooling)
        t_h = np.logical_and(is_turn, heating)

        heating_kinematics_by_temp["Turn magnitude [degrees]"].append(
            norm_hist(b_temp[t_h], temperature_bins, np.abs(b_angle)[t_h]))
        cooling_kinematics_by_temp["Turn magnitude [degrees]"].append(
            norm_hist(b_temp[t_c], temperature_bins, np.abs(b_angle)[t_c]))

    for k in sim_kinematics_by_temp:
        fig, ax = pl.subplots()
        bvars_h = utils.bootstrap(np.vstack(heating_kinematics_by_temp[k]), 1000, np.nanmean)
        bvars_c = utils.bootstrap(np.vstack(cooling_kinematics_by_temp[k]), 1000, np.nanmean)
        pl.errorbar(temp_bc, np.nanmean(bvars_h, axis=0), np.nanstd(bvars_h, axis=0), label="Heating", color="C1")
        pl.errorbar(temp_bc, np.nanmean(bvars_c, axis=0), np.nanstd(bvars_c, axis=0), label="Cooling", color="C0")
        pl.ylabel(k)
        pl.xlabel("Temperature [C]")
        pl.legend()
        pf.remove_spines(ax)
        pf.format_legend(ax)
        fig.savefig(path.join(plot_dir, "S5_simulation_"+k[:k.find('[')]+"_bydirection.pdf"))

    corr_temp_centers = np.array([20, 22, 24, 26, 28, 30])

    displacement_pairs = {"Direction": [], "Previous": [], "Current": [], "Temperature": [], "Fish ID": []}
    angle_pairs = {"Direction": [], "Previous": [], "Current": [], "Temperature": [], "Fish ID": []}
    for bout_key in mm_bouts["fishmodel_F32_B18"]:  # sim indices
        bouts = filter_bouts(mm_bouts["fishmodel_F32_B18"][bout_key], 320, 2562, border_size)
        for i in range(1, bouts.shape[0]):
            prev = bouts.iloc[i-1]
            current = bouts.iloc[i]

            if (current["Original Index"] - prev["Original Index"]) > 1.5:
                continue  # bouts are not part of same trajectory

            if current["Prev Delta T"] < -delta_cut:
                direction = "Cooling"
            elif current["Prev Delta T"] > delta_cut:
                direction = "Heating"
            else:
                continue
            try:
                ix_temp = np.where(np.abs(current["Temperature"] - corr_temp_centers) <= 1)[0][0]
            except IndexError:
                continue
            temperature = corr_temp_centers[ix_temp]
            displacement_pairs["Direction"].append(direction)
            displacement_pairs["Previous"].append(prev["Displacement"])
            displacement_pairs["Current"].append(current["Displacement"])
            displacement_pairs["Temperature"].append(temperature)
            displacement_pairs["Fish ID"].append(bout_key)

            # limit to actual turns
            if np.abs(np.rad2deg(current["Angle change"])) < turn_cut or np.abs(np.rad2deg(prev["Angle change"])) < turn_cut:
                continue

            angle_pairs["Direction"].append(direction)
            angle_pairs["Previous"].append(prev["Angle change"])
            angle_pairs["Current"].append(current["Angle change"])
            angle_pairs["Temperature"].append(temperature)
            angle_pairs["Fish ID"].append(bout_key)

    for k in angle_pairs.keys():
        angle_pairs[k] = np.hstack(angle_pairs[k])
        displacement_pairs[k] = np.hstack(displacement_pairs[k])

    fig = plot_corr_straps(angle_pairs, corr_temp_centers, 1000)
    pl.xticks([20.0, 22.5, 25.0, 27.5, 30.0])
    fig.savefig(path.join(plot_dir, "S5_sim_angle_correlations.pdf"))

    fig = plot_corr_straps(displacement_pairs, corr_temp_centers, 1000)
    pl.xticks([20.0, 22.5, 25.0, 27.5, 30.0])
    fig.savefig(path.join(plot_dir, "S5_sim_displacement_correlations.pdf"))

    # simulation occupancy and kl divergences

    def temp_convert_cold(ypos):
        return (18-26)*ypos/1464 + 26

    def temp_convert_hot(ypos):
        return (24-32)*ypos/1464 + 32

    def kl_divergence(data_a: np.ndarray, data_b: np.ndarray, bins: np.ndarray) -> Tuple[float, float]:
        c_a = np.histogram(data_a, bins=bins)[0].astype(float)
        p_a = c_a / np.sum(c_a)
        c_b = np.histogram(data_b, bins=bins)[0].astype(float)
        p_b = c_b / np.sum(c_b)
        d_kl = 0
        d_kl_uni = 0
        uni_prob = 1 / (bins.size - 1)
        for i in range(bins.size-1):
            if p_b[i] > 0:
                d_kl += p_a[i] * np.log(p_a[i] / p_b[i])
                d_kl_uni += p_a[i] * np.log(p_a[i] / uni_prob)
            else:
                if p_a[i] > 0:
                    d_kl += np.inf
        return d_kl, d_kl_uni

    def load_fish_sim_data(filename: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        with open(filename, "rb") as fish_file:
            example_fish = pickle.load(fish_file)
            keys = example_fish.keys()
            cold_key = [k for k in keys if "F26_B18" in k][0]
            hot_key = [k for k in keys if "F32_B24" in k][0]
        return example_fish[cold_key], example_fish[hot_key]

    # collect all real fish temperatures by experiment type
    fish_pickles = utils.find_all_fish_pickle_paths(gradient_folder)
    real_pos = {}
    for pick in fish_pickles:
        if "F26_B18" in pick:
            k = "cold"
        elif "F32_B24" in pick:
            k = "hot"
        else:
            raise ValueError(f"Could not identify gradient type for file {pick}")
        if k not in real_pos:
            real_pos[k] = []
        fish = pd.read_pickle(pick)
        real_pos[k].append(np.array(fish["Temperature"]))

    for k in real_pos:
        real_pos[k] = np.hstack(real_pos[k])

    T_cold, T_hot = load_fish_sim_data(path.join(simulation_folder,'Experiment_Fish_model_constant.fishexp'))
    TdT_cold, TdT_hot = load_fish_sim_data(path.join(simulation_folder,'Experiment_Fish_model_gradient.fishexp'))
    MM3_cold, MM3_hot = load_fish_sim_data(path.join(simulation_folder,'Fish_model_3.fishexp'))
    MM2_cold, MM2_hot = load_fish_sim_data(path.join(simulation_folder,'Fish_model_2.fishexp'))
    MM1_cold, MM1_hot = load_fish_sim_data(path.join(simulation_folder,'Fish_model_1.fishexp'))
    MM0_cold, MM0_hot = load_fish_sim_data(path.join(simulation_folder,'Fish_model_0.fishexp'))

    with open(path.join(simulation_folder,'3rdorderFish_model_3.fishexp'), 'rb') as fish_file:
        MM3_all = pickle.load(fish_file)["fishmodel_F32_B18"]

    cold_bins = np.linspace(18.5, 25.5, 50)
    cold_bc = cold_bins[:-1] + np.diff(cold_bins)/2
    hot_bins = np.linspace(24.5, 31.5, 50)
    hot_bc = hot_bins[:-1] + np.diff(hot_bins)/2
    all_bins = np.linspace(18.5, 31.5, 50)
    all_bc = all_bins[:-1] + np.diff(all_bins)/2

    def plot_sim_histogram(sim_list: List, bins: np.ndarray, converter: callable, comp_data: np.ndarray):
        sim_temps = np.hstack([converter(sim_list[elem][:, 1]) for elem in sim_list])
        if comp_data is not None:
            divergence, div_uni = kl_divergence(comp_data, sim_temps, bins)
            sim_hist = np.histogram(sim_temps, bins=bins)[0].astype(float)
            sim_hist /= sim_hist.max()  # convert to occupancy
            comp_hist = np.histogram(comp_data, bins=bins)[0].astype(float)
            comp_hist /= comp_hist.max()
            bincenters = bins[:-1] + np.diff(bins)/2
            pl.plot(bincenters, sim_hist, label=f"KL divergence: {np.round(divergence, 3)}", color="C1")
            pl.plot(bincenters, comp_hist, color='k', label=f"dKL Uniform: {np.round(div_uni, 3)}")
        else:
            sim_hist = np.histogram(sim_temps, bins=bins)[0].astype(float)
            sim_hist /= sim_hist.max()  # convert to occupancy
            bincenters = bins[:-1] + np.diff(bins) / 2
            pl.plot(bincenters, sim_hist)
        pl.xlabel("Temperature [C]")
        pl.ylabel("Preference strength")
        pl.ylim(0, 1)
        pl.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        pl.legend()
        sns.despine()

    fig = pl.figure()
    plot_sim_histogram(T_cold, cold_bins, temp_convert_cold, real_pos["cold"])
    fig.savefig(path.join(plot_dir, "F2_T_sim_Cold.pdf"))

    fig = pl.figure()
    plot_sim_histogram(T_hot, hot_bins, temp_convert_hot, real_pos["hot"])
    fig.savefig(path.join(plot_dir, "F2_T_sim_Hot.pdf"))

    fig = pl.figure()
    plot_sim_histogram(TdT_cold, cold_bins, temp_convert_cold, real_pos["cold"])
    fig.savefig(path.join(plot_dir, "F2_TdT_sim_Cold.pdf"))

    fig = pl.figure()
    plot_sim_histogram(TdT_hot, hot_bins, temp_convert_hot, real_pos["hot"])
    fig.savefig(path.join(plot_dir, "F2_TdT_sim_Hot.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM3_cold, cold_bins, temp_convert_cold, real_pos["cold"])
    fig.savefig(path.join(plot_dir, "F5_MM3_sim_Cold.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM3_hot, hot_bins, temp_convert_hot, real_pos["hot"])
    fig.savefig(path.join(plot_dir, "F5_MM3_sim_Hot.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM2_cold, cold_bins, temp_convert_cold, real_pos["cold"])
    fig.savefig(path.join(plot_dir, "F5_MM2_sim_Cold.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM2_hot, hot_bins, temp_convert_hot, real_pos["hot"])
    fig.savefig(path.join(plot_dir, "F5_MM2_sim_Hot.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM1_cold, cold_bins, temp_convert_cold, real_pos["cold"])
    fig.savefig(path.join(plot_dir, "F5_MM1_sim_Cold.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM1_hot, hot_bins, temp_convert_hot, real_pos["hot"])
    fig.savefig(path.join(plot_dir, "F5_MM1_sim_Hot.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM0_cold, cold_bins, temp_convert_cold, real_pos["cold"])
    fig.savefig(path.join(plot_dir, "S5_MM0_sim_Cold.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM0_hot, hot_bins, temp_convert_hot, real_pos["hot"])
    fig.savefig(path.join(plot_dir, "S5_MM0_sim_Hot.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM3_all, all_bins, temp_convert_hot, None)
    fig.savefig(path.join(plot_dir, "S5_MM3_sim_All.pdf"))

    cold_0, hot_0 = load_fish_sim_data(path.join(simulation_folder,'Fish_model_0.fishexp'))
    temp_cold_0 = np.hstack([temp_convert_cold(cold_0[elem][:, 1]) for elem in cold_0])
    temp_hot_0 = np.hstack([temp_convert_hot(hot_0[elem][:, 1]) for elem in hot_0])
    cold_1, hot_1 = load_fish_sim_data(path.join(simulation_folder,'Fish_model_1.fishexp'))
    temp_cold_1 = np.hstack([temp_convert_cold(cold_1[elem][:, 1]) for elem in cold_1])
    temp_hot_1 = np.hstack([temp_convert_hot(hot_1[elem][:, 1]) for elem in hot_1])
    cold_2, hot_2 = load_fish_sim_data(path.join(simulation_folder,'Fish_model_2.fishexp'))
    temp_cold_2 = np.hstack([temp_convert_cold(cold_2[elem][:, 1]) for elem in cold_2])
    temp_hot_2 = np.hstack([temp_convert_hot(hot_2[elem][:, 1]) for elem in hot_2])
    cold_3, hot_3 = load_fish_sim_data(path.join(simulation_folder,'Fish_model_3.fishexp'))
    temp_cold_3 = np.hstack([temp_convert_cold(cold_3[elem][:, 1]) for elem in cold_3])
    temp_hot_3 = np.hstack([temp_convert_hot(hot_3[elem][:, 1]) for elem in hot_3])

    mm_kl_divergences = {"Model": [], "dKL": [], "Gradient": []}
    mm_kl_divergences["Model"].append("Random")
    mm_kl_divergences["Gradient"].append("Cold avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["cold"], temp_cold_0, cold_bins)[1])
    mm_kl_divergences["Model"].append("Order 0")
    mm_kl_divergences["Gradient"].append("Cold avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["cold"], temp_cold_0, cold_bins)[0])
    mm_kl_divergences["Model"].append("Order 1")
    mm_kl_divergences["Gradient"].append("Cold avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["cold"], temp_cold_1, cold_bins)[0])
    mm_kl_divergences["Model"].append("Order 2")
    mm_kl_divergences["Gradient"].append("Cold avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["cold"], temp_cold_2, cold_bins)[0])
    mm_kl_divergences["Model"].append("Order 3")
    mm_kl_divergences["Gradient"].append("Cold avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["cold"], temp_cold_3, cold_bins)[0])
    mm_kl_divergences["Model"].append("Random")
    mm_kl_divergences["Gradient"].append("Hot avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["hot"], temp_hot_0, hot_bins)[1])
    mm_kl_divergences["Model"].append("Order 0")
    mm_kl_divergences["Gradient"].append("Hot avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["hot"], temp_hot_0, hot_bins)[0])
    mm_kl_divergences["Model"].append("Order 1")
    mm_kl_divergences["Gradient"].append("Hot avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["hot"], temp_hot_1, hot_bins)[0])
    mm_kl_divergences["Model"].append("Order 2")
    mm_kl_divergences["Gradient"].append("Hot avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["hot"], temp_hot_2, hot_bins)[0])
    mm_kl_divergences["Model"].append("Order 3")
    mm_kl_divergences["Gradient"].append("Hot avoidance")
    mm_kl_divergences["dKL"].append(kl_divergence(real_pos["hot"], temp_hot_3, hot_bins)[0])

    fig = pl.figure()
    sns.stripplot(data=mm_kl_divergences, x="Model", y="dKL", hue="Gradient")
    sns.despine()
    fig.savefig(path.join(plot_dir, "F5_ModelDivergences.pdf"))
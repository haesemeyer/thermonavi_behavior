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
from typing import List, Tuple, Optional, Union
import seaborn as sns
import argparse
import scipy.stats as sts

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

    plot_dir = "REVISION_simulation_compare"
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    sim_kinematics_by_temp = {
        "Displacement [mm]": [],
        "IBI [ms]": [],
        "Turn magnitude [degrees]": []
    }

    # for bout edge filtering as in regular experiments
    border_size = 4 * 7  # since simulations run in pixels

    with open(path.join(simulation_folder, "Fish_model_fullsim_transition_bouts_250618.pkl"), 'rb') as bout_file:
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
        m = np.nanmean(bvars, axis=0)
        e = np.nanstd(bvars, axis=0)
        ax.fill_between(temp_bc, m - e, m + e, color='C1', alpha=0.4)
        ax.plot(temp_bc, m, color='C1', marker='.')
        pl.ylabel(k)
        pl.xlabel("Temperature [C]")
        pl.xticks([20, 22.5, 25, 27.5, 30])
        pf.remove_spines(ax)
        fig.savefig(path.join(plot_dir, "REVISION_S6A-C_simulation_"+k[:k.find('[')]+".pdf"))

    heating_kinematics_by_temp = {"Turn magnitude [degrees]": []}
    cooling_kinematics_by_temp = {"Turn magnitude [degrees]": []}
    for bout_key in mm_bouts["fishmodel_F32_B18"]:  # sim indices
        bouts = filter_bouts(mm_bouts["fishmodel_F32_B18"][bout_key], 320, 2562, border_size)

        b_angle = np.rad2deg(bouts["Angle change"])
        b_dt = bouts["Prev Delta T"]
        b_temp = bouts["Temperature"]

        cooling = b_dt < -delta_cut
        heating = b_dt > delta_cut

        # for turns magnitude limit to actual turns
        is_turn = np.abs(b_angle) > turn_cut
        t_c = np.logical_and(is_turn, cooling)
        t_h = np.logical_and(is_turn, heating)

        heating_kinematics_by_temp["Turn magnitude [degrees]"].append(
            norm_hist(b_temp[t_h], temperature_bins, np.abs(b_angle)[t_h]))
        cooling_kinematics_by_temp["Turn magnitude [degrees]"].append(
            norm_hist(b_temp[t_c], temperature_bins, np.abs(b_angle)[t_c]))

    for k in heating_kinematics_by_temp:
        # compute p-values
        hk_by_temp = np.vstack(heating_kinematics_by_temp[k])
        ck_by_temp = np.vstack(cooling_kinematics_by_temp[k])
        p_vals = np.full(temp_bc.size, np.nan)
        for dix in range(p_vals.size):
            da = hk_by_temp[:, dix]
            db = ck_by_temp[:, dix]
            p_vals[dix] = sts.ranksums(da[np.isfinite(da)], db[np.isfinite(db)]).pvalue
        p_vals *= p_vals.size  # Bonferroni correction

        fig, ax = pl.subplots()
        bvars_h = utils.bootstrap(hk_by_temp, 1000, np.nanmean)
        bvars_c = utils.bootstrap(ck_by_temp, 1000, np.nanmean)
        m = np.nanmean(bvars_h, axis=0)
        e = np.nanstd(bvars_h, axis=0)
        y_max = np.max(m)
        ax.fill_between(temp_bc, m - e, m + e, color='C1', alpha=0.4)
        ax.plot(temp_bc, m, color='C1', marker='.', label="Heating")
        m = np.nanmean(bvars_c, axis=0)
        e = np.nanstd(bvars_c, axis=0)
        y_max = max([y_max, np.max(m)])
        ax.fill_between(temp_bc, m - e, m + e, color='C0', alpha=0.4)
        ax.plot(temp_bc, m, color='C0', marker='.', label="Cooling")

        for pv, xv in zip(p_vals, temp_bc):
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
            ax.scatter(xv, y_max, marker=marker, color='k')

        pl.ylabel(k)
        pl.xlabel("Temperature [C]")
        pl.legend()
        pf.remove_spines(ax)
        pf.format_legend(ax)
        fig.savefig(path.join(plot_dir, "REVISION_S6D_simulation_"+k[:k.find('[')]+"_bydirection.pdf"))

    corr_temp_centers = np.array([20, 22, 24, 26, 28, 30])

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

    fig = plot_corr_straps(angle_pairs, corr_temp_centers, 1000)
    pl.xticks([20.0, 22.5, 25.0, 27.5, 30.0])
    fig.savefig(path.join(plot_dir, "REVISION_S6E_sim_angle_correlations.pdf"))

    # simulation occupancy and kl divergences

    def temp_convert_cold(ypos):
        return (18-26)*ypos/1464 + 26

    def temp_convert_hot(ypos):
        return (24-32)*ypos/1464 + 32

    def kl_divergence(p_a: np.ndarray, data_b: np.ndarray, bins: np.ndarray) -> Tuple[float, float]:
        c_b = np.histogram(data_b, bins=bins)[0].astype(float)
        p_b = c_b / np.sum(c_b)
        uni_prob = 1 / (bins.size - 1)
        log_ratio = np.log(p_a / p_b)
        log_ratio[np.logical_and(p_b==0, p_a==0)] = 0  # lim(x->0) np.log(x/x) = 0
        d_kl = np.sum(p_a * log_ratio)
        d_kl_uni = np.sum(p_a * np.log(p_a / uni_prob))
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
    real_pos_list = {}
    for pick in fish_pickles:
        if "F26_B18" in pick:
            k = "cold"
        elif "F32_B24" in pick:
            k = "hot"
        else:
            raise ValueError(f"Could not identify gradient type for file {pick}")
        if k not in real_pos_list:
            real_pos_list[k] = []
        fish = pd.read_pickle(pick)
        real_pos_list[k].append(np.array(fish["Temperature"]))

    real_pos = {}
    for k in real_pos_list:
        real_pos[k] = np.hstack(real_pos_list[k])  # prepare for confidence by bootstrapping across fish

    TdT_cold, TdT_hot = load_fish_sim_data(path.join(simulation_folder,'Experiment_Fish_model_gradient.fishexp'))
    MM0_cold, MM0_hot = load_fish_sim_data(path.join(simulation_folder,'Fish_model_0_order_250618.fishexp'))
    MMstim_cold, MMstim_hot = load_fish_sim_data(path.join(simulation_folder,'Fish_model_transition_250618.fishexp'))

    with open(path.join(simulation_folder,'Fish_model_fullsim_transition_250618.fishexp'), 'rb') as fish_file:
        MMstim_all = pickle.load(fish_file)["fishmodel_F32_B18"]

    cold_bins = np.linspace(18.5, 25.5, 50)
    cold_bc = cold_bins[:-1] + np.diff(cold_bins)/2
    hot_bins = np.linspace(24.5, 31.5, 50)
    hot_bc = hot_bins[:-1] + np.diff(hot_bins)/2
    all_bins = np.linspace(18.5, 31.5, 50)
    all_bc = all_bins[:-1] + np.diff(all_bins)/2

    def plot_sim_histogram(sim_list: List, bins: np.ndarray, converter: callable, comp_data: Optional[np.ndarray],
                           name: Optional[str]=None):
        sim_temps = np.hstack([converter(sim_list[elem][:, 1]) for elem in sim_list])
        if comp_data is not None:
            p_comp_data = np.histogram(comp_data, bins=bins)[0].astype(float)
            p_comp_data /= p_comp_data.sum()
            divergence, div_uni = kl_divergence(p_comp_data, sim_temps, bins)
            sim_hist = np.histogram(sim_temps, bins=bins)[0].astype(float)
            sim_hist /= sim_hist.max()  # convert to occupancy
            comp_hist = np.histogram(comp_data, bins=bins)[0].astype(float)
            comp_hist /= comp_hist.max()
            bincenters = bins[:-1] + np.diff(bins)/2
            pl.plot(bincenters, sim_hist, label=f"{name} KL divergence: {np.round(divergence, 3)}")
            pl.plot(bincenters, comp_hist, color='k', label=f"dKL Uniform: {np.round(div_uni, 3)}")
        else:
            sim_hist = np.histogram(sim_temps, bins=bins)[0].astype(float)
            sim_hist /= sim_hist.max()  # convert to occupancy
            bincenters = bins[:-1] + np.diff(bins) / 2
            pl.plot(bincenters, sim_hist)
        pl.xlabel("Temperature [C]")
        pl.ylabel("Occupancy")
        pl.ylim(0, 1)
        pl.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        pl.legend()
        sns.despine()

    fig = pl.figure()
    plot_sim_histogram(TdT_cold, cold_bins, temp_convert_cold, real_pos["cold"])
    pl.xticks([20, 22.5, 25])
    fig.savefig(path.join(plot_dir, "REVISION_S1H_TdT_sim_Cold.pdf"))

    fig = pl.figure()
    plot_sim_histogram(TdT_hot, hot_bins, temp_convert_hot, real_pos["hot"])
    pl.xticks([25, 27.5, 30])
    fig.savefig(path.join(plot_dir, "REVISION_S1G_TdT_sim_Hot.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM0_cold, cold_bins, temp_convert_cold, real_pos["cold"], "Constant")
    plot_sim_histogram(MMstim_cold, cold_bins, temp_convert_cold, real_pos["cold"], "Stimulus")
    pl.xticks([20, 22.5, 25])
    fig.savefig(path.join(plot_dir, "REVISION_6C_MM_Cold.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MM0_hot, hot_bins, temp_convert_hot, real_pos["hot"], "Constant")
    plot_sim_histogram(MMstim_hot, hot_bins, temp_convert_hot, real_pos["hot"], "Stimulus")
    pl.xticks([25, 27.5, 30])
    fig.savefig(path.join(plot_dir, "REVISION_6B_MM_Hot.pdf"))

    fig = pl.figure()
    plot_sim_histogram(MMstim_all, all_bins, temp_convert_hot, None)
    pl.xticks([20, 22.5, 25, 27.5, 30])
    fig.savefig(path.join(plot_dir, "REVISION_S6F_MMstim_sim_All.pdf"))

    temp_cold_0 = np.vstack([temp_convert_cold(MM0_cold[elem][:, 1])[:168000] for elem in MM0_cold])
    temp_hot_0 = np.vstack([temp_convert_hot(MM0_hot[elem][:, 1])[:168000] for elem in MM0_hot])
    temp_cold_stim = np.vstack([temp_convert_cold(MMstim_cold[elem][:, 1])[:168000] for elem in MMstim_cold])
    temp_hot_stim = np.vstack([temp_convert_hot(MMstim_hot[elem][:, 1])[:168000] for elem in MMstim_hot])

    p_cold = np.histogram(real_pos["cold"], cold_bins)[0].astype(float)
    p_cold /= p_cold.sum()
    p_hot = np.histogram(real_pos["hot"], hot_bins)[0].astype(float)
    p_hot /= p_hot.sum()

    indexer = np.arange(200)
    mm_kl_divergences = {"Model": [], "dKL": [], "Gradient": []}
    mm_kl_divergences["Model"].append("Random")
    mm_kl_divergences["Gradient"].append("Cold avoidance")
    kl_cold_chance = kl_divergence(p_cold, temp_cold_0, cold_bins)[1]
    mm_kl_divergences["dKL"].append(kl_cold_chance)
    mm_kl_divergences["Model"].append("Random")
    mm_kl_divergences["Gradient"].append("Hot avoidance")
    kl_hot_chance = kl_divergence(p_hot, temp_hot_0, hot_bins)[1]
    mm_kl_divergences["dKL"].append(kl_hot_chance)

    for i in range(100):
        index = np.random.choice(indexer, 200)
        mm_kl_divergences["Model"].append("Constant")
        mm_kl_divergences["Gradient"].append("Cold avoidance")
        mm_kl_divergences["dKL"].append(kl_divergence(p_cold, temp_cold_0[index].ravel(), cold_bins)[0])
        mm_kl_divergences["Model"].append("Stimulus")
        mm_kl_divergences["Gradient"].append("Cold avoidance")
        mm_kl_divergences["dKL"].append(kl_divergence(p_cold, temp_cold_stim[index].ravel(), cold_bins)[0])

        mm_kl_divergences["Model"].append("Constant")
        mm_kl_divergences["Gradient"].append("Hot avoidance")
        mm_kl_divergences["dKL"].append(kl_divergence(p_hot, temp_hot_0[index].ravel(), hot_bins)[0])
        mm_kl_divergences["Model"].append("Stimulus")
        mm_kl_divergences["Gradient"].append("Hot avoidance")
        mm_kl_divergences["dKL"].append(kl_divergence(p_hot, temp_hot_stim[index].ravel(), hot_bins)[0])

    def bootstrap_test_kl(p_comp: np.ndarray, sim_a: Union[np.ndarray, float], sim_b: np.ndarray, bins: np.ndarray, n_boot=1000) -> float:
        """
        Two-sided bootstrap test comparing the KL divergence between a fixed comparison distribution and two simulation distributions
        :param p_comp: The comparison probability distribution across the same bins as bins
        :param sim_a: n_sims x datapoints data from the first simulation or a fixed KL-divergence value against which to test
        :param sim_b: n_sims x datapoints data from the second simulation
        :param bins: The bins to use for KL divergence calculation
        :param n_boot: The number of bootstrap samples to use for the test (note that this determines p-value granularity)
        :return: p-value for the KL divergence difference
        """
        if type(sim_a) != np.ndarray:
            kl_boot = np.full(n_boot, np.nan)
            kl_diff_real = kl_divergence(p_comp, sim_b.ravel(), bins)[0] - sim_a
            indices = np.arange(sim_b.shape[0])
            for boot in range(n_boot):
                ix_boot = np.random.choice(indices, indices.size, replace=True)
                bsam = sim_b[ix_boot]
                kl_boot[boot] = kl_divergence(p_comp, bsam.ravel(), bins)[0]
            return np.sum(kl_boot <= sim_a)/n_boot if kl_diff_real>0 else np.sum(kl_boot >= sim_a)/n_boot
        kl_diff_boot = np.full(n_boot, np.nan)
        kl_diff_real = np.abs(kl_divergence(p_comp, sim_a.ravel(), bins)[0] -
                              kl_divergence(p_comp, sim_b.ravel(), bins)[0])
        data_join = np.vstack((sim_a, sim_b))
        indices = np.arange(data_join.shape[0])
        for boot in range(n_boot):
            np.random.shuffle(indices)
            bs_a = data_join[indices[:sim_a.shape[0]]]
            bs_b = data_join[indices[sim_a.shape[0]:]]
            kl_diff_boot[boot] = np.abs(kl_divergence(p_comp, bs_a.ravel(), bins)[0] -
                                        kl_divergence(p_comp, bs_b.ravel(), bins)[0])
        return np.sum(kl_diff_boot >= kl_diff_real)/n_boot

    # get p-values for key comparisons - we subsample the simulations across time to make this efficient - since
    # (simulated) fish do not move 10 times per second, sampling time at 10 Hz should still be sufficient to obtain
    # an accurate representation of the chamber distribution
    p_cold_chance_vs_stim = bootstrap_test_kl(p_cold, kl_cold_chance, temp_cold_stim[:, ::10], cold_bins, 10_000)
    p_cold_0_vs_stim = bootstrap_test_kl(p_cold, temp_cold_0[:, ::10], temp_cold_stim[:, ::10], cold_bins, 10_000)
    p_hot_chance_vs_stim = bootstrap_test_kl(p_hot, kl_hot_chance, temp_hot_stim[:, ::10], hot_bins, 10_000)
    p_hot_0_vs_stim = bootstrap_test_kl(p_hot, temp_hot_0[:, ::10], temp_hot_stim[:, ::10], hot_bins, 10_000)

    fig = pl.figure()
    # in the following we pick standard deviation as the error metric since our samples are already from a bootstrap
    # in other words, the standard deviation across these samples is the nonparametric standard error of the data
    sns.pointplot(data=mm_kl_divergences, x="Model", y="dKL", hue="Gradient",
                  order=["Random", "Constant", "Stimulus"], errorbar='sd', join=False)
    pl.text(0, 0.3, f"Cold chance vs stim {p_cold_chance_vs_stim}")
    pl.text(0, 0.28, f"Cold 0 vs stim {p_cold_0_vs_stim}")
    pl.text(0, 0.26, f"Hot chance vs stim {p_hot_chance_vs_stim}")
    pl.text(0, 0.24, f"Hot 0 vs stim {p_hot_0_vs_stim}")
    sns.despine()
    fig.savefig(path.join(plot_dir, "REVISION_6D_ModelDivergences.pdf"))
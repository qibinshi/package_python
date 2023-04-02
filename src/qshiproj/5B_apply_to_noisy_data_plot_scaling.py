"""
Apply the model to noise data
e.g. deep earthquakes

author: Qibin Shi
"""

import matplotlib
import numpy
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from denoiser_util import get_vp_vs
from scipy.stats import binned_statistic
from obspy import read_events
plt.rcParams['axes.axisbelow'] = True
matplotlib.rcParams.update({'font.size': 35})

def main():
    model_dir = 'Release_Middle_augmentation_P4Hz_150s'
    fig_dir = model_dir + '/apply_minSNR2_fixWin600_max2.5falloff_multipeak'
    # model_dir = 'Release_Middle_augmentation_S4Hz_150s_removeCoda_SoverCoda20_1980-2021-include-noFMS'
    # fig_dir = model_dir + '/apply_minSNR2_fixWin800'
    csv_file = fig_dir + '/source_measurements.csv'

    # %% Read the saved source parameters of global earthquakes
    meta_result = pd.read_csv(csv_file, low_memory=False)

    evid = meta_result.source_id.to_numpy()
    mb = meta_result.source_magnitude.to_numpy()
    nbin = meta_result.num_bin.to_numpy()
    nsta = meta_result.num_station.to_numpy()
    Es_deno = meta_result.Es_denoised.to_numpy()
    Es_noisy = meta_result.Es_noisy.to_numpy()
    dura_deno = meta_result.duration_denoised.to_numpy()
    dura_noisy = meta_result.duration_noisy.to_numpy()
    falloff_deno = meta_result.falloff_denoised.to_numpy()
    falloff_noisy = meta_result.falloff_noisy.to_numpy()
    corner_freq_deno = meta_result.corner_freq_denoised.to_numpy()
    corner_freq_noisy = meta_result.corner_freq_noisy.to_numpy()
    ctr_speed_noisy = meta_result.centroid_speed_noisy.to_numpy()
    ctr_speed_deno = meta_result.centroid_speed_denoised.to_numpy()
    ctr_dir_noisy = meta_result.centroid_direction_noisy.to_numpy()
    ctr_dir_deno = meta_result.centroid_direction_denoised.to_numpy()
    depth = meta_result.source_depth_km.to_numpy().astype(np.float32)
    num_peak = meta_result.num_peaks.to_numpy()
    num_peak_dir = meta_result.num_peaks_dir.to_numpy()


    # %% Select events
    ind_select = np.where(np.logical_and(np.logical_and(nbin > 5, nsta > 20), depth > 99.0))[0]

    mb = mb[ind_select]
    evid = evid[ind_select]
    depth = depth[ind_select]
    Es_deno = Es_deno[ind_select]
    Es_noisy = Es_noisy[ind_select]
    dura_deno = dura_deno[ind_select]
    dura_noisy = dura_noisy[ind_select]
    falloff_deno = falloff_deno[ind_select]
    falloff_noisy = falloff_noisy[ind_select]
    ctr_dir_deno = ctr_dir_deno[ind_select]
    ctr_dir_noisy = ctr_dir_noisy[ind_select]
    ctr_speed_deno = ctr_speed_deno[ind_select]
    ctr_speed_noisy = ctr_speed_noisy[ind_select]
    corner_freq_deno = corner_freq_deno[ind_select]
    corner_freq_noisy = corner_freq_noisy[ind_select]
    num_peak = num_peak[ind_select]
    num_peak_dir = num_peak_dir[ind_select]

    # %% Convert mw to moment
    mw = np.exp(0.741 + 0.21 * mb) - 0.785

    for i in range(len(evid)):
        filename = "complete_mag_catalog/" + str(evid[i]) + ".xml"
        print(filename)
        cat = read_events(filename)

        for mag in cat[0].magnitudes:
            if mag.magnitude_type == "Mw" or mag.magnitude_type == "MW" or mag.magnitude_type == "Mww" or mag.magnitude_type == "MWW":
                mw[i] = mag.mag
                print("##### ", mag.magnitude_type, mw[i])
                break
    #####
    for i in range(len(evid)):
        if mb[i] > 6.9 and mw[i] > 7.5:
            print("Mb", mb[i], "Mw", mw[i], "dp", depth[i], "evid", evid[i])
    #####

    lg_moment = (mw + 6.07) * 1.5
    moment = 10 ** lg_moment
    lg_mg = np.arange(16, 22.1, 0.5, dtype=np.float64)

    # %% get 1D velocity profile
    vp = np.ones(len(depth), dtype=np.float64)
    vs = np.ones(len(depth), dtype=np.float64)
    den = np.ones(len(depth), dtype=np.float64)
    # vmod = 'PREM'
    vmod = 'AK135F'
    vmodel = vmod + '.txt'
    for i in range(len(depth)):
        vp[i], vs[i], den[i] = get_vp_vs(depth[i], vmodel=vmodel)

    ##################### Plot scaling relations ######################
    plt.close('all')
    cmap = matplotlib.cm.bwr.reversed()
    cmap1 = matplotlib.cm.hot.reversed()
    fig, ax = plt.subplots(12, 4, figsize=(50, 100), constrained_layout=True)

    v1 = 0
    v2 = 600
    v3 = 5.0
    v4 = 8.0

    ##################### Duration in time
    # %% duration--mw
    ind_m0_clean = \
    np.where(np.logical_and(10 ** (lg_moment / 4 - 3.55) > dura_deno, 10 ** (lg_moment / 4 - 4.45) < dura_deno))[0]
    ind_m0_noisy = \
    np.where(np.logical_and(10 ** (lg_moment / 4 - 3.55) > dura_noisy, 10 ** (lg_moment / 4 - 4.45) < dura_noisy))[0]
    ino = ax[0, 0].scatter(moment, dura_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ide = ax[0, 1].scatter(moment, dura_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    cbr = plt.colorbar(ide, ax=ax[0, 1])
    cbr.set_label('Depth (km)')

    # %% Fitting lines
    #####
    bins_edge = np.arange(16.9, 22.0, 0.4)
    bin_means, _, _ = binned_statistic(lg_moment[ind_m0_noisy], np.log10(dura_noisy[ind_m0_noisy]), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1] + 0.2), bin_means)
    lg_E = lg_mg * a - b
    print('T--M', a, b)
    ax[0, 0].scatter(10 ** (bins_edge[:-1] + 0.2), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None", edgecolors='g')
    ax[0, 0].plot(10 ** lg_mg, 10 ** lg_E, '--', color='g', linewidth=5, alpha=1.0, label=r'T~$M_0^{0.23}$')

    bin_means, _, _ = binned_statistic(lg_moment[ind_m0_clean], np.log10(dura_deno[ind_m0_clean]), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1] + 0.2), bin_means)
    lg_E = lg_mg * a - b
    print('T--M', a, b)
    ax[0, 1].scatter(10 ** (bins_edge[:-1] + 0.2), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None", edgecolors='g')
    ax[0, 1].plot(10 ** lg_mg, 10 ** lg_E, '--', color='g', linewidth=5, alpha=1.0, label=r'T~$M_0^\frac{1}{4}$')
    #####

    for k in range(2):
        lg_t = lg_mg / 4
        # ax[0, k].plot(10 ** lg_mg, 10 ** (lg_t - 4.0), '--g', linewidth=5, alpha=1, label=r'T~$M_0^\frac{1}{4}$')
        ax[0, k].plot(10 ** lg_mg, 10 ** (lg_t - 3.55), '--k', linewidth=1, alpha=1)
        ax[0, k].plot(10 ** lg_mg, 10 ** (lg_t - 4.45), '--k', linewidth=1, alpha=1)

        lg_t = lg_mg / 3
        ax[0, k].plot(10 ** lg_mg, 10 ** (lg_t - 5.5), '--k', linewidth=3, alpha=1, label=r'T~$M_0^\frac{1}{3}$')
        ax[0, k].legend(loc='lower right')

    # %% Ts--depth
    ts_clean = dura_deno[ind_m0_clean] * np.power((1e19 / moment[ind_m0_clean]), 0.25) * vs[ind_m0_clean] / 4.5
    ts_noisy = dura_noisy[ind_m0_noisy] * np.power((1e19 / moment[ind_m0_noisy]), 0.25) * vs[ind_m0_noisy] / 4.5
    dp_clean = depth[ind_m0_clean]
    dp_noisy = depth[ind_m0_noisy]
    ino = ax[0, 2].scatter(dp_noisy, ts_noisy, marker='o', s=200, c=mw[ind_m0_noisy], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ide = ax[0, 3].scatter(dp_clean, ts_clean, marker='o', s=200, c=mw[ind_m0_clean], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    cbr = plt.colorbar(ide, ax=ax[0, 3])
    cbr.set_label(r'$M_W$')

    bins_edge = np.arange(100, 800, 50)
    try:
        bin_means, _, _ = binned_statistic(dp_clean, ts_clean, statistic='mean', bins=bins_edge)
        bin_stdvs, _, _ = binned_statistic(dp_clean, ts_clean, statistic='std', bins=bins_edge)
        ax[0, 3].errorbar((bins_edge[:-1] + 25), bin_means, yerr=bin_stdvs, fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=2,capsize=15,capthick=2)

        bin_means, _, _ = binned_statistic(dp_noisy, ts_noisy, statistic='mean', bins=bins_edge)
        bin_stdvs, _, _ = binned_statistic(dp_noisy, ts_noisy, statistic='std', bins=bins_edge)
        ax[0, 2].errorbar((bins_edge[:-1] + 25), bin_means, yerr=bin_stdvs, fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=2,capsize=15,capthick=2)
    except:
        pass

    ##################### Rupture speed
    doppler_clean = (ctr_speed_deno * 1)
    doppler_noisy = (ctr_speed_noisy * 1)
    ax[3, 0].scatter(moment, doppler_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[3, 1].scatter(moment, doppler_clean, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[3, 2].scatter(depth, doppler_noisy, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[3, 3].scatter(depth, doppler_clean, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    ax[10, 0].scatter(moment, dura_noisy*corner_freq_noisy, marker='o', s=200,
                     c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[10, 1].scatter(moment, dura_deno *corner_freq_deno,  marker='o', s=200,
                     c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[10, 2].scatter( depth, dura_noisy*corner_freq_noisy, marker='o', s=200,
                        c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[10, 3].scatter( depth, dura_deno *corner_freq_deno,  marker='o', s=200,
                        c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    ax[11, 0].scatter(moment[ind_m0_clean], num_peak_dir[ind_m0_clean], marker='o', s=200, c=depth[ind_m0_clean], edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[11, 1].scatter(moment[ind_m0_clean], num_peak[ind_m0_clean], marker='o', s=200, c=depth[ind_m0_clean], edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[11, 2].scatter(depth[ind_m0_clean], num_peak_dir[ind_m0_clean], marker='o', s=200, c=mw[ind_m0_clean], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[11, 3].scatter(depth[ind_m0_clean], num_peak[ind_m0_clean], marker='o', s=200, c=mw[ind_m0_clean], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    # %% fc - T
    ax[2, 0].scatter(dura_noisy[ind_m0_noisy], 1 / corner_freq_noisy[ind_m0_noisy], marker='o', s=200,
                     c=depth[ind_m0_noisy], edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[2, 1].scatter(dura_deno[ind_m0_clean], 1 / corner_freq_deno[ind_m0_clean], marker='o', s=200,
                     c=depth[ind_m0_clean], edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[2, 2].scatter(dura_noisy[ind_m0_noisy], 1 / corner_freq_noisy[ind_m0_noisy], marker='o', s=200,
                     c=mw[ind_m0_noisy], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[2, 3].scatter(dura_deno[ind_m0_clean], 1 / corner_freq_deno[ind_m0_clean], marker='o', s=200, c=mw[ind_m0_clean],
                     edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[2, 0].plot([0, 20], [0, 20], linestyle='dashed', color='k', linewidth=3, alpha=1.0)
    ax[2, 1].plot([0, 20], [0, 20], linestyle='dashed', color='k', linewidth=3, alpha=1.0)
    ax[2, 2].plot([0, 20], [0, 20], linestyle='dashed', color='k', linewidth=3, alpha=1.0)
    ax[2, 3].plot([0, 20], [0, 20], linestyle='dashed', color='k', linewidth=3, alpha=1.0)

    # %% 1/fc--depth
    ind_m0_clean = np.where(np.logical_and(10**(lg_moment/4 - 3.6) > 1 / corner_freq_deno, 10**(lg_moment/4 - 4.4) < 1 / corner_freq_deno))[0]
    ind_m0_noisy = np.where(np.logical_and(10**(lg_moment/4 - 3.6) > 1 / corner_freq_noisy, 10**(lg_moment/4 - 4.4) < 1 / corner_freq_noisy))[0]

    ##################### Duration as 1/ corner frequency
    # %% corner frequency
    ax[1, 0].scatter(moment, 1 / corner_freq_noisy, marker='o', s=200, c=depth,
                     edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[1, 1].scatter(moment, 1 / corner_freq_deno, marker='o', s=200, c=depth,
                     edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)

    lg_t = lg_mg / 4
    # ax[1, 1].plot(10 ** lg_mg, 10 ** (lg_t - 4.0), '--g', linewidth=5, alpha=1, label=r'T~$M_0^\frac{1}{4}$')
    ax[1, 0].plot(10 ** lg_mg, 10 ** (lg_t - 3.6), '--k', linewidth=1, alpha=1)
    ax[1, 0].plot(10 ** lg_mg, 10 ** (lg_t - 4.4), '--k', linewidth=1, alpha=1)
    ax[1, 1].plot(10 ** lg_mg, 10 ** (lg_t - 3.6), '--k', linewidth=1, alpha=1)
    ax[1, 1].plot(10 ** lg_mg, 10 ** (lg_t - 4.4), '--k', linewidth=1, alpha=1)
    lg_t = lg_mg / 3
    ax[1, 0].plot(10 ** lg_mg, 10 ** (lg_t - 5.5), '--k', linewidth=3, alpha=1, label=r'$f_c$~$M_0^\frac{1}{3}$')
    ax[1, 1].plot(10 ** lg_mg, 10 ** (lg_t - 5.5), '--k', linewidth=3, alpha=1, label=r'$f_c$~$M_0^\frac{1}{3}$')

    #####
    bins_edge = np.arange(17.25, 21.7, 0.4)
    bin_means, _, _ = binned_statistic(lg_moment[ind_m0_noisy], np.log10(1 / corner_freq_noisy[ind_m0_noisy]), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1] + 0.2), bin_means)
    print('1/fc--M', a, b)
    lg_E = lg_mg * a - b
    ax[1, 0].scatter(10 ** (bins_edge[:-1] + 0.2), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None", edgecolors='g')
    ax[1, 0].plot(10 ** lg_mg, 10 ** lg_E, '--', color='g', linewidth=5, alpha=1.0, label=r'$f_c$~$M_0^{0.22}}$')
    ax[1, 0].legend(loc='lower right')

    bin_means, _, _ = binned_statistic(lg_moment[ind_m0_clean], np.log10(1 / corner_freq_deno[ind_m0_clean]), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1] + 0.2), bin_means)
    print('1/fc--M', a, b)
    lg_E = lg_mg * a - b
    ax[1, 1].scatter(10 ** (bins_edge[:-1] + 0.2), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None", edgecolors='g')
    ax[1, 1].plot(10 ** lg_mg, 10 ** lg_E, '--', color='g', linewidth=5, alpha=1.0, label=r'$f_c$~$M_0^{0.24}$')
    ax[1, 1].legend(loc='lower right')
    #####

    ts_clean = 1 / corner_freq_deno[ind_m0_clean] * np.power(10, (19-lg_moment[ind_m0_clean])/4) * vs[ind_m0_clean] / 4.5
    ts_noisy = 1 / corner_freq_noisy[ind_m0_noisy] * np.power(10, (19-lg_moment[ind_m0_noisy])/4) * vs[ind_m0_noisy] / 4.5
    dp_clean = depth[ind_m0_clean]
    dp_noisy = depth[ind_m0_noisy]
    ax[1, 2].scatter(dp_noisy, ts_noisy, marker='o', s=200, c=mw[ind_m0_noisy], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[1, 3].scatter(dp_clean, ts_clean, marker='o', s=200, c=mw[ind_m0_clean], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    bins_edge = np.arange(100, 800, 50)
    try:
        bin_means, _, _ = binned_statistic(dp_clean, ts_clean, statistic='mean', bins=bins_edge)
        bin_stdvs, _, _ = binned_statistic(dp_clean, ts_clean, statistic='std', bins=bins_edge)
        ax[1, 3].errorbar((bins_edge[:-1] + 25), bin_means, yerr=bin_stdvs, fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=2,capsize=15,capthick=2)

        bin_means, _, _ = binned_statistic(dp_noisy, ts_noisy, statistic='mean', bins=bins_edge)
        bin_stdvs, _, _ = binned_statistic(dp_noisy, ts_noisy, statistic='std', bins=bins_edge)
        ax[1, 2].errorbar((bins_edge[:-1] + 25), bin_means, yerr=bin_stdvs, fmt='o',ms=30,mec='g',mfc='g',ecolor='g',elinewidth=2,capsize=15,capthick=2)
    except:
        pass

    # %% stress drop
    rad_noisy = 0.32 * vs * 1000 / corner_freq_noisy
    rad_deno = 0.32 * vs * 1000 / corner_freq_deno
    sdrop_noisy = 7 * moment / (rad_noisy ** 3 * 16 * 1e6)
    sdrop_deno = 7 * moment / (rad_deno ** 3 * 16 * 1e6)
    ax[6, 0].scatter(moment, sdrop_noisy, marker='o', s=200, c=depth,
                     edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[6, 1].scatter(moment, sdrop_deno, marker='o', s=200, c=depth,
                     edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[6, 2].scatter(depth, sdrop_noisy, marker='o', s=200, c=mw,
                     edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[6, 3].scatter(depth, sdrop_deno, marker='o', s=200, c=mw,
                     edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    bins_edge = np.arange(17, 21.5, 0.5)
    bin_means, _, _ = binned_statistic(lg_moment, np.log10(sdrop_noisy), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1] + 0.25), bin_means)
    lg_E = bins_edge * a - b
    print('stress drop--M', a, b)
    ax[6, 0].scatter(10 ** (bins_edge[:-1] + 0.25), 10 ** bin_means, marker='s', s=500, linewidth=5,
                     facecolor="None", edgecolors='g')
    ax[6, 0].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0,
                  label=r'$\Delta\sigma$~$M_0^{0.37}$')
    ax[6, 0].legend(loc='lower right')

    bin_means, _, _ = binned_statistic(lg_moment, np.log10(sdrop_deno), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1] + 0.25), bin_means)
    lg_E = bins_edge * a - b
    print('stress drop--M', a, b)
    ax[6, 1].scatter(10 ** (bins_edge[:-1] + 0.25), 10 ** bin_means, marker='s', s=500, linewidth=5,
                     facecolor="None", edgecolors='g')
    ax[6, 1].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0,
                  label=r'$\Delta\sigma$~$M_0^{0.31}$')
    ax[6, 1].legend(loc='lower right')

    ##################### Radiated energy
    Es_noisy = Es_noisy * moment * moment * 5**5 * 3 / (vp**5 * den) * 1.0
    Es_deno = Es_deno * moment * moment * 5**5 * 3 / (vp**5 * den) * 1.0
    ax[4, 0].scatter(moment, Es_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[4, 1].scatter(moment, Es_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[4, 2].scatter(depth, Es_noisy, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[4, 3].scatter(depth, Es_deno, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    bins_edge = np.arange(17, 21.5, 0.5)

    bin_means, _, _ = binned_statistic(lg_moment, np.log10(Es_noisy), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1]+0.25), bin_means)
    lg_E = bins_edge * a - b
    print('E-M', a, b)
    ax[4, 0].scatter(10 ** (bins_edge[:-1] + 0.25), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None", edgecolors='g')
    ax[4, 0].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$E_R$~$M_0^{1.48}$')
    ax[4, 0].legend(loc='lower right')

    bin_means, _, _ = binned_statistic(lg_moment, np.log10(Es_deno), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1]+0.25), bin_means)
    lg_E = bins_edge * a - b
    print('E-M', a, b)
    ax[4, 1].scatter(10 ** (bins_edge[:-1] + 0.25), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None", edgecolors='g')
    ax[4, 1].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$E_R$~$M_0^{1.30}$')
    ax[4, 1].legend(loc='lower right')

    ##################### Apparent stress
    app_stress_noisy = Es_noisy * 1e3 * den * vs * vs / moment
    app_stress_deno = Es_deno * 1e3 * den * vs * vs / moment
    ax[5, 0].scatter(moment, app_stress_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[5, 1].scatter(moment, app_stress_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[5, 2].scatter(depth, app_stress_noisy, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[5, 3].scatter(depth, app_stress_deno, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    bins_edge = np.arange(17, 21.5, 0.5)
    bin_means, _, _ = binned_statistic(lg_moment, np.log10(app_stress_noisy), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1] + 0.25), bin_means)
    lg_E = bins_edge * a - b
    print('apparent stress--M', a, b)
    ax[5, 0].scatter(10 ** (bins_edge[:-1] + 0.25), 10 ** bin_means, marker='s', s=500, linewidth=5,
                     facecolor="None", edgecolors='g')
    ax[5, 0].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0,
                  label=r'$\sigma_a$~$M_0^{0.53}$')
    ax[5, 0].legend(loc='lower right')

    bin_means, _, _ = binned_statistic(lg_moment, np.log10(app_stress_deno), statistic='mean', bins=bins_edge)
    a, b = line_fit((bins_edge[:-1] + 0.25), bin_means)
    lg_E = bins_edge * a - b
    print('apparent stress--M', a, b)
    ax[5, 1].scatter(10 ** (bins_edge[:-1] + 0.25), 10 ** bin_means, marker='s', s=500, linewidth=5,
                     facecolor="None", edgecolors='g')
    ax[5, 1].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0,
                  label=r'$\sigma_a$~$M_0^{0.35}$')
    ax[5, 1].legend(loc='lower right')

    # %% radiation efficiency
    rat_noisy = app_stress_noisy * 2 / sdrop_noisy
    rat_deno = app_stress_deno * 2 / sdrop_deno
    ax[7, 0].scatter(moment, rat_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[7, 1].scatter(moment, rat_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[7, 2].scatter(depth, rat_noisy, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[7, 3].scatter(depth, rat_deno, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    bins_edge = np.arange(17, 21.5, 0.5)
    bin_means, _, _ = binned_statistic(lg_moment, np.log10(rat_noisy), statistic='mean', bins=bins_edge)
    ax[7, 0].scatter(10 ** (bins_edge[:-1] + 0.25), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None",
                     edgecolors='g')
    bin_means, _, _ = binned_statistic(lg_moment, np.log10(rat_deno), statistic='mean', bins=bins_edge)
    ax[7, 1].scatter(10 ** (bins_edge[:-1] + 0.25), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None",
                     edgecolors='g')

    bins_edge = np.arange(100, 800, 50)
    bin_means, _, _ = binned_statistic(depth, np.log10(rat_noisy), statistic='mean', bins=bins_edge)
    ax[7, 2].scatter((bins_edge[:-1] + 25), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None",
                     edgecolors='g')
    # a, b = line_fit((bins_edge[:-1] + 25)/100, bin_means)
    # lg_E = bins_edge/100 * a - b
    # print('Efficiency-Depth', a/100, b)
    # ax[8, 2].plot(bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$\eta = 10^{-(0.0001xH+1.4)}$')
    # ax[8, 2].legend(loc='lower right')

    bin_means, _, _ = binned_statistic(depth, np.log10(rat_deno), statistic='mean', bins=bins_edge)
    ax[7, 3].scatter((bins_edge[:-1] + 25), 10 ** bin_means, marker='s', s=500, linewidth=5, facecolor="None",
                     edgecolors='g')
    # a, b = line_fit((bins_edge[:-1] + 25)/100, bin_means)
    # lg_E = bins_edge/100 * a - b
    # print('Efficiency-Depth', a/100, b)
    # ax[8, 3].plot(bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$\eta = 10^{-(0.0002xH+1.5)}$')
    # ax[8, 3].legend(loc='lower right')

    # try:
    #     bin_means, _, _ = binned_statistic(depth, np.log10(rat_deno), statistic='mean',
    #                                        bins=[100, 200, 300, 400, 500, 600, 800])
    #     bin_stdvs, _, _ = binned_statistic(depth, np.log10(rat_deno), statistic='std',
    #                                        bins=[100, 200, 300, 400, 500, 600, 800])
    #     ax[8, 3].errorbar([150, 250, 350, 450, 550, 650], 10**bin_means, yerr=bin_stdvs, fmt='o', ms=30, mec='g', mfc='g',
    #                       ecolor='g', elinewidth=9, capsize=15, capthick=9)
    #
    #     bin_means, _, _ = binned_statistic(depth, np.log10(rat_noisy), statistic='mean',
    #                                        bins=[100, 200, 300, 400, 500, 600, 800])
    #     bin_stdvs, _, _ = binned_statistic(depth, np.log10(rat_noisy), statistic='std',
    #                                        bins=[100, 200, 300, 400, 500, 600, 800])
    #     ax[8, 2].errorbar([150, 250, 350, 450, 550, 650], 10**bin_means, yerr=bin_stdvs, fmt='o', ms=30, mec='g', mfc='g',
    #                       ecolor='g', elinewidth=9, capsize=15, capthick=9)
    # except:
    #     pass

    ##################### Fracture energy
    rad_noisy = 0.32 * vs * 1000 * dura_noisy
    rad_deno = 0.32 * vs * 1000 * dura_deno
    sdrop_noisy = 7 * moment / (rad_noisy ** 3 * 16 * 1e6)
    sdrop_deno = 7 * moment / (rad_deno ** 3 * 16 * 1e6)
    slip_noisy = moment / (1e9 * den * vs * vs * rad_noisy * rad_noisy * np.pi)
    slip_deno = moment / (1e9 * den * vs * vs * rad_deno * rad_deno * np.pi)
    E_frac_noisy = (sdrop_noisy * 1e6 / 2 - Es_noisy * 1e9 * den * vs * vs / moment) * slip_noisy
    E_frac_deno = (sdrop_deno * 1e6 / 2 - Es_deno * 1e9 * den * vs * vs / moment) * slip_deno
    s_h = 1e-3
    s_c = 1e-3
    ind_m0_clean = \
    np.where(np.logical_and(np.logical_and(10 ** (lg_moment / 4 - 3.55) > dura_deno, 10 ** (lg_moment / 4 - 4.45) < dura_deno), E_frac_deno>0))[0]
    ind_m0_noisy = \
    np.where(np.logical_and(np.logical_and(10 ** (lg_moment / 4 - 3.55) > dura_noisy, 10 ** (lg_moment / 4 - 4.45) < dura_noisy), E_frac_noisy>0))[0]
    ax[8, 0].scatter(slip_noisy[ind_m0_noisy], E_frac_noisy[ind_m0_noisy], marker='o', s=200, c=depth[ind_m0_noisy], edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[8, 1].scatter(slip_deno[ind_m0_clean], E_frac_deno[ind_m0_clean], marker='o', s=200, c=depth[ind_m0_clean], edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[8, 2].scatter(slip_noisy[ind_m0_noisy], E_frac_noisy[ind_m0_noisy], marker='o', s=200, c=mw[ind_m0_noisy], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[8, 3].scatter(slip_deno[ind_m0_clean], E_frac_deno[ind_m0_clean], marker='o', s=200, c=mw[ind_m0_clean], edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)

    bins_edge = np.arange(-2, 2, 0.1)
    a, b = line_fit(np.log10(slip_noisy[ind_m0_noisy]), np.log10(E_frac_noisy[ind_m0_noisy]))
    lg_E = bins_edge * a - b
    print('FracE-Slip', a, b)
    ax[8, 0].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$G$~$S^{2}$')
    ax[8, 2].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$G$~$S^{2}$')
    ax[8, 0].legend(loc='lower right')

    a, b = line_fit(np.log10(slip_deno[ind_m0_clean]), np.log10(E_frac_deno[ind_m0_clean]))
    lg_E = bins_edge * a - b
    print('FracE-Slip', a, b)
    ax[8, 1].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$G$~$S^{1.95}$')
    ax[8, 3].plot(10 ** bins_edge, 10 ** lg_E, '-', color='g', linewidth=5, alpha=1.0, label=r'$G$~$S^{1.95}$')
    ax[8, 1].legend(loc='lower right')

    ##################### Falloff power
    ax[9, 0].scatter(moment, falloff_noisy, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[9, 1].scatter(moment, falloff_deno, marker='o', s=200, c=depth, edgecolors='k', cmap=cmap, vmin=v1, vmax=v2)
    ax[9, 2].scatter(depth, falloff_noisy, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)
    ax[9, 3].scatter(depth, falloff_deno, marker='o', s=200, c=mw, edgecolors='k', cmap=cmap1, vmin=v3, vmax=v4)


    ####################################
    for i in [0, 1, 2, 3]:
        ax[2, i].set_xlim(0, 20)
        ax[2, i].set_ylim(0, 20)
        ax[3, i].set_ylim(-0.05, 0.6)
        ax[4, i].set_ylim(1e8, 1e18)
        ax[5, i].set_ylim(1e-3, 1e2)
        ax[6, i].set_ylim(1e-2, 1e3)
        ax[7, i].set_ylim(1e-2, 1e0)
        ax[8, i].set_ylim(1e2, 1e10)
        ax[8, i].set_xlim(1e-2, 3e1)
        ax[9, i].set_ylim(0, 3.5)
        ax[10, i].set_ylim(0.2, 5.2)
        ax[11, i].set_ylim(0.2, 5.2)

    for i in [0, 1]:
        for j in [0, 1]:
            ax[j, i].set_ylim(3e-1, 8e1)
        for j in [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]:
            ax[j, i].set_xlim(1e16, 1e22)
    for i in [2, 3]:
        for j in [0, 1]:
            ax[j, i].set_ylim(0, 20)
        for j in [0, 1, 3, 4, 5, 6, 7, 9]:
            ax[j, i].set_xlim(50, 750)

    for i in [0, 1]:
        for j in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
            ax[j, i].set_xscale("log")
        for j in [0, 1, 4, 5, 6, 7, 8]:
            ax[j, i].set_yscale("log")
    for i in [2, 3]:
        for j in [4, 5, 6, 7, 8]:
            ax[j, i].set_yscale("log")
        ax[8, i].set_xscale("log")

    for i in range(12):
        for j in range(4):
            ax[i, j].grid(which='major', color='#DDDDDD', linewidth=1)
            ax[i, j].grid(which='minor', color='#EEEEEE', linestyle='--', linewidth=1)
            ax[i, j].minorticks_on()
    for i in [0, 1]:
        for j in [1, 7, 9]:
            ax[j, i].set_xlabel('Moment (Nm)', fontsize=40)
        ax[2, i].set_xlabel('Duration (s)', fontsize=40)
        ax[8, i].set_xlabel('Slip (m)', fontsize=40)
        ax[11, i].set_xlabel('Moment (Nm)', fontsize=40)
    for i in [2, 3]:
        ax[2, i].set_xlabel('Duration (s)', fontsize=40)
        ax[7, i].set_xlabel('Depth (km)', fontsize=40)
        ax[9, i].set_xlabel('Depth (km)', fontsize=40)
        ax[11, i].set_xlabel('Depth (km)', fontsize=40)

    ax[0, 0].set_ylabel('Duration (s)', fontsize=40)
    ax[1, 0].set_ylabel(r'1/$f_c$ ($Hz^{-1}$)', fontsize=40)
    ax[2, 0].set_ylabel(r'1/$f_c$ ($Hz^{-1}$)', fontsize=40)
    ax[3, 0].set_ylabel('Doppler strength', fontsize=40)
    ax[4, 0].set_ylabel('Radiated energy', fontsize=40)
    ax[5, 0].set_ylabel('Apparent stress', fontsize=40)
    ax[6, 0].set_ylabel(r'Stress drop (MPa)', fontsize=40)
    ax[7, 0].set_ylabel(r'Radiation efficiency', fontsize=40)
    ax[8, 0].set_ylabel(r'Fracture energy ($Jm^{-2}$)', fontsize=40)
    ax[9, 0].set_ylabel('Falloff power', fontsize=40)
    ax[10, 0].set_ylabel('T $f_c$', fontsize=40)
    ax[11, 0].set_ylabel('# peaks', fontsize=40)
    ax[0, 2].set_ylabel('Scaled duration', fontsize=40)
    ax[1, 2].set_ylabel('Scaled duration', fontsize=40)

    ax[0, 0].set_title('noisy data', fontsize=40)
    ax[0, 1].set_title('denoised data', fontsize=40)
    ax[0, 2].set_title('noisy data', fontsize=40)
    ax[0, 3].set_title('denoised data', fontsize=40)
    ax[11, 0].set_title('Stack', fontsize=40)
    ax[11, 1].set_title('Stack and stretch', fontsize=40)
    ax[11, 2].set_title('Stack', fontsize=40)
    ax[11, 3].set_title('Stack and stretch', fontsize=40)

    # plt.savefig(fig_dir + '/scaling_deepfocus.pdf')
    # plt.savefig(fig_dir + '/scaling_intermediate.pdf')
    plt.savefig(fig_dir + '/scaling' + vmod + '.pdf')

    # %% Vr/Vp histograms
    doppler_clean = doppler_clean[ind_m0_clean]
    num_peak_dir = num_peak_dir[ind_m0_clean]
    num_peak = num_peak[ind_m0_clean]
    depth = depth[ind_m0_clean]
    mw = mw[ind_m0_clean]

    plt.close('all')
    fig, ax = plt.subplots(2, 3, figsize=(30, 30), constrained_layout=True)
    bins = np.linspace(0, 1, 20)
    ax[0, 0].hist(doppler_clean, bins=bins, density=True, histtype='stepfilled',
                  color='0.7', alpha=1, label='100-700 km', lw=2)
    ax[0, 0].hist(doppler_clean[depth > 300.0], bins=bins, density=True, histtype='step',
               color='b', alpha=1, label='300-700 km', lw=3)
    ax[0, 0].hist(doppler_clean[depth < 300.0], bins=bins, density=True, histtype='step',
               color='r', alpha=1, label='100-300 km', lw=3)
    ax[0, 0].set_xlabel('Doppler strength', fontsize=44)
    ax[0, 0].set_ylabel('density', fontsize=40)
    ax[0, 0].set_ylim(0, 7)
    ax[0, 0].set_xlim(-0.05, 0.6)
    ax[0, 0].set_title('Doppler effect', fontsize=40)
    ax[0, 0].legend(loc='upper right')

    ax[1, 0].hist(doppler_clean, bins=bins, density=True, histtype='stepfilled',
                  color='0.9', alpha=1, label='M5.5-8.5', lw=2)
    ax[1, 0].hist(doppler_clean[mw <= 6.0], bins=bins, density=True, histtype='step',
                  color='k', alpha=1, label='M5.5-6.0', lw=2)
    ax[1, 0].hist(doppler_clean[np.where(np.logical_and(mw <= 7.0, mw > 6.0))[0]], bins=bins, density=True, histtype='step',
                  color='orange', alpha=1, label='M6.0-7.0', lw=2)
    ax[1, 0].hist(doppler_clean[mw > 7.0], bins=bins, density=True, histtype='step',
                  color='purple', alpha=1, label='M7.0-8.5', lw=2)
    ax[1, 0].set_xlabel('Doppler strength', fontsize=44)
    ax[1, 0].set_ylabel('density', fontsize=40)
    ax[1, 0].set_ylim(0, 8)
    ax[1, 0].set_xlim(-0.05, 0.6)
    ax[1, 0].legend(loc='upper right')



    bins = np.linspace(0.5, 4.5, 5)
    ax[0, 1].hist(num_peak, bins=bins, density=True, histtype='stepfilled',
               color='0.7', alpha=1, label='100-700 km', lw=2)
    ax[0, 1].hist(num_peak[depth > 300.0], bins=bins, density=True, histtype='step',
               color='b', alpha=1, label='300-700 km', lw=2)
    ax[0, 1].hist(num_peak[depth < 300.0], bins=bins, density=True, histtype='step',
               color='r', alpha=1, label='100-300 km', lw=2)
    ax[0, 1].set_xlabel('# peaks', fontsize=44)
    ax[0, 1].set_ylabel('density', fontsize=40)
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_xlim(0.2, 5.2)
    ax[0, 1].set_title('Sub-event complexity (stack stretched P)', fontsize=40)

    ax[0, 2].hist(num_peak_dir, bins=bins, density=True, histtype='stepfilled',
               color='0.7', alpha=1, label='100-700 km', lw=2)
    ax[0, 2].hist(num_peak_dir[depth > 300.0], bins=bins, density=True, histtype='step',
               color='b', alpha=1, label='300-700 km', lw=2)
    ax[0, 2].hist(num_peak_dir[depth < 300.0], bins=bins, density=True, histtype='step',
               color='r', alpha=1, label='100-300 km', lw=2)
    ax[0, 2].set_xlabel('# peaks', fontsize=44)
    ax[0, 2].set_ylabel('density', fontsize=40)
    ax[0, 2].set_ylim(0, 1)
    ax[0, 2].set_xlim(0.2, 5.2)
    ax[0, 2].set_title('Sub-event complexity (stack P)', fontsize=40)

    ax[1, 1].hist(num_peak, bins=bins, density=True, histtype='stepfilled',
                  color='0.9', alpha=1, label='M5.5-8.5', lw=2)
    ax[1, 1].hist(num_peak[mw <= 6.0], bins=bins, density=True, histtype='step',
                  color='k', alpha=1, label='M5.5-6.0', lw=2)
    ax[1, 1].hist(num_peak[np.where(np.logical_and(mw <= 7.0, mw > 6.0))[0]], bins=bins, density=True, histtype='step',
                  color='orange', alpha=1, label='M6.0-7.0', lw=2)
    ax[1, 1].hist(num_peak[mw > 7.0], bins=bins, density=True, histtype='step',
                  color='purple', alpha=1, label='M7.0-8.5', lw=2)
    ax[1, 1].set_xlabel('# peaks', fontsize=44)
    ax[1, 1].set_ylabel('density', fontsize=40)
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_xlim(0.2, 5.2)

    ax[1, 2].hist(num_peak_dir, bins=bins, density=True, histtype='stepfilled',
                  color='0.9', alpha=1, label='M5.5-8.5', lw=2)
    ax[1, 2].hist(num_peak_dir[mw <= 6.0], bins=bins, density=True, histtype='step',
                  color='k', alpha=1, label='M5.5-6.0', lw=2)
    ax[1, 2].hist(num_peak_dir[np.where(np.logical_and(mw <= 7.0, mw > 6.0))[0]], bins=bins, density=True, histtype='step',
                  color='orange', alpha=1, label='M6.0-7.0', lw=2)
    ax[1, 2].hist(num_peak_dir[mw > 7.0], bins=bins, density=True, histtype='step',
                  color='purple', alpha=1, label='M7.0-8.5', lw=2)
    ax[1, 2].set_xlabel('# peaks', fontsize=44)
    ax[1, 2].set_ylabel('density', fontsize=40)
    ax[1, 2].set_ylim(0, 1)
    ax[1, 2].set_xlim(0.2, 5.2)





    plt.savefig(fig_dir + '/histograms.pdf')


# %% grid search the best a and b
def line_fit(x, y, da=0.01, db=0.05):
    l2_min = 10000.0
    a_best = 2.00
    b_best = 5.00
    for a in np.arange(-1.0, 2.5, da, dtype=np.float64):
        for b in np.arange(-20, 30, db, dtype=np.float64):
            line_func = x * a - b
            l2 = np.sum(np.square(y - line_func))
            if l2 < l2_min:
                l2_min = l2
                a_best = a
                b_best = b

    return a_best, b_best


if __name__ == '__main__':
    main()


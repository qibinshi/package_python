"""
Prepare noisy teleseismic data

@author: Qibin Shi (qibins@uw.edu)
"""
import glob
import time
import h5py
import argparse
import numpy as np
import pandas as pd
from distaz import DistAz
from functools import partial
from multiprocessing import Pool
from obspy.taup import TauPyModel
from obspy import read_events, read_inventory, read, UTCDateTime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--phase', default='P', type=str, help='earthquake phase')
    parser.add_argument('-r', '--maxsnr', default=10000, type=int, help='upper-bound signal-noise ratio')
    parser.add_argument('-n', '--threads', default=24, type=int, help='number of processes')
    args = parser.parse_args()
    # %% Directories of raw and reformatted data
    # workdir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M5.5-6.0/2000_2021/'
    # datadir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M5.5-6.0/matfiles_for_denoiser/'
    workdir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M6.0plus/2000-2021/'
    datadir = '/data/whd01/qibin_data/raw_data_for_DenoTe/M6.0plus/matfiles_for_denoiser/'

    # %% Quake catalog from xml files
    cat = read_events(workdir + "*.xml")
    print(len(cat), "events in total")

    # %% Recording length
    if args.phase == 'P':
        halftime = 150.0
    else:
        halftime = 250.0

    samplerate = 10
    npts = int(halftime*2*samplerate)
    all_quake = np.zeros((0, npts, 3), dtype=np.double)

    # %% Multi-processing for multi-events
    since = time.time()
    partial_func = partial(one_event,
                           directory=workdir,
                           halftime=halftime,
                           maxsnr=args.maxsnr,
                           mindep=100,
                           phase=args.phase)

    num_proc = args.threads  # num_proc = os.cpu_count()
    with Pool(processes=num_proc) as pool:
        print("Number of multi-processing threads: ", num_proc)
        result = pool.map(partial_func, cat)
    print("All are processed. Time elapsed: %.2f s" % (time.time() - since))

    # %% Merge from threads
    meta = pd.DataFrame(columns=[
        "source_id",
        "source_origin_time",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "source_magnitude",
        "source_strike",
        "source_dip",
        "source_rake",
        "station_network_code",
        "station_code",
        "station_location_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "trace_snr_db",
        "trace_mean_0",
        "trace_stdv_0",
        "trace_mean_1",
        "trace_stdv_1",
        "trace_mean_2",
        "trace_stdv_2",
        "distance",
        "takeoff_p",
        "takeoff_phase",
        "azimuth"])

    for i in range(len(cat)):
        all_quake = np.append(all_quake, result[i][0], axis=0)
        print(i, 'th quake added')
        meta = pd.concat([meta, result[i][1]], ignore_index=True)
        print(i, 'th metadata added')
        print('------------')

    with h5py.File(datadir + 'M6_deep100km_allSNR_' + args.phase + '.hdf5', 'w') as f:
        f.create_dataset("quake", data=all_quake)

    meta.to_csv(datadir + 'M6_deep100km_allSNR_' + args.phase + '.csv', sep=',', index=False)
    print("Total traces of data:", all_quake.shape[0])
    print("All is saved! Time elapsed: %.2f s" % (time.time() - since))


def one_event(ev, directory=None, halftime=None, freq=4, rate=10, mindep=100, phase='P', maxsnr=100000):
    """Default 10Hz sampling rate
        Raw data is 1 hour before and after the origin time."""

    if phase == 'P':
        cmp = 2  # Z
    elif phase == 'S':
        cmp = 0  # T (after rotation)
    else:
        cmp = 1

    npts = int(rate * halftime * 2)

    all_quake = np.zeros((0, npts, 3), dtype=np.double)
    one_quake = np.zeros((npts, 3), dtype=np.double)

    evnm = str(ev.resource_id)[13:]
    evlo = ev.origins[0].longitude
    evla = ev.origins[0].latitude
    evdp = ev.origins[0].depth / 1000.0
    org_t = ev.origins[0].time
    evmg = ev.magnitudes[0].mag
    strike = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.strike
    dip = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.dip
    rake = ev.focal_mechanisms[0].nodal_planes.nodal_plane_1.rake
    # pre_filt = (0.004, 0.005, 10.0, 12.0)

    # format the metadata
    meta = pd.DataFrame(columns=[
        "source_id",
        "source_origin_time",
        "source_latitude_deg",
        "source_longitude_deg",
        "source_depth_km",
        "source_magnitude",
        "source_strike",
        "source_dip",
        "source_rake",
        "station_network_code",
        "station_code",
        "station_location_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "trace_snr_db",
        "trace_mean_0",
        "trace_stdv_0",
        "trace_mean_1",
        "trace_stdv_1",
        "trace_mean_2",
        "trace_stdv_2",
        "distance",
        "takeoff_phase",
        "azimuth"])

    if evdp > mindep:
        # %% Loop over stations
        for sta in glob.glob(directory + evnm + 'stas/*xml'):
            inv = read_inventory(sta)
            stnw = inv[0].code
            stco = inv[0][0].code
            stla = inv[0][0].latitude
            stlo = inv[0][0].longitude
            stlc = inv[0][0][0].location_code
            result = DistAz(stla, stlo, evla, evlo)
            distdeg = result.getDelta()
            azimuth = result.getAz()
            backazi = result.getBaz()

            try:
                st0 = read(directory + evnm + "waves/" + stnw + "." + stco + "." + stlc + ".?H?_*")
                st = st0.copy()
                # st.remove_response(inventory=inv, output='VEL', pre_filt=pre_filt)
                st.filter("lowpass", freq=freq)
            except:
                continue

            st.resample(rate)
            st.merge(fill_value=np.nan)
            model = TauPyModel(model="iasp91")
            arrivals = model.get_travel_times(source_depth_in_km=evdp, distance_in_degree=distdeg, phase_list=[phase])
            tphase = UTCDateTime(org_t + arrivals[0].time)
            takeoff_phase = arrivals[0].takeoff_angle
            st.trim(tphase - halftime, tphase + halftime)

            if len(st) >= 3 and len(st[0].data) >= npts and len(st[1].data) >= npts and len(st[2].data) >= npts:
                if phase != 'P':
                    st.rotate(method="NE->RT", back_azimuth=backazi)
                noise_amp = np.std(np.array(st[cmp].data)[int(rate*halftime-600): int(rate*halftime-100)])
                quake_amp = np.std(np.array(st[cmp].data)[int(rate*halftime): int(rate*halftime+500)])

                if quake_amp < (noise_amp * maxsnr):
                    for i in range(3):
                        one_quake[:, i] = np.array(st[i].data)[0:npts]

                    one_quake[np.isnan(one_quake)] = 0
                    scale_mean = np.mean(one_quake, axis=0, keepdims=True)
                    scale_stdv = np.std(one_quake, axis=0, keepdims=True) + 1e-12
                    one_quake = (one_quake - scale_mean) / scale_stdv
                    all_quake = np.append(all_quake, one_quake[np.newaxis, :, :], axis=0)

                    # %% Store the metadata of event-station
                    meta = pd.concat([meta, pd.DataFrame(data={
                        "source_id": evnm,
                        "source_origin_time": org_t,
                        "source_latitude_deg": "%.3f" % evla,
                        "source_longitude_deg": "%.3f" % evlo,
                        "source_depth_km": "%.3f" % evdp,
                        "source_magnitude": evmg,
                        "source_strike": strike,
                        "source_dip": dip,
                        "source_rake": rake,
                        "station_network_code": stnw,
                        "station_code": stco,
                        "station_location_code": stlc,
                        "station_latitude_deg": stla,
                        "station_longitude_deg": stlo,
                        "trace_snr_db": "%.3f" % (quake_amp / (noise_amp + 1e-12)),
                        "trace_mean_0": scale_mean[0, 0],
                        "trace_stdv_0": scale_stdv[0, 0],
                        "trace_mean_1": scale_mean[0, 1],
                        "trace_stdv_1": scale_stdv[0, 1],
                        "trace_mean_2": scale_mean[0, 2],
                        "trace_stdv_2": scale_stdv[0, 2],
                        "distance": distdeg,
                        "takeoff_phase": takeoff_phase,
                        "azimuth": azimuth}, index=[0])], ignore_index=True)

    return all_quake, meta


if __name__ == '__main__':
    main()

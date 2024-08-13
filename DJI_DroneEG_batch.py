#
# DJI_DroneEG_batch.py -- Batch process DJI drone / Eye-Gaze pairing tests, generating keys over a range of segment
#                       sizes.
# Description -- Batch processing of DJI Mini 3 Pro drone time series and HoloLens eye-gaze time series from
#               drone / eye-gaze pairing tests. For each pairing test, the function creates an object of DroneEyePair
#               class that contains the drone time series and eye-gaze time series.
#
# by Christopher Abel
# Revision History
# ----------------
# 08/07/2024 -- Original. Copied and modified from DroneEG_batch.py, which performed the same function for
#               CoDrone EDU / HoloLens eye-pairing tests.
#
# -------------------------------------------------
import csv
import math

import DroneEyePair
import matplotlib.pyplot as plt
import matplotlib as mpl
import EGCustomPlots as EGplt
import EyeMatchUtils as EGutil


def main():
    eg_file_path = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Data Sets\\DJI Drone\\'
                    'DroneTracker3\\080624\\')
    drone_file_path = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Data Sets\\DJI Drone\\'
                       'Drone Flight Path\\080624\\')
    eg_file_list = ['08062024_134813.csv']
    drone_file_list = ['08062024_134824.csv']
    write_keygen_summ_csv = True

    #
    # Write headers for keygen summary .csv file
    #
    if write_keygen_summ_csv:
        file_summ_out = open('keygen_summary.csv', 'w', newline='')
        writer_summ_out = csv.writer(file_summ_out)
        writer_summ_out.writerow(['Test', 'tquant_step (s)', 'Total Bits', 'Total Bit Match (%)',
                                  'Test Time (s)', 'Num Keys Gen', 'KGR (keys/s)', 'Avg. Entropy Drone Keys (bits)',
                                  'Avg. Entropy EG Keys (bits)', 'Avg. Key Bit Match (%)', 'Eye-Gaze Outliers Removed',
                                  'Lowpass Filtering'])

    for test_num in range(0, len(eg_file_list)):
        eg_file_name = 'eye_tracker_' + eg_file_list[test_num]
        drone_file_name = 'randx_patt_' + drone_file_list[test_num]
        print('Test ', test_num + 1)
        print('======================================')
        print('\tDrone data file:', drone_file_path + drone_file_name)
        print('\tEye-gaze data file:', eg_file_path + eg_file_name)
        eg_start = 3.4
        eg_end = 165.0
        tquant_start = 12.0
        tquant_end = 160.0
        nbits_quant = 2
        nkey_bits = 128
        max_bit_correction = 0.11
        #
        # Signal-Conditioning Settings
        #
        rem_outliers = True
        interp='Akima'
        lowpass = True
        quant_abs = False

        for tquant_step in [0.4, 0.6, 0.8, 1.0, 1.4, 2.0]:
        # for tquant_step in [1.0]:
            test1 = DroneEyePair.DroneEyePair(drone_file_path + drone_file_name, eg_file_path + eg_file_name, eg_start,
                                              eg_end, tquant_start, tquant_end, tquant_step, rem_outliers=rem_outliers,
                                              interp=interp, lowpass=lowpass, nbits_per_seg=nbits_quant,
                                              quant_abs_val=quant_abs, save_ext=False)

            # Calculate total number of bits and matching rate for all bits generated in this test, with the current
            # value of tquant_step.
            [nbits_total, e1, e2, mism, total_match_fract] = EGutil.key_stats(test1.key_drone, test1.key_eg)

            # Split Quantized key strings into individual keys of length nkey_bits.
            num_keys_gen = 0
            ek_dr_avg = 0
            ek_eg_avg = 0
            match_fract_avg = 0
            for i in range(0, math.floor(min(len(test1.key_drone), len(test1.key_eg)) / nkey_bits)):
                key_dr_current = test1.key_drone[nkey_bits * i: nkey_bits * i + (nkey_bits - 1)]
                key_eg_current = test1.key_eg[nkey_bits * i: nkey_bits * i + (nkey_bits - 1)]
                [nbits, ek_dr, ek_eg, mism, match_fract] = EGutil.key_stats(key_dr_current, key_eg_current)
                if match_fract >= 1.0 - max_bit_correction:
                    num_keys_gen += 1
                    ek_dr_avg += ek_dr
                    ek_eg_avg += ek_eg
                    match_fract_avg += match_fract
            ek_dr_avg /= max(num_keys_gen, 1)   # prevent divide-by-zero
            ek_eg_avg /= max(num_keys_gen, 1)   # prevent divide-by-zero
            match_fract_avg /= max(num_keys_gen, 1)     # prevent divide-by-zero

            # Print results of current tquant_step value to screen
            print('\ttquant_step = {0:.4f}s'.format(tquant_step))
            print('\t------------------------')
            print('\t\tDrone key string    = {0}'.format(test1.key_drone))
            print('\t\tEye-gaze key string = {0}'.format(test1.key_eg))
            time_key_gen = min(test1.flight.t_interp[-1], test1.eyegaze.eg_time[-1] + test1.t_shift) - tquant_start
            print('\t\tKey Generation Time = {0:.2f}s'.format(time_key_gen))
            print('\t\tTotal key bits generated = {0} bits'.format(nbits_total))
            print('\t\tTotal Percentage of bit matching = {0:.3f}%'.format(100 * total_match_fract))
            print('\t\tTotal number of keys generated = ', num_keys_gen)
            print('\t\tKey Generation Rate = {0:.4f} keys/s'.format(num_keys_gen / time_key_gen))
            print('\t\tAverage Entropy for keys generated: Drone = {0:.3f} bits\t Eye-Gaze = {1:.3f} bits'.
                  format(ek_dr_avg, ek_eg_avg))
            print('\t\tAverage matching percentage for keys generated = {0:.2f}%'.format(100.0 * match_fract_avg))

            # Print results of current tquant step to summary .csv file
            rem_outliers_val = 1 if rem_outliers else 0
            lowpass_val = 1 if lowpass else 0
            if write_keygen_summ_csv:
                writer_summ_out.writerow([test_num + 1, tquant_step, nbits_total, 100 * total_match_fract, time_key_gen,
                                          num_keys_gen, num_keys_gen / time_key_gen, ek_dr_avg, ek_eg_avg,
                                          100 * match_fract_avg, rem_outliers_val, lowpass_val])

    #
    # Close open output summary file
    #
    if write_keygen_summ_csv:
        file_summ_out.close()


# Main code
if __name__ == '__main__':
    main()

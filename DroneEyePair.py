"""
    DroneEyePair -- Class that generates key bit sequences from drone random flight path and eye-gaze path.
                This is targeted for use with a DJI Mini 3 Pro drone.

    Description -- Python class that segments, quantizes, and generates key bit sequences for two time series:
            DJI Mini 3 Pro drone flight path and a corresponding HoloLens eye-gaze path.

    by Christopher Abel
    Revision History
    ----------------
    07/29/2024 -- Original
    08/05/2024 -- Initial version that generates keys, and calculates bit matching between drone and eye-gaze time
                series, for a random-X pattern.
    08/13/2024 -- Modified normalization of eye-gaze pattern to use calc_norm_factor2 function from EyeMatchUtils.
                Plot drone - eye-gaze position error and calculate error standard deviation, and segment and segment
                error standard deviations.  Added capability for constructor to invert eye-gaze hitpoint X values
                to account for case of drone facing observer.
    09/01/2024 -- Modified align_drone_eg() to ensure that the entire calibration sequence is captured when searching
                for the times at which the first minimum and first maximum positions occur.  Added normalization of
                Y time series data and plotting of Y as well as X.
"""
import csv
import math

from matplotlib import pyplot as plt
import numpy as np
import scipy
import EGCustomPlots as EGplt
import EyeMatchUtils as EGutil
import DJIDronePath as DJId
import HoloEGPath as Holo


class DroneEyePair:
    # Default constructor
    def __init__(self, drone_file, eg_file, eg_start, eg_end, tquant_start, tquant_end, tquant_step, eg_inv=False, rem_outliers=True, interp='Akima', lowpass=True, nbits_per_seg=2, quant_abs_val=True, save_ext=False):
        #
        # Read in drone flight path and HoloLens eye-gaze data into a flight object and an eyegaze object.
        #
        #   The flight object includes removal of consecutive duplicate position points, and conversion of latitude,
        #   longitude, and height to drone x, y, and z time series, with interpolation to a constant 200ms timestep.
        #
        #   The eyegaze object includes optional removal of eye blinks, interpolation to a constant timestep, and
        #   application of a smoothing filter.
        #
        self.flight = DJId.DJIDronePath(drone_file)
        self.eyegaze = Holo.HoloEGPath(eg_file, eg_start, eg_end, rem_outliers=rem_outliers, interp=interp, tstep=0.02, lowpass=lowpass)
        #
        # If eg_inv = True, invert the x values of eye-gaze hitpoint.  This should be the case if the drone is
        # facing towards the observer, rather than away from the observer, during the test.
        #
        if eg_inv:
            self.eyegaze.hitpoint[:, 0] = -self.eyegaze.hitpoint[:, 0]

        # Find time indices at which to align drone and eye-gaze at start of drone calibration pattern
        alignment_indices = self.align_drone_eg(0.5, 10)
        self.drone_align_st, self.drone_align_end, self.eg_align_st, self.eg_align_end = alignment_indices
        # print(f'Drone align at t = {self.flight.t_interp[self.drone_align_st]}'
        #       f'\tEye-Gaze align at t= {self.eyegaze.eg_time[self.eg_align_st]}')

        #
        # Calculate alignment parameters:
        #   self.t_shift = time difference between the drone maximum X value and eye-gaze hitpoint maximum X value
        #       during calibration.  This value is used to align the normalized drone and eye-gaze time series.
        #   self.drone_start_pos = [x, y, z] values corresponding to center position of drone locus.
        #   self.eg_xyz_start = [x, y, z] values corresponding to center position of eye-gaze hitpoint locus.
        #
        t_min_dr, x_min_dr, t_max_dr, x_max_dr, t_max_vel_dr = DroneEyePair.find_alignment_param(self.flight.t_interp, self.flight.x, self.drone_align_st, self.drone_align_end, 1)
        t_min_eg, x_min_eg, t_max_eg, x_max_eg, t_max_vel_eg = DroneEyePair.find_alignment_param(self.eyegaze.eg_time, self.eyegaze.hitpoint[:, 0], self.eg_align_st, self.eg_align_end, 10)
        self.t_shift = t_max_dr - t_max_eg
        self.drone_start_pos = self.find_locus_center()[1]
        self.eg_xyz_start = self.find_locus_center()[0]

        #
        # Normalize drone and eye-gaze paths
        #
        self.dr_norm_x = self.flight.x
        m_normx, b_normx = EGutil.calc_norm_factor2(self.flight.t_interp, self.flight.x, self.eyegaze.eg_time
                                                    + self.t_shift, self.eyegaze.hitpoint[:, 0],
                                                    0, 160, alpha=0.02)
        self.eg_norm_x = m_normx * self.eyegaze.hitpoint[:, 0] + b_normx
        self.dr_norm_y = self.flight.y
        m_normy, b_normy = EGutil.calc_norm_factor2(self.flight.t_interp, self.flight.y, self.eyegaze.eg_time
                                                    + self.t_shift, self.eyegaze.hitpoint[:, 1],
                                                    0, 160, alpha=0.02)
        self.eg_norm_y = m_normy * self.eyegaze.hitpoint[:, 1] + b_normy

        #
        # Calculate position error (drone(t) - eye-gaze-hitpoint(t)) vs. time
        #
        self.pos_err_stdev, self.pos_err = EGutil.calc_error_wvf(self.flight.t_interp, self.dr_norm_x,
                                                       self.eyegaze.eg_time + self.t_shift, self.eg_norm_x)

        #
        # Segment and quantize drone and eye-gaze time series (x(t), y(t)). Calculate standard dev. of drone segments
        # and of segment error (drone_seg - eye-gaze_seg)
        #
        self.key_gen(nbits_per_seg, tquant_start, tquant_end, tquant_step, quant_abs_val)
        self.drone_seg_stdev = np.std(np.array(self.drone_segs, np.float64), ddof=1)
        self.seg_err_stdev = np.std(np.array(self.drone_segs, np.float64) - np.array(self.eg_segs, np.float64), ddof=1)

        #
        # Write selected outputs to .csv files for convenient analysis in other tools.
        #
        if save_ext:
            with open('eg_data.csv', 'w', newline='') as eg_out_file:
                print('\tWriting eye-gaze data to eg_data.csv...')
                eg_writer = csv.writer(eg_out_file)
                for i in range(0, len(self.eyegaze.eg_time)):
                    eg_writer.writerow([self.eyegaze.eg_time[i], self.eyegaze.hitpoint[i, 0], self.eg_norm_x[i], self.eyegaze.hitpoint[i, 1]])
            with open('drone_data.csv', 'w', newline='') as drone_out_file:
                print('\tWriting drone data to drone_data.csv...')
                drone_writer = csv.writer(drone_out_file)
                for i in range(0, len(self.flight.t_interp)):
                    drone_writer.writerow([self.flight.t_interp[i], self.flight.x[i], self.dr_norm_x[i], self.flight.y[i]])
            with open('eg_seg_data.csv', 'w', newline='') as eg_out_file:
                print('\tWriting eye-gaze segments to eg_seg_data.csv...')
                eg_writer = csv.writer(eg_out_file)
                for i in range(0, len(self.eg_segs)):
                    eg_writer.writerow([i, self.eg_segs[i], self.eg_seg_indicies[i][0], self.eg_seg_indicies[i][1]])
            with open('drone_seg_data.csv', 'w', newline='') as drone_out_file:
                print('\tWriting drone segments to drone_seg_data.csv...')
                drone_writer = csv.writer(drone_out_file)
                for i in range(0, len(self.drone_segs)):
                    drone_writer.writerow([i, self.drone_segs[i], self.drone_seg_indices[i][0], self.drone_seg_indices[i][1]])

    def align_drone_eg(self, fract_start=0.1, cal_phase_dur=10.0):
        # Method that finds an indices in the time vectors for the drone pattern and eye-gaze
        # that represent the times at which the drone is beginning to move left in
        # the calibration section of the pattern.
        #
        # The method finds the first index at which the x velocity of the drone and x velocity
        # of eye-gaze hitpoint both first reach fract_start * peak negative velocity in first
        # cal_phase_dur seconds of the drone flight path.
        #
        # It returns the indexes of the time arrays for both drone flight and eye-gaze at
        # the start of calibration and end of calibration.
        #
        drone_x_vel = EGutil.calc_vel(self.flight.t_interp, self.flight.x, 1)
        eg_x_vel = EGutil.calc_vel(self.eyegaze.eg_time, self.eyegaze.hitpoint[:, 0], 10)

        # Find minimum values of dx/dt in first cal_phase_dur seconds
        drone_search_end = np.argwhere(self.flight.t_interp >= self.flight.t_interp[0] + cal_phase_dur)[0][0]
        drone_x_vel_min = np.min(drone_x_vel[:drone_search_end, 1])
        eg_search_end = np.argwhere(self.eyegaze.eg_time >= self.eyegaze.eg_time[0] + cal_phase_dur)[0][0]
        eg_x_vel_min = np.min(eg_x_vel[:eg_search_end, 1])

        # Find indices corresponding to first crossing time of fract_start * vel_min
        t_drone_cal_start = drone_x_vel[np.argwhere(drone_x_vel[:, 1] <= fract_start * drone_x_vel_min)[0][0], 0]
        t_eg_cal_start = eg_x_vel[np.argwhere(eg_x_vel[:, 1] <= fract_start * eg_x_vel_min)[0][0], 0]
        drone_cal_st = np.argwhere(self.flight.t_interp >= t_drone_cal_start)[0][0]
        eg_cal_st = np.argwhere(self.eyegaze.eg_time >= t_eg_cal_start)[0][0]

        # Find indices corresponding to t_drone_cal_start + cal_phase_dur, and t_eg_cal_start + cal_phase_dur
        drone_cal_end = np.argwhere(self.flight.t_interp >= t_drone_cal_start + cal_phase_dur)[0][0]
        eg_cal_end = np.argwhere(self.eyegaze.eg_time >= t_eg_cal_start + cal_phase_dur)[0][0]

        # Return indices corresponding to [tstart_drone_cal, tend_drone_cal, tstart_eg_cal, tend_eg_cal]
        return [drone_cal_st, drone_cal_end, eg_cal_st, eg_cal_end]

    def find_locus_center(self):
        # Method to find, and return the (X, Y) location of the "starting point" of the eye-gaze locus,
        # and the (X, Y, Z) location of the drone starting point.
        #   Inputs: eg_time -- the eye-gaze time returned by the align_drone_eg method.
        #   Returns list: [[x_eye_gaze_start, y_eye_gaze_start], [x_drone_start, y_drone_start, z_drone_start]]
        #
        # Note that the X values are calculated simply as the average of the min and max values of x
        # during the calibration period, while Y and Z are calculated as the mean values during this period.
        #
        # x_eg_start = np.mean(self.eyegaze.hitpoint[self.eg_align_st:self.eg_align_end, 0])
        x_eg_start = 0.5 * (np.max(self.eyegaze.hitpoint[self.eg_align_st : self.eg_align_end + 1, 0]) +
                            np.min(self.eyegaze.hitpoint[self.eg_align_st : self.eg_align_end + 1, 0]))
        y_eg_start = np.mean(self.eyegaze.hitpoint[self.eg_align_st:self.eg_align_end + 1, 1])
        z_eg_start = self.eyegaze.hitpoint[self.eg_align_st, 2]
        # x_drone_start = np.mean(self.flight.x[self.drone_align_st:self.drone_align_end + 1])
        x_drone_start = 0.5 * (np.max(self.flight.x[self.drone_align_st:self.drone_align_end + 1]) +
                               np.min(self.flight.x[self.drone_align_st:self.drone_align_end + 1]))
        y_drone_start = np.mean(self.flight.y[self.drone_align_st:self.drone_align_end + 1])
        z_drone_start = np.mean(self.flight.z[self.drone_align_st:self.drone_align_end + 1])
        return [[x_eg_start, y_eg_start, z_eg_start], [x_drone_start, y_drone_start, z_drone_start]]

    def key_gen(self, nbits_per_seg, tquant_start, tquant_end, tquant_step, quant_abs_val):
        #
        # Split drone and eye-gaze time series (x(t), y(t)) into segments of tquant_step length.
        #
        self.drone_segs, self.drone_seg_indices = EGutil.calc_seg_disp(self.flight.t_interp, self.dr_norm_x, tquant_start,
                                                             tquant_end, tquant_step, quant_abs_val)
        self.eg_segs, self.eg_seg_indicies = EGutil.calc_seg_disp(self.eyegaze.eg_time + self.t_shift, self.eg_norm_x,
                                                        tquant_start, tquant_end, tquant_step, quant_abs_val)

        #
        # Choose optimum partition thresholds for nbits_per_seg bits of quantization per segement.
        #
        nbits_per_seg = 2
        self.drone_x_part = EGutil.partition_calc(self.drone_segs, nbits_per_seg)
        self.eg_x_part = EGutil.partition_calc(self.eg_segs, nbits_per_seg)
        #
        # Quantize drone and eye-gaze segments based on optimal partition thresholds
        #
        code = [0, 1, 3, 2]  # 2-bits, gray-coded
        drone_dx_q = np.digitize(self.drone_segs, self.drone_x_part)
        eg_dx_q = np.digitize(self.eg_segs, self.eg_x_part)
        #
        # Convert sequence of n-bit values to key strings consisting of bits.
        #
        self.key_drone = ''
        self.key_eg = ''
        for i in range(0, len(drone_dx_q)):
            if nbits_per_seg == 1:
                self.key_drone += "{0:1b}".format(code[drone_dx_q[i] - 1])
            else:
                self.key_drone += "{0:02b}".format(code[drone_dx_q[i] - 1])
        for i in range(0, len(eg_dx_q)):
            if nbits_per_seg == 1:
                self.key_eg += "{0:1b}".format(code[eg_dx_q[i] - 1])
            else:
                self.key_eg += "{0:02b}".format(code[eg_dx_q[i] - 1])

    @staticmethod
    def find_alignment_param(t_arr, pos_arr, index_st, index_end, avg_samples):
        #
        # Calculate and return the following parameters from a position vs. time series:
        #
        #
        # Maximum and minimum values of position over calibration period, and times at which they occur
        pos_max = np.max(pos_arr[index_st : index_end + 1])
        t_max = t_arr[index_st + np.argmax(pos_arr[index_st : index_end + 1])]
        pos_min = np.min(pos_arr[index_st : index_end + 1])
        t_min = t_arr[index_st + np.argmin(pos_arr[index_st : index_end + 1])]
        #
        # Time at which maximum positive velocity occurs during calibration period
        vel_arr = EGutil.calc_vel(t_arr[index_st : index_end + 1], pos_arr[index_st : index_end + 1], avg_samples)
        index_vel_max = np.argmax(vel_arr[:, 1])
        t_max_vel = vel_arr[index_vel_max, 0]
        #
        # Return values
        return [t_min, pos_min, t_max, pos_max, t_max_vel]


def main ():
    eg_filepath = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research'
                    '\\Data Sets\\DJI Drone\\DroneTracker3\\')
    drone_filepath = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Data Sets'
                '\\DJI Drone\\Drone Flight Path\\')
    test_date = '082624\\'
    # eg_filename, drone_filename, eg_start = ['eye_tracker_07232024_192335.csv', 'randx_patt_07232024_192433.csv', 3.0]
    # eg_filename, drone_filename, eg_start = ['eye_tracker_07232024_192522.csv', 'randx_patt_07232024_192616.csv', 2.0]
    # eg_filename, drone_filename, eg_start = ['eye_tracker_07232024_192707.csv', 'randx_patt_07232024_192757.csv', 0.75]
    # eg_filename, drone_filename, eg_start = ['eye_tracker_08062024_134813.csv', 'randx_patt_08062024_134824.csv', 3.4]
    # eg_filename, drone_filename, eg_start = ['eye_tracker_08062024_134429.csv', 'randx_patt_08062024_134437.csv', 2.0]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08102024_110205.csv',
    #                                                   'randx_patt_08102024_110242.csv', 2.5, False]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08112024_075955.csv',
    #                                                  'randx_patt_08112024_080146.csv', 0.5, False]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08112024_075955.csv',
    #                                                  'randx_patt_08112024_080146.csv', 0.5, False]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08112024_080435.csv',
    #                                                  'randx_patt_08112024_080534.csv', 2.0, False]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08112024_080826.csv',
    #                                                  'randx_patt_08112024_080940.csv', 6.0, True]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08112024_081230.csv',
    #                                                  'randx_patt_08112024_081308.csv', 0.0, True]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08112024_201319.csv',
    #                                                  'randx_patt_08112024_201350.csv', 1.0, True]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08122024_092630.csv',
    #                                                  'randx_patt_08122024_092701.csv', 3.0, False]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08122024_092948.csv',
    #                                                  'randx_patt_08122024_093101.csv', 2.0, True]
    # eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08122024_200239.csv',
    #                                                  'randx_patt_08122024_200320.csv', 0.0, False]
    eg_filename, drone_filename, eg_start, eg_inv = ['eye_tracker_08262024_150602.csv',
                                                     'randxy_patt_08262024_150642.csv', 0.0, False]

    print(f'Drone flight time series: {drone_filepath + test_date + drone_filename}')
    print(f'Eye-gaze time series: {eg_filepath + test_date + eg_filename}')
    eg_end = 165.0
    tquant_start = 12.0
    tquant_end = 150.0
    tquant_step = 1.0
    rem_outliers = True
    interp = 'Akima'
    lowpass = True
    quant_abs_val = False
    pair_test = DroneEyePair(drone_filepath + test_date + drone_filename, eg_filepath + test_date + eg_filename, eg_start,
                             eg_end, tquant_start, tquant_end, tquant_step, eg_inv=eg_inv, rem_outliers=rem_outliers,
                             interp=interp, lowpass=lowpass, quant_abs_val=quant_abs_val, save_ext=True)
    print(f'\tDrone approx. {pair_test.flight.dist_tako:.2f} m from observer')
    print(f'\tInitial drone height = {pair_test.flight.y[0]:.2f} m')
    print(f'\tDrone speed level = {pair_test.flight.speedlev:.2f}')
    print(f'\tDrone Z axis rotated {180 / math.pi * pair_test.flight.theta:.4f} degrees relative to North.')
    print(f'\tTime shift = {pair_test.t_shift:.4f} s')
    print(f'\tKey from Drone Flight = {pair_test.key_drone}')
    print(f'\tKey from Eye Gaze     = {pair_test.key_eg}')
    bit_count, bit_diff, ones_dr, ones_eg = EGutil.key_bit_comp(pair_test.key_drone, pair_test.key_eg)
    if len(pair_test.key_drone) == len(pair_test.key_eg):
        print(f'\tKeys have same length = {bit_count} bits')
    else:
        print(f'\tKeys have different lengths:')
        print(f'\t\tDrone key = {len(pair_test.key_drone)} bits')
        print(f'\t\tEye-gaze key = {len(pair_test.key_eg)} bits')
    print(f'\tThere are {bit_diff} bit differences between the keys ({100.0 * (1 - bit_diff / bit_count):.4f}% matching)')
    print(f'\tEntropy(Drone key) = {EGutil.entropy_calc(ones_dr, len(pair_test.key_drone))} bits')
    print(f'\tEntropy(Eye-gaze key) = {EGutil.entropy_calc(ones_eg, len(pair_test.key_eg))} bits')

    #
    # Calculate position error (drone(t) - eye-gaze-hitpoint(t)) vs. time
    #
    print('\tStd dev of position error = {0:.4f}'.format(pair_test.pos_err_stdev))
    #
    # Print drone and eye-gaze quantization thresholds and standard deviation of segment error
    #
    print('\n\tDrone quantization thresholds    {0:.4f}, {1:.4f}, {2:.4f}'.format(pair_test.drone_x_part[0],
                                                                pair_test.drone_x_part[1], pair_test.drone_x_part[2]))
    print('\tEye-Gaze quantization thresholds {0:.4f}, {1:.4f}, {2:.4f}'.format(pair_test.eg_x_part[0],
                                                                pair_test.eg_x_part[1], pair_test.eg_x_part[2]))
    print('\tStd dev of drone X segments = {0:.5f}'.format(pair_test.drone_seg_stdev))
    print('\tStd dev of (drone - eye-gaze) X segment error = {0:.5f}'.format(pair_test.seg_err_stdev))

    #
    # Plot raw values of drone X and eye-gaze X vs. time without any time shift
    #
    plot_start = 0
    fig1, (axes1a, axes1b, axes1c, axes1d) = plt.subplots(4, 1)
    param_dict = {'title': 'Drone X vs. Time', 'xlabel': 'drone_time (s)', 'ylabel': 'X (m)',
                  'legends': ['Drone X']}
    EGplt.plot_multiline(axes1a, [pair_test.flight.x],[pair_test.flight.t_interp], param_dict)
    axes1a.set_xlim(0, tquant_end)
    axes1a.grid(visible=True, which='both', axis='both')
    param_dict = {'title': 'Eye-Gaze X vs. Time', 'xlabel': 'eg_time (s)', 'ylabel': 'X (m)',
                  'legends': ['Eye-Gaze X']}
    EGplt.plot_multiline(axes1b, [pair_test.eyegaze.hitpoint[plot_start:, 0]],
                         [pair_test.eyegaze.eg_time[plot_start:]], param_dict)
    axes1b.set_xlim(0.0, tquant_end)
    axes1b.grid(visible=True, which='both', axis='both')
    param_dict = {'title': 'Drone Y vs. Time', 'xlabel': 'drone_time (s)', 'ylabel': 'Y (m)',
                  'legends': ['Drone Y']}
    EGplt.plot_multiline(axes1c, [pair_test.flight.y], [pair_test.flight.t_interp], param_dict)
    axes1c.set_xlim(0, tquant_end)
    axes1c.grid(visible=True, which='both', axis='both')
    param_dict = {'title': 'Eye-Gaze Y vs. Time', 'xlabel': 'drone_time (s)', 'ylabel': 'Y (m)',
                  'legends': ['Eye-Gaze Y']}
    EGplt.plot_multiline(axes1d, [pair_test.eyegaze.hitpoint[plot_start:, 1]],
                         [pair_test.eyegaze.eg_time[plot_start:]], param_dict)
    axes1d.set_xlim(0, tquant_end)
    axes1d.grid(visible=True, which='both', axis='both')

    #
    # Plot normalized drone and eye-gaze X vs. time
    #
    egplt_start = 0
    drplt_start = 0
    fig2, axes2 = plt.subplots()
    param_dict = {'title': 'Eye-Gaze and Drone X vs. Time', 'xlabel': 'Time (s)', 'ylabel': 'X (m)',
                  'legends': ['Eye-Gaze X', 'Drone X']}
    EGplt.plot_multiline(axes2, [pair_test.eg_norm_x[egplt_start:], pair_test.dr_norm_x[drplt_start:]],
                         [pair_test.eyegaze.eg_time[egplt_start:] + pair_test.t_shift,
                         pair_test.flight.t_interp[drplt_start:]], param_dict)
    axes2.grid(visible=True, which='both', axis='both')
    axes2.set_xlim(0, tquant_end)

    #
    # Plot normalized drone and eye-gaze Y vs. time
    #
    fig3, axes3 = plt.subplots()
    param_dict = {'title': 'Eye-Gaze and Drone Y vs. Time', 'xlabel': 'Time (s)', 'ylabel': 'Y (m)',
                  'legends': ['Eye-Gaze Y', 'Drone Y']}
    EGplt.plot_multiline(axes3, [pair_test.eg_norm_y[egplt_start:], pair_test.dr_norm_y[drplt_start:]],
                         [pair_test.eyegaze.eg_time[egplt_start:] + pair_test.t_shift,
                          pair_test.flight.t_interp[drplt_start:]], param_dict)
    axes3.grid(visible=True, which='both', axis='both')
    axes3.set_xlim(0, tquant_end)

    #
    # Plot position error (drone(t) - eye-gaze-hitpoint(t)) vs. time
    #
    fig4, axes4 = plt.subplots()
    param_dict = {'title': 'Drone - Eye-Gaze X Position Error', 'xlabel': 'Time (s)', 'ylabel': 'Error (m)',
                  'legends': ['Drone - Eye-Gaze']}
    EGplt.plot_multiline(axes4, [pair_test.pos_err[drplt_start:]],
                         [pair_test.flight.t_interp[drplt_start:]], param_dict)
    axes4.grid(visible=True, which='both', axis='both')
    axes4.set_xlim(0, tquant_end)

    plt.show()


# Main code
if __name__ == '__main__':
    main()
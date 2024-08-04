"""
    DroneEyePair -- Class that

    Description ...

    by Christopher Abel
    Revision History
    ----------------
    07/29/2024 -- Original
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
    def __init__(self, drone_file, eg_file, eg_start, eg_end, tquant_start, tquant_end, tquant_step, nbits_per_seg=2, quant_abs_val=True, save_ext=False):
        self.flight = DJId.DJIDronePath(drone_file)
        self.eyegaze = Holo.HoloEGPath(eg_file, eg_start, eg_end, rem_outliers=True, interp='Akima', tstep=0.02, lowpass=True)

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
        self.drone_pkpk = EGutil.calc_norm_factor(self.flight.t_interp, self.flight.x, 0, 10, alpha=0.0)
        self.eg_pkpk = EGutil.calc_norm_factor(self.eyegaze.eg_time, self.eyegaze.hitpoint[:, 0], 0, 10, alpha=0.0)
        self.dr_norm_x = EGutil.normalize_path(self.flight.x, self.drone_start_pos[0], self.drone_pkpk)
        self.eg_norm_x = EGutil.normalize_path(self.eyegaze.hitpoint[:, 0], self.eg_xyz_start[0], self.eg_pkpk)

        #
        # Segment and quantize drone and eye-gaze time series (x(t), y(t))
        #
        self.key_gen(nbits_per_seg, tquant_start, tquant_end, tquant_step, quant_abs_val)
        # drone_segs, drone_seg_indices = EGutil.calc_seg_disp(self.flight.t_interp, self.dr_norm_x, tquant_start, tquant_end, tquant_step, quant_abs_val)
        # eg_segs, eg_seg_indicies = EGutil.calc_seg_disp(self.eyegaze.eg_time + self.t_shift, self.eg_norm_x, tquant_start, tquant_end, tquant_step, quant_abs_val)
        # self.drone_x_part = EGutil.partition_calc(drone_segs, nbits_per_seg)
        # self.eg_x_part = EGutil.partition_calc(eg_segs, nbits_per_seg)
        # code = [0, 1, 3, 2] # 2-bits, gray-coded
        # drone_dx_q = np.digitize(drone_segs, self.drone_x_part)
        # eg_dx_q = np.digitize(eg_segs, self.eg_x_part)
        # self.key_drone = ''
        # self.key_eg = ''
        # for i in range(0, len(drone_dx_q)):
        #     if nbits_per_seg == 1: self.key_drone += "{0:1b}".format(code[drone_dx_q[i] - 1])
        #     else: self.key_drone += "{0:02b}".format(code[drone_dx_q[i] - 1])
        # for i in range(0, len(eg_dx_q)):
        #     if nbits_per_seg == 1: self.key_eg += "{0:1b}".format(code[eg_dx_q[i] - 1])
        #     else: self.key_eg += "{0:02b}".format(code[eg_dx_q[i] - 1])

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

        # Find indices corresponding to cal_phase_dur of time from the start of the pattern
        drone_cal_end = np.argwhere(self.flight.t_interp >= self.flight.t_interp[0] + cal_phase_dur)[0][0]
        eg_cal_end = np.argwhere(self.eyegaze.eg_time >= self.eyegaze.eg_time[0] + cal_phase_dur)[0][0]

        # Find minimum values of dx/dt in first cal_phase_dur seconds
        drone_x_vel_min = np.min(drone_x_vel[:drone_cal_end, 1])
        eg_x_vel_min = np.min(eg_x_vel[:eg_cal_end, 1])

        # Find indices corresponding to first crossing time of fract_start * vel_min
        t_drone_cal_start = drone_x_vel[np.argwhere(drone_x_vel[:, 1] <= fract_start * drone_x_vel_min)[0][0], 0]
        t_eg_cal_start = eg_x_vel[np.argwhere(eg_x_vel[:, 1] <= fract_start * eg_x_vel_min)[0][0], 0]
        drone_cal_st = np.argwhere(self.flight.t_interp >= t_drone_cal_start)[0][0]
        eg_cal_st = np.argwhere(self.eyegaze.eg_time >= t_eg_cal_start)[0][0]

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
                    '\\Data Sets\\DJI Drone\\DroneTracker3\\072324\\')
    drone_filepath = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Data Sets'
                '\\DJI Drone\\Drone Flight Path\\072324\\')
    # eg_filename, drone_filename, eg_start = ['eye_tracker_07232024_192335.csv', 'randx_patt_07232024_192433.csv', 3.0]
    eg_filename, drone_filename, eg_start = ['eye_tracker_07232024_192522.csv', 'randx_patt_07232024_192616.csv', 2.0]
    # eg_filename, drone_filename, eg_start = ['eye_tracker_07232024_192707.csv', 'randx_patt_07232024_192757.csv', 0.75]

    print(f'Drone flight time series: {drone_filepath + drone_filename}')
    print(f'Eye-gaze time series: {eg_filepath + eg_filename}')
    eg_end = 45.0
    tquant_start = 12.0
    tquant_end = 35.0
    tquant_step = 1.0
    quant_abs_val = False
    pair_test = DroneEyePair(drone_filepath + drone_filename, eg_filepath + eg_filename, eg_start, eg_end, tquant_start, tquant_end, tquant_step, quant_abs_val=quant_abs_val, save_ext=True)
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
    # Plot raw values of drone X and eye-gaze X vs. time without any time shift
    #
    plot_start = 0
    fig1, (axes1a, axes1b) = plt.subplots(2, 1)
    param_dict = {'title': 'Drone X vs. Time', 'xlabel': 'drone_time (s)', 'ylabel': 'X (m)',
                  'legends': ['Drone X']}
    EGplt.plot_multiline(axes1a, [pair_test.flight.x], [pair_test.flight.t_interp], param_dict)
    axes1a.set_xlim(0, 40.0)
    axes1a.grid(visible=True, which='both', axis='both')
    param_dict = {'title': 'Eye-Gaze X vs. Time', 'xlabel': 'eg_time (s)', 'ylabel': 'X (m)',
                  'legends': ['Eye-Gaze X']}
    EGplt.plot_multiline(axes1b, [pair_test.eyegaze.hitpoint[plot_start:, 0]],
                         [pair_test.eyegaze.eg_time[plot_start:]], param_dict)
    axes1b.set_xlim(0.0, 40.0)
    axes1b.grid(visible=True, which='both', axis='both')

    #
    # Plot normalized drone and eye-gaze X vs. time
    #
    egplt_start = 0
    drplt_start = 0
    fig2, axes2 = plt.subplots()
    param_dict = {'title': 'Eye-Gaze and Drone X vs. Time', 'xlabel': 'Time (s)', 'ylabel': 'X (m)',
                  'legends': ['Eye-Gaze X', 'Drone X']}
    EGplt.plot_multiline(axes2, [pair_test.eg_norm_x[egplt_start:], pair_test.dr_norm_x[drplt_start:]],
                         [pair_test.eyegaze.eg_time[egplt_start:] + pair_test.t_shift, pair_test.flight.t_interp[drplt_start:]], param_dict)
    axes2.grid(visible=True, which='both', axis='both')

    plt.show()


# Main code
if __name__ == '__main__':
    main()
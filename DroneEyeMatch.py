#
# DroneEyeMatch.py -- Class containing fields and methods for data from
#           drone flight and eye-gaze capture files.
#
# Description -- Class that contains data from:
#       -- Drone flight log .csv file
#       -- Eye-gaze position capture .csv file
#       Includes methods to read data from files into np.arrays for each
#       variable, plot position / direction vs. time, and plot eye-gaze
#       hitpoints superimposed on 2D projection of drone path.
#
# by Christopher Abel
# Revision History
# ----------------
# 03/12/2024 -- Original
# 04/09/2024 -- Added interpolation options to read_eg_file().
# 04/12/2024 -- Added interpolation options to read_drone_file().
# 04/15/2024 -- Added outlier removal to read_eg_file(), and Savitzky-Golay filtering to read_eg_file()
#               and read_drone_file().  Added calculation of entropy in key strings.
# 04/22/2024 -- Added in shift removal (via the pos_shift_comp() function) to read_drone_file() to remove gross
#               shifts in y vector returned by drone sensor.
#
# -------------------------------------------------
import csv
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator, CubicSpline
import scipy
import EGCustomPlots as EGplt
import EyeMatchUtils as EGutil


class DroneEyeMatch:
    # Constructor for one data set
    def __init__(self, eye_gaze_file, drone_file, drone_patt, chop_start=0.0, chop_end=1000.0):
        self.drone_file = drone_file
        self.eye_gaze_file = eye_gaze_file
        self.drone_patt = drone_patt

        # Read data from eye-gaze and drone files into arrays
        interp = 'Akima'
        tstep_interp = 0.02
        rem_outliers = True
        rem_shifts = True
        lowpass = True
        [self.eg_date, self.eg_start_time, self.eg_time, self.eg_hitpoint, self.eg_direction,
         self.eg_origin, self.eg_time_outliers] = self.read_eg_file(chop_start=chop_start, chop_end=chop_end,
                                                                    rem_outliers=rem_outliers, out_level=0.1,
                                                                    interp=interp, tstep=tstep_interp, lowpass=lowpass)
        [self.drone_date, self.drone_time, self.drone_pos] = self.read_drone_file(rem_shifts=rem_shifts, interp=interp,
                                                                                  tstep=tstep_interp, lowpass=lowpass)

        # Uncomment these lines to save selected time-series data to eg_data.csv and drone_data.csv for
        # convenient analysis in Matlab.
        with open('eg_data.csv', 'w', newline='') as eg_out_file:
            print('Writing eye-gaze data to eg_data.csv...')
            eg_writer = csv.writer(eg_out_file)
            for i in range(0, len(self.eg_time)):
                eg_writer.writerow([self.eg_time[i], self.eg_hitpoint[i, 0], self.eg_hitpoint[i, 1]])
        with open('drone_data.csv', 'w', newline='') as drone_out_file:
            print('Writing drone data to drone_data.csv...')
            drone_writer = csv.writer(drone_out_file)
            for i in range(0, len(self.drone_time)):
                drone_writer.writerow([self.drone_time[i], self.drone_pos[i, 0], self.drone_pos[i, 1],
                                       self.drone_pos[i, 2]])
        with open('eg_time_outliers.csv', 'w', newline='') as outlier_file:
            print('Writing eye-gaze outlier time points to eg_time_outliers.csv...')
            outlier_writer = csv.writer(outlier_file)
            for i in range(0, len(self.eg_time_outliers)):
                outlier_writer.writerow([self.eg_time_outliers[i]])

        # Find time indices at which to align drone and eye-gaze at start of drone calibration pattern
        alignment_indices = self.align_drone_eg(0.5, 10)
        self.drone_alignment_index = alignment_indices[0]
        self.eg_alignment_index = alignment_indices[2]

        # Find drone start position, and eye-gaze hitpoint start, and time at which X calibration sequence begins
        self.drone_start_pos = self.find_locus_center()[1]
        self.eg_xyz_start = self.find_locus_center()[0]

        # Calculate drone X, Y, Z velocities, and eye-gaze hitpoint X and Y velocities
        avg_samples = 20
        self.t_algn_dr = self.drone_time[self.drone_alignment_index:] - self.drone_time[self.drone_alignment_index]
        self.pos_algn_dr = self.drone_pos[self.drone_alignment_index:]
        self.t_algn_eg = self.eg_time[self.eg_alignment_index:] - self.eg_time[self.eg_alignment_index]
        self.pos_algn_eg = self.eg_hitpoint[self.eg_alignment_index:]
        self.drone_x_vel = self.calc_vel(self.t_algn_dr, self.pos_algn_dr[:, 0], avg_samples)
        self.drone_y_vel = self.calc_vel(self.t_algn_dr, self.pos_algn_dr[:, 1], avg_samples)
        self.drone_z_vel = self.calc_vel(self.t_algn_dr, self.pos_algn_dr[:, 2], avg_samples)
        self.eg_x_vel = self.calc_vel(self.t_algn_eg, self.pos_algn_eg[:, 0], avg_samples)
        self.eg_y_vel = self.calc_vel(self.t_algn_eg, self.pos_algn_eg[:, 1], avg_samples)

        # Normalize drone and eye-gaze hitpoint loci
        dr_dx, eg_dx = self.normalize_dr_eg()
        drone_norm_factor = math.fabs(2.0 / dr_dx)
        eg_norm_factor = math.fabs(2.0 / eg_dx)
        self.pos_norm_dr = drone_norm_factor*(self.pos_algn_dr - np.array(self.drone_start_pos).reshape(1, 3))
        self.pos_norm_eg = eg_norm_factor*(self.pos_algn_eg - np.array(self.eg_xyz_start).reshape(1, 3))

        # Calculate segment X and Y displacement arrays. Quantize and create keys.
        t_quant_start = 12.0
        t_quant_step = 1.0
        num_bits_per_seg = 2
        self.seg_disp = self.calc_seg_disp(t_quant_start, t_quant_step)
        # partition = [0.0552, 0.1296, 0.2510, 0.0419, 0.0871, 0.1853]
        # partition = [0.1296, 0.0871]
        partition = self.calc_opt_partition(num_bits_per_seg)
        code = [0, 1, 3, 2]
        # code = [0, 1]
        self.key_dr, self.key_eg = self.key_gen(partition, code)

    def read_eg_file(self, chop_start=0.0, chop_end=1000.0, rem_outliers=False, out_level=0.1, interp='None',
                     tstep=0.02, lowpass=False):
        #
        # Read eye-gaze data file, load samples into np.arrays.
        # interp and tstep control interpolation of data to a smaller fixed timestep.
        #   interp = 'None' -- no interpolation (default)
        #          = 'Linear' -- linear interpolation
        #          = 'Akima' -- Akima 1D cubic interpolation
        #          = 'Cubic' -- Cubic spline interpolation
        #   tstep = Fixed time step (s) used for interpolation.  Ignored if interp='None'.
        #
        with open(self.eye_gaze_file, 'r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)   # Read first line of file; contains test start date
            eg_date = datetime.datetime.strptime(header[1], '%m/%d/%Y %H:%M:%S')
            column_header = next(reader)    # Read second line of file; contains field headers
            #
            # Read remaining rows of eye_gaze file.
            #
            date_time = []
            t = []
            hitpoint = []
            direction = []
            origin = []
            sample_count = 0
            for row in reader:
                sample_count += 1
                #
                # Store two different time values
                #   date_time[i] = Python datetime object whose value corresponds to
                #               the time captured from column 1.
                #   t[i] = Time in seconds = date_time[i] - date_time[0]
                #
                date_time.append(datetime.datetime.strptime(row[1], '%H:%M:%S.%f'))
                t.append((datetime.datetime.strptime(row[1], '%H:%M:%S.%f')
                          - date_time[0]).total_seconds())
                hitpoint.append(row[6:9])
                direction.append(row[9:12])
                origin.append(row[12:15])
            # Find the first index of the time array which is greater than or equal to chop_start
            start = 0
            while t[start] < chop_start:
                start += 1
            # Find the first index of the time array which is greater than or equal to chop_end
            end = 0
            while (t[end] < chop_end) and (end < len(t) - 1):
                end += 1
            #
            # Return only the starting datetime value, eg_start_time = date_time[start], and
            # the array of time values, in seconds, from this datetime -- eg_time.
            # If interp = 'None', return time, hitpoint, direction, and origin arrays without
            # interpolation. Else, interpolate data using selected algorithm.
            #
            eg_start_time = date_time[start]
            eg_time = np.array(t[start:end], dtype=np.float32) - float(t[start])
            eg_hitpoint = np.array(hitpoint[start:end], dtype=np.float32)
            eg_direction = np.array(direction[start:end], dtype=np.float32)
            eg_origin = np.array(origin[start:end], dtype=np.float32)
            #
            # If rem_outliers = True, then use rem_outliers() function to remove outlier values from hitpoint,
            #   direction, and origin.
            #
            eg_time_outliers = []
            if rem_outliers:
                eg_data = np.concatenate((eg_hitpoint, eg_direction, eg_origin), 1)
                eg_time, eg_data, eg_time_outliers = EGutil.rem_outliers(eg_time, eg_data, out_level, 3)
                eg_hitpoint = eg_data[:, 0:3]
                eg_direction = eg_data[:, 3:6]
                eg_origin = eg_data[:, 6:9]
            #
            # Interpolate and upsample to a higher sample rate (lower sampling period) if interp is something
            # other than 'None'.
            #
            if not interp == 'None':
                npts = math.floor((eg_time[-1] - eg_time[0]) / tstep) + 1
                t_interp = np.linspace(eg_time[0], eg_time[0] + (npts - 1) * tstep, npts)
                hitp_interp = EGutil.interp_1d_array(t_interp, eg_time, eg_hitpoint, interp)
                dir_interp = EGutil.interp_1d_array(t_interp, eg_time, eg_direction,interp)
                origin_interp = EGutil.interp_1d_array(t_interp, eg_time, eg_origin, interp)
                eg_time = t_interp
                eg_hitpoint = hitp_interp
                eg_direction = dir_interp
                eg_origin = origin_interp
            #
            # Lowpass filter dataset. Use Savitzky-Golay filter of order sg_ord and window length = sg_len
            # to filter eg_hitpoint dataset.
            #
            if lowpass:
                sg_ord = 0
                sg_len = 25
                eg_hitpoint[:, 0] = scipy.signal.savgol_filter(eg_hitpoint[:, 0], sg_len, sg_ord)
                eg_hitpoint[:, 1] = scipy.signal.savgol_filter(eg_hitpoint[:, 1], sg_len, sg_ord)
                eg_hitpoint[:, 2] = scipy.signal.savgol_filter(eg_hitpoint[:, 2], sg_len, sg_ord)
        # Return captured data
        return [eg_date, eg_start_time, eg_time, eg_hitpoint, eg_direction, eg_origin, eg_time_outliers]

    def read_drone_file(self, rem_shifts=False, interp='None', tstep=0.02, lowpass=False):
        #
        # Read drone flight data file, load samples into np.arrays
        #
        with open(self.drone_file, 'r', newline='') as file:
            reader = csv.reader(file)
            date_time = []
            t = []
            pos = []
            sample_count = 0
            for row in reader:
                if row[0] == self.drone_patt:
                    if sample_count == 0:
                        start_date_time = datetime.datetime.strptime(row[1], '%m/%d/%Y %H:%M:%S.%f')
                    sample_date_time = datetime.datetime.strptime(row[1], '%m/%d/%Y %H:%M:%S.%f')
                    date_time.append((sample_date_time - start_date_time).total_seconds())
                    t.append(row[2])
                    # Mapping drone X axis (row[3]) to Z axis, drone Y axis to X axis, and drone Z
                    # axis to Y axis.  This orientation matches the orientation of XYZ for the eye-gaze.
                    pos.append([row[4], row[5], row[3]])
                    sample_count += 1
            drone_date = np.array(date_time, dtype=np.float32)
            drone_time = np.array(t, dtype=np.float32)
            drone_pos = np.array(pos, dtype=np.float32)
            drone_pos[:, 0] = -drone_pos[:, 0]  # Negate x-axis values to match X axis of eye-gaze
            #
            # Remove non-physical shifts in y position that arise from presumed inaccuracies in drone height sensor.
            #
            if rem_shifts:
                yshift_thresh = 0.075
                drone_pos[:, 1] = EGutil.pos_shift_comp(drone_pos[:, 1], yshift_thresh)
            #
            # Interpolate and upsample to a higher sample rate (lower sampling period) if interp is something
            # other than 'None'. If interpolating, use rem_duplicates() function to remove consecutive duplicate
            # time samples in dataset.
            #
            if not interp == 'None':
                drone_time, drone_pos = EGutil.rem_duplicates(drone_time, drone_pos, 1e-5)
                npts = math.floor((drone_time[-1] - drone_time[0]) / tstep) + 1
                t_interp = np.linspace(drone_time[0], drone_time[0] + (npts - 1) * tstep, npts)
                drone_pos_interp = EGutil.interp_1d_array(t_interp, drone_time, drone_pos, interp)
                drone_time = t_interp
                drone_pos = drone_pos_interp
            #
            # Lowpass filter dataset. Use Savitzky-Golay filter of order sg_ord and window length = sg_len.
            #
            if lowpass:
                sg_ord = 0
                sg_len = 25
                drone_pos[:, 0] = scipy.signal.savgol_filter(drone_pos[:, 0], sg_len, sg_ord)
                drone_pos[:, 1] = scipy.signal.savgol_filter(drone_pos[:, 1], sg_len, sg_ord)
                drone_pos[:, 2] = scipy.signal.savgol_filter(drone_pos[:, 2], sg_len, sg_ord)
        # Return captured data
        return [drone_date, drone_time, drone_pos]

    #
    # Methods to return specified elements in data lists
    #
    def get_eg_at_index(self, index):
        # Return list consisting of: eye-gaze [time, hitpoint, direction, origin] at row = index
        return [self.eg_time[index], self.eg_hitpoint[index, :], self.eg_direction[index, :],
                self.eg_origin[index, :]]

    def print_eg_at_index(self, index):
        # Primarily for debugging -- formatted print of values returned by get_eg_at_index
        eg = self.get_eg_at_index(index)
        print(f'Eye gaze values at index = {index}:')
        print('\tt = {0:.3f}s\tHitpoint [X, Y, Z] = [{1:.4f}, {2:.4f}, {3:.4f}]m'.
              format(eg[0], eg[1][0], eg[1][1], eg[1][2]))
        print('\tDirection = [{0:.3f}, {1:.3f}, {2:.3f}]'.format(eg[2][0], eg[2][1], eg[2][2]), end='')
        print('  Origin = [{0:.3f}, {1:.3f}, {2:.3f}]m'.format(eg[3][0], eg[3][1], eg[3][2]))

    def get_drone_at_index(self, index):
        # Return list consisting of: drone [date, time, location[x, y, z]] at row = index
        return [self.drone_date[index], self.drone_time[index], self.drone_pos[index, :]]

    def print_drone_at_index(self, index):
        # Primarily for debugging -- formatted print of values returned by get_drone_at_index
        drone = self.get_drone_at_index(index)
        print(f'Drone position at index = {index}')
        print(f'\tTime = {drone[1]:.3f}\tPosition = [{drone[2][0]:.3f}, {drone[2][1]:.3f},'
              f'{drone[2][2]:.3f}]')

    def align_drone_eg(self, fract_start=0.1, cal_phase_dur=10.0):
        # Method that finds an indices in the time vectors for the drone pattern and eye-gaze
        # that represent the times at which the drone is beginning to move left in
        # the calibration section of the pattern.
        #
        # The method finds the first index at which the x velocity of the drone and x velocity
        # of eye-gaze hitpoint both first reach fract_start * peak negative velocity in first
        # cal_phase_dur seconds of the drone flight path.
        #
        drone_x_vel = DroneEyeMatch.calc_vel(self.drone_time, self.drone_pos[:, 0], 20)
        eg_x_vel = DroneEyeMatch.calc_vel(self.eg_time, self.eg_hitpoint[:, 0], 20)

        # Find indices corresponding to cal_phase_dur of time from the start of the pattern
        drone_cal_end = np.argwhere(self.drone_time >= self.drone_time[0] + cal_phase_dur)[0][0]
        eg_cal_end = np.argwhere(self.eg_time >= self.eg_time[0] + cal_phase_dur)[0][0]

        # Find minimum values of dx/dt in first cal_phase_dur seconds
        drone_x_vel_min = np.min(drone_x_vel[:drone_cal_end, 1])
        eg_x_vel_min = np.min(eg_x_vel[:eg_cal_end, 1])

        # Find indices corresponding to first crossing time of fract_start * vel_min
        t_drone_cal_start = drone_x_vel[np.argwhere(drone_x_vel[:, 1] <= fract_start * drone_x_vel_min)[0][0], 0]
        t_eg_cal_start = eg_x_vel[np.argwhere(eg_x_vel[:, 1] <= fract_start * eg_x_vel_min)[0][0], 0]
        drone_cal_start = np.argwhere(self.drone_time >= t_drone_cal_start)[0][0]
        eg_cal_start = np.argwhere(self.eg_time >= t_eg_cal_start)[0][0]

        # Return indices and corresponding time values
        return [drone_cal_start, self.drone_time[drone_cal_start], eg_cal_start, self.eg_time[eg_cal_start]]

    @staticmethod
    def find_index_at_time(time_val, time_arr):
        # Method to find the first index in a time array at which time >= time_val
        # Assumes that time_arr is passed as a numpy array.  If time_val exceeds the
        # maximum value of time in time_arr, then return index = -1.
        if np.max(time_arr) < time_val:
            return -1
        else:
            return np.argwhere(time_arr >= time_val)[0][0]

    def find_locus_center(self):
        # Method to find, and return the (X, Y) location of the "starting point" of the eye-gaze locus (just
        # before the start of the drone calibration movement), and the (X, Y, Z) location of the drone starting
        # point.
        #   Inputs: eg_time -- the eye-gaze time returned by the align_drone_eg method.
        #   Returns list: [[x_eye_gaze_start, y_eye_gaze_start], [x_drone_start, y_drone_start, z_drone_start]]
        #
        # The method calculates the (X, Y) values for the eye-gaze by averaging x and y over the range of
        # t = 0 to t = 0.75*eg_time[eg_cal_start]
        eg_average_range = 0.75
        x_eg_start = np.mean(self.eg_hitpoint[self.eg_alignment_index:, 0])
        y_eg_start = np.mean(self.eg_hitpoint[self.eg_alignment_index:, 1])
        z_eg_start = self.eg_hitpoint[self.eg_alignment_index, 2]
        x_drone_start = np.mean(self.drone_pos[self.drone_alignment_index:, 0])
        y_drone_start = np.mean(self.drone_pos[self.drone_alignment_index:, 1])
        z_drone_start = np.mean(self.drone_pos[self.drone_alignment_index:, 2])
        return [[x_eg_start, y_eg_start, z_eg_start], [x_drone_start, y_drone_start, z_drone_start]]

    def normalize_dr_eg(self):
        # Method to find scale factors that normalize the drone flight and eye-gaze hitpoint
        # based on maximum x displacement during calibration movement.
        #
        # Find first 2 zero-crossing times of drone and eye-gaze x velocity.  Search for maximum
        # negative X deflection in time from 0 to 2nd zero-crossing time of dX/dt.
        #
        dr_index = self.find_index_at_time(10.0, self.t_algn_dr)
        x_sort_dr = np.sort(self.pos_algn_dr[:dr_index, 0])
        dr_dx = x_sort_dr[math.ceil(0.98*len(x_sort_dr))] - x_sort_dr[math.floor(0.02*len(x_sort_dr))]
        eg_index = self.find_index_at_time(10.0, self.t_algn_eg)
        x_sort_eg = np.sort(self.pos_algn_eg[:eg_index, 0])
        eg_dx = x_sort_eg[math.ceil(0.98*len(x_sort_eg))] - x_sort_eg[math.floor(0.02*len(x_sort_eg))]
        # Return max neg X values for drone and eye-gaze
        return [dr_dx, eg_dx]

    @staticmethod
    def calc_vel(time_arr, pos_arr, avg_samples=1):
        # Calculate velocity for the pos_arr position array.
        #
        # pos_arr is a 1D array (X), the method will
        # return a 2D array with 2 columns (time, and dX/dt).
        #
        # Each velocity sample is calculated as a weighted average over
        # the previous avg_samples samples.
        #
        deriv = []
        for i in range(avg_samples, len(time_arr)):
            delta_x = pos_arr[i] - pos_arr[i - avg_samples]
            delta_t = time_arr[i] - time_arr[i - avg_samples]
            if delta_t > 0:
                deriv.append([time_arr[i], delta_x / delta_t])
        return np.array(deriv)

    @staticmethod
    def sign_hyst(x_arr, hyst):
        # Method that returns a 1D array of values {-1, 1}, with each element of the returned array
        # defined as follows:
        #
        #   y[i] = -1 if x_arr[i] < Vthresh
        #   y[i] = +1 if x_arr[i] >= Vthresh
        #
        #   Vthresh is a dynamic quantity defined by
        #   If y[i-1] = -1, Vthresh = hyst
        #      y[i-1] = +1, Vthresh = -hyst
        #
        y = [-1 if x_arr[0] < 0 else 1]
        # Define starting value of y and Vthresh
        vthresh = math.fabs(hyst) if x_arr[0] < 0 else -math.fabs(hyst)
        # Calculate values of y and vthresh for remaining elements of x_arr
        for i in range(1, len(x_arr)):
            x_sgn = -1 if x_arr[i] < vthresh else 1
            vthresh = (1 if x_sgn == y[i-1] else -1)*vthresh
            y.append(x_sgn)
        # Convert to numpy array and return
        return np.array(y)

    @staticmethod
    def find_zcross(sgn_arr, edge='both'):
        # Method to report indices of sgn_arr at which zero crossings occur.
        #
        # Inputs:
        #   sgn_arr = numpy array of values in {-1, 1}, generated by the sign_hyst method.
        #   edge = 'rising', 'falling', or 'both' -- determines which crossing type is
        #   reported.
        #
        y = np.diff(sgn_arr)
        if edge == 'rising':
            return np.argwhere(y == 2)
        elif edge == 'falling':
            return np.argwhere(y == -2)
        else:
            return np.argwhere(y != 0)

    def calc_seg_disp(self, t_start, delta_t, disp_abs_val=True):
        # Method to produce 1D arrays of x-segment displacements and y-segment displacements,
        # where each segment is delta_t long.
        #
        # Find starting indices for normalized eye-gaze and drone position following end
        # of calibration cycles.
        dr_start_quant = self.find_index_at_time(t_start, self.t_algn_dr)
        eg_start_quant = self.find_index_at_time(t_start, self.t_algn_eg)
        # Calculate number of segments
        num_segs = math.floor((min(self.t_algn_dr[-1], self.t_algn_eg[-1]) - t_start) / delta_t)
        # Generate arrays of delta_x and delta_y for drone movement and eye-gaze
        dx_dr = []
        dy_dr = []
        dx_eg = []
        dy_eg = []
        seg_indices = []
        for i in range(0, num_segs):
            dr_seg_start = self.find_index_at_time(t_start + i * delta_t, self.t_algn_dr)
            dr_seg_end = self.find_index_at_time(t_start + (i + 1) * delta_t, self.t_algn_dr)
            eg_seg_start = self.find_index_at_time(t_start + i * delta_t, self.t_algn_eg)
            eg_seg_end = self.find_index_at_time(t_start + (i + 1) * delta_t, self.t_algn_eg)
            seg_indices.append([dr_seg_start, eg_seg_start])
            # Append segment x and displacements to corresponding arrays
            if disp_abs_val:
                dx_dr.append(math.fabs(self.pos_norm_dr[dr_seg_end, 0] - self.pos_norm_dr[dr_seg_start, 0]))
                dy_dr.append(math.fabs(self.pos_norm_dr[dr_seg_end, 1] - self.pos_norm_dr[dr_seg_start, 1]))
                dx_eg.append(math.fabs(self.pos_norm_eg[eg_seg_end, 0] - self.pos_norm_eg[eg_seg_start, 0]))
                dy_eg.append(math.fabs(self.pos_norm_eg[eg_seg_end, 1] - self.pos_norm_eg[eg_seg_start, 1]))
            else:
                dx_dr.append(self.pos_norm_dr[dr_seg_end, 0] - self.pos_norm_dr[dr_seg_start, 0])
                dy_dr.append(self.pos_norm_dr[dr_seg_end, 1] - self.pos_norm_dr[dr_seg_start, 1])
                dx_eg.append(self.pos_norm_eg[eg_seg_end, 0] - self.pos_norm_eg[eg_seg_start, 0])
                dy_eg.append(self.pos_norm_eg[eg_seg_end, 1] - self.pos_norm_eg[eg_seg_start, 1])
        # Return arrays of segment displacements
        return [dx_dr, dy_dr, dx_eg, dy_eg, seg_indices]

    def calc_opt_partition(self, nbits):
        """
        Method to calculate the optimal quantization partitions for drone X, drone Y, eye-gaze X, and eye-gaze Y
        independently.
        Args:
            nbits: Number of bits of quantization for each threshold
        Returns:
            partition: list of partition thresholds, with 2^nbits - 1 thresholds each for drone X, Y, eye-gaze X, Y
        """
        drone_x_partition = EGutil.partition_calc(self.seg_disp[0], nbits)
        drone_y_partition = EGutil.partition_calc(self.seg_disp[1], nbits)
        eg_x_partition = EGutil.partition_calc(self.seg_disp[2], nbits)
        eg_y_partition = EGutil.partition_calc(self.seg_disp[3], nbits)
        partition = drone_x_partition.copy()
        partition.extend(drone_y_partition)
        partition.extend(eg_x_partition)
        partition.extend(eg_y_partition)
        # Return list of all partition values [drone_x, drone_y, eg_x, eg_y]
        return partition

    def key_gen(self, partition, code):
        # Method to quantize dx and dy segment lengths, concatenating the values for form binary keys.
        #
        # partition will have 4*(2^n - 1) thresholds (2^n -1 for drone_dx, 2^n - 1 for drone_dy,
        # 2^n - 1 for eye-gaze_dx, and 2^n - 1 for eye_gaze_dy)
        # for n-bit quantization
        nthresh = len(partition)//4
        [dx_dr, dy_dr, dx_eg, dy_eg, seg_indices] = self.seg_disp

        # Quantize drone dx segments; append to drone key
        if nthresh == 1:
            dx_dr_q = np.digitize(dx_dr, np.array([partition[0]]))
        else:
            dx_dr_q = np.digitize(dx_dr, np.array(partition[0:nthresh-1]))
        key_dr = ''
        for i in range(0, len(dx_dr_q)):
            if nthresh == 1:
                key_dr += "{0:1b}".format(code[dx_dr_q[i] - 1])
            else:
                key_dr += "{0:02b}".format(code[dx_dr_q[i] - 1])
        # Quantize drone dy segments; append to drone key
        if nthresh == 1:
            dy_dr_q = np.digitize(dy_dr, np.array([partition[1]]))
        else:
            dy_dr_q = np.digitize(dy_dr, np.array(partition[nthresh:2*nthresh - 1]))
        for i in range(0, len(dy_dr_q)):
            if nthresh == 1:
                key_dr += "{0:1b}".format(code[dy_dr_q[i] - 1])
            else:
                key_dr += "{0:02b}".format(code[dy_dr_q[i] - 1])

        # Quantize eye-gaze dx segments; append to eye-gaze key
        if nthresh == 1:
            dx_eg_q = np.digitize(dx_eg, np.array([partition[2]]), code)
        else:
            dx_eg_q = np.digitize(dx_eg, np.array(partition[2*nthresh : 3*nthresh - 1]), code)
        key_eg = ''
        for i in range(0, len(dx_eg_q)):
            if nthresh == 1:
                key_eg += "{0:1b}".format(code[dx_eg_q[i] - 1])
            else:
                key_eg += "{0:02b}".format(code[dx_eg_q[i] - 1])
        # Quantize eye-gaze dy segments; append to eye-gaze key
        if nthresh == 1:
            dy_eg_q = np.digitize(dy_eg, np.array([partition[3]]), code)
        else:
            dy_eg_q = np.digitize(dy_eg, np.array(partition[3*nthresh : 4*nthresh - 1]), code)
        for i in range(0, len(dy_eg_q)):
            if nthresh == 1:
                key_eg += "{0:1b}".format(code[dy_eg_q[i] - 1])
            else:
                key_eg += "{0:02b}".format(code[dy_eg_q[i] - 1])
        # Return keys
        return [key_dr, key_eg]


def main():
    eg_file_path = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Data Sets\\CoDrone\\'
                    'DroneTracker3\\')
    # eg_file_name = 'eye_tracker_03242024_121300.csv'
    # eg_file_name = 'eye_tracker_03242024_122057.csv'
    # eg_file_name = 'eye_tracker_03242024_122440.csv'
    # eg_file_name = 'eye_tracker_03242024_122605.csv'
    # eg_file_name = 'eye_tracker_04082024_211307.csv'
    # eg_file_name = 'eye_tracker_04082024_211540.csv'
    eg_file_name = 'eye_tracker_04082024_211754.csv'
    eg_chop_start = 6.1
    # eg_chop_start = 0.0
    eg_chop_end = 59.0
    drone_file_path = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Data Sets\\CoDrone\\'
                       'Drone Flight Path\\')
    # drone_file_name = 'drone_path03242024_121322.csv'
    # drone_file_name = 'drone_path03242024_122102.csv'
    # drone_file_name = 'drone_path03242024_122449.csv'
    # drone_file_name = 'drone_path03242024_122626.csv'
    # drone_file_name = 'drone_path04082024_211353.csv'
    # drone_file_name = 'drone_path04082024_211609.csv'
    drone_file_name = 'drone_path04082024_211814.csv'
    test1 = DroneEyeMatch(eg_file_path + eg_file_name, drone_file_path + drone_file_name, 'Random_Limits',
                          eg_chop_start, eg_chop_end)

    print('\nDrone data file:', test1.drone_file)
    print('Eye-gaze data file:', test1.eye_gaze_file)

    # Test print_eg_at_index() and get_eg_at_index() methods
    print(f'Eye-Gaze App Start: {test1.eg_date.strftime("%m-%d-%Y %H:%M:%S")}')
    print(f'Eye-Gaze Collection Start: {test1.eg_start_time.strftime("%H:%M:%S.%f")}')
    test1.print_eg_at_index(0)

    # Test print_drone_at_index() and get_drone_at_index() methods
    print('\nDrone values')
    test1.print_drone_at_index(0)

    # Print out indexes and time values to align drone and eye-gaze
    print('\nDrone alignment index = {0}\ttime = {1:.3f}'.format(test1.drone_alignment_index,
                                                                 test1.drone_time[test1.drone_alignment_index]))
    print('Eye-Gaze alignment index = {0}\ttime = {1:.3f}'.format(test1.eg_alignment_index,
                                                                  test1.eg_time[test1.eg_alignment_index]))

    # Find and print eye-gaze (X, Y) and drone (X, Y, Z) starting points
    # eg_xyz_start, drone_xyz_start = test1.find_locus_start_points(test1.eg_alignment_index)
    print('\nEye-gaze start point, x={0:.3f}\ty={1:.3f}'.format(test1.eg_xyz_start[0], test1.eg_xyz_start[1]))
    print('Drone start point, x={0:.3f}\ty={1:.3f}\tz={2:.3f}'.
          format(test1.drone_start_pos[0], test1.drone_start_pos[1], test1.drone_start_pos[2]))

    # Print out segment displacement array
    [dx_dr, dy_dr, dx_eg, dy_eg, seg_indices] = test1.seg_disp
    file_out = 'segment_displacements.csv'
    print('\nWriting segment displacements to .csv ...')
    with open(file_out, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Segment Num', 'Drone(dx)', 'Eye-Gaze(dx)', 'Drone(dy)', 'Eye-Gaze(dy)',
                         'Drone Time(s)', 'Eye-Gaze Time(s)'])
        for i in range(0, len(dx_dr)):
            writer.writerow([i, dx_dr[i], dx_eg[i], dy_dr[i], dy_eg[i], test1.t_algn_dr[seg_indices[i][0]],
                             test1.t_algn_eg[seg_indices[i][1]]])

    # Calculate optimum x and y partitions for drone and eye-gaze
    drone_x_partition = EGutil.partition_calc(test1.seg_disp[0], 2)
    drone_y_partition = EGutil.partition_calc(test1.seg_disp[1], 2)
    eg_x_partition = EGutil.partition_calc(test1.seg_disp[2], 2)
    eg_y_partition = EGutil.partition_calc(test1.seg_disp[3], 2)
    print('\nOptimal partitions')
    print('\tDrone X: ', drone_x_partition)
    print('\tDrone Y: ', drone_y_partition)
    print('\tEye-Gaze X: ', eg_x_partition)
    print('\tEye-Gaze Y: ', eg_y_partition)

    # Print out keys
    print('Drone key    = {0}'.format(test1.key_dr))
    print('Eye-gaze key = {0}'.format(test1.key_eg))
    bit_count = 0
    bit_diff_count = 0
    ones_count_dr = 0
    ones_count_eg = 0
    for bit in range(0, len(test1.key_dr)):
        bit_count += 1
        if (test1.key_dr[bit] != test1.key_eg[bit]):
            bit_diff_count += 1
        if test1.key_dr[bit] == '1':
            ones_count_dr += 1
        if test1.key_eg[bit] == '1':
            ones_count_eg += 1
    print('{0} bits are different out of {1} bits'.format(bit_diff_count, bit_count))
    print("The drone key has {0} 1's; the eye-gaze key has {1} 1's".format(ones_count_dr, ones_count_eg))
    print('The drone key has {0} bits of entropy'.format(EGutil.entropy_calc(ones_count_dr, bit_count)))
    print('The eye-gaze key has {0} bits of entropy'.format(EGutil.entropy_calc(ones_count_eg, bit_count)))

    # Write drone_time and eye-gaze time to separate .csv file
    # filename_dr_out = 'drone_out.csv'
    # filename_eg_out = 'eg_out.csv'
    # with open(filename_dr_out, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for i in range(0, len(test1.drone_time)):
    #         writer.writerow([i, test1.drone_time[i]])
    # with open(filename_eg_out, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     for i in range(0, len(test1.eg_time)):
    #         writer.writerow([i, test1.eg_time[i]])

    ##########################################################
    # Test plotting helper methods
    # Start eye-gaze plotting at index whose time >= the drone center location
    plot_start = np.argwhere(test1.eg_time >= 0.0)[0][0]
    # plot_start = 0
    # print(f'\nplot_start = {plot_start}')

    # Figure 1 -- Eye-Gaze Hitpoint locus
    fig1, axes1 = plt.subplots()
    param_dict = {'title': 'Eye-Gaze Hitpoint Locus', 'xlabel': 'X (m)', 'ylabel': 'Y (m)', 'marker': 'b+'}
    EGplt.plot_locus_2d(axes1, test1.eg_hitpoint[plot_start:, 0], test1.eg_hitpoint[plot_start:, 1], param_dict)

    # Figure 3 -- Drone XY plane locus
    fig3, axes3 = plt.subplots()
    param_dict = {'title': 'Drone XY-Plane Locus', 'xlabel': 'X (m)', 'ylabel': 'Y (m)', 'marker': 'r+'}
    EGplt.plot_locus_2d(axes3, test1.drone_pos[:, 0], test1.drone_pos[:,1], param_dict)

    # Figure 4 -- Drone XYZ locus
    # fig4 = plt.figure()
    # EGplt.plot_3d(fig4, test1.drone_pos[:, 0], test1.drone_pos[:, 2], test1.drone_pos[:, 1])

    # Figure 5 -- Eye-Gaze Hitpoint X, Y vs. Time
    fig5, axes5 = plt.subplots()
    param_dict = {'title': 'Eye-Gaze X and Y vs. Time', 'xlabel': 'eg_time (s)', 'ylabel': 'X, Y (m)',
                  'legends': ['X', 'Y']}
    EGplt.plot_multiline(axes5, [test1.eg_hitpoint[plot_start:, 0], test1.eg_hitpoint[plot_start:, 1]],
                         [test1.eg_time[plot_start:], test1.eg_time[plot_start:]], param_dict)

    # Plot drone X velocity vs. time
    fig7, axes7 = plt.subplots()
    param_dict = {'title': 'Drone X and and dX/dt vs. Time', 'xlabel': 'Time (s)', 'ylabel': 'X (m) / dX/dt (m/s)',
                  'legends': ['Drone X', 'Drone X Velocity']}
    EGplt.plot_multiline(axes7, [test1.pos_algn_dr[:, 0], test1.drone_x_vel[:, 1]],
                         [test1.t_algn_dr, test1.drone_x_vel[:, 0]], param_dict)
    fig7a, axes7a = plt.subplots()
    param_dict = {'title': 'Eye-Gaze X and and dX/dt vs. Time', 'xlabel': 'Time (s)', 'ylabel': 'X (m) / dX/dt (m/s)',
                  'legends': ['Eye-Gaze X', 'Eye-Gaze X Velocity']}
    EGplt.plot_multiline(axes7a, [test1.pos_algn_eg[:, 0], test1.eg_x_vel[:, 1]],
                         [test1.t_algn_eg, test1.eg_x_vel[:, 0]], param_dict)

    # Plot drone X position and eye-gaze X position vs. time, and drone Y and eye-gaze Y, with time aligned
    # Use set_xlim method to set x-axis limit to smaller end time of drone and eye-gaze data.
    fig8, (axes8a, axes8b) = plt.subplots(2, 1)
    param_dict = {'title': 'Drone Y and Eye-Gaze Y vs. Time', 'xlabel': '', 'ylabel': 'Y (m)',
                  'legends': ['Drone Y', 'Eye-Gaze Y']}
    EGplt.plot_multiline(axes8a, [test1.pos_norm_dr[:, 1], test1.pos_norm_eg[:, 1]],
                         [test1.t_algn_dr, test1.t_algn_eg], param_dict)
    axes8a.set_xlim([0, min(test1.t_algn_dr[-1], test1.t_algn_eg[-1])])
    param_dict = {'title': 'Drone X and Eye-Gaze X vs. Time', 'xlabel': 'Time (s)', 'ylabel': 'X (m)',
                  'legends': ['Drone X', 'Eye-Gaze X']}
    EGplt.plot_multiline(axes8b, [test1.pos_norm_dr[:, 0], test1.pos_norm_eg[:, 0]],
                         [test1.t_algn_dr, test1.t_algn_eg], param_dict)
    axes8b.set_xlim([0, min(test1.t_algn_dr[-1], test1.t_algn_eg[-1])])

    plt.show()


# Main code
if __name__ == '__main__':
    main()

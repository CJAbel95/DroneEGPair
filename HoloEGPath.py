"""
    HoloEGPath -- Class that contains a sampled drone-tracking eye gaze time series,
            and methods to pre-process and clean the data in preparation for pairing with
            drone flight path time series.

    Description ...

    by Christopher Abel
    Revision History
    ----------------
    07/27/2024 -- Original
"""
import csv
import datetime
import math
from matplotlib import pyplot as plt
import numpy as np
import scipy
import EGCustomPlots as EGplt
import EyeMatchUtils as EGutil


class HoloEGPath:
    # Default constructor
    def __init__(self, eye_gaze_file, eg_start, eg_end, rem_outliers=True, interp='Akima', tstep=0.02, lowpass=True):
        self.eye_gaze_file = eye_gaze_file
        # Read data from eye-gaze and drone files into arrays
        out_lev = 0.1
        [self.eg_date, self.start_time, self.eg_time, self.hitpoint, self.direction,
         self.origin] = read_eg_file(self.eye_gaze_file, chop_start=eg_start, chop_end=eg_end)
        #
        # If rem_outliers = True, then use rem_outliers() function to remove outlier values from hitpoint,
        #   direction, and origin.
        eg_time_outliers = []
        if rem_outliers:
            eg_data = np.concatenate((self.hitpoint, self.direction, self.origin), 1)
            self.eg_time, eg_data, eg_time_outliers = EGutil.rem_outliers(self.eg_time, eg_data, out_lev, 3)
            self.hitpoint = eg_data[:, 0:3]
            self.direction = eg_data[:, 3:6]
            self.origin = eg_data[:, 6:9]
        #
        # Interpolate and sample at a different sampling period if interp is unequal to 'None'.
        #
        if not interp == 'None':
            npts = math.floor((self.eg_time[-1] - self.eg_time[0]) / tstep) + 1
            t_interp = np.linspace(self.eg_time[0], self.eg_time[0] + (npts - 1) * tstep, npts)
            hitp_interp = EGutil.interp_1d_array(t_interp, self.eg_time, self.hitpoint, interp)
            self.eg_time = t_interp
            self.hitpoint = hitp_interp
        #
        # Lowpass filter dataset. Use Savitzky-Golay filter of order sg_ord and window length = sg_len
        # to filter eg_hitpoint dataset.
        #
        if lowpass:
            sg_ord = 0
            sg_len = 25
            self.hitpoint[:, 0] = scipy.signal.savgol_filter(self.hitpoint[:, 0], sg_len, sg_ord)
            self.hitpoint[:, 1] = scipy.signal.savgol_filter(self.hitpoint[:, 1], sg_len, sg_ord)
            self.hitpoint[:, 2] = scipy.signal.savgol_filter(self.hitpoint[:, 2], sg_len, sg_ord)


def read_eg_file(eg_file, chop_start=0.0, chop_end=1000.0):
    #
    # Read eye-gaze data file, load samples into np.arrays.
    #
    with open(eg_file, 'r', newline='') as file:
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
    #
    eg_start_time = date_time[start]
    eg_time = np.array(t[start:end], dtype=np.float64) - float(t[start])
    eg_hitpoint = np.array(hitpoint[start:end], dtype=np.float64)
    eg_direction = np.array(direction[start:end], dtype=np.float64)
    eg_origin = np.array(origin[start:end], dtype=np.float64)
    return [eg_date, eg_start_time, eg_time, eg_hitpoint, eg_direction, eg_origin]


def main():
    # eg_file_path = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research'
    #                 '\\Data Sets\\DJI Drone\\DroneTracker3\\080624\\')
    eg_file_path = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research'
                    '\\Data Sets\\DJI Drone\\DroneTracker3\\')
    filepth_date = '082624\\'
    # eg_file_name = 'eye_tracker_07232024_192335.csv'
    # eg_file_name = 'eye_tracker_08062024_133856.csv'
    # eg_file_name = 'eye_tracker_08062024_134429.csv'
    # eg_file_name = 'eye_tracker_08062024_134813.csv'
    # eg_file_name = 'eye_tracker_08102024_105325.csv'
    # eg_file_name = 'eye_tracker_08102024_105711.csv'
    # eg_file_name = 'eye_tracker_08102024_110205.csv'
    # eg_file_name = 'eye_tracker_08102024_110533.csv'
    # eg_file_name = 'eye_tracker_08112024_075955.csv'
    # eg_file_name = 'eye_tracker_08112024_080435.csv'
    # eg_file_name = 'eye_tracker_08112024_080826.csv'
    # eg_file_name = 'eye_tracker_08112024_081230.csv'
    # eg_file_name = 'eye_tracker_08112024_200316.csv'
    # eg_file_name = 'eye_tracker_08112024_200623.csv'
    # eg_file_name = 'eye_tracker_08112024_200946.csv'
    # eg_file_name = 'eye_tracker_08112024_201319.csv'
    # eg_file_name = 'eye_tracker_08122024_092630.csv'
    # eg_file_name = 'eye_tracker_08122024_092948.csv'
    # eg_file_name = 'eye_tracker_08122024_093349.csv'
    # eg_file_name = 'eye_tracker_08122024_093720.csv'
    # eg_file_name = 'eye_tracker_08122024_195210.csv'
    # eg_file_name = 'eye_tracker_08122024_195528.csv'
    # eg_file_name = 'eye_tracker_08122024_195859.csv'
    # eg_file_name = 'eye_tracker_08122024_200239.csv'
    # eg_file_name = 'eye_tracker_08242024_193358.csv'
    # eg_file_name = 'eye_tracker_08242024_193517.csv'
    # eg_file_name = 'eye_tracker_08242024_193625.csv'
    # eg_file_name = 'eye_tracker_08262024_070838.csv'
    eg_file_name = 'eye_tracker_08262024_150602.csv'
    eg_start = 0.0
    eg_end = 160.0
    rem_outliers = False
    interp = 'None'
    lowpass = False
    eg_test1 = HoloEGPath(eg_file_path + filepth_date + eg_file_name, eg_start, eg_end, rem_outliers=rem_outliers, interp=interp, lowpass=lowpass)

    # Figure 1 -- Eye-Gaze Hitpoint X and Y vs. Time
    plot_start = 0
    fig1, axes1 = plt.subplots()
    param_dict = {'title': 'Eye-Gaze X and Y vs. Time', 'xlabel': 'eg_time (s)', 'ylabel': 'X, Y (m)',
                  'legends': ['X', 'Y']}
    EGplt.plot_multiline(axes1, [eg_test1.hitpoint[plot_start:, 0], eg_test1.hitpoint[plot_start:, 1]],
                         [eg_test1.eg_time[plot_start:], eg_test1.eg_time[plot_start:]], param_dict)
    axes1.grid(visible=True, which='both', axis='both')

    plt.show()


# Main code
if __name__ == '__main__':
    main()
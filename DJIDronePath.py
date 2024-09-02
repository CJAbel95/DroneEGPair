"""
    DJIDronePath -- Class that contains a sampled DJI drone flight path time series,
            and methods to pre-process and clean the data in preparation for pairing with
            eye-gaze time series.

    Description ...

    by Christopher Abel
    Revision History
    ----------------
    07/23/2024 -- Original
    08/13/2024 -- Store dist_tako (distance from takeoff to pattern start) and speedlev as attributes.

"""
import csv
import math
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import scipy

import EGCustomPlots as EGplt
import EyeMatchUtils as EGutil


class DJIDronePath:
    # Constructor
    def __init__(self, drone_file, interp='Akima', tstep= 0.2, drone_type='DJI Mini 3 Pro', lowpass=False):
        self.drone_type = drone_type
        self.drone_file = drone_file
        #
        # Read raw data from drone file
        #
        [self.start_time, self.takeoff_lat, self.takeoff_long, sidx, snum,
                timee, timed, alt, latit, longit, self.dist_tako, self.speedlev] = read_dji_file(self.drone_file)
        #
        # Remove samples with duplicate positions
        [sidxr, snumr, timeer, altr, latitr, longitr] = rem_duplicates([sidx, snum, timee, alt, latit, longit])
        #
        # Rotate longitutde and latitude axes to form X and Z axes relative to drone perspective.
        self.dji_convtoxyz(snumr, timeer, latitr, longitr, altr)
        #
        # Interpolate, and optionally lowpass filter, flight path to fixed time step, tstep
        # [self.t_interp, self.x, self.y, self.z] = xyz_interp_and_filt(timeer, self.x, self.y, self.z,
        #                                                               interp=interp, lowpass=False)
        npts = math.floor((timeer[-1] - timeer[0]) / tstep) + 1
        self.t_interp = np.linspace(timeer[0], timeer[0] + (npts - 1) * tstep, npts) if (not interp == 'None') else timeer
        if not interp == 'None':
            self.x = EGutil.interp_1d_array(self.t_interp, timeer, self.x, interp)
            self.y = EGutil.interp_1d_array(self.t_interp, timeer, self.y, interp)
            self.z = EGutil.interp_1d_array(self.t_interp, timeer, self.z, interp)
        if (not interp == 'None') and lowpass:
            self.x = scipy.signal.savgol_filter(self.x, 5, 0)
            self.y = scipy.signal.savgol_filter(self.y, 5, 0)
            self.z = scipy.signal.savgol_filter(self.z, 5, 0)

    def dji_convtoxyz(self, snum, timeer, latitr, longitr, altr):
        """
        dji_convToXYZ -- Method to convert the latitude, longitude, and altitude coordinates of a DJI drone into
                X, Y, and Z coordinates, where +Z points in the forward direction from the front of the drone,
                and +X points toward the right. +Y points upward.
        """
        #
        # Find indices corresponding to endpoints of flight segments 1, 2, and 3 -- the segments
        # which we know consist only of movement in the +/- x directions.
        # Calculate angle formed by longitude / latitude in each segment, and average the three
        # to obtain the estimated angle of rotation.
        self.theta = 0.0
        for seg in [1, 2, 3]:
            seg_start = int(np.where(snum == seg)[0][0])
            seg_end = int(np.where( snum == seg)[0][-1])
            # Segments 1 and 3 go in -x direction; segment 2 in positive x
            delta_longit = ((-1) ** seg) * (longitr[seg_end] - longitr[seg_start])
            delta_latit = ((-1) ** seg) * (latitr[seg_end] - latitr[seg_start])
            theta_seg = math.atan2(delta_latit, delta_longit)
            if theta_seg < -0.75 * math.pi:
                theta_seg += 2.0 * math.pi # If angle is close to -pi, then add 2*pi
            self.theta += theta_seg
        self.theta /= 3
        # print(f'Angle of rotation = {180 / math.pi * theta} degrees')
        #
        # Create X and Z time series from longitude and latitude. Create Y time series directly
        # from altitude.
        self.x = math.cos(self.theta) * longitr + math.sin(self.theta) * latitr
        self.z = math.cos(self.theta) * latitr - math.sin(self.theta) * longitr
        self.y = altr


def read_dji_file(filename):
    """
    read_dji_file -- Read .csv file containing recorded drone flight path vs.
                time. Return arrays of time, altitude, latitude, and longitude.

    :param filename: Path and name of .csv file containing the time series for
                    the drone flight path.
    :return: list consisting of
            start_time
            takeoff_lat, takeoff_long
            samp_index
            segment_num
            time_elapsed
            time_dateobj
            altitude
            delta_lat, delta_long
    """
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        #
        # Capture datetime and position at takeoff and start of test
        #
        takeoff = next(reader)  # Capture latitude and longitude at takeoff
        takeoff_lat = float(takeoff[5])
        takeoff_long = float(takeoff[7])
        start_pos = next(reader) # 2nd line contains starting time, altitude, latitude and longitude
        start_time = datetime.strptime(start_pos[2], ' %m/%d/%Y %H:%M:%S')
        start_lat = float(start_pos[6])
        start_long = float(start_pos[8])
        speedlevel = float(start_pos[10])
        start_height = float(start_pos[4])
        lat_shift = 111111.0 * (start_lat - takeoff_lat)
        long_shift = 111111.0 * (start_long - takeoff_long)
        dist_from_takeoff = np.sqrt(lat_shift ** 2 + long_shift ** 2)
        #
        # Read each additional line, capturing sample index, elapsed time, elapsed time
        # from Date object, altitude, latitude, longitude, and segment index in lists.
        #
        samp = []
        t_elapsed = []
        t_date = []
        dalt = []
        dlat = []
        dlong = []
        seg_num = []
        for row in reader:
            samp.append(int(row[1]))
            t_elapsed.append(float(row[2]))
            # Capture datetime from first sample row
            if (int(row[1]) == 0): datetime_0 = datetime.strptime(row[3], ' %m/%d/%Y %H:%M:%S.%f')
            # Subtract initial datetime from each subsequent datetime value
            t_date.append((datetime.strptime(row[3], ' %m/%d/%Y %H:%M:%S.%f')
                           - datetime_0).total_seconds())
            dalt.append(float(row[4]))
            dlat.append(float(row[5]))
            dlong.append(float(row[6]))
            seg_num.append(int(row[7]))
        #
        # Convert lists to numpy arrays.
        #   time_elapsed = array of elapsed time obtained from the Kotlin TimeMark.elapsedNow() method
        #   time_dateobj = array of elapsed time obtained from Kotlin Date class
        #   altitude = drone height in (m)
        #   delta_lat = change in drone latitude relative to time = 0 in (m)
        #   delta_long = change in drone longitude relative to time = 0 in (m)
        #
        samp_index = np.array(samp, dtype=np.int64)
        time_elapsed = np.array(t_elapsed, dtype=np.float64) - t_elapsed[0]
        time_dateobj = np.array(t_date, dtype=np.float32)
        altitude = np.array(dalt, dtype=np.float32)
        delta_lat = 111111.0 * (np.array(dlat, dtype=np.float64) - dlat[0])
        delta_long = 111111.0 * (np.array(dlong, dtype=np.float64) - dlong[0])
        segment_num = np.array(seg_num, dtype=np.int16)
        #
        # Return list of [start_time, takeoff_lat, takeoff_long, samp_index, segment_num, time_elapsed,
        #   time_dateobj, altitude, delta_lat, delta_long]
        return [start_time, takeoff_lat, takeoff_long, samp_index, segment_num, time_elapsed,
                time_dateobj, altitude, delta_lat, delta_long, dist_from_takeoff, speedlevel]


def rem_duplicates(time_series):
    """
    rem_duplicates --

    :param time_series:
    :return:
    """
    samp_index, segment_num, time_elapsed, altitude, delta_lat, delta_long = time_series
    #
    # Create empty numpy arrays to hold time series arrays after duplicate value removal
    #
    samp_index_red = np.array(samp_index[0], dtype=np.int64)
    segment_num_red = np.array(segment_num[0], dtype=np.int32)
    time_elapsed_red = np.array(time_elapsed[0], dtype=np.float64)
    altitude_red = np.array(altitude[0], dtype=np.float64)
    delta_lat_red = np.array(delta_lat[0], dtype=np.float64)
    delta_long_red = np.array(delta_long[0], dtype=np.float64)
    for i in range(1, samp_index.size):
        if ((abs(altitude[i] - altitude[i-1]) > 1.0e-6) or (abs(delta_lat[i] - delta_lat[i-1]) > 1.0e-6)
                or (abs(delta_long[i] - delta_long[i-1]) > 1.0e-6)):
            samp_index_red = np.append(samp_index_red, samp_index[i])
            segment_num_red = np.append(segment_num_red, segment_num[i])
            time_elapsed_red = np.append(time_elapsed_red, time_elapsed[i])
            altitude_red = np.append(altitude_red, altitude[i])
            delta_lat_red = np.append(delta_lat_red, delta_lat[i])
            delta_long_red = np.append(delta_long_red, delta_long[i])
    #
    # Return list of arrays with duplicate position points eliminated
    #
    return [samp_index_red, segment_num_red, time_elapsed_red, altitude_red,
            delta_lat_red, delta_long_red]


def main():
    filepath = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Data Sets'
                '\\DJI Drone\\Drone Flight Path\\')
    filepth_date = '082624\\'
    # filename = 'randx_patt_08062024_134101.csv'
    # filename = 'randx_patt_08062024_134437.csv'
    # filename = 'randx_patt_08062024_134824.csv'
    # filename = 'randx_patt_08102024_110242.csv'
    # filename = 'randx_patt_08112024_080146.csv'
    # filename = 'randx_patt_08112024_080534.csv'
    # filename = 'randx_patt_08112024_080940.csv'
    # filename = 'randx_patt_08112024_081308.csv'
    # filename = 'randx_patt_08112024_200332.csv'
    # filename = 'randx_patt_08112024_200658.csv'
    # filename = 'randx_patt_08112024_201030.csv'
    # filename = 'randx_patt_08112024_201030.csv'
    # filename = 'randx_patt_08112024_201350.csv'
    # filename = 'randx_patt_08122024_092701.csv'
    # filename = 'randx_patt_08122024_093101.csv'
    # filename = 'randx_patt_08122024_093433.csv'
    # filename = 'randx_patt_08122024_093802.csv'
    # filename = 'randx_patt_08122024_195240.csv'
    # filename = 'randx_patt_08122024_195610.csv'
    # filename = 'randx_patt_08122024_195946.csv'
    # filename = 'randx_patt_08122024_200320.csv'
    # filename = 'randxy_patt_08242024_134014.csv'
    # filename = 'randxy_patt_08242024_193420.csv'
    # filename = 'randxy_patt_08262024_071019.csv'
    filename = 'randxy_patt_08262024_150642.csv'
    print(f'Drone flight path file: {filepath}{filepth_date}{filename}')

    flight1 = DJIDronePath(filepath + filepth_date + filename, 'Akima', 0.2, lowpass=False)
    print(f'\tDrone Z axis rotated {180 / math.pi * flight1.theta:.4f} degrees relative to North.')
    print(f'\tDrone height = {flight1.y[0]:.2f} m\tDistance from Takeoff = {flight1.dist_tako:.3f} m'
          f'\tSpeedlev = {flight1.speedlev:.2f}')


    #
    # Read in data file separately in order to plot latitude and longitude vs. time
    #
    [start_time, takeoff_lat, takeoff_long, sidx, snum,
     timee, timed, alt, latit, longit, dist_tako, speedlev] = read_dji_file(filepath + filepth_date + filename)
    [sidxr, snumr, timeer, altr, latitr, longitr] = rem_duplicates([sidx, snum, timee, alt, latit, longit])
    fig0, (axes0a, axes0b, axes0c) = plt.subplots(3, 1)
    param_dict = {'title': 'Latitude vs. Time', 'xlabel': 't (s)', 'ylabel': 'Delta(Latitude) (m)', 'legends': ['Latitude']}
    EGplt.plot_multiline(axes0a, [latitr], [timeer], param_dict)
    axes0a.grid(visible=True, which='both', axis='both')
    param_dict = {'title': 'Longitude vs. Time', 'xlabel': 't (s)', 'ylabel': 'Delta(Longitude) (m)', 'legends': ['Longitude']}
    EGplt.plot_multiline(axes0b, [longitr], [timeer], param_dict)
    axes0b.grid(visible=True, which='both', axis='both')
    param_dict = {'title': 'Altitude vs. Time', 'xlabel': 't (s)', 'ylabel': 'Altitude (m)', 'legends': ['Altitude']}
    EGplt.plot_multiline(axes0c, [altr], [timeer], param_dict)
    axes0c.grid(visible=True, which='both', axis='both')


    #
    # Plot X, Y, Z vs. time
    #
    fig1, axes1 = plt.subplots()
    param_dict = {'title': 'X and Z vs. Time', 'xlabel': 't (s)', 'ylabel': 'X (m), Z (m)', 'legends': ['X', 'Z']}
    EGplt.plot_multiline(axes1, [flight1.x, flight1.z],
                         [flight1.t_interp, flight1.t_interp], param_dict)
    axes1.grid(visible=True, which='both', axis='both')

    fig2, axes2 = plt.subplots()
    param_dict = {'title': 'Y vs. Time', 'xlabel': 't (s)', 'ylabel': 'Y (m)', 'legends': ['Y']}
    EGplt.plot_multiline(axes2, [flight1.y],[flight1.t_interp], param_dict)
    axes2.grid(visible=True, which='both', axis='both')

    plt.show()


# Main code
if __name__ == '__main__':
    main()
"""
    DJIDronePath -- Class that defines an object that contains a sampled
            DJI drone flight path time series, and methods to pre-process and
            clean the data prior.

    Description ...

    by Christopher Abel
    Revision History
    ----------------
    07/23/2024 -- Original

"""
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import EGCustomPlots as EGplt


class DJIDronePath:
    # Constructor
    def __init__(self):
        self.drone_type = 'DJI Mini 3 Pro'


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
        start_height = float(start_pos[4])
        lat_shift = 111111.0 * (start_lat - takeoff_lat)
        long_shift = 111111.0 * (start_long - takeoff_long)
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
                time_dateobj, altitude, delta_lat, delta_long]


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
    filepath = ('C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research'
                '\\DJI Drones\\Saved Data\\072324\\')
    filename = 'randx_patt_07232024_102440.csv'
    print(f'Drone flight path file: {filepath}{filename}')
    [start_time, takeoff_lat, takeoff_long, samp_index, segment_num,
        time_elapsed, time_dateobj, altitude, delta_lat, delta_long] = read_dji_file(filepath + filename)
    print(f'\tTest Date: {start_time.strftime('%m/%d/%Y')}\tTime: {start_time.strftime('%H:%M:%S')}')
    print(f'\tTotal sample points = {samp_index.size}')

    [samp_index_red, segment_num_red, time_elapsed_red, altitude_red, delta_lat_red, delta_long_red] \
        = rem_duplicates([samp_index, segment_num, time_elapsed, altitude, delta_lat, delta_long])

    #
    # Plot delta_lat and delta_long vs. time
    #
    fig1, (axes1a, axes1b) = plt.subplots(2, 1)
    param_dict = {'title': 'Latitude and Longitude vs. Time', 'xlabel': 't (s)', 'ylabel': 'delta(Latitude), delta(Longitude) (m)', 'legends': ['Latitude', 'Longitude']}
    EGplt.plot_multiline(axes1a,[delta_lat, delta_long],
                         [time_elapsed, time_elapsed], param_dict)
    axes1a.grid(visible=True, which='both', axis='both')
    param_dict = {'title': 'Segment Number vs. Time', 'xlabel': 't (s)',
                  'ylabel': 'Segment Number', 'legends': ['Segment Number']}
    EGplt.plot_multiline(axes1b, [segment_num], [time_elapsed], param_dict)
    axes1b.grid(visible=True, which='both', axis='both')

    #
    # Plot delta_lat and delta_long vs. time after removing duplicates
    #
    fig2, (axes2a, axes2b) = plt.subplots(2, 1)
    param_dict = {'title': 'Latitude and Longitude vs. Time', 'xlabel': 't (s)',
                  'ylabel': 'delta(Latitude), delta(Longitude) (m)', 'legends': ['Latitude', 'Longitude']}
    EGplt.plot_multiline(axes2a, [delta_lat_red, delta_long_red],
                         [time_elapsed_red, time_elapsed_red], param_dict)
    axes2a.grid(visible=True, which='both', axis='both')
    param_dict = {'title': 'Segment Number vs. Time', 'xlabel': 't (s)',
                  'ylabel': 'Segment Number', 'legends': ['Segment Number']}
    EGplt.plot_multiline(axes2b, [segment_num_red], [time_elapsed_red], param_dict)
    axes2b.grid(visible=True, which='both', axis='both')

    plt.show()


# Main code
if __name__ == '__main__':
    main()
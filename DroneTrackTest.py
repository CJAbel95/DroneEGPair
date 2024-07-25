#
# DroneTrackTest.py -- Brief Description
# Description -- Details..
#
# by Christopher Abel
# Revision History
# ----------------
# 2/5/2024
#
# -------------------------------------------------
import csv
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt


class DroneTrackTest:
    # Constructor for drone tracking test object
    def __init__(self, param_file, data_file, chop_init_final=True):
        # Read test parameters from parameter file, and assign to attributes
        test_params = DroneTrackTest.read_param_file(param_file)
        self.parameter_file = param_file
        self.data_file = data_file
        self.program_name = test_params[0]
        self.test_date = test_params[1]
        self.zval = test_params[2]
        self.yval = test_params[3]
        self.xylim = test_params[4]
        self.xyvel = test_params[5]
        # Read data from data file, and assign to attributes
        data_list = DroneTrackTest.read_data_file(data_file)
        self.num_records_pre = len(data_list[0])
        #
        # If chop_init_final is True, then chop off the first
        # 10% and last 10% of the input data in order to eliminate
        # transient effects from calculations.
        #
        if chop_init_final:
            start = math.ceil(0.1*len(data_list[0]))
            end = math.floor(0.9*len(data_list[0])) + 1
        else:
            start = 0
            end = len(data_list)
        self.sample_time = data_list[0][start:end]
        self.frame_count = data_list[1][start:end]
        self.eye_gaze = data_list[2][start:end]
        self.drone_pos = data_list[3][start:end]
        self.eye_gaze_dir = data_list[4][start:end]
        self.eye_gaze_origin = data_list[5][start:end]
        self.num_records_post = len(self.sample_time)

    def print_test_params(self):
        # Method that prints out test parameters
        print("DroneTrackTest -- Test Parameters:")
        print("\tParameter file: {0}\n\tData File:{1}".format(self.parameter_file, self.data_file))
        print('\tTest date: {0}'.format(self.test_date.isoformat(' ')), end=', ')
        print('\tZ = {0:.2f}m, Ycenter = {1:.2f}m, XY limt = +/- {2:.2f}m, Velocity = {3:.2f}m/s'
              .format(self.zval, self.yval, self.xylim, self.xyvel))
        eye_gaze_origin_avg = [np.mean([row[0] for row in self.eye_gaze_origin]), np.mean([row[1] for row in self.eye_gaze_origin]),
                               np.mean([row[2] for row in self.eye_gaze_origin])]
        print('\tAvg. eye-gaze origin={0}'.format(eye_gaze_origin_avg))

    #
    # Method to read the parameter file, and return a list containing the
    # date/time of the test, and the drone movement parameters.
    #
    @staticmethod
    def read_param_file(param_file):
        with open(param_file, newline="") as file:
            #
            # First row of file contains program name, date, and time
            #
            row1 = file.readline()
            row1_elements = row1.split(',')
            program_name = row1_elements[0]
            month = int(row1_elements[1].split('/')[0])
            day = int(row1_elements[1].split('/')[1])
            year = int(row1_elements[1].split('/')[2])
            hour = int(row1_elements[2].split(':')[0])
            minute = int(row1_elements[2].split(':')[1])
            seconds = int(row1_elements[2].split(':')[2].split(".")[0])
            test_date = datetime.datetime(year, month, day, hour, minute, seconds)
            #
            # Second row contains the drone Z value.
            # Third row contains the y center value.
            # Fourth row contains the XY displacement limit.
            # Fifth value contains the velocity.
            #
            row2 = file.readline()
            zval = float(row2.rstrip().split(' ')[2][0:-1])
            row3 = file.readline()
            yval = float(row3.rstrip().split(' ')[2][0:-1])
            row4 = file.readline()
            xylim = float(row4.rstrip().split(' ')[2][0:-1])
            row5 = file.readline()
            xyvel = float(row5.rstrip().split(' ')[2][0:-4])
        #
        # Return list containing test datetime and the parameter values
        #
        return [program_name, test_date, zval, yval, xylim, xyvel]

    #
    # Method to read data from the eye_tracker.csv file
    #
    @staticmethod
    def read_data_file(data_file):
        with open(data_file, newline='') as file:
            reader = csv.reader(file)
            skip = -2
            record_count = 0
            time = []
            frame_count = []
            eye_gaze = []
            drone_pos = []
            eye_gaze_dir = []
            eye_gaze_origin = []
            for row in reader:
                # Skip header rows
                if skip < 0:
                    skip = skip + 1
                else:
                    record_count = record_count + 1
                    time.append(float(row[3]))
                    frame_count.append(int(row[5]))
                    eye_gaze.append([float(row[6]), float(row[7]), float(row[8])])
                    drone_pos.append([float(row[9]), float(row[10]), float(row[11])])
                    eye_gaze_dir.append([float(row[12]), float(row[13]), float(row[14])])
                    eye_gaze_origin.append([float(row[15]), float(row[16]), float(row[17])])
        return [time, frame_count, eye_gaze, drone_pos, eye_gaze_dir, eye_gaze_origin]

    #
    # Methods to return specified elements in data lists
    #
    def get_time_pt(self, index):
        return self.sample_time[index]

    def get_frame_ct(self, index):
        return self.frame_count[index]

    def get_eye_gaze(self, index):
        return self.eye_gaze[index]

    def get_eye_gaze_dir(self, index):
        return self.eye_gaze_dir[index]

    def get_drone_pos(self, index):
        return self.drone_pos[index]

    def get_eye_gaze_origin(self, index):
        return self.eye_gaze_origin[index]

    def get_record(self, index):
        return [self.get_time_pt(index), self.get_frame_ct(index), self.get_eye_gaze(index),
                self.get_eye_gaze_dir(index), self.get_eye_gaze_origin(index), self.get_drone_pos(index)]

    def print_record(self, index):
        record = self.get_record(index)
        print("Index={0},\n\tTime={1}ms, Frame={2}, Eye-gaze hitpoint={3}, ".format(index, record[0],
                                record[1], record[2]))
        print("\tEye-gaze dir={0}, Eye-gaze origin={1}, Drone pos={2}".format(record[3], record[4], record[5]))

    #
    # Method to calculate center of eye-gaze hitpoint locus on transparent window.
    #       Finds the X and Y coordinates of the center of
    #       a set of eye-gaze hitpoints. Returns the coordinates as a 2-element list.
    #
    #       Calculate center of eye-gaze hitpoints from median.
    #       Calculate effective upper and lower x and y limits of
    #       hitpoint locus from ecdf function.
    #
    def find_eyegaze_center(self):
        eye_gaze_center = [np.median([eye_gaze_sample[0] for eye_gaze_sample in self.eye_gaze]),
                           np.median([eye_gaze_sample[1] for eye_gaze_sample in self.eye_gaze])]

        #
        # Find upper and lower limts of x position of eye-gaze hitpoints.
        # Examine only those points whose y values are within 1 standard deviation of eye-gaze center.
        #
        eye_gaze_stdev_y = np.std([sample[1] for sample in self.eye_gaze], ddof=1)
        eye_gaze_x_central_y = []
        for eye_gaze_sample in self.eye_gaze:
            if -eye_gaze_stdev_y < eye_gaze_sample[1] - eye_gaze_center[1] < eye_gaze_stdev_y:
                eye_gaze_x_central_y.append(eye_gaze_sample[0])
        eye_gaze_x_central_y.sort()
        x_lower_lim = eye_gaze_x_central_y[int(0.02*len(eye_gaze_x_central_y))]
        x_upper_lim = eye_gaze_x_central_y[int(0.98*len(eye_gaze_x_central_y))]

        #
        # Find upper and lower limts of y position of eye-gaze hitpoints.
        # Examine only those points whose x values are within 1 standard deviation of eye-gaze center.
        #
        eye_gaze_stdev_x = np.std([sample[0] for sample in self.eye_gaze], ddof=1)
        eye_gaze_y_central_x = []
        for eye_gaze_sample in self.eye_gaze:
            if -eye_gaze_stdev_x < eye_gaze_sample[0] - eye_gaze_center[0] < eye_gaze_stdev_x:
                eye_gaze_y_central_x.append(eye_gaze_sample[1])
        eye_gaze_y_central_x.sort()
        y_lower_lim = eye_gaze_y_central_x[int(0.02 * len(eye_gaze_y_central_x))]
        y_upper_lim = eye_gaze_y_central_x[int(0.98 * len(eye_gaze_y_central_x))]

        # Return eye-gaze center and x, y upper and lower limit estimates
        return[eye_gaze_center, x_lower_lim, x_upper_lim, y_lower_lim, y_upper_lim]

    #
    # Method to calculate the vector of eye-gaze hitpoints on Z plane of drone, and
    # corresponding vector of (drone-position - eye_gaze) errors. Return both vectors.
    #
    def eye_gaze_drone_track_error(self):
        eye_gaze_hitpoints = []
        eye_gaze_errors = []
        eye_gaze_errors_mag = []
        # Loop through all records
        for i in range(0, self.num_records_post - 1):
            # Calculate number of eye-gaze direction vectors required to get from Z at eye-gaze origin
            # to Z at drone position.
            multiplier = (self.drone_pos[i][2] - self.eye_gaze_origin[i][2])/self.eye_gaze_dir[i][2]
            # Use this multiplier to find the x and y coordinates of the eye-gaze hitpoint in the
            # drone Z plane.
            eye_gaze_hitpoint_x = self.eye_gaze_origin[i][0] + multiplier*self.eye_gaze_dir[i][0]
            eye_gaze_hitpoint_y = self.eye_gaze_origin[i][1] + multiplier * self.eye_gaze_dir[i][1]
            eye_gaze_hitpoints.append([eye_gaze_hitpoint_x, eye_gaze_hitpoint_y])
            # Calculate the X and Y location errors between the drone position and eye-gaze hitpoint
            eye_gaze_error_x = self.drone_pos[i][0] - eye_gaze_hitpoint_x
            eye_gaze_error_y = self.drone_pos[i][1] - eye_gaze_hitpoint_y
            eye_gaze_errors.append([eye_gaze_error_x, eye_gaze_error_y])
            eye_gaze_errors_mag.append(math.sqrt(eye_gaze_error_x**2 + eye_gaze_error_y**2))
        # Return eye_gaze_hitpoints, eye_gaze_errors, and eye_gaze_errors_mag vectors
        return [eye_gaze_hitpoints, eye_gaze_errors, eye_gaze_errors_mag]

    #
    # Improved method to calculate eye-gaze hitpoint error
    #
    def eye_gaze_drone_track_error2(self):
        # Get drone center and limits
        drone_center_x = 0
        drone_center_y = self.yval
        drone_limit_x = self.xylim
        drone_limit_y = self.xylim

        # Get center and x, y limits of eye-gaze hitpoints on window
        (eye_gaze_center, x_lower_lim, x_upper_lim, y_lower_lim, y_upper_lim) = self.find_eyegaze_center()
        eye_gaze_center_x = eye_gaze_center[0]
        eye_gaze_center_y = eye_gaze_center[1]
        delta_x_lim = np.mean([eye_gaze_center_x - x_lower_lim, x_upper_lim - eye_gaze_center_x])
        delta_y_lim = np.mean([eye_gaze_center_y - y_lower_lim, y_upper_lim - eye_gaze_center_y])

        # Calculate error between eye-gaze hitpoint and drone position.  Measure eye-gaze and drone
        # position relative to eye-gaze center and drone center.
        eye_gaze_hitpoints = []
        eye_gaze_error = []
        eye_gaze_error_mag = []
        for i in range(0, len(self.drone_pos)):
            eye_gaze_hit = [(self.eye_gaze[i][0] - eye_gaze_center_x)*self.xylim/delta_x_lim + 0.0,
                                (self.eye_gaze[i][1] - eye_gaze_center_y)*self.xylim/delta_y_lim + self.yval]
            error_sample = [self.drone_pos[i][0] - eye_gaze_hit[0], self.drone_pos[i][1] - eye_gaze_hit[1]]
            eye_gaze_hitpoints.append(eye_gaze_hit)
            eye_gaze_error.append(error_sample)
            eye_gaze_error_mag.append(math.sqrt(error_sample[0]**2 + error_sample[1]**2))
        return [eye_gaze_hitpoints, eye_gaze_error, eye_gaze_error_mag]

    #
    # Method to calculate root means-square-error of eye-gaze hitpoint, with an option to eliminate outliers
    #
    @staticmethod
    def rmse_calc(error_vector, reject_outliers=False, outlier_level=1.0):
        #print("reject_outliers = " + str(reject_outliers) + " outlier = " + str(outlier_level))
        npts = len(error_vector)
        npts_included = 0
        sum_sq_error = 0
        if not reject_outliers:
            for error in error_vector:
                sum_sq_error = sum_sq_error + error**2
            npts_included = npts
        else:
            for error in error_vector:
                if abs(error) <= outlier_level:
                    npts_included = npts_included + 1
                    sum_sq_error = sum_sq_error + error**2
        mse = sum_sq_error/npts_included
        return math.sqrt(mse)

def main():
    #file_path = "C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Hololens\\Matlab Code\\Selected_Data\\"
    file_path = "D:\\OneDrive\\Cleveland State\\Thesis Research\\Hololens\\Matlab Code\\Selected_Data\\"
    #data_files = [["drone1_parameters_02042024_z10_y1p5.csv", "eye_tracker_02042024_z10_y1p5.csv"]]
    data_files = [["drone1_parameters_02052024_z3_y1p5.csv", "eye_tracker_02052024_z3_y1p5.csv"]]
    drone_test1 = DroneTrackTest(file_path + data_files[0][0], file_path + data_files[0][1])
    drone_test1.print_test_params()
    print()
    drone_test1.print_record(10)
    (eye_gaze_center, x_lower_lim, x_upper_lim, y_lower_lim, y_upper_lim) = drone_test1.find_eyegaze_center()
    print('\nEye-gaze center = ', eye_gaze_center)
    print('X limits = [{0:.5f}, {1:.5f}]\tY limits = [{2:.5f}, {3:.5f}]'.format(x_lower_lim,
                                                                                x_upper_lim, y_lower_lim, y_upper_lim))
    eye_gaze_error_mag = drone_test1.eye_gaze_drone_track_error()[2]
    print('\nRMS Eye-gaze hitpoint error = {0:.3f}\t RMS after removing outliers = {1:.3f}'.
          format(DroneTrackTest.rmse_calc(eye_gaze_error_mag), DroneTrackTest.rmse_calc(eye_gaze_error_mag, True, 0.3)))
    (eye_gaze_hitpoints_droneZ, eye_gaze_error, eye_gaze_error_mag2) = drone_test1.eye_gaze_drone_track_error2()
    print('Using new algorithm: ', end='')
    print('RMS Eye-gaze hitpoint error = {0:.3f}\t RMS after removing outliers = {1:.3f}'.
          format(DroneTrackTest.rmse_calc(eye_gaze_error_mag2), DroneTrackTest.rmse_calc(eye_gaze_error_mag2, True, 0.3)))

    # Plot drone position and corresponding eye-gaze hitpoints estimated at drone plane
    fig1, ax1 = plt.subplots()
    ax1.plot([row[0] for row in drone_test1.drone_pos], [row[1] for row in drone_test1.drone_pos], 'b-',
             [row[0] for row in eye_gaze_hitpoints_droneZ], [row[1] for row in eye_gaze_hitpoints_droneZ], 'r.')
    plt.show()


if __name__ == '__main__':
    main()

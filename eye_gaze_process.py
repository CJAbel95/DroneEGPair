#
# eye_gaze_process.py -- Brief Description
# Description -- Details..
#
# by Christopher Abel
# Revision History
# ----------------
# 
#
# -------------------------------------------------
from DroneTrackTest import DroneTrackTest


def main():
    file_path = "C:\\Users\\abelc\\OneDrive\\Cleveland State\\Thesis Research\\Hololens\\Matlab Code\\Selected_Data\\"
    #file_path = "D:\\OneDrive\\Cleveland State\\Thesis Research\\Hololens\\Matlab Code\\Selected_Data\\"
    data_files = [
        ["drone1_parameters_02052024_z3_y1p5.csv", "eye_tracker_02052024_z3_y1p5.csv"],
        ["drone1_parameters_02052024_z3_y1p5_vel0p6.csv", "eye_tracker_02052024_z3_y1p5_vel0p6.csv"],
        ["drone1_parameters_02052024_z3_y1p5_vel0p8.csv", "eye_tracker_02052024_z3_y1p5_vel0p8.csv"],
        ["drone1_parameters_02042024_z5_y1p5.csv", "eye_tracker_02042024_z5_y1p5.csv"],
        ["drone1_parameters_02052024_z5_y1p5_vel0p6.csv", "eye_tracker_02052024_z5_y1p5_vel0p6.csv"],
        ["drone1_parameters_02052024_z5_y1p5_vel0p8.csv", "eye_tracker_02052024_z5_y1p5_vel0p8.csv"],
        ["drone1_parameters_02042024_z10_y1p5.csv", "eye_tracker_02042024_z10_y1p5.csv"],
        ["drone1_parameters_02052024_z15_y1p5_vel0p4.csv", "eye_tracker_02052024_z15_y1p5_vel0p4.csv"],
        ["drone1_parameters_02042024_z20_y1p5.csv", "eye_tracker_02042024_z20_y1p5.csv"],
        ["drone1_parameters_02132024_z20_y2_xy2.csv", "eye_tracker_02132024_z20_y2_xy2.csv"],
        ["drone1_parameters_02132024_z30_y3_xy3.csv", "eye_tracker_02132024_z30_y3_xy3.csv"],
        ["drone1_parameters_02132024_z50_y3_xy3.csv", "eye_tracker_02132024_z50_y3_xy3.csv"],
        ["drone1_parameters_02262024_z15_xy2_vel0p6.csv", "eye_tracker_02262024_z15_xy2_vel0p6.csv"],
        ["drone1_parameters_02262024_z15_xy2_vel0p6_set2.csv", "eye_tracker_02262024_z15_xy2_vel0p6_set2.csv"],
        ["drone1_parameters_02262024_z20_xy2_vel0p6.csv", "eye_tracker_02262024_z20_xy2_vel0p6.csv"],
        ["drone1_parameters_02262024_z30_xy3_vel0p7.csv", "eye_tracker_02262024_z30_xy3_vel0p7.csv"],
        ["drone1_parameters_02262024_z30_xy3_vel0p8.csv", "eye_tracker_02262024_z30_xy3_vel0p8.csv"],
        ["drone1_parameters_02262024_z50_xy4_vel1p0.csv", "eye_tracker_02262024_z50_xy4_vel1p0.csv"],
        ["drone1_parameters_02262024_z50_xy4_vel1p0_set2.csv", "eye_tracker_02262024_z50_xy4_vel1p0_set2.csv"],
        ["drone1_parameters_03032024_z20_xy2_vel0p8.csv", "eye_tracker_03032024_z20_xy2_vel0p8.csv"],
        ["drone1_parameters_03032024_z20_xy2_vel1p0.csv", "eye_tracker_03032024_z20_xy2_vel1p0.csv"],
        ["drone1_parameters_03032024_z20_xy2_vel2p0.csv", "eye_tracker_03032024_z20_xy2_vel2p0.csv"],
        ["drone1_parameters_03032024_z30_xy3_vel0p6.csv", "eye_tracker_03032024_z30_xy3_vel0p6.csv"],
        ["drone1_parameters_03032024_z30_xy3_vel1p0.csv", "eye_tracker_03032024_z30_xy3_vel1p0.csv"],
        ["drone1_parameters_03032024_z30_xy3_vel2p0.csv", "eye_tracker_03032024_z30_xy3_vel2p0.csv"],
        ["drone1_parameters_03042024_z5_xy1_vel0p6.csv", "eye_tracker_03042024_z5_xy1_vel0p6.csv"],
        ["drone1_parameters_03042024_z5_xy1_vel0p8.csv", "eye_tracker_03042024_z5_xy1_vel0p8.csv"],
        ["drone1_parameters_03042024_z5_xy1_vel1p0.csv", "eye_tracker_03042024_z5_xy1_vel1p0.csv"],
        ["drone1_parameters_03042024_z5_xy1_vel1p5.csv", "eye_tracker_03042024_z5_xy1_vel1p5.csv"],
        ["drone1_parameters_03042024_z10_xy1_vel0p6.csv", "eye_tracker_03042024_z10_xy1_vel0p6.csv"],
        ["drone1_parameters_03042024_z10_xy1_vel1p0.csv", "eye_tracker_03042024_z10_xy1_vel1p0.csv"],
        ["drone1_parameters_03042024_z10_xy1_vel1p5.csv", "eye_tracker_03042024_z10_xy1_vel1p5.csv"]
        ]
    rms_error_lists = []

    #
    # Open csv file to output RMSE data vs. drone parameter values
    #
    file_out_obj = open(file_path + "dronetracker1_rmse.csv", "w")
    # Write headers
    file_out_obj.write("Z (m),Y (m),XY (m),Vel (m/s),rmse (m),rmse_wo_outliers (m)\n")

    #
    # Loop through list of tests, extracting drone parameters and data records for each test.
    #
    dataset_count = 0
    for file_set in data_files:
        dataset_count += 1
        param_file = file_path + file_set[0]
        data_file = file_path + file_set[1]
        dronetest = DroneTrackTest(param_file, data_file)

        # Print drone parameter values for current test.
        dronetest.print_test_params()

        # Calculate eye-gaze hitpoint vs drone-position errors
        error_lists = dronetest.eye_gaze_drone_track_error2()
        eye_gaze_errors_mag = error_lists[2]

        # Calculate RMS error for current test, with and without outliers
        rmse_eye_gaze = DroneTrackTest.rmse_calc(eye_gaze_errors_mag)
        rmse_eye_gaze_wo_outliers = DroneTrackTest.rmse_calc(eye_gaze_errors_mag, True, 0.3)

        print("\tAt Drone Z plane, RMS Error = {0:.3f}m, RMS Error w/o Outliers = {1:.3f}m"
              .format(rmse_eye_gaze, rmse_eye_gaze_wo_outliers))
        rms_error_lists.append([dronetest.zval, dronetest.yval, dronetest.xylim, dronetest.xyvel,
                                rmse_eye_gaze, rmse_eye_gaze_wo_outliers])
        file_out_obj.write("{0:.2f},{1:.2f},{2:.2f},{3:.2f},{4:.3f},{5:.3f}\n".format(dronetest.zval, dronetest.yval, dronetest.xylim,
                                                dronetest.xyvel,rmse_eye_gaze,rmse_eye_gaze_wo_outliers))
    print(f"\nProcessed {dataset_count} data sets.")

    #
    # Close output file
    #
    file_out_obj.close()


if __name__ == '__main__':
    main()

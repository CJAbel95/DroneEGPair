#
# yshift_comp_test.py -- Test the pos_shift_comp() function in EyeMatchUtils.py
# Description -- Details..
#
# by Christopher Abel
# Revision History
# ----------------
# 04/22/2024 -- Original
#
# -------------------------------------------------
import csv
import numpy as np
import matplotlib.pyplot as plt
import EyeMatchUtils as EGutil


def main():
    time_list = []
    y_list = []
    # Read in data file
    filename = 'drone_data.csv'
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            time_list.append(row[0])
            y_list.append(row[2])
    # Convert time and y lists to numpy arrays
    time_arr = np.array(time_list, dtype=np.float32)
    y_arr = np.resize(np.array(y_list, dtype=np.float32), (len(y_list), 1))

    # Create y_adj array from y_arr by removing shifts
    y_adj = EGutil.pos_shift_comp(y_arr, 0.075)

    # Plot y and y_adj vs time
    fig1, ax1 = plt.subplots()
    ax1.plot(time_arr, y_arr, 'b-+')
    ax1.plot(time_arr, y_adj, 'r-o')

    plt.show()


# Main code
if __name__ == '__main__':
    main()

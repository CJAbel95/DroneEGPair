#
# numpy_stats_test.py -- Brief Description
# Description -- Details..
#
# by Christopher Abel
# Revision History
# ----------------
# 02/22/2024 -- Original
#
# -------------------------------------------------
import random

import numpy as np
import scipy as sp


def main():
    x = [-0.4, 1.2, 0.8, -0.75, 0.3]
    mean_x = np.mean(x)
    median_x = np.median(x)
    std_x = np.std(x, ddof=1)
    print('Mean of ', x, ' = ', mean_x)
    print('Median = {0:.4f}\tStd. dev. = {1:.4f}'.format(median_x, std_x))
    z = []
    z.append([0, 0, 0])
    z.append([-0.05, 0.025, 0.01])
    z_x_mean = np.mean([row[0] for row in z])
    print(z_x_mean)

    #
    # Generate random list and use ecdf function from SciPy.
    #
    rand_list = []
    for i in range(1,1000000):
        rand_list.append(random.random())
    print('Mean = {0:.5f}\tstd = {1:.5f}'.format(np.mean(rand_list), np.std(rand_list, ddof=1)))
    sorted_list = sorted(rand_list)
    nval = len(sorted_list)
    print('5% point = {0:.5f}\t95% point = {1:.5f}'.format(sorted_list[int(0.05*nval)], sorted_list[int(0.95*nval)]))
    #rand_list_cdf = sp.stats.ecdf(rand_list).cdf
    #print(rand_list_cdf[0])

# Main code
if __name__ == '__main__':
    main()

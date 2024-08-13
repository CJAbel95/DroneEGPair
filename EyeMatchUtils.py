#
# EyeMatchUtils.py -- Additional utililty functions used in Drone-Eye-Gaze
#                   matching routines.
# Description -- Includes functions to interpolate multi-variable arrays
#               vs. time.
#
# by Christopher Abel
# Revision History
# ----------------
# 04/09/2024 -- Original
# 04/14/2024 -- Addded rem_duplicates() function.
# 04/15/2024 -- Added rem_outliers() function and entropy_calc() function.
# 04/22/2024 -- Added pos_shift_comp() and partition_calc() functions.
# 07/29/2024 -- Added calc_vel() function.
# 08/07/2024 -- Added key_stats() function.
# 08/13/2024 -- Added calc_norm_factor2() and calc_error_wvf() functions.
#
# -------------------------------------------------
import math

import numpy as np
import scipy.stats
from scipy.interpolate import CubicSpline, Akima1DInterpolator
import matplotlib.pyplot as plt


def interp_1d_array(xnew, x, y_arr, interp='Linear'):
    """
    Interpolates an n-dimensional dependent variable with a single independent variable.
    For example, if x is a 1-d array of values of an independent variable, then the array
    of dependent variables will look like [y1, y2, y3, ..., yn], with each column (yi)
    representing a unique function of x.

    The function returns an array of n columns, with each column containing the values of
    the corresponding dependent variable interpolated at the new array of independent values (xnew).

    Args:
        xnew (float): New column array (shape = (mnew, 1)) of mnew independent variable values.
        x (float): 1D array of m independent variable values.
        y_arr (2D array of float): m x n array of dependent values.  Each row contains the values
                of the n dependent functions evaluated at the corresponding row of x.
        interp (string): 'Linear' -- performs linear interpolation of y1 - yn vs. x.
                        'Cubic' -- performs a cubic spline interpolation of y1 - yn vs. x.
                        'Akima' -- performs an Akima1D cubic interpolation of y1 - yn vs. x.

    Returns: y_interp (2D array of float): mnew x n array of dependent values. Each row contains the
                values of the n dependent functions evaluated at the corresponding row of xnew.

    """
    # Check whether the y array is 1D or 2D
    if y_arr.ndim == 2:
        num_ydim = y_arr.shape[1]  # number of columns of y array; corresponds to the number of dependent functions.
        #
        # For each column in the y array, create a new column, ynew, with the same functional
        # dependence on x, but interpolated to the values contained in xnew.
        #
        for icol in range(0, num_ydim):
            y = y_arr[:, icol]  # Dependent values of current column, input to program
            if interp == 'Linear':
                ynew = np.interp(xnew, x, y)
            elif interp == 'Cubic':
                spline = CubicSpline(x, y)
                ynew = spline(xnew)
            else:
                spline = Akima1DInterpolator(x, y)
                ynew = spline(xnew)
            #
            # For the first column of y, copy to the first column of a new array, y_interp.
            # Concatenate subsequent columns to y_interp.
            #
            if icol == 0:
                y_interp = ynew.reshape(len(ynew), 1)
            else:
                y_interp = np.concatenate((y_interp, ynew.reshape(len(ynew), 1)), 1)
    else:
        #
        # For a 1D y array, interpolate over xnew, forming a new 1D array, y_interp
        #
        if interp == 'Linear':
            y_interp = np.interp(xnew, x, y_arr)
        elif interp == 'Cubic':
            spline = CubicSpline(x, y_arr)
            y_interp = spline(xnew)
        else:
            spline = Akima1DInterpolator(x, y_arr)
            y_interp = spline(xnew)
    # Return interpolated array
    return y_interp


def rem_duplicates(t, data, tolerance):
    """
    Remove consecutive duplicate samples from dataset.  This seems to happen in the datasets from
    the Codrone-Edu drone.

    Args:
        t: 1D numpy array of time values
        data: Numpy array (1D or 2D) of time-series data; each column of array is a separate
                time-dependent variable.
        tolerance: If abs(t[n] - t[n-1]) < tolerance, then t[n] is considered equal to t[n-1].

    Returns: List consisting of [t, data] with duplicate samples removed.
    """
    #
    # Loop through array of time elements. If a time value is equal to (within
    # an error threshold) that of the previous element, remove that row from
    # the time array and from the data array.
    #
    t_rem = t.copy()
    data_rem = data.copy()
    i = 1
    while i < len(t_rem):
        # Compare current time point to previous time point. If they are identical, delete current
        # time point, but do not advance row index.
        if abs(t_rem[i] - t_rem[i - 1]) < tolerance:
            t_rem = np.delete(t_rem, i)
            # If data is a 2D array, delete row i. If it is a 1D array, delete element i
            if data_rem.ndim == 2:
                data_rem = np.delete(data_rem, i, 0)
            else:
                data_rem = np.delete(data, i)
        # If current time point is different from previous time point, increment row index.
        else:
            i += 1
    return [t_rem, data_rem]


def rem_outliers(t, data, out_level, num_avg=20):
    """
    Identify and remove outliers in time-series dataset. Calculate moving average of previous num_avg points in
    data.  For the i'th data point, if abs(data[i] - mov_avg[i]) > out_level, then record time[i] to list
    of deleted values, and remove point from time array and data array.

    Args:
        t: 1D numpy array of time values
        data: Numpy array (1D or 2D) of time-series data; each column of array is a separate
                time-dependent variable.
        out_level: Outlier tolerance.  If abs(data[i] - mov_avg[i]) > out_level, then point i is an outlier.
        num_avg: Number of previous points to average when calculating moving average.

    Returns: List consisting of [t, data, t_outliers] with t and data having outliers removed.  Note that t_outliers
            is itself a list of the time points (i.e. values of original t array) that were removed.
    """
    # If data is a 2D array, then mov_avg is a list containing # elements = # columns in data.
    # Each element holds the moving average of the data in the corresponding column.
    # If data is a 1D array, then mov_avg is a floating-point number.

    # Create copies of the original time and data arrays.
    t_outliers = []
    t_rem = t.copy()
    # If the data array is a 1D array, convert it into a 2D array with multiple rows and 1 column.
    if data.ndim == 1:
        data_rem = np.resize(data, (len(data), 1))
    else:
        data_rem = data.copy()
    ncols = data_rem.shape[1]
    # For each time point, compare the value in each column of data[] with the average of the previous
    # num_avg points.
    i = 0
    while i < len(t_rem):
        outlier = False
        # Examine each column of data_rem, row i, to see if it is an outlier.  If any row is an outlier,
        # set outlier = True. Ignore the first num_avg points.
        if i >= num_avg:
            for j in range(0, ncols):
                if abs(data_rem[i, j] - np.mean(data_rem[i - num_avg:i, j])) > out_level:
                    outlier = True
        # If point is not an outlier, increment index i to look at the next point.
        # If it is an outlier, remove the current time and data point.
        if not outlier:
            i += 1
        else:
            t_outliers.append(t_rem[i])  # Append time point to list of outlier times.
            t_rem = np.delete(t_rem, i)
            data_rem = np.delete(data_rem, i, 0)
    # Return list of [t_rem, data_rem] with outliers removed.
    return [t_rem, data_rem, t_outliers]


def pos_shift_comp(y, dy_thresh):
    """
    Compensate for presumed non-physical shifts in a dependent variable in a time series.  For example, if a vector
    containing the position of an object vs. time exhibits an apparent non-physical shift in one time instant (i.e.
    a large change in position in one time step), measure the dy associated with that step, and subtract it from
    subsequent points.  If additional shifts appear later in the sequence, then these additional shifts should
    be accumulated and subtracted from subsequent points in the vector.

    Args:
        y: 1D numpy array (or 2D array with 1 column) containing the dependent variable of a time-series.
        dy_thresh: Threshold for shifts in y that are compensated; if abs(y[i] - y[i-1]) > dy_thresh, then
                (y[i] - y[i-1]) is added to the quantity total_accum_shift, which is then subtracted from subsequent
                values of y.

    Returns:
        y_adj: array of same dimension as y with accumulated excess dy shift removed from y.
    """
    # Begin by copying y to y_adj. Shift values will be subtracted from y_adj.
    y_adj = y.copy()
    #
    # Step through y array.  If the absolute value of the delta between two consecutive points exceeds
    # dy_thresh, then add this delta into the accumulated adjustment that is subtracted from each
    # subsequent point of y.
    #
    total_accum_shift = 0
    # Handle the case that y is a 1D array
    if y.ndim == 1:
        for i in range(1, len(y)):
            if abs(y[i] - y[i-1]) >= dy_thresh:
                total_accum_shift += y[i] - y[i-1]
            # For each value of y_adj, subtract the total accumulated shift
            y_adj[i] = y_adj[i] - total_accum_shift
    # Handle the case that y is a 2D array
    else:
        for i in range(1, len(y)):
            if abs(y[i, 0] - y[i-1, 0]) >= dy_thresh:
                total_accum_shift += y[i, 0] - y[i-1, 0]
            y_adj[i, 0] = y_adj[i, 0] - total_accum_shift
    # Return the y array with shifts removed
    return y_adj


def partition_calc(x, nbits):
    """
    Given a random data set, x, this function calculates the set of thresholds which partition x
    into approximately equal subsets, each of equal probability of occurrence.  For nbits bits,
    the function calculates 2^nbits - 1 thresholds.

    Args:
        x: numpy array of random samples.
        nbits: Number of bits of quantization; that is, x will be divided into 2^nbits levels.

    Returns: list of 2^nbits - 1 thresholds, thresholds[].
    """
    # Calculate empirical CDF of set of random samples
    x_cdf = scipy.stats.ecdf(np.resize(x, len(x)))
    thresholds = []
    nsamp = len(x)
    # Find each of the 2^nbits - 1 thresholds from the quantiles array of x_cdf
    for i_thresh in range(1, 2**nbits):
        index = round(nsamp * i_thresh / 2**nbits) - 1
        thresholds.append(x_cdf.cdf.quantiles[index])
    # Return list of thresholds
    return thresholds


def key_bit_comp(key1, key2):
    bit_count = min(len(key1), len(key2))
    bit_diff_count = 0
    ones_count_k1 = 0
    ones_count_k2 = 0
    for bit in range(0, bit_count):
        if (key1[bit] != key2[bit]):
            bit_diff_count += 1
        if key1[bit] == '1':
            ones_count_k1 += 1
        if key2[bit] == '1':
            ones_count_k2 += 1
    return [bit_count, bit_diff_count, ones_count_k1, ones_count_k2]


def entropy_calc(ones_count, bits_count):
    """
    Approximates information entropy in a bit string using Shannon's formula.

    Args:
        ones_count: Number of 1's in bit string
        bits_count: Total number of bits in bit string

    Returns: Entropy, in bits
    """
    prob1 = ones_count / bits_count
    prob0 = 1.0 - prob1
    return -(prob1 * math.log2(prob1) + prob0 * math.log2(prob0)) * bits_count


def calc_vel(time_arr, pos_arr, avg_samples=1):
    """
    calc_vel(): Calculate the velocity time series corresponding to a position time series.

    :param time_arr: A 1D numpy array of time values.
    :param pos_arr:  A 1D numpy array of position values.
    :param avg_samples: Number of samples in weighted average used to calculate velocity.
    :return: 2D numpy array with columns (time, dX/dt).
    """
    deriv = []
    for i in range(avg_samples, len(time_arr)):
        delta_x = pos_arr[i] - pos_arr[i - avg_samples]
        delta_t = time_arr[i] - time_arr[i - avg_samples]
        if delta_t > 0:
            deriv.append([time_arr[i], delta_x / delta_t])
    return np.array(deriv)


def normalize_path(pos_arr, pos_center, pkpk_range):
    #
    # Subtract center value of position array and multiply by scale factor
    #
    return 2.0 / pkpk_range * (pos_arr - pos_center)


def calc_norm_factor(time_arr, pos_arr, t_start, t_end, alpha=0.02):
    """
    Calculate normalization factor, which can be used to scale a time series.
    The normalization factor is equal to the difference between the max - min
    of the time series over the time range [t_start, t_end].
    After subtracting the average value and dividing by this factor, the time series
    values will span the range [-0.5, 0.5].

    The function includes a parameter alpha, which represents the outlier
    parameter.  If alpha > 0.000001, then max = LUB of (1 - alpha) fraction
    of the time series, and min = GLB of alpha fraction of the time series.
    For example, if alpha = 0.02 (default), then max = 98% of time series, and
    min = 2% of time series.

    :param time_arr: Numpy 1D array of time points in time series.
    :param pos_arr: Numpy 1D array of dependent values of time series.
    :param t_start: Start of time range over which max - min is calculated -- [t_start, t_end]
    :param t_end: End of time range over which max - min is calculated
    :param alpha: Fractional value representing outliers. Range [0, 0.5).
    :return: max - min of time series.
    """
    #
    # Find indices in time_arr corresponding to t_start and t_end
    #
    i_start = np.argwhere(time_arr >= t_start)[0][0]
    i_end = np.argwhere(time_arr <= t_end)[-1][0]
    #
    # Find alpha (lower) and 1 - alpha (upper) limits of pos_ar[i_start : i_end + 1].
    #
    # Note that if alpha is close to 0, then simply return the maximum and minimum values
    # of the array.
    #
    if alpha > 1.0e-6:
        pos_sorted = sorted(pos_arr[i_start: i_end + 1])
        pos_lower = pos_sorted[int(alpha * len(pos_sorted))]
        pos_upper = pos_sorted[len(pos_sorted) - int(alpha * len(pos_sorted))]
    else:
        pos_lower = np.min(pos_arr[i_start : i_end + 1])
        pos_upper = np.max(pos_arr[i_start : i_end + 1])
    #
    # Return scale factor
    return pos_upper - pos_lower


def calc_norm_factor2(time1_arr, x1_arr, time2_arr, x2_arr, t_start, t_end, alpha=0.02):
    """
    Calculate the slope (m) and intercept (b) of the linear mapping between the x1 time series
    values and the x2 time series.  That is,

    x1 = m * x2 + b

    m and b are determined from the extrema (max and min) of the two arrays.  In other words,
    max(x1) = m * max(x2) + b and min(x1) = m * min(x2) + b.

    The function includes a factor alpha that sets the extrema slightly inside the true max
    and min limits, in order to eliminate outliers.  With alpha > 0,
        max(x) >= (1 - alpha)% of values in the x array.
        min(x) >= alpha% of values in the x array.

    :param time1_arr:
    :param x1_arr:
    :param time2_arr:
    :param x2_arr:
    :param t_start
    :param t_end
    :param alpha:
    :return:
    """
    # Find indices in time_arr corresponding to t_start and t_end
    i1_start = np.argwhere(time1_arr >= t_start)[0][0]
    i1_end = np.argwhere(time1_arr <= t_end)[-1][0]
    i2_start = np.argwhere(time2_arr >= t_start)[0][0]
    i2_end = np.argwhere(time2_arr <= t_end)[-1][0]
    #
    # Find the extreme values (max, min) of x1_arr and x2_arr
    #
    if alpha > 1.0e-6:
        x1_sorted = sorted(x1_arr[i1_start: i1_end + 1])
        x1_max = x1_sorted[len(x1_sorted) - int(alpha * len(x1_sorted))]
        x1_min = x1_sorted[int(alpha * len(x1_sorted))]
        x2_sorted = sorted(x2_arr[i2_start: i2_end + 1])
        x2_max = x2_sorted[len(x2_sorted) - int(alpha * len(x2_sorted))]
        x2_min = x2_sorted[int(alpha * len(x2_sorted))]
    else:
        x1_max = np.max(x1_arr[i1_start: i1_end + 1])
        x1_min = np.min(x1_arr[i1_start: i1_end + 1])
        x2_max = np.max(x2_arr[i1_start: i1_end + 1])
        x2_min = np.min(x2_arr[i1_start: i1_end + 1])
    #
    # Calculate slope and intercept relating x1_arr to x2_arr.  That is, the values of
    # m and b such that x1_arr = m * x2_arr + b
    #
    m = (x1_max - x1_min) / (x2_max - x2_min)
    b = x1_max - m * x2_max
    # return m and b
    return [m, b]


def calc_error_wvf(t1, x1, t2, x2):
    """

    :param t1:
    :param x1:
    :param t2:
    :param x2:
    :return:
    """
    # Interpolate the values of x2 at the t1 array.
    x2_interp = interp_1d_array(t1, t2, x2, 'Akima')
    #
    # Calculate error e = x1 - x2_interp at each time point in the t1 array.
    # Also calculate standard deviation of error.
    #
    err = x1 - x2_interp
    err_rms = np.std(err, ddof=1)
    # Return calculated standard deviation and error time series.
    return [err_rms, err]


def calc_seg_disp(time_arr, pos_arr, t_start, t_end, delta_t, disp_abs_val=True):
    #
    # Find starting and ending indices corresponding to time_arr = t_start
    # and time_arr = t_end
    #
    index_st = np.argwhere(time_arr >= t_start)[0][0]
    index_end = np.argwhere(time_arr <= t_end)[-1][0]
    # Calculate number of segments
    num_segs = math.floor((t_end - t_start) / delta_t)
    #
    # Split pos_arr time series into num_segs segments
    #
    segments = []
    seg_indices = []
    t_seg_start = t_start
    for i in range(0, num_segs):
        t_seg_end = t_seg_start + delta_t
        seg_index_st = np.argwhere(time_arr >= t_seg_start)[0][0]
        seg_index_end = np.argwhere(time_arr <= t_seg_end)[-1][0]
        seg_indices.append([seg_index_st, seg_index_end])
        if disp_abs_val:
            segments.append(math.fabs(pos_arr[seg_index_end] - pos_arr[seg_index_st]))
        else:
            segments.append(pos_arr[seg_index_end] - pos_arr[seg_index_st])
        t_seg_start = t_seg_end
    # Return lists of segment indicies and segment displacements
    return [segments, seg_indices]


def key_stats(key1, key2):
    """
    Calculate the following statistics for two keys consisting of bit strings
        -- key length
        -- entropy
        -- count of mismatched bits
        -- bit matching fraction

    Args:
        key1: String of bits, for example -- '100101100...'
        key2: String of bits to be compared with key1

    Returns: List consisting of
        key_length: Number of bits in key1 and key2 that are compared.  If the lengths are not equal, only the first
                n bits, where n = min(length(key1), length(key2)) are compared.
        entropy_key1, entropy_key2 = Information entropy of each key, in units of bits
        mismatch_count = number of bits mismatched between key1 and key2, in a position-by-position comparison
        bit_match_fract = (key_length - mismatch_count)/key_length

    """
    nbits = min(len(key1), len(key2))
    mismatch_count = 0
    ones_count_key1 = 0
    ones_count_key2 = 0
    for bit in range(0, nbits):
        if key1[bit] != key2[bit]:
            mismatch_count += 1
        if key1[bit] == '1':
            ones_count_key1 += 1
        if key2[bit] == '1':
            ones_count_key2 += 1
    entropy_key1 = entropy_calc(ones_count_key1, nbits)
    entropy_key2 = entropy_calc(ones_count_key2, nbits)
    bit_match_fract = (nbits - mismatch_count)/nbits
    #
    # Return values
    return [nbits, entropy_key1, entropy_key2, mismatch_count, bit_match_fract]


def main():
    # 1D array of independent variable
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)
    #
    # Form 3 1D arrays of dependent variables; combine to form a 3-column array
    # of dependent variables, r.
    #
    x = 4.0 * np.random.default_rng(23).random((11,))
    y = np.array([-1.2, 3.1, 2.3, -6.1, -0.5, 1.5, 2.5, 8.1, 3.78, -1.2, 0.9])
    z = np.array([0.3 * (xval - 3.0) ** 2 - 0.03 * xval ** 3 for xval in x])
    r = np.hstack((x.reshape(11, 1), y.reshape(11, 1), z.reshape(11, 1)))

    # Form a new independent variable array, with a much smaller step size
    tstep = 0.1
    npts = int((t[-1] - t[0]) / tstep) + 1
    # tf = np.linspace(t[0], t[-1], npts).reshape(npts, 1)
    tf = np.linspace(t[0], t[-1], npts)

    # Test partition_calc() on a random array
    xrand_arr = np.random.standard_normal(400)
    #print('xrand_arr = ', xrand_arr)
    print('xrand_arr thresholds = ', partition_calc(xrand_arr, 1))
    fig0, ax0 = plt.subplots()
    ax0.hist(xrand_arr, 20)

    # linear interpolation
    ylin = np.interp(tf, t, y)
    rlin = interp_1d_array(tf, t, r, interp='Linear')
    xlin = rlin[:, 0]
    # cubic spline
    spline = CubicSpline(t, y)
    ycub = spline(tf)
    rcub = interp_1d_array(tf, t, r, interp='Cubic')
    xcub = rcub[:, 0]
    # Akima spline
    spline = Akima1DInterpolator(t, y)
    yakim = spline(tf)
    rakim = interp_1d_array(tf, t, r, interp='Akima')
    xakim = rakim[:, 0]

    # Plot y vs. x and interpolated curves
    fig1, ax1 = plt.subplots()
    ax1.plot(t, y, 'bo')
    ax1.plot(t, x, 'r+')
    # ax1.plot(xf, ylin, 'r--')
    # ax1.plot(tf, xlin, 'r-')
    ax1.plot(tf, xcub, 'r-')
    ax1.plot(tf, ycub, 'b-')
    ax1.plot(tf, xakim, 'm-.')
    ax1.plot(tf, yakim, 'm-.')
    plt.show()

    # Test rem_duplicates() function.  4th sample is a duplicate of 3rd sample, and 6th is a duplicate of the 5th.
    t1 = np.array([0.0, 0.02, 0.04, 0.05, 0.05, 0.07, 0.07])
    x1 = np.random.default_rng(18).random((7, 2))
    x1[4, :] = x1[3, :]
    x1[6, :] = x1[5, :]
    print('\nBefore rem_duplicates()....')
    print('\tt1  \tx1')
    print('\t--  \t--')
    for i in range(0, len(t1)):
        print('\t{0:.2f}\t{1:.2f}, {2:.2f}'.format(t1[i], x1[i, 0], x1[i, 1]))
    t1, x1 = rem_duplicates(t1, x1, 1e-5)
    print('\nAfter rem_duplicates()....')
    print('\tt1  \tx1')
    print('\t--  \t--')
    for i in range(0, len(t1)):
        print('\t{0:.2f}\t{1:.2f}, {2:.2f}'.format(t1[i], x1[i, 0], x1[i, 1]))

    # Test rem_outliers() function.
    t2 = np.arange(0.0, 10.0, 0.02)
    x2 = np.random.default_rng(18).random((len(t2), 1))
    y2 = 2.0 * x2
    x2[200] = 9.3
    x2[375] = -7.8
    x2[376] = -8.3
    y2[175] = -14.0
    y2[317] = 16.0
    r2 = np.concatenate((x2, y2), 1)
    t3, r3, t_outliers = rem_outliers(t2, r2, 2.0, 20)
    fig2, (ax2a, ax2b) = plt.subplots(2, 1)
    ax2a.plot(t2, x2, 'b+', t3, r3[:, 0], 'ro-')
    ax2b.plot(t2, y2, 'b+', t3, r3[:, 1], 'ro-')
    plt.show()


# Main code
if __name__ == '__main__':
    main()

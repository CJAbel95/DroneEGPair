#
# EGCustomPlots.py -- Wrapper / helper functions to create plots of eye-gaze and drone position
# Description -- Wrapper / helper functions which call Matplotlib methods, and which have been customized
#               for use with plotting eye-gaze hitpoint and drone position data.
#
# by Christopher Abel
# Revision History
# ----------------
# 03/24/2024 -- Original
#
# -------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np


def plot_locus_2d(axes, x, y, param_dict=None):
    # 2D scatter plot, Y vs. X values. Assumes that a call to fig, axes = matplotlib.pyplot.subplots()
    # is made prior to calling this function.
    #
    # Inputs:
    #   axes = Axes object for figure to be generated; passed to function.
    #   x = numpy array of X values
    #   y = numpy array of Y values
    #   param_dict = Optional dictionary containing values for title, xlabel, ylabel, and marker type.
    #       The entire dictionary is optional; if it exists, each individual key is optional.
    #
    # Extract values from param_dict
    title = 'Scatter Plot' if (param_dict == None or not 'title' in param_dict) else param_dict['title']
    xlabel = '' if (param_dict == None or not 'xlabel' in param_dict) else param_dict['xlabel']
    ylabel = '' if (param_dict == None or not 'ylabel' in param_dict) else param_dict['ylabel']
    marker = 'b.' if (param_dict == None or not 'marker' in param_dict) else param_dict['marker']
    axes.plot(x, y, marker)
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)


def plot_multiline(axes, y_list, x_list, param_dict=None):
    # Multiline 2D plot; multiple Y arrays plotted vs same X array.
    # Assumes that a call to fig, axes = matplotlib.pyplot.subplots()
    # is made prior to calling this function.
    #
    # Inputs:
    #   axes = Axes object for figure to be generated; passed to function.
    #   x_list = list of numpy arrays, each of which contains an array of X values
    #   y_list = List of numpy arrays, each of which contains Y values to be plotted against the X array values
    #   param_dict = Optional dictionary containing values for title, xlabel, ylabel, and marker type.
    #       The entire dictionary is optional; if it exists, each individual key is optional.
    #
    # Extract values from param_dict
    title = 'Multiline Plot' if (param_dict == None or not 'title' in param_dict) else param_dict['title']
    xlabel = '' if (param_dict == None or not 'xlabel' in param_dict) else param_dict['xlabel']
    ylabel = '' if (param_dict == None or not 'ylabel' in param_dict) else param_dict['ylabel']
    marker_list = ['bo-', 'r+--', 'mx-.', 'g+-']
    num_lines = len(y_list)
    handles = []
    if param_dict != None and 'legends' in param_dict:
        for i in range(num_lines):
            handle, = axes.plot(x_list[i], y_list[i], marker_list[i % len(marker_list)], label=param_dict['legends'][i])
            handles.append(handle)
        axes.legend(handles=handles)
    else:
        for i in range(num_lines):
            axes.plot(x_list[i], y_list[i], marker_list[i])
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)


def plot_2d_vs_t(fig, x, y, t, param_dict=None):
    # 3D plot of X and Y vs. time. Assumes that a call to fig = matplotlib.pyplot.figure()
    # is made prior to calling this function.
    #
    # Inputs:
    #   fig = Figure object for figure to be generated; passed to function.
    #   x = numpy array of X values
    #   y = numpy array of Y values
    #   t = numpy array of time values
    #   param_dict = Optional dictionary containing values for title, tlabel, xlabel, ylabel, and marker type.
    #       The entire dictionary is optional; if it exists, each individual key is optional.
    #
    # Extract parameters from param_dict
    title = 'X and Y vs. Time' if (param_dict == None or not 'title' in param_dict) else param_dict['title']
    tlabel = 'Time (s)' if (param_dict == None or not 'tlabel' in param_dict) else param_dict['tlabel']
    xlabel = 'X (m)' if (param_dict == None or not 'xlabel' in param_dict) else param_dict['xlabel']
    ylabel = 'Y (m)' if (param_dict == None or not 'ylabel' in param_dict) else param_dict['ylabel']
    marker = '-o' if (param_dict == None or not 'marker' in param_dict) else param_dict['marker']
    axes = fig.add_subplot(projection='3d')
    axes.plot(t, x, y, marker)
    axes.set_title(title)
    axes.set_xlabel(tlabel)
    axes.set_ylabel(xlabel)
    axes.set_zlabel(ylabel)


def plot_3d(fig, x, y, z, param_dict=None):
    # 3D plot of X, Y, Z.  I'll add further capability later.
    axes = fig.add_subplot(projection='3d')
    axes.plot(x, y, z, 'ro-')


def main():
    rng = np.random.default_rng()
    x = 2.0 * rng.random(size=50, dtype=np.float32) - 1.0
    y = 5.0 * rng.random(size=50, dtype=np.float32) + 3.5
    fig1, axes1 = plt.subplots()
    param_dict = {'title':'Rand Y vs Rand X', 'xlabel':'X (m)', 'ylabel':'Y (m)', 'marker':'b+-'}
    plot_locus_2d(axes1, x, y, param_dict)

    fig2, axes2 = plt.subplots()
    t = np.array([0.05 * i for i in range(0, 50)], np.float32)
    param_dict = {'title': 'X and Y vs. Time', 'xlabel': 't (s)', 'ylabel': 'X, Y (m)', 'legends': ['X', 'Y']}
    plot_multiline(axes2, [x, y], [t, t], param_dict)

    fig3 = plt.figure()
    param_dict = {'title': '3D plot: X and Y vs. Time', 'tlabel': 'Time (sec)', 'xlabel': 'X (m)',
                  'ylabel': 'Y (m)', 'marker': 'b+-'}
    plot_2d_vs_t(fig3, x, y, t, param_dict)

    plt.show()


# Main code
if __name__ == '__main__':
    main()

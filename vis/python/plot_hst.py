#! /usr/bin/env python

# Script for plotting 1D data from Athena++ .hst files.

# Run "plot_hst.py -h" for help.

# Python modules
import argparse

# Athena++ modules
import athena_read


# Main function
def main(**kwargs):

    #  extract inputs
    input_file = kwargs['input']
    variables = kwargs['variables']
    output_file = kwargs['output']
    x_log = kwargs['xlog']
    y_log = kwargs['ylog']

    # read data
    data = athena_read.hst(input_file)

    # check vraiable names are valid, and set x/y data
    if variables not in data:
        print('Invalid input variable name, valid names are:')
        for key in data:
            print(key)
        raise RuntimeError

    y_vals = data[variables]
    x_vals = data["time"]

    # Load Python plotting modules
    if output_file != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # Plot data
    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.xlabel("time")
    plt.ylabel(variables)
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.show()


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='name of input (hst) file')
    parser.add_argument('-o', '--output',
                        default='show',
                        help='image filename; omit to display to screen')
    parser.add_argument('-v', '--variables',
                        help='comma-separated list of variables to be plotted')
    parser.add_argument('-xlog', '--xlog',
                        action='store_true',
                        help='plot log of x-values')
    parser.add_argument('-ylog', '--ylog',
                        action='store_true',
                        help='plot log of y-values')

    args = parser.parse_args()
    main(**vars(args))

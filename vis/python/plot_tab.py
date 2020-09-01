#! /usr/bin/env python

"""
Script for plotting 1D data from Athena++ .tab files.

Run "plot_tab.py -h" for help.
"""

# Python modules
import argparse

# Athena++ modules
import athena_read


# Main function
def main(**kwargs):

    # get input file and read data
    input_file = kwargs['input']
    data = athena_read.tab(input_file)

    # get variable names, check they are valid, and set x/y data
    variables = kwargs['variables']
    if variables not in data:
        print('Invalid input variable name, valid names are:')
        for key in data:
            print(key)
        raise RuntimeError

    y_vals = data[variables]
    x_vals = []
    if "x1v" in data:
        x_vals = data["x1v"]
    else:
        if "x2v" in data:
            x_vals = data["x2v"]
        else:
            x_vals = data["x3v"]

    # print(data)

    # Load Python plotting modules
    output_file = kwargs['output']
    if output_file != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # Plot data
    plt.figure()
    plt.plot(x_vals, y_vals, 's')
    plt.show()



# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input',
                        help='name of input (tab) file')
    parser.add_argument('-o','--output',
                        default='show',
                        help='name of output image file; omit to display to screen')
    parser.add_argument('-v','--variables',
                        help='comma-separated list of variables to be plotted')

    args = parser.parse_args()
    main(**vars(args))

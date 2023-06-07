#! /usr/bin/env python

# Script for plotting and animating 1D data from Athena++ .tab files.
# Example make a single plot for one .tab file use:

# Example to animate a sequentially numbered list of .tab files use:

# Run "plot_tab.py -h" for help.

# Python modules
import argparse
import matplotlib.widgets as mwidgets
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import mpl_toolkits.axes_grid1
import athena_read
mpl.rcParams['animation.embed_limit'] = 2**32


# custom animation class Player
# is derived class of FuncAnimation
class Player(FuncAnimation):
    def __init__(self, fig, func, frames=None, init_func=None,
                 fargs=None, save_count=None, mini=0, maxi=100,
                 pos=(0.125, 0.94), **kwargs):
        self.i = 0
        self.min = mini
        self.max = maxi
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.update,
                               frames=self.play(), init_func=init_func,
                               fargs=None, save_count=None,
                               interval=200, cache_frame_data=False, **kwargs)

    # version that stops when reaching end of plot list
    # def play(self):
    #    while self.runs:
    #        if self.i > self.min and self.i < self.max:
    #            self.i = self.i+self.forwards-(not self.forwards)
    #            yield self.i
    #        elif self.i == self.min and self.forwards:
    #            self.i+=1
    #            yield self.i
    #        elif self.i == self.max and not self.forwards:
    #            self.i-=1
    #            yield self.i
    #        else:
    #            self.stop()
    #            yield self.i

    # version that loops forever
    def play(self):
        while self.runs:
            self.i = self.i+self.forwards-(not self.forwards)
            if self.i > self.max:
                self.i = self.min
            if self.i < self.min:
                self.i = self.max
            yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.event_source.interval = 200
        self.forwards = True
        self.start()

    def fastforward(self, event=None):
        self.event_source.interval = 100
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.event_source.interval = 200
        self.forwards = False
        self.start()

    def fastbackward(self, event=None):
        self.event_source.interval = 100
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.i > self.min and self.i < self.max:
            self.i = self.i+self.forwards-(not self.forwards)
        elif self.i == self.min and self.forwards:
            self.i += 1
        elif self.i == self.max and not self.forwards:
            self.i -= 1
        self.func(self.i)
        self.slider.set_val(self.i)
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.64, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        fbax = divider.append_axes("right", size="80%", pad=0.05)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ffax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = mwidgets.Button(playerax, label='$\u29CF$')
        self.button_fastback = mwidgets.Button(fbax, label='$\u25C0\u25C0$')
        self.button_back = mwidgets.Button(bax, label='$\u25C0$')
        self.button_stop = mwidgets.Button(sax, label='$\u25A0$')
        self.button_forward = mwidgets.Button(fax, label='$\u25B6$')
        self.button_fastforward = mwidgets.Button(ffax, label='$\u25B6\u25B6$')
        self.button_oneforward = mwidgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_fastback.on_clicked(self.fastbackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_fastforward.on_clicked(self.fastforward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = mwidgets.Slider(sliderax, '', self.min, self.max,
                                      valinit=self.i, valfmt='%0.0f',
                                      valstep=1)
        self.slider.on_changed(self.set_pos)

    def set_pos(self, i):
        self.i = int(self.slider.val)
        self.func(self.i)

    def update(self, i):
        self.slider.set_val(i)


# Main function
def main(**kwargs):

    # Load Python plotting modules
    output_file = kwargs['output']
    if output_file != 'show':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    # get input filename
    input_file = kwargs['input']
    nfiles = int(kwargs['nfiles'])

    fprefix = input_file[:-9]
    fnumber = int(input_file[-9:-4])
    fnames = [fprefix+str(fnumber+i).zfill(5)+'.tab' for i in range(nfiles)]
    data = []
    for n in range(nfiles):
        data.append(athena_read.tab(fnames[n]))

    # read data, get variable names and check they are valid
    yvar = kwargs['variables']
    if yvar not in data[0]:
        print('Invalid input variable name, valid names are:')
        for key in data[0]:
            print(key)
        raise RuntimeError
    # set x/y data
    y_vals = []
    x_vals = []
    for n in range(nfiles):
        y_vals.append(data[n][yvar])
    if 'x1v' in data[0]:
        xvar = 'x1v'
    if 'x2v' in data[0]:
        xvar = 'x2v'
    if 'x3v' in data[0]:
        xvar = 'x3v'
    for n in range(nfiles):
        x_vals.append(data[n][xvar])

    # print(xvar)
    # print(data)
    # print(y_vals)
    # print(x_vals)

    # make single plot
    if (nfiles == 1):
        # Plot data
        plt.figure()
        plt.plot(x_vals[0], y_vals[0], '.')
        plt.show()

    # make animation with multiple files
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        def update_func(i):
            ax.clear()
            ax.plot(x_vals[i], y_vals[i], '.')
            # if xlim != (None,None):
            #     ax.set_xlim(xlim)
            # if ylim != (None,None):
            #     ax.set_ylim(ylim)
            ax.set_title('Time=%f'%data[i]['time'])  # noqa
            ax.set_xlabel(xvar)
            ax.set_ylabel(yvar)
        Player(fig, update_func, maxi=(nfiles-1))
        plt.show()
        # to save movie as mp4 use following instead of 'Player'
        # anim=FuncAnimation(fig, update_func)
        # plt.show()
        # FFwriter = animation.FFMpegWriter(fps=20)
        # anim.save('./animation.mp4', writer = FFwriter)


# Execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='name of input (tab) file')
    parser.add_argument('-o', '--output',
                        default='show',
                        help='image filename; omit to display to screen')
    parser.add_argument('-v', '--variables',
                        help='comma-separated list of variables to be plotted')
    parser.add_argument('-n', '--nfiles',
                        default=1,
                        help='number of files to be plotted for animations')

    args = parser.parse_args()
    main(**vars(args))

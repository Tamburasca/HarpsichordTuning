from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
from timeit import default_timer
import logging

from Tuning import parameters

logging.basicConfig(format=parameters.myformat,
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
if parameters.DEBUG:
    logging.getLogger().setLevel(logging.DEBUG)


class MPmatplot(Process):
    def __init__(self, queue, **kwargs):
        super().__init__(daemon=True)
        self.__queue = queue
        self.__tuning = kwargs.get('tuning')
        self.__a1 = kwargs.get('a1')
        self.__firstplot = True
        self.__resolution = parameters.RATE / parameters.SLICE_LENGTH * 1.5
        plt.ion()  # Stop matplotlib windows from blocking
        logging.debug(
            "Resolution incl. Hanning apodization (Hz/channel) ~ {0}"
            .format(str(self.__resolution)))

    @staticmethod
    def pie(axes, displaced, key_pressed):
        """
        matplotlib subroutine for pie inlet
        """
        # print(axes.__dict__)
        # delete all patches and texts from inset_pie axes that piled up
        while axes.patches:
            axes.patches.pop()
        while axes.texts:
            axes.texts.pop()
        axes.text(x=0,
                  y=0,
                  s=key_pressed,
                  fontdict={'fontsize': 20,
                            'horizontalalignment': 'center',
                            'verticalalignment': 'center'})
        # outer pie
        axes.pie(
            [-displaced, 100 + displaced] if displaced < 0 else [
                displaced, 100 - displaced],
            startangle=90,
            colors=['red' if displaced < 0 else 'green', 'white'],
            counterclock=displaced < 0,
            labels=(
                "{0:.0f} cent".format(displaced) if key_pressed else '', ''
            ),
            wedgeprops=dict(width=.6,
                            edgecolor='k',
                            lw=.5))
        # inner pie
        axes.pie([1],
                 # 2 cents within the target means key is well tuned
                 # paint pie white (default) to make it opaque
                 colors='y' if key_pressed and -2 < displaced < 2 else 'w',
                 radius=.4
                 )

        return

    @staticmethod
    def eventcollection(axes, peak_list, f_meas):
        """
        vertical bar subroutine
        """
        # remove all previous collections from axes, reverse order
        while axes.collections:
            axes.collections.pop()
        y_axis0, y_axis1 = axes.get_ylim()
        if parameters.DEBUG:
            yevents = EventCollection(
                positions=peak_list,
                color='tab:orange',
                lineoffset=(y_axis0 + y_axis1) / 2,
                linelength=abs(y_axis0) + y_axis1,
                linewidth=1.
            )
        else:
            yevents = EventCollection(
                positions=peak_list,
                color='tab:orange',
                linelength=-2 * y_axis0,
                lineoffset=0.,
                linewidth=2.
            )
        axes.add_collection(yevents)
        yevents1 = EventCollection(positions=f_meas,
                                   color='tab:red',
                                   linelength=-2 * y_axis0,
                                   lineoffset=y_axis0,
                                   linewidth=2.
                                   )
        axes.add_collection(yevents1)

        return

    def run(self):
        # run eternally
        while True:
            # fetch parameter from queue, block till message is available
            dic = self.__queue.get(block=True)
            # check if there're already some messages more than that just picked
            if self.__queue.qsize() > 1:
                logging.warning("{0} messages in MP queue".format(
                    self.__queue.qsize()))
            _start = default_timer()
            t1 = dic.get('t1')
            yfft = dic.get('yfft')
            ymax = max(yfft)
            fmin = dic.get('fmin')
            fmax = dic.get('fmax')

            displayed_title = "{0:s} (a1={1:3.0f} Hz)".format(self.__tuning,
                                                              self.__a1)
            info_text = "Resolution: {2:3.1f} Hz/channel\n" \
                        "Audio shape: {0} [slices, samples]\n" \
                        "Slice shift: {1:d} samples".format(
                            dic.get('slices').shape,
                            dic.get('step'),
                            self.__resolution)
            info_color = 'red' if dic.get('slices').shape[0] > 3 else 'black'
            font_title = {'family': 'serif',
                          'color': 'darkred',
                          'weight': 'normal',
                          'size': 14}

            # Matplotlib block
            if self.__firstplot:
                # Setup figure, axis, lines, text and initiate plot once
                # and copy background
                fig = plt.gcf()
                ax1 = fig.add_subplot(111)
                fig.set_size_inches(12, 6)
                fig.canvas.set_window_title(
                    'Digital String Tuner (c) Ralf Antonius Timmermann')
                # inset_axes with nested pie and equal aspect ratio
                inset_pie = ax1.inset_axes(
                    bounds=[0.65, 0.5, 0.35, 0.5],
                    zorder=5)  # default
                inset_pie.axis('equal')
                # define plot
                ln1, = ax1.plot(t1, yfft)
                text = ax1.text(fmax, ymax, '',
                                verticalalignment='top',
                                horizontalalignment='right',
                                fontsize=12,
                                fontweight='bold'
                                )
                text1 = ax1.text(fmin, ymax, '',
                                 horizontalalignment='left',
                                 verticalalignment='top')
                ax1.set_title(label=displayed_title,
                              loc='right',
                              fontdict=font_title)
                ax1.set_xlabel('Frequency/Hz')
                ax1.set_ylabel('Intensity/arb. units')
                ax1background = fig.canvas.copy_from_bbox(ax1.bbox)
            else:
                ln1.set_xdata(t1)
                ln1.set_ydata(yfft)
            # set attributes of subplot
            ax1.set_xlim([fmin, fmax])
            # permit some percentages of margin to the x-axes
            ax1.set_ylim([-0.04 * ymax, 1.025 * ymax])
            text.set_x(fmax)
            text.set_y(ymax)
            # call nested pie inset
            self.pie(axes=inset_pie,
                     displaced=dic.get('off', 0.),
                     key_pressed=dic.get('key', ''))
            text1.set_text(info_text)
            text1.set_color(info_color)
            text1.set_x(fmin)
            text1.set_y(ymax)
            # plot vertical bars
            self.eventcollection(ax1,
                                 dic.get('peaklist'),
                                 dic.get('f_measured'))
            # Rescale the axis so that the data can be seen in the plot
            # if you know the bounds of your data you could just set this
            # once, such that the axis don't keep changing
            ax1.relim()
            ax1.autoscale_view()

            if self.__firstplot:
                plt.pause(0.0001)
                self.__firstplot = False
            else:
                # restore background
                fig.canvas.restore_region(ax1background)
                # redraw just the points
                ax1.draw_artist(ln1)
                ax1.draw_artist(text)
                ax1.draw_artist(text1)
                # fill in the axes rectangle
                fig.canvas.blit(ax1.bbox)

            # resume audio streaming, expect retardation for status change
            fig.canvas.flush_events()

            _stop = default_timer()
            logging.debug("time utilized for matplotlib: {0:.2f} ms".format(
                (_stop - _start) * 1000.))

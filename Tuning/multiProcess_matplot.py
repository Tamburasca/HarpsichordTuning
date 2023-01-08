from multiprocessing import Process
import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
from matplotlib.axes import Axes
from numpy.fft import rfftfreq
from numpy import ndarray
from timeit import default_timer
import logging
from typing import List
from queue import Queue
# internal
from Tuning import parameters


class MPmatplot(Process):
    def __init__(self, queue: Queue, **kwargs):
        super().__init__(daemon=True)
        self.__firstplot = True
        # factor accounts for the Gaussian apodization
        self.__resolution = parameters.RATE / parameters.SLICE_LENGTH * 2.62
        self.__t1 = rfftfreq(parameters.SLICE_LENGTH,
                             1./parameters.RATE)
        self.__queue = queue
        self.__tuning = kwargs.get('tuning')
        self.__a1 = kwargs.get('a1')
        logging.debug(
            "Resolution incl. Gaussian apodization (Hz/channel) ~ {0}"
            .format(self.__resolution))

    @staticmethod
    def pie(axes: Axes, displaced: float, key_pressed: str) -> None:
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

    @staticmethod
    def eventcollection(axes: Axes, peak_list: List, f_meas: ndarray) -> None:
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
                linewidth=1.5
            )
        axes.add_collection(yevents)
        yevents1 = EventCollection(positions=f_meas,
                                   color='tab:red',
                                   linelength=-2 * y_axis0,
                                   lineoffset=y_axis0,
                                   linewidth=1.5
                                   )
        axes.add_collection(yevents1)

    def run(self) -> None:
        """
        Matplotlib commands swapped to a proprietary process. Run in an
        endless loop. Variables (in dict) are passed from animate through a
        queue.
        :return:
        """
        ln1, ln2, text1, ax1background = None, None, None, None
        plt.ion()  # Stop matplotlib windows from blocking
        plt.rcParams['keymap.quit'].remove('q')  # disable key q from closing the window
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        fig.canvas.set_window_title(
            'Digital String Tuner (c) Ralf Antonius Timmermann')
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('Frequency/Hz')
        ax1.set_ylabel('Intensity/arb. units')
        # inset_axes with nested pie and equal aspect ratio
        inset_pie = ax1.inset_axes(
            bounds=[0.65, 0.5, 0.35, 0.5],
            zorder=5)  # default
        inset_pie.axis('equal')
        displayed_title = "{0:s} (a1={1:3.0f} Hz)".format(self.__tuning,
                                                          self.__a1)
        font_title = {'family': 'serif',
                      'color': 'darkred',
                      'weight': 'normal',
                      'size': 14}

        # run eternally
        while True:
            # fetch parameter from queue, block till message is available
            # time consumed from putting to the queue till getting <.6 ms
            dic = self.__queue.get(block=True)
            _start = default_timer()
            # check if there are already some messages more than those picked
            qsize = self.__queue.qsize()
            if qsize > 0:
                logging.warning("{0} messages in MP queue".format(qsize))
            baseline = dic.get('baseline')
            noise_toggle = dic.get('noise_toggle')
            yfft = dic.get('yfft') if not noise_toggle or baseline is None else baseline
            ymax = max(yfft)
            fmin = dic.get('fmin')
            fmax = dic.get('fmax')
            info_text = "Resolution: {2:3.1f} Hz (-6 dB Main Lobe Width)\n" \
                        "Audio shape: {0} [slices, samples]\n" \
                        "Slice shift: {1:d} samples\n" \
                        "{3}".format(
                            dic.get('slices').shape,
                            dic.get('step'),
                            self.__resolution,
                            "" if baseline is None else "Noise threshold: Yes")
            info_color = 'red' if dic.get('slices').shape[0] > 3 else 'black'
            if self.__firstplot:
                # Setup line, define plot, text, and copy background once
                ln1, = ax1.plot(self.__t1, yfft)
                # ln2, = ax1.plot(self.__t1, baseline)
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
                ax1background = fig.canvas.copy_from_bbox(ax1.bbox)
            else:
                ln1.set_xdata(self.__t1)
                ln1.set_ydata(yfft)
                # ln2.set_xdata(self.__t1)
                # ln2.set_ydata(baseline)
            # set attributes of subplot
            ax1.set_xlim([fmin, fmax])
            # permit some percentages of margin to the x-axes
            ax1.set_ylim([-0.04 * ymax, 1.025 * ymax])
            if noise_toggle:
                ln1.set_color('orange')
            elif not noise_toggle and ln1.get_color() == 'orange':
                ln1.set_color('#1f77b4')
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
            # if you know the bounds of your data you could just set this once,
            # such that the axis don't keep changing
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

            logging.debug("Time utilized for matplot: {0:.2f} ms".format(
                (default_timer() - _start) * 1_000)
            )

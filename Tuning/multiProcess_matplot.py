import logging
from _tkinter import TclError
from multiprocessing import Process, Queue
from timeit import default_timer

import matplotlib
import matplotlib.pyplot as plt
# from PyQt6.QtCore import Qt  # Wayland issues need to be solved
from matplotlib.axes import Axes
from matplotlib.collections import EventCollection
from numpy.fft import rfftfreq
from numpy.typing import NDArray

# internal
import parameters


class MPmatplot(Process):
    def __init__(
            self,
            queue: Queue,
            a1: float,
            tuning: str
    ) -> None:
        """
        Matplotlib process instance
        :param queue:
        :param a1: pitch level
        :param tuning: temperament key word
        """
        super().__init__(daemon=True)
        self.__firstplot: bool = True
        # factor accounts for the Gaussian apodization
        self.__resolution: float = parameters.RATE / parameters.SLICE_LENGTH * 2.62
        self.__t1: NDArray = rfftfreq(
            parameters.SLICE_LENGTH,
            1. / parameters.RATE
        )
        self.__queue: Queue = queue
        self.__a1: float = a1
        self.__tuning: str = tuning
        logging.debug(
            "Resolution with Gaussian apodization (Hz/channel) ~ {0}"
            .format(self.__resolution))

    @staticmethod
    def pie(
            axes: Axes,
            displaced: float,
            key_pressed: str
    ) -> None:
        """
        matplotlib subroutine for pie inlet
        """
        # print(axes.__dict__)
        # delete all patches and texts from inset_pie axes that piled up
        while axes.patches:
            axes.patches[0].remove()
        while axes.texts:
            axes.texts[0].remove()
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
                            lw=0.))
        # inner pie
        axes.pie(x=[1],
                 # 1 cent within the target means key is well tuned
                 # paint pie white (default) to make it opaque
                 colors='y' if key_pressed and -1 < displaced < 1 else 'w',
                 radius=.4
                 )

    @staticmethod
    def eventcollection(
            axes: Axes,
            peak_list: list[float],
            f_meas: NDArray
    ) -> None:
        """
        vertical bar subroutine
        """
        # remove all previous collections from axes, reverse order
        while axes.collections:
            axes.collections[0].remove()
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
                                   linelength=-1.5 * y_axis0,
                                   lineoffset=y_axis0,
                                   linewidth=1.2
                                   )
        axes.add_collection(yevents1)

    @staticmethod
    def on_press(event):
        print('press', event.key)

    @staticmethod
    def do_nothing():
        print("Use ctrl-y to exit program!")
        pass

    def run(self) -> None:
        """
        Matplotlib commands swapped to a proprietary process. Run in an
        endless loop. Variables (in dict) are passed from animate through a
        queue.
        :return:
        """
        ln1, ln2, text, text1, ax1background = None, None, None, None, None
        plt.ion()  # Stop matplotlib windows from blocking
        print(f"Matplotlib backend used: {matplotlib.get_backend()}")
        plt.rcParams['backend'] = 'TkAgg'
        plt.rcParams['keymap.quit'].remove('q')  # disable key q from closing the window
        plt.rcParams['keymap.quit'].remove('ctrl+w')
        plt.rcParams['keymap.quit'].remove('cmd+w')
        fig = plt.gcf()
        # self.__cid = fig.canvas.mpl_connect('key_press_event', self.on_press)
        # disable closing figure button in the upper toolbar
        win = fig.canvas.manager.window  # type: ignore
        # TkAgg
        win.protocol('WM_DELETE_WINDOW', self.do_nothing)
        # QtAgg
        # win.setWindowFlags(win.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        # win.setWindowFlags(win.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)
        fig.set_size_inches(12, 6)
        fig.canvas.manager.set_window_title(
            'Digital String Tuner (c) Ralf Antonius Timmermann')
        ax1 = fig.add_subplot(111)
        ax1.set_xlabel('Frequency/Hz')
        ax1.set_ylabel('Intensity/arb. units')
        # inset_axes with nested pie and equal aspect ratio
        inset_pie = ax1.inset_axes(
            bounds=(0.65, 0.5, 0.35, 0.5),
            zorder=5)  # default
        # inset_pie.axis('equal') # worked with matplotlib 3.9.1
        inset_pie.set_aspect('equal', 'box')  # seems to work with matplotlib 3.10.6
        displayed_title = "{0:s} (A4 = {1:3.0f} Hz)".format(self.__tuning,
                                                            self.__a1)
        font_title = {'family': 'serif',
                      'color': 'darkred',
                      'weight': 'normal',
                      'size': 14}

        # run eternally until error occurs (window closed abnormally) or is
        # exited elsewhere via ctrl-y
        while True:
            # try:
            #     # will throw an error if the window has been closed abnormally
            #     _ = win.winfo_id()
            # except TclError:
            #     logging.info("Matplotlib window closed by user abnormally ...")
            #     break
            # fetch parameter from queue, block till message is available
            # time consumed from putting to the queue till getting <.6 ms
            dic = self.__queue.get(block=True)
            _start = default_timer()
            # check if there are already some messages more than those picked
            # relocated to FFTonLiveAudio.py
            # qsize = self.__queue.qsize()
            # if qsize > 0:
            #    logging.warning("{0} messages in MP queue".format(qsize))
            baseline = dic.get('baseline')
            noise_toggle = dic.get('noise_toggle')
            yfft = dic.get('yfft') \
                if not noise_toggle or baseline is None \
                else baseline
            ymax = max(yfft)
            fmin = dic.get('fmin')
            fmax = dic.get('fmax')
            if noise_toggle:
                measure_status = "Measuring"
            elif baseline is None:
                measure_status = "No"
            else:
                measure_status = "Yes"
            # obsolete
            # " Resolution: {2:3.1f} Hz (-6 dB Main Lobe Width)\n"
            info_text = \
                f" Audio shape: {dic.get('slices').shape} [slices, samples]\n" \
                f" Slice shift: {dic.get('step'):d} samples\n" \
                f" Peaks (no): {dic.get('npeaks')}\n" \
                f" Noise background: {measure_status}"
            info_color = 'red' if dic.get('slices').shape[0] > 3 else 'black'
            if self.__firstplot:
                # Setup line, define plot, text, and copy background once
                ln1, = ax1.plot(self.__t1, yfft)
                # ln2, = ax1.plot(self.__t1, baseline)
                text = ax1.text(x=fmax, y=ymax, s='',
                                verticalalignment='top',
                                horizontalalignment='right',
                                fontsize=12,
                                fontweight='bold'
                                )
                text1 = ax1.text(x=fmin, y=ymax, s='',
                                 horizontalalignment='left',
                                 verticalalignment='top')
                ax1.set_title(label=displayed_title,
                              loc='right',
                              fontdict=font_title)
                ax1background = fig.canvas.copy_from_bbox(ax1.bbox) # type: ignore
            else:
                ln1.set_xdata(self.__t1)
                ln1.set_ydata(yfft)
                # ln2.set_xdata(self.__t1)
                # ln2.set_ydata(baseline)
            # set attributes of subplot
            ax1.set_xlim((fmin, fmax))
            if ymax == 0.: continue
            # permit some percentages of margin to the x-axes
            ax1.set_ylim((-0.04 * ymax, 1.025 * ymax))
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
                fig.canvas.restore_region(ax1background) # type: ignore
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

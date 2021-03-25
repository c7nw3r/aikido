import logging

from aikido.__api__.dojo_listener import DojoListener
from aikido.__api__.listener.event.on_after_backpropagation import OnAfterBackpropagation
from aikido.__api__.listener.event.on_dan_finished import OnDanFinished
from aikido.__api__.listener.event.on_dan_started import OnDanStarted


class LiveLossPlotListener(DojoListener):
    """
    DojoListener implementation which renders a livelossplot after finishing a dan.
    """

    def __init__(self):
        try:
            from livelossplot import PlotLosses
            import matplotlib.pyplot as plt
            import matplotlib as mpl

            plt.style.use('dark_background')
            plt.rcParams['axes.facecolor'] = '#282828'

            self.liveloss = None
            self.total_dan_loss = 0.0
            self.PlotLosses = PlotLosses
        except ImportError:
            logging.error("-" * 100)
            logging.error("no livelossplot installation found. see https://pypi.org/project/livelossplot/")
            logging.error("-" * 100)
            return

    def dan_started(self, event: OnDanStarted):
        self.liveloss = self.PlotLosses()
        self.total_dan_loss = 0.0

    def after_backprogagation(self, event: OnAfterBackpropagation):
        self.total_dan_loss += event.loss.wrapped.item()

    def dan_finished(self, event: OnDanFinished):
        self.liveloss.update({"loss": self.total_dan_loss})
        self.liveloss.draw()

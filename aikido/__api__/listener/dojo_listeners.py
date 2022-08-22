from typing import List

from aikido.__api__.dojo import DojoListener
from aikido.dojo.listener.cuda_listener import CudaListener
from aikido.dojo.listener.stopwatch_listener import StopwatchListener
from aikido.dojo.listener.tqdm_listener import TqdmListener


class DojoListeners:

    def __init__(self, listeners: List[DojoListener]):
        self.listeners = [listener for listener in listeners]
        self._register_if_necessary(TqdmListener())
        self._register_if_necessary(StopwatchListener())
        self._register_if_necessary(CudaListener())
        self.listeners.sort(key=lambda x: x.get_order())

    def _register_if_necessary(self, listener: DojoListener):
        for _listener in self.listeners:
            if type(_listener) is type(listener):
                return
        self.listeners.append(listener)

    def __iter__(self):
        return self.listeners.__iter__()

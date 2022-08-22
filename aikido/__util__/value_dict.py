from collections import OrderedDict


class ValueDict(OrderedDict):

    def __iter__(self):
        for key in super().__iter__():
            yield self.get(key)

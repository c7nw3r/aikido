from collections import OrderedDict
from typing import Optional, TypeVar, Generic, Callable

T = TypeVar("T")


class LRUCache(Generic[T]):

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def invalidate_cache(self, key: str):
        if key in self.cache:
            del self.cache[key]

    def get_from_cache(self, key: str) -> Optional[T]:
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put_into_cache(self, key: str, value: T):
        if self.capacity <= 0:
            return

        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def get_or_else(self, key: str, callback: Callable[[str], T]) -> T:
        if key not in self.cache:
            value = callback(key)
            self.put_into_cache(key, value)
            return value
        return self.cache[key]
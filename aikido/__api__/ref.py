import typing

from dataclasses import dataclass

T = typing.TypeVar("T")


@dataclass
class Ref(typing.Generic[T]):
    wrapped: T

    def set_wrapped(self, wrapped: T):
        self.wrapped = wrapped

    def get_wrapped(self) -> T:
        return self.wrapped

    def __getitem__(self, item):
        return self.wrapped[item]

    def apply(self, callable: typing.Callable[[T], T]):
        self.wrapped = callable(self.wrapped)

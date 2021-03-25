import typing

from dataclasses import dataclass

T = typing.TypeVar("T")


@dataclass
class Ref(typing.Generic[T]):
    wrapped: T
    desc: str = "default"

    def set_wrapped(self, wrapped: T, desc: str = "default"):
        self.wrapped = wrapped
        self.desc = desc

    def get_wrapped(self) -> T:
        return self.wrapped

    def __getitem__(self, item):
        return self.wrapped[item]

    def apply(self, callable: typing.Callable[[T], T]):
        self.wrapped = callable(self.wrapped)

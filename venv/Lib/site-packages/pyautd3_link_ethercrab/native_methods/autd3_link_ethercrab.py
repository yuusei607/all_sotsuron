import enum


class Status(enum.IntEnum):
    Error = 0
    Lost = 1
    StateChanged = 2
    Resumed = 4

    @classmethod
    def from_param(cls, obj):
        return int(obj)  # pragma: no cover

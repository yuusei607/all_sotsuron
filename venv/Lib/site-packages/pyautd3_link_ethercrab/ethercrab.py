import ctypes
from collections.abc import Callable
from typing import Self

from pyautd3.driver.link import Link
from pyautd3.native_methods.autd3capi_driver import LinkPtr
from pyautd3.native_methods.utils import ConstantADT, _to_null_terminated_utf8, _validate_ptr
from pyautd3.utils import Duration

from pyautd3_link_ethercrab.native_methods.autd3_capi_link_ethercrab import EtherCrabOption as EtherCrabOption_
from pyautd3_link_ethercrab.native_methods.autd3_capi_link_ethercrab import NativeMethods as LinkEtherCrab
from pyautd3_link_ethercrab.native_methods.autd3_capi_link_ethercrab import Status as Status_

ErrHandlerFunc = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint8)  # type: ignore[arg-type]


class Status(metaclass=ConstantADT):
    _inner: Status_
    _msg: str

    @classmethod
    def __private_new__(cls: type["Status"], inner: Status_, msg: str) -> "Status":
        ins = super().__new__(cls)
        ins._inner = inner
        ins._msg = msg
        return ins

    def __new__(cls: type["Status"]) -> "Status":
        raise NotImplementedError

    def __repr__(self: Self) -> str:
        return f"{self._msg}"

    def __eq__(self: Self, other: object) -> bool:
        if not isinstance(other, Status):
            return False
        return self._inner == other._inner

    def __hash__(self: Self) -> int:
        return self._inner.__hash__()  # pragma: no cover

    @staticmethod
    def Lost() -> "Status":  # noqa: N802
        return Status.__private_new__(Status_.Lost, "")

    @staticmethod
    def StateChanged() -> "Status":  # noqa: N802
        return Status.__private_new__(Status_.StateChanged, "")

    @staticmethod
    def Error() -> "Status":  # noqa: N802
        return Status.__private_new__(Status_.Error, "")

    @staticmethod
    def Resumed() -> "Status":  # noqa: N802
        return Status.__private_new__(Status_.Resumed, "")


class EtherCrabOption:
    ifname: str | None
    state_check_period: Duration
    sync0_period: Duration
    sync_tolerance: Duration
    sync_timeout: Duration

    def __init__(
        self: Self,
        *,
        ifname: str | None = None,
        state_check_period: Duration | None = None,
        sync0_period: Duration | None = None,
        sync_tolerance: Duration | None = None,
        sync_timeout: Duration | None = None,
    ) -> None:
        self.ifname = ifname
        self.state_check_period = state_check_period or Duration.from_millis(100)
        self.sync0_period = sync0_period or Duration.from_millis(2)
        self.sync_tolerance = sync_tolerance or Duration.from_micros(1)
        self.sync_timeout = sync_timeout or Duration.from_secs(10)

    def _inner(self: Self) -> EtherCrabOption_:
        return EtherCrabOption_(
            _to_null_terminated_utf8(self.ifname) if self.ifname else None,
            self.sync0_period._inner,
            self.state_check_period._inner,
            self.sync_tolerance._inner,
            self.sync_timeout._inner,
        )


class EtherCrab(Link):
    _err_handler: Callable[[int, Status], None]
    _option: EtherCrabOption

    def __init__(self: Self, err_handler: Callable[[int, Status], None], option: EtherCrabOption) -> None:
        super().__init__()
        self._err_handler = err_handler
        self._option = option

    def _resolve(self: Self) -> LinkPtr:
        def callback_native(_context: ctypes.c_void_p, slave: ctypes.c_uint32, status: ctypes.c_uint8) -> None:  # pragma: no cover
            err = bytes(bytearray(128))  # pragma: no cover
            status_ = Status_(int(status))  # pragma: no cover
            LinkEtherCrab().link_ether_crab_status_get_msg(status_, err)  # pragma: no cover
            self._err_handler(int(slave), Status.__private_new__(status_, err.decode("utf-8").rstrip(" \t\r\n\0")))  # pragma: no cover

        self._err_handler_f = ErrHandlerFunc(callback_native)  # pragma: no cover

        return _validate_ptr(  # pragma: no cover
            LinkEtherCrab().link_ether_crab(
                self._err_handler_f,  # type: ignore[arg-type]
                ctypes.c_void_p(0),
                self._option._inner(),
            ),
        )

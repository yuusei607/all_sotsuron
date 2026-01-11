import numpy as np
import time
from pyautd3 import (
    AUTD3,
    Controller,
    Focus,
    Bessel,
    Plane,
    FocusOption,
    BesselOption,
    PlaneOption,
    Hz,
    FociSTM,
    Silencer,
    Sine,
    SineOption,
    Intensity,
    EulerAngles,
    rad,
    Phase,
)
# from pyautd3.link.ethercrab import EtherCrab, EtherCrabOption, Status
from pyautd3.link.simulator import Simulator


# def err_handler(idx: int, status: Status) -> None:
#     print(f"Device[{idx}]: {status}")
link = Simulator("127.0.0.1:8080")
w = AUTD3.DEVICE_WIDTH
if __name__ == "__main__":
    with Controller.open(
        [
            AUTD3(pos=[-w/2, 0.0, 0.0], rot=[1, 0, 0, 0]),
            AUTD3(pos=[w/2, 0.0, 0.0], rot=[1, 0, 0, 0]),
            # AUTD3(pos=[0.0, 0.0, w], rot=EulerAngles.XYZ(0*rad, np.pi / 2*rad, 0*rad)),
            # AUTD3(pos=[2*w, 0.0, 0.0], rot=EulerAngles.XYZ(0*rad, -np.pi / 2*rad, 0*rad)),
            ],
        link,
    ) as autd:
        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(Silencer())

        
        m = Sine(
            freq=150 * Hz,
            option=SineOption(),
        )

        # g = Bessel(
        #     pos=autd.center(),
        #     direction=[0.0, 0.0, 1.0],
        #     theta=np.pi/10 * rad,
        #     option=BesselOption(
        #         intensity=Intensity.MAX,
        #         phase_offset=Phase.ZERO,
        #     ),
        # )
        g = Plane(
            direction=[0, 0, 1],
            option=PlaneOption(
                intensity=Intensity.MAX,
                phase_offset=Phase.ZERO,
            ),
        )


        autd.send((m, g))
       


        _ = input()

        autd.close()

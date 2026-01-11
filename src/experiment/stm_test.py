import numpy as np
import random
import msvcrt
import time
from pyautd3 import (
    AUTD3,
    Controller,
    Focus,
    FocusOption,
    Hz,
    Silencer,
    Sine,
    SineOption,
    EulerAngles,
    rad,
    FociSTM,
    ControlPoints,
    ControlPoint,
    Static,
    GainSTM,
    GainSTMOption,
    GainSTMMode,
    Intensity,
    Phase,
    
)
from pyautd3.gain.holo import GSPAT, EmissionConstraint, GSPATOption, Pa, Naive, NaiveOption
from pyautd3.modulation import Fourier, FourierOption, Custom
from pyautd3.link.ethercrab import EtherCrab, EtherCrabOption, Status
from pyautd3.link.simulator import Simulator

w = AUTD3.DEVICE_WIDTH
h = AUTD3.DEVICE_HEIGHT

def err_handler(idx: int, status: Status) -> None:
    # print(f"Device[{idx}]: {status}")
    pass
import numpy as np
import math
from pyautd3 import (
    AUTD3, Controller, FociSTM, Hz
)
# ... (他のimportはそのまま)

# ... (generate_abc などの関数はそのまま)

def get_valid_stm_freqs(point_num, base_clock=40000):
    """
    point_num が固定のとき、設定可能な stm_freq のリストを返す関数
    条件: (stm_freq * point_num) が base_clock の約数であること
    つまり、stm_sampling_freq が base_clock の約数であればよい
    """
    valid_freqs = []
    # base_clockの約数を全探索（あるいは主要なものだけ抽出）
    # ここではサンプリング周波数としてあり得る値を探索
    for i in range(1, base_clock + 1):
        if base_clock % i == 0:
            sampling_freq = i # これが「点の切り替え速度」
            stm_freq = sampling_freq / point_num
            # 実験的に意味のある範囲（例えば0.5Hz以上）だけ残す
            if stm_freq >= 0.5: 
                valid_freqs.append(stm_freq)
    

    return sorted(valid_freqs)

link = Simulator("127.0.0.1:8080")
if __name__ == "__main__":
    with Controller.open(
        [
            AUTD3(pos=[0.0, 2*h, 10.0], rot=EulerAngles.ZYZ(np.pi * rad, - np.pi/2 * rad, 0.0 * rad)), 
            AUTD3(pos=[0.0, 2*h, w+10.0], rot=EulerAngles.ZYZ(np.pi * rad, - np.pi/2 * rad, 0.0 * rad)), 
            AUTD3(pos=[0.0, h, w+10.0], rot=EulerAngles.ZYZ(np.pi * rad, - np.pi/2 * rad, 0.0 * rad)), 
            AUTD3(pos=[0.0, h, 10.0], rot=EulerAngles.ZYZ(np.pi * rad, - np.pi/2 * rad, 0.0 * rad)), 
            AUTD3(pos=[178.0, h, 0.0], rot=EulerAngles.ZYZ(np.pi * rad, 0.0 * rad, 0.0 * rad)), 
            AUTD3(pos=[178.0, 2*h, 0.0], rot=EulerAngles.ZYZ(np.pi * rad, 0.0 * rad, 0.0 * rad)), 
            AUTD3(pos=[178.0 + w, 2*h, 0.0], rot=EulerAngles.ZYZ(np.pi * rad, 0.0 * rad, 0.0 * rad)), 
            AUTD3(pos=[178.0 + w, h, 0.0], rot=EulerAngles.ZYZ(np.pi * rad, 0.0 * rad, 0.0 * rad)), 
            AUTD3(pos=[2*w-2.0, h-141.2, 0.0], rot=[1, 0, 0, 0]), 
            AUTD3(pos=[2*w-2.0, 2*h-141.2, 0.0], rot=[1, 0, 0, 0]), 
            AUTD3(pos=[565.0, 2*h, w], rot=EulerAngles.ZYZ(np.pi * rad, np.pi/2 * rad, 0.0 * rad)),
            AUTD3(pos=[565.0, h, w], rot=EulerAngles.ZYZ(np.pi * rad, np.pi/2 * rad, 0.0 * rad)),
            AUTD3(pos=[565.0, h, 2*w], rot=EulerAngles.ZYZ(np.pi * rad, np.pi/2 * rad, 0.0 * rad)),
            AUTD3(pos=[565.0, 2*h, 2*w], rot=EulerAngles.ZYZ(np.pi * rad, np.pi/2 * rad, 0.0 * rad)),
        ],
        
        
        EtherCrab(err_handler=err_handler, option=EtherCrabOption()),
    ) as autd:
        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(Silencer())
        center = np.array([1.5*w, h, 200.0])
        radius = 3.0

        # --------------- parameter to be tuned directly ------------------
        point_num = 10
        stm_freqs = get_valid_stm_freqs(point_num)
        stm_idx = 0
        am_freq = 10.0
        # -----------------------------------------------------------------
        # --------------- parameter to be calculated ----------------------
        # velocity = 2 * np.pi * radius * stm_freq
        
        # -----------------------------------------------------------------


           

        while True:
            g = FociSTM(
                foci=(
                    ControlPoints(
                        points=[
                            ControlPoint(
                                point=center + radius * np.array([np.cos(theta), np.sin(theta), 0]),
                                phase_offset=Phase.ZERO,
                            ),
                            # ControlPoint(
                            #     point=center - radius * np.array([np.cos(theta), np.sin(theta), 0]),
                            #     phase_offset=Phase.ZERO,
                            # ),
                        ],
                        intensity=Intensity.MAX,
                    )
                    for theta in (2.0 * np.pi * i / point_num for i in range(point_num))
                ),
                config=stm_freqs[stm_idx] * Hz,
            )
            if am_freq < 5:
                m = Static(intensity=0xff) 
            else:
                m = Sine(
                    freq=am_freq * Hz,
                    option=SineOption(intensity=0xff),
                )

            autd.send((m, g))

            key = msvcrt.getch().decode('utf-8').lower()
            if key == 'a':
                am_freq += 10.0
            elif key == "s":
                stm_idx += 1
            elif key == "d":
                am_freq -= 10.0
            elif key == "f":
                stm_idx -= 1 
            elif key == "q":
                break
            print(f"am_freq: {am_freq}, stm_freq: {stm_freqs[stm_idx]}")

        autd.close()
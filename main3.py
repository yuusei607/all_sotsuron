import numpy as np
import random
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
    Static,
    
)
from pyautd3.modulation import Fourier, FourierOption, Custom
from pyautd3.link.ethercrab import EtherCrab, EtherCrabOption, Status

w = AUTD3.DEVICE_WIDTH
h = AUTD3.DEVICE_HEIGHT

def err_handler(idx: int, status: Status) -> None:
    # print(f"Device[{idx}]: {status}")
    pass
def generate_abc():
    
    # 1. まず、合計値 S を 0 から 256 の間で決める
    total_sum = random.randint(0, 256)
    
    # 2. 0 から S の間に「2つの仕切り」をランダムに置く
    #    (0からtotal_sumまでの値を2回選ぶ)
    break1 = random.randint(0, total_sum)
    break2 = random.randint(0, total_sum)
    
    # 3. 2つの仕切りをソートする (0 <= p1 <= p2 <= total_sum)
    p1 = min(break1, break2)
    p2 = max(break1, break2)
    
    # 4. 3つの区間の長さが a, b, c になる
    a = p1        # 0 から p1 まで
    b = p2 - p1   # p1 から p2 まで
    c = total_sum - p2 # p2 から total_sum まで
    
    # (a+b+c = p1 + (p2-p1) + (total_sum-p2) = total_sum となる)
    
    return a, b, c


if __name__ == "__main__":
    with Controller.open(
        [
            AUTD3(pos=[-0.7071 * w - 20.0, 0.0, 0.7071 * w + 12.0], rot=EulerAngles.ZYZ(0 * rad, np.pi/4 * rad, 0 * rad)), # 左前
            AUTD3(pos=[- 0.7071 * w - 20.0 , h, 0.7071 * w + 12.0], rot=EulerAngles.ZYZ(0 * rad, np.pi/4 * rad, 0 * rad)), # 左後ろ
            AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), # 真ん中手前
            AUTD3(pos=[0.0, h, 0.0], rot=[1, 0, 0, 0]), # 真ん中後ろ
            AUTD3(pos=[w + 20.0, 0.0, 9.0], rot=EulerAngles.ZYZ(0 * rad, -np.pi/4 * rad, 0 * rad)), # 右前
            AUTD3(pos=[w + 20.0, h, 9.0], rot=EulerAngles.ZYZ(0 * rad, -np.pi/4 * rad, 0 * rad)), # 右後ろ
        ],
        # [
        #     AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
        #     AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
        #     AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
        #     AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
        #     AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
        #     AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]),
        # ],
        EtherCrab(err_handler=err_handler, option=EtherCrabOption()),
    ) as autd:
        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(Silencer())
        center = np.array([0.0, 0.0, 150.0])
        point_num = 200
        radius = 3.0

        # g = FociSTM(
        #         foci=(
        #             center + radius * np.array([np.cos(theta), np.sin(theta), 0])
        #             for theta in (2.0 * np.pi * i / point_num for i in range(point_num))
        #         ),
        #         config=5.0 * Hz,
        #     )   
        g = Focus(
            pos=autd.center() + np.array([0.0, 0.0, 250.0]),
            option=FocusOption(),
        )

        stimuli_array = np.array([])
        sumpling_freq = 4000.0
        num_sumple = int(sumpling_freq*5)
        t = np.arange(num_sumple)
        while True:
            a, b, c = generate_abc()
            print(f'a, b, c = {a}, {b}, {c}')
            stimuli_array = a*np.sin(2*np.pi*30*t/sumpling_freq) + b*np.sin(2*np.pi*200*t/sumpling_freq) + c

            m = Custom(
                    buffer=stimuli_array,
                    sampling_config=sumpling_freq * Hz
            )
            
            autd.send((m, g))
            time.sleep(5)
            # if input():
            #     break

   

        autd.close()

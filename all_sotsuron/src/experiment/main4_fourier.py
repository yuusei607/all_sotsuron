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
def generate_three_vars_sum_to_N(N=255):

    dividers = sorted(random.sample(range(N + 1), 2))

    r1, r2 = dividers[0], dividers[1]
    
    a = r1              # 最初の区切りまで (0 - r1)
    b = r2 - r1         # 最初の区切りから2番目の区切りまで (r1 - r2)
    c = N - r2          # 2番目の区切りから最後まで (r2 - N)
    
    return a, b, c

if __name__ == "__main__":
    with Controller.open(
        [
            AUTD3(pos=[-2.0, 0.0, 196.0], rot=EulerAngles.ZYZ(0 * rad, np.pi/2 * rad, 0 * rad)), # 左前
            AUTD3(pos=[-2.0 , h, 195.0], rot=EulerAngles.ZYZ(0 * rad, np.pi/2 * rad, 0 * rad)), # 左後ろ
            AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0]), # 真ん中手前
            AUTD3(pos=[0.0, h, 0.0], rot=[1, 0, 0, 0]), # 真ん中後ろ
            AUTD3(pos=[184.0, 0.0, 1.2], rot=EulerAngles.ZYZ(0 * rad, -np.pi/2 * rad, 0 * rad)), # 右前
            AUTD3(pos=[185.0, h, 1.2], rot=EulerAngles.ZYZ(0 * rad, -np.pi/2 * rad, 0 * rad)), # 右後ろ
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
        # g = FociSTM(
        #         foci=(
        #             center + radius * np.array([np.cos(theta), np.sin(theta), 0])
        #             for theta in (2.0 * np.pi * i / point_num for i in range(point_num))
        #         ),
        #         config=5.0 * Hz,
        #     )   
        # g = Focus(
        #     pos=np.array([130.0, 200.0, 400.0]),
        #     option=FocusOption(),
        # )

       
        g = FociSTM(
            foci=(
                np.array([w/2, h + 0.5 *y - 50, 200.0]) for y in np.arange(0, 100, 1)
            ),
            config=1.0 * Hz,
        )
        
       
        while True:
            
            a, b, c = generate_three_vars_sum_to_N()
            print(f'a, b, c = {a}, {b}, {c}')
            m1 = Sine(freq=30 * Hz, option=SineOption(intensity=0xff))
            m2 = Static(intensity=0xff)
            m = Fourier(
                components=[
                    Sine(freq=30 * Hz, option=SineOption(intensity=a)),
                    Sine(freq=200 * Hz, option=SineOption(intensity=b)),
                ],
                option=FourierOption(
                    scale_factor=None,
                    clamp=False,
                    offset=c//2,
                ),
            )
            autd.send((m1, g))
            _ = input()
            # if input():
            #     break

   

        autd.close()

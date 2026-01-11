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
from pyautd3.link.simulator import Simulator

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

        l_1 = [np.array([1.5 * w, h + 5 * y - 50, 300.0]) for y in np.arange(0, 10, 1)]
        l_2 = [np.array([1.5 * w, h + 45 - 5 * y_2 - 50, 300.0]) for y_2 in np.arange(0, 10, 1)]

        g = FociSTM(
            # foci=l_1 + l_2,
            foci=[np.array([1.5 * w, h, 300.0]), np.array([1.5 * w, h+5.0, 300.0])],
            config=16.0 * Hz,
        )
        # g = FociSTM(
        #     foci=(
        #         [np.array([1.5 * w, h, 300.0]), np.array([1.5 * w, h+5.0, 300.0])],
        #     ),
        #     config=1.0 * Hz,
        # )
       
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
            autd.send((m2, g))
            _ = input()
            # if input():
            #     break

   

        autd.close()

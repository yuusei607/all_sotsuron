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


def generate_points(distance):
    points = []
    current_point = np.array([random.uniform(0.0, 10.0), random.uniform(0.0, 10.0), 0.0])
    points.append(center + current_point)

    for _ in range(999):
        while True:
                angle = random.uniform(0, 2 * np.pi)
                next_x = current_point[0] + distance * np.cos(angle)
                next_y = current_point[1] + distance * np.sin(angle)
                
                if 0.0 <= next_x <= 10.0 and 0.0 <= next_y <= 10.0:
                    current_point = np.array([next_x, next_y, 0.0])
                    points.append(center + current_point)
                    break
    return points

w = AUTD3.DEVICE_WIDTH
h = AUTD3.DEVICE_HEIGHT

def err_handler(idx: int, status: Status) -> None:
    # print(f"Device[{idx}]: {status}")
    pass


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

        autd.send(Silencer.disable())
        center = np.array([1.5*w, h, 200.0])
        radius = 3.0

        # --------------- parameter to be tuned directly ------------------
        point_num = 10
        stm_freqs = get_valid_stm_freqs(point_num)
        stm_idx = 0
        am_freq = 100.0
        # -----------------------------------------------------------------
        # --------------- parameter to be calculated ----------------------
        # velocity = 2 * np.pi * radius * stm_freq
        
        # ------------------------------------------------------------
  
        m = Sine(
                freq=am_freq * Hz,
                option=SineOption(intensity=255),
            )
        m1 = Static(intensity=200)
        # count = 0
        # while True:

        #     if count < 3:
        #         dist = 0.05
        #     elif count < 6:
        #         dist = 0.5
        #     else:
        #         dist = 4.0
        #     if count % 3 == 0:
        #         velo = 10
        #     elif count % 3 == 1:
        #         velo = 100
        #     else:
        #         velo = 1000

        #     fre = velo / (1000*dist)
        #     g = FociSTM(
        #             foci=generate_points(dist),
        #             config=fre * Hz,
        #         )
        #     autd.send((m, g))
        #     _ = input()
        #     count += 1

        distances = [0.05, 0.5, 4.0]
        velocities = [10, 100, 1000]
        am_freqs = [0, 20, 100]
        count = 0
        for dist in distances:
            for velo in velocities:
                for am_freq in am_freqs:
                    count += 1
                    print(f'generating {count}th stimulus')
                    # --- 1. AM変調 (Modulation) の切り替えロジック ---
                    if am_freq == 0:
                        # 0HzならStatic (無変調)
                        # intensity=255 にしておくとSineの最大振幅と揃います
                        m = Static(intensity=255) 
                        mod_name = "Static (0Hz)"
                    else:
                        # それ以外ならSine波
                        m = Sine(
                            freq=am_freq * Hz,
                            option=SineOption(intensity=255),
                        )
                        mod_name = f"Sine {am_freq}Hz"

                    # --- 2. STM周波数の計算 ---
                    # 提示された式: frequency = velocity / (1000 * distance)
                    fre = velo / (1000 * dist)

                    # --- 3. ログ出力 ---
                    print(f"Testing: Dist={dist}, Velo={velo}, STM_Freq={fre:.2f}Hz, AM={mod_name}")

                    # --- 4. STMの設定 ---
                    g = FociSTM(
                        foci=generate_points(dist),
                        config=fre * Hz,
                    )

                    # --- 5. デバイスへ送信 ---
                    autd.send((m, g))
                    
                    # --- 6. 待機 (Enterで次へ) ---
                    input("Press Enter for next pattern...")

        autd.close()
        
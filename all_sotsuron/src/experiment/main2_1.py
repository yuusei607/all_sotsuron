import numpy as np
import time
from pyautd3 import (
    AUTD3,
    Controller,
    Focus,
    FocusOption,
    Hz,
    FociSTM,
    Silencer,
    Sine,
    SineOption,
    Intensity,
    WithSegment,
    Segment,
    transition_mode,
    SwapSegmentModulation
)
from pyautd3.link.ethercrab import EtherCrab, EtherCrabOption, Status


def err_handler(idx: int, status: Status) -> None:
    print(f"Device[{idx}]: {status}")


if __name__ == "__main__":
    with Controller.open(
        [AUTD3(pos=[0.0, 0.0, 0.0], rot=[1, 0, 0, 0])],
        EtherCrab(err_handler=err_handler, option=EtherCrabOption()),
    ) as autd:
        firmware_version = autd.firmware_version()
        print(
            "\n".join(
                [f"[{i}]: {firm}" for i, firm in enumerate(firmware_version)],
            ),
        )

        autd.send(Silencer())

        # 共通で使用する変調を定義
        m = Sine(
            150 * Hz,
            option=SineOption(),
        )

        # 切り替える2つの焦点座標を定義
        center = autd.center() + np.array([0.0, 0.0, 150.0])
        pos_a = center + np.array([20.0, 0.0, 0.0])
        pos_b = center - np.array([20.0, 0.0, 0.0])

        gain_a = Focus(
            pos=pos_a,
            option=FocusOption(),
        )
        gain_b = Focus(
            pos=pos_b,
            option=FocusOption(),
        )

        print("Setting up initial state...")

        # --- 初期状態の設定 ---
        # 1. まず、最初の焦点AをSegment.S0 (表舞台) に設定し、即時有効化
        
        WithSegment(
            inner=(m, gain_a),
            segment=Segment.S0,
            transition_mode=transition_mode.Immediate(),
        )
        
        # 2. 次の焦点BをSegment.S1 (舞台袖) に準備しておく
        WithSegment(
            inner=(m, gain_b),
            segment=Segment.S1,
            transition_mode=transition_mode.Later(), # まだ切り替えない
        )
        
        

        print("Starting focus transition loop... (Press Ctrl+C to stop)")

        try:
            # --- 切り替えループ ---
            while True:
                # 1秒待機 (現在、焦点Aが出力されている)
                print("-> Focus at point A")
                time.sleep(1)

                # Segmentをスワップ -> S1にあった焦点Bが有効になる
                SwapSegmentModulation(Segment.S1, transition_mode.Immediate())
                
                # S0が舞台袖になったので、次の焦点Aを準備しておく
      
                WithSegment(
                    inner=(m, gain_a),
                    segment=Segment.S0,
                    transition_mode=transition_mode.Later(),
                )
                

                # 1秒待機 (現在、焦点Bが出力されている)
                print("-> Focus at point B")
                time.sleep(1)
                
                # 再度Segmentをスワップ -> S0にあった焦点Aが有効になる
                SwapSegmentModulation(Segment.S0, transition_mode.Immediate())

                # S1が舞台袖になったので、次の焦点Bを準備しておく
            
                WithSegment(
                    inner=(m, gain_b),
                    segment=Segment.S1,
                    transition_mode=transition_mode.Later(),
                )
                
            

        except KeyboardInterrupt:
            pass

        print("\nStopping...")
import tkinter as tk
from tkinter import filedialog
import numpy as np
import random
import json
import math
from datetime import datetime

# AUTD3関連のインポート
from pyautd3 import (
    AUTD3, Controller, FocusOption, Hz, Silencer, Sine, SineOption,
    EulerAngles, rad, FociSTM, Static
)
from pyautd3.link.simulator import Simulator
from pyautd3.link.ethercrab import EtherCrab, EtherCrabOption, Status

# --- 設定値 ---
w = AUTD3.DEVICE_WIDTH
h = AUTD3.DEVICE_HEIGHT
CANVAS_SIZE = 800  # 描画領域のサイズ (ピクセル)
NODE_RADIUS = 15   # 点の大きさ
def err_handler(idx: int, status: Status) -> None:
    # print(f"Device[{idx}]: {status}")
    pass


# --- 幾何計算用の関数 (既存コードより) ---
def generate_points(distance, center):
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

# --- GUIアプリケーションクラス ---
class TactileMapApp:
    def __init__(self, root, autd_controller):
        self.root = root
        self.autd = autd_controller
        self.root.title("Tactile Spatial Arrangement Task")

        # AUTD座標の中心設定など
        self.center = np.array([1.5 * w, h, 200.0])
        
        # 刺激パラメータの生成 (27パターン)
        self.stimuli = self._generate_stimuli_params()
        
        # ドラッグアンドドロップ管理用変数
        self.drag_data = {"x": 0, "y": 0, "item": None}

        # GUIパーツの作成
        self._create_widgets()
        self._draw_nodes()

    def _generate_stimuli_params(self):
        """27パターンの刺激パラメータを生成してリストで返す"""
        distances = [0.05, 0.5, 4.0]
        velocities = [10, 100, 1000]
        am_freqs = [0, 20, 100]
        
        params = []
        idx = 0
        for dist in distances:
            for velo in velocities:
                for am in am_freqs:
                    # STM周波数の計算
                    freq = velo / (1000 * dist)
                    params.append({
                        "id": idx,
                        "dist": dist,
                        "velo": velo,
                        "am_freq": am,
                        "stm_freq": freq,
                        "x": random.randint(50, CANVAS_SIZE - 50), # 初期位置ランダム
                        "y": random.randint(50, CANVAS_SIZE - 50),
                        "color": "#{:06x}".format(random.randint(0, 0xFFFFFF)) # 識別用ランダムカラー（本来は統一しても良い）
                    })
                    idx += 1
        return params

    def _create_widgets(self):
        # メインキャンバス
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 操作パネル
        panel = tk.Frame(self.root)
        panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        btn_save = tk.Button(panel, text="Save & Quit", command=self.save_and_quit, height=2, bg="lightblue")
        btn_save.pack(side=tk.RIGHT)
        
        lbl_inst = tk.Label(panel, text="Drag circles to arrange by similarity. Click to feel.", font=("Arial", 12))
        lbl_inst.pack(side=tk.LEFT)

        # キャンバスのイベントバインド
        self.canvas.tag_bind("token", "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind("token", "<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind("token", "<B1-Motion>", self.on_drag)

    def _draw_nodes(self):
        """初期ノードの描画"""
        for s in self.stimuli:
            x, y = s["x"], s["y"]
            # 円を描画 (tagにIDを埋め込む)
            item_id = self.canvas.create_oval(
                x - NODE_RADIUS, y - NODE_RADIUS,
                x + NODE_RADIUS, y + NODE_RADIUS,
                fill="lightgrey", outline="black", width=2,
                tags=("token", f"stim_{s['id']}")
            )
            # 番号を描画
            self.canvas.create_text(x, y, text=str(s['id']), tags=("token", f"stim_{s['id']}"))

    # --- イベントハンドラ ---
    def on_press(self, event):
        """クリックされたときに呼ばれる：刺激提示とドラッグ開始"""
        # クリック位置に最も近いアイテムを探す
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags(item)
        
        # タグからIDを抽出 (stim_XX という形式)
        stim_id = -1
        for tag in tags:
            if tag.startswith("stim_"):
                stim_id = int(tag.split("_")[1])
                break
        
        if stim_id != -1:
            # 1. 刺激の再生
            self.play_stimulus(stim_id)
            
            # 2. ドラッグ開始の準備
            self.drag_data["item"] = item # テキストか円の片方だけ掴むのを防ぐため、本来はグループ化が必要だが簡易的に実装
            # ※簡易実装のため、テキストをクリックするとテキストだけ動く可能性があります。
            #   実用時は find_enclosed などでセットで動かすか、円だけ当たり判定を持たせると良いです。
            #   ここでは「円」をクリックしたと仮定して処理します。
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def on_drag(self, event):
        """ドラッグ中：位置更新"""
        item = self.drag_data["item"]
        if item:
            delta_x = event.x - self.drag_data["x"]
            delta_y = event.y - self.drag_data["y"]
            
            # 同じタグを持つオブジェクト（円とテキスト）をまとめて移動
            tags = self.canvas.gettags(item)
            for tag in tags:
                if tag.startswith("stim_"):
                    # そのIDを持つすべての要素を移動
                    self.canvas.move(tag, delta_x, delta_y)
                    
                    # 内部データの座標も更新（後で保存するため）
                    stim_id = int(tag.split("_")[1])
                    self.stimuli[stim_id]["x"] += delta_x
                    self.stimuli[stim_id]["y"] += delta_y
                    break

            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def on_release(self, event):
        """ドロップ時"""
        self.drag_data["item"] = None

    def play_stimulus(self, stim_id):
        """AUTD3へ刺激送信"""
        params = self.stimuli[stim_id]
        print(f"Playing ID:{stim_id} | Dist:{params['dist']}, Velo:{params['velo']}, AM:{params['am_freq']}")

        # AMの設定
        if params['am_freq'] == 0:
            m = Static(intensity=255)
        else:
            m = Sine(freq=params['am_freq'] * Hz, option=SineOption(intensity=255))

        # STMの設定
        g = FociSTM(
            foci=generate_points(params['dist'], self.center),
            config=params['stm_freq'] * Hz,
        )

        self.autd.send((m, g))

    def save_and_quit(self):
        """結果を保存して終了"""
        filename = f"experiment_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 必要なデータだけ抽出して保存
        export_data = []
        for s in self.stimuli:
            export_data.append({
                "id": s["id"],
                "parameters": {
                    "dist": s["dist"],
                    "velo": s["velo"],
                    "am_freq": s["am_freq"]
                },
                "position": {
                    "x": s["x"],
                    "y": s["y"]
                }
            })
            
        with open(filename, "w") as f:
            json.dump(export_data, f, indent=4)
        
        print(f"Saved to {filename}")
        self.root.destroy()


# --- メイン処理 ---
if __name__ == "__main__":
    # シミュレータ設定 (実機の場合はここを変更)
    # link = Simulator("127.0.0.1:8080")

    # Controllerの設定（デバイス構成は元のコードのまま）
    # ※デバイス配置コードが長いため省略していますが、元のコードの `with Controller.open(...)` の中身を使用します
    # geometry_config = [
    #     AUTD3(pos=[0.0, 2*h, 10.0], rot=EulerAngles.ZYZ(np.pi * rad, - np.pi/2 * rad, 0.0 * rad)),
    #     # ... (中略：ここに元の14台分の設定を入れてください) ...
    # ]
    # ※デバッグ用に1台だけで起動する場合は以下を使用
    # geometry_config = [AUTD3(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0])] 

    # 注: 以下は既存コードのデバイス構成を簡略化して書きます。実際には元の長いリストを使ってください。
    # 動作確認のため、ここだけは適切に書き換えてください。
    
    # 既存コードと同じ構成を再現する関数やリストを用意してください
    # ここでは例として「autdコンテキストの中でGUIを起動する」構造を示します

    # --- 実際の起動フロー ---
    # 元のコードのデバイス構成リストをここにコピペしてください
    devices = [
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
        ]
    
    # GUIを起動
    # AUTD3の接続コンテキストの中でTkinterのmainloopを回すのが重要です
    try:
        # ※ Simulatorなどは適宜変更してください
        with Controller.open(devices, EtherCrab(err_handler=err_handler, option=EtherCrabOption())) as autd:
            autd.send(Silencer.disable())
            
            # Tkinterウィンドウの作成
            root = tk.Tk()
            app = TactileMapApp(root, autd)
            
            # アプリ開始
            root.mainloop()
            
    except Exception as e:
        print(f"Error: {e}")
        # Simulator接続エラーなどの場合、GUIだけでテストしたいならここを工夫する
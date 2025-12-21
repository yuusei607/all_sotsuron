import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import random
import json
import math
import itertools
from collections import defaultdict
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
CANVAS_SIZE = 800   # 描画領域のサイズ (ピクセル)
ARENA_RADIUS = 350
NODE_RADIUS = 20    # 点の大きさ
ITEMS_PER_TRIAL = 5 # ★変更: 1回の提示数を5個に

def err_handler(idx: int, status: Status) -> None:
    pass

# --- Greedy法によるトライアル生成クラス ---
class GreedyTrialGenerator:
    def __init__(self, num_items, items_per_trial):
        self.num_items = num_items
        self.items_per_trial = items_per_trial
        self.trials = []
        self._generate()

    def _generate(self):
        """全ペアを網羅するまでgreedyにリストを作成"""
        all_items = list(range(self.num_items))
        all_pairs = list(itertools.combinations(all_items, 2))
        pair_counts = defaultdict(int)
        
        print(f"Generating trials for {self.num_items} items, {self.items_per_trial} per trial...")
        
        while True:
            # まだ出ていないペアを探す
            missing_pairs = [p for p in all_pairs if pair_counts[p] < 1]
            if not missing_pairs:
                break
            
            # 1. 不足ペアから1つ選んで核にする
            target_pair = random.choice(missing_pairs)
            current_trial_items = set(target_pair)
            
            # 2. 残りの枠を「未評価ペアを最も多く含むアイテム」で埋める
            while len(current_trial_items) < self.items_per_trial:
                best_candidate = -1
                max_new_pairs = -1
                
                candidates = [i for i in all_items if i not in current_trial_items]
                random.shuffle(candidates)
                
                for cand in candidates:
                    new_pairs_count = 0
                    for existing in current_trial_items:
                        pair = tuple(sorted((cand, existing)))
                        if pair_counts[pair] < 1:
                            new_pairs_count += 1
                    
                    if new_pairs_count > max_new_pairs:
                        max_new_pairs = new_pairs_count
                        best_candidate = cand
                
                if best_candidate != -1:
                    current_trial_items.add(best_candidate)
                else:
                    current_trial_items.add(random.choice(candidates))
            
            # 記録
            trial_list = list(current_trial_items)
            self.trials.append(trial_list)
            
            for p in itertools.combinations(trial_list, 2):
                pair_counts[tuple(sorted(p))] += 1
                
        print(f"Generated {len(self.trials)} trials to cover all pairs.")

# --- 幾何計算用の関数 ---
def generate_points(distance, center):
    points = []
    # ランダムウォークの初期点
    current_point = np.array([random.uniform(0.0, 10.0), random.uniform(0.0, 10.0), 0.0])
    points.append(center + current_point)
    for _ in range(999): # 1000点分
        while True:
            angle = random.uniform(0, 2 * np.pi)
            next_x = current_point[0] + distance * np.cos(angle)
            next_y = current_point[1] + distance * np.sin(angle)
            # 1cm (10mm) 平方の範囲内に収める
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
        self.root.title("Tactile Spatial Arrangement Task (8 Stimuli)")

        # AUTD座標の中心設定
        self.center = np.array([1.5 * w, h, 200.0])
        
        # 1. 刺激パラメータの生成 (N=8)
        self.all_params = self._generate_stimuli_params()
        
        # 2. トライアルリストの生成
        generator = GreedyTrialGenerator(num_items=len(self.all_params), items_per_trial=ITEMS_PER_TRIAL)
        self.trial_list = generator.trials
        self.current_trial_idx = 0
        
        # 結果保存用
        self.results = [None] * len(self.trial_list)

        # ドラッグ管理用
        self.drag_data = {"x": 0, "y": 0, "item": None}

        # GUIパーツ作成
        self._create_widgets()
        
        # 最初のトライアルを開始
        self.load_trial()

    def _generate_stimuli_params(self):
        """★変更: 8パターンの刺激パラメータを生成"""
        distances = [0.05, 4.0] 
        velocities = [10, 1000] # 100を削除
        am_freqs = [0, 100]     # 20を削除
        
        params = []
        idx = 0
        for dist in distances:
            for velo in velocities:
                for am in am_freqs:
                    freq = velo / (1000 * dist)
                    params.append({
                        "id": idx,
                        "dist": dist,
                        "velo": velo,
                        "am_freq": am,
                        "stm_freq": freq,
                        "color": "#{:06x}".format(random.randint(0, 0xFFFFFF))
                    })
                    idx += 1
        return params

    def _create_widgets(self):
        # 上部情報パネル
        info_panel = tk.Frame(self.root)
        info_panel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.lbl_progress = tk.Label(info_panel, text="", font=("Arial", 14, "bold"))
        self.lbl_progress.pack(side=tk.LEFT)

        # メインキャンバス
        self.canvas = tk.Canvas(self.root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 円形アリーナの描画
        cx, cy = CANVAS_SIZE / 2, CANVAS_SIZE / 2
        self.canvas.create_oval(
            cx - ARENA_RADIUS, cy - ARENA_RADIUS,
            cx + ARENA_RADIUS, cy + ARENA_RADIUS,
            outline="lightgrey", width=3, dash=(5, 5),
            tags="arena_bg"
        )
        self.canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="black", outline="")

        # 操作パネル
        panel = tk.Frame(self.root)
        panel.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Nextボタン
        self.btn_next = tk.Button(panel, text="Next Trial", command=self.next_trial, height=2, width=15, bg="lightblue")
        self.btn_next.pack(side=tk.RIGHT, padx=5)

        # Prevボタン
        self.btn_prev = tk.Button(panel, text="< Back", command=self.prev_trial, height=2, width=10, bg="lightgrey")
        self.btn_prev.pack(side=tk.RIGHT, padx=5)
        
        lbl_inst = tk.Label(panel, text="円の内側全体を使って配置してください。\n似ているものは近く、違うものは遠くへ。", font=("Arial", 11), fg="red")
        lbl_inst.pack(side=tk.LEFT)

        # キャンバスのイベント
        self.canvas.tag_bind("token", "<ButtonPress-1>", self.on_press)
        self.canvas.tag_bind("token", "<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind("token", "<B1-Motion>", self.on_drag)

    def load_trial(self):
        """現在のトライアルIDに基づいてキャンバスをリセット・再描画"""
        items = self.canvas.find_all()
        for item in items:
            tags = self.canvas.gettags(item)
            if "arena_bg" not in tags:
                self.canvas.delete(item)
        
        self.lbl_progress.config(text=f"Trial: {self.current_trial_idx + 1} / {len(self.trial_list)}")
        
        if self.current_trial_idx == len(self.trial_list) - 1:
            self.btn_next.config(text="Finish & Save", bg="orange")
        else:
            self.btn_next.config(text="Next Trial", bg="lightblue")
            
        if self.current_trial_idx == 0:
            self.btn_prev.config(state=tk.DISABLED)
        else:
            self.btn_prev.config(state=tk.NORMAL)

        current_items_indices = self.trial_list[self.current_trial_idx]
        
        saved_data = self.results[self.current_trial_idx]
        saved_positions = {}
        if saved_data is not None:
            for item in saved_data["items"]:
                saved_positions[item["id"]] = (item["x"], item["y"])
        
        cx, cy = CANVAS_SIZE / 2, CANVAS_SIZE / 2
        init_radius = ARENA_RADIUS - 40 
        angle_step = 2 * math.pi / len(current_items_indices)
        
        self.current_canvas_items = [] 
        
        for i, item_idx in enumerate(current_items_indices):
            param = self.all_params[item_idx]
            
            if item_idx in saved_positions:
                x, y = saved_positions[item_idx]
            else:
                angle = i * angle_step + random.uniform(-0.1, 0.1)
                x = cx + init_radius * math.cos(angle)
                y = cy + init_radius * math.sin(angle)
            
            self._draw_single_node(x, y, param)
            
            self.current_canvas_items.append({
                "id": param["id"],
                "tag": f"stim_{param['id']}",
                "x": x,
                "y": y
            })

    def _draw_single_node(self, x, y, param):
        tag = f"stim_{param['id']}"
        color = param["color"]
        self.canvas.create_oval(
            x - NODE_RADIUS, y - NODE_RADIUS,
            x + NODE_RADIUS, y + NODE_RADIUS,
            fill="lightgrey", outline="black", width=2,
            activefill=color,
            tags=("token", tag)
        )
        self.canvas.create_text(x, y, text=str(param['id']), tags=("token", tag))

    def _save_current_screen(self):
        trial_data = {
            "trial_index": self.current_trial_idx,
            "items": []
        }
        for item in self.current_canvas_items:
            tag = item["tag"]
            coords = self.canvas.coords(tag)
            if coords:
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2
                trial_data["items"].append({
                    "id": item["id"],
                    "x": cx,
                    "y": cy,
                    "params": self.all_params[item["id"]]
                })
        self.results[self.current_trial_idx] = trial_data

    def next_trial(self):
        self._save_current_screen()
        if self.current_trial_idx < len(self.trial_list) - 1:
            self.current_trial_idx += 1
            self.load_trial()
        else:
            self.save_and_quit()

    def prev_trial(self):
        self._save_current_screen()
        if self.current_trial_idx > 0:
            self.current_trial_idx -= 1
            self.load_trial()

    def on_press(self, event):
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags(item)
        
        stim_id = -1
        for tag in tags:
            if tag.startswith("stim_"):
                stim_id = int(tag.split("_")[1])
                self.canvas.tag_raise(tag)
                break
        
        if stim_id != -1:
            self.play_stimulus(stim_id)
            self.drag_data["item"] = item
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def on_drag(self, event):
        item = self.drag_data["item"]
        if item:
            delta_x = event.x - self.drag_data["x"]
            delta_y = event.y - self.drag_data["y"]
            
            tags = self.canvas.gettags(item)
            target_tag = ""
            for tag in tags:
                if tag.startswith("stim_"):
                    target_tag = tag
                    break
            
            coords = self.canvas.coords(target_tag)
            cur_cx = (coords[0] + coords[2]) / 2
            cur_cy = (coords[1] + coords[3]) / 2
            
            next_cx = cur_cx + delta_x
            next_cy = cur_cy + delta_y
            
            center_x, center_y = CANVAS_SIZE / 2, CANVAS_SIZE / 2
            vec_x = next_cx - center_x
            vec_y = next_cy - center_y
            dist = math.sqrt(vec_x**2 + vec_y**2)
            
            max_dist = ARENA_RADIUS - NODE_RADIUS
            real_delta_x = delta_x
            real_delta_y = delta_y
            
            if dist > max_dist:
                scale = max_dist / dist
                clamped_cx = center_x + vec_x * scale
                clamped_cy = center_y + vec_y * scale
                real_delta_x = clamped_cx - cur_cx
                real_delta_y = clamped_cy - cur_cy

            self.canvas.move(target_tag, real_delta_x, real_delta_y)
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def on_release(self, event):
        self.drag_data["item"] = None
        try:
            self.autd.send(Static(intensity=0))
        except Exception as e:
            print(f"Warning (Stop): Communication failed. Ignored. ({e})")

    def play_stimulus(self, stim_id):
        params = self.all_params[stim_id]
        print(f"Playing ID:{stim_id} | Dist:{params['dist']}, Velo:{params['velo']}, AM:{params['am_freq']}")

        if params['am_freq'] == 0:
            m = Static(intensity=255)
        else:
            m = Sine(freq=params['am_freq'] * Hz, option=SineOption(intensity=255))

        g = FociSTM(
            foci=generate_points(params['dist'], self.center),
            config=params['stm_freq'] * Hz,
        )

        try:
            self.autd.send((m, g))
        except Exception as e:
             print(f"Warning (Play): Communication failed. Ignored. ({e})")

    def save_and_quit(self):
        # ★変更: ファイル名に 8stimuli を追加
        filename = f"experiment_result_8stimuli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        valid_results = [r for r in self.results if r is not None]

        final_export = {
            "config": {
                "num_items_total": len(self.all_params),
                "items_per_trial": ITEMS_PER_TRIAL,
                "total_trials": len(self.trial_list)
            },
            "trials": valid_results
        }
        
        with open(filename, "w") as f:
            json.dump(final_export, f, indent=4)
        
        messagebox.showinfo("Done", f"Experiment finished!\nSaved to {filename}")
        self.root.destroy()


# --- メイン処理 ---
if __name__ == "__main__":
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

    try:
        # with Controller.open(devices, Simulator("127.0.0.1:8080")) as autd:
        with Controller.open(devices, EtherCrab(err_handler=err_handler, option=EtherCrabOption())) as autd:
            autd.send(Silencer.disable())
            
            root = tk.Tk()
            app = TactileMapApp(root, autd)
            root.mainloop()
            
    except Exception as e:
        print(f"Error: {e}")
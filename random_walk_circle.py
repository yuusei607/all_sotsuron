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
NODE_RADIUS = 20    # 点の大きさ（操作しやすいよう少し大きくしました）
ITEMS_PER_TRIAL = 5 # 1回の提示数

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
# --- GUIアプリケーションクラス ---
class TactileMapApp:
    def __init__(self, root, autd_controller):
        self.root = root
        self.autd = autd_controller
        self.root.title("Tactile Spatial Arrangement Task (Multi-arrangement)")

        # AUTD座標の中心設定
        self.center = np.array([1.5 * w, h, 200.0])
        
        # 1. 刺激パラメータの生成 (N=18)
        self.all_params = self._generate_stimuli_params()
        
        # 2. トライアルリストの生成
        generator = GreedyTrialGenerator(num_items=len(self.all_params), items_per_trial=ITEMS_PER_TRIAL)
        self.trial_list = generator.trials
        self.current_trial_idx = 0
        
        # 結果保存用（行ったり来たりできるよう、あらかじめ枠を作っておく）
        self.results = [None] * len(self.trial_list)

        # ドラッグ管理用
        self.drag_data = {"x": 0, "y": 0, "item": None}

        # GUIパーツ作成
        self._create_widgets()
        
        # 最初のトライアルを開始
        self.load_trial()

    def _generate_stimuli_params(self):
        """18パターンの刺激パラメータを生成"""
        distances = [0.05, 4.0] 
        velocities = [10, 100, 1000]
        am_freqs = [0, 20, 100]
        
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

        # --- 修正: 円形アリーナの描画 ---
        # キャンバスの中心計算
        cx, cy = CANVAS_SIZE / 2, CANVAS_SIZE / 2
        
        # アリーナ（白い円）を描画
        # 背景が白なので、少しグレーの枠線で円を表現します
        self.canvas.create_oval(
            cx - ARENA_RADIUS, cy - ARENA_RADIUS,
            cx + ARENA_RADIUS, cy + ARENA_RADIUS,
            outline="lightgrey", width=3, dash=(5, 5), # 点線で見やすく
            tags="arena_bg" # タグをつけておく（消さないように管理するため）
        )
        # 中心点もあると配置しやすいので描いておく
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
        # "arena_bg" というタグがついた背景円以外を削除する
        # (全部 delete("all") すると円も消えちゃうので)
        items = self.canvas.find_all()
        for item in items:
            tags = self.canvas.gettags(item)
            if "arena_bg" not in tags: # 背景以外を消す
                self.canvas.delete(item)
        
        # 進捗表示更新
        self.lbl_progress.config(text=f"Trial: {self.current_trial_idx + 1} / {len(self.trial_list)}")
        
        # ボタン制御
        if self.current_trial_idx == len(self.trial_list) - 1:
            self.btn_next.config(text="Finish & Save", bg="orange")
        else:
            self.btn_next.config(text="Next Trial", bg="lightblue")
            
        if self.current_trial_idx == 0:
            self.btn_prev.config(state=tk.DISABLED)
        else:
            self.btn_prev.config(state=tk.NORMAL)

        # アイテムリスト
        current_items_indices = self.trial_list[self.current_trial_idx]
        
        # 保存データ確認
        saved_data = self.results[self.current_trial_idx]
        saved_positions = {}
        if saved_data is not None:
            for item in saved_data["items"]:
                saved_positions[item["id"]] = (item["x"], item["y"])
        
        # --- 初期配置ロジック ---
        # 論文では「円の周囲（Seating area）」に置くのが基本ですが、
        # 操作しやすさ優先で「円の内側の縁」に並べるようにします。
        
        cx, cy = CANVAS_SIZE / 2, CANVAS_SIZE / 2
        # 初期配置の半径（アリーナより少し小さくする）
        init_radius = ARENA_RADIUS - 40 
        angle_step = 2 * math.pi / len(current_items_indices)
        
        self.current_canvas_items = [] 
        
        for i, item_idx in enumerate(current_items_indices):
            param = self.all_params[item_idx]
            
            if item_idx in saved_positions:
                x, y = saved_positions[item_idx]
            else:
                angle = i * angle_step + random.uniform(-0.1, 0.1)
                # アリーナの境界付近に配置
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
        """現在の画面の状態をself.resultsに一時保存"""
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
        # インデックスを指定して保存（上書き可能にする）
        self.results[self.current_trial_idx] = trial_data

    def next_trial(self):
        """次のトライアルへ"""
        self._save_current_screen() # まず保存
        
        if self.current_trial_idx < len(self.trial_list) - 1:
            self.current_trial_idx += 1
            self.load_trial()
        else:
            self.save_and_quit()

    def prev_trial(self):
        """前のトライアルへ（追加）"""
        self._save_current_screen() # 戻る前にも念のため現状を保存しておく（戻ってまた進んだときに維持するため）
        
        if self.current_trial_idx > 0:
            self.current_trial_idx -= 1
            self.load_trial()

    # --- イベントハンドラ（通信エラー対策済み） ---
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
        """ドラッグ中：円の外に出ないように制限する"""
        item = self.drag_data["item"]
        if item:
            # マウスの移動量
            delta_x = event.x - self.drag_data["x"]
            delta_y = event.y - self.drag_data["y"]
            
            # --- ここから制限ロジック ---
            # 1. 現在のアイテムの中心座標を取得
            tags = self.canvas.gettags(item)
            target_tag = ""
            for tag in tags:
                if tag.startswith("stim_"):
                    target_tag = tag
                    break
            
            # アイテムの現在位置 (x1, y1, x2, y2)
            coords = self.canvas.coords(target_tag)
            cur_cx = (coords[0] + coords[2]) / 2
            cur_cy = (coords[1] + coords[3]) / 2
            
            # 2. 移動「予定」の座標
            next_cx = cur_cx + delta_x
            next_cy = cur_cy + delta_y
            
            # 3. キャンバス中心からの距離を計算
            center_x, center_y = CANVAS_SIZE / 2, CANVAS_SIZE / 2
            vec_x = next_cx - center_x
            vec_y = next_cy - center_y
            dist = math.sqrt(vec_x**2 + vec_y**2)
            
            # 4. 制限半径 (円の半径 - アイテムの半径)
            # これを超えるとアイテムが円からはみ出す
            max_dist = ARENA_RADIUS - NODE_RADIUS
            
            # 5. もしはみ出るなら、境界線上に押し戻す
            real_delta_x = delta_x
            real_delta_y = delta_y
            
            if dist > max_dist:
                # 距離を max_dist に縮める比率
                scale = max_dist / dist
                clamped_cx = center_x + vec_x * scale
                clamped_cy = center_y + vec_y * scale
                
                # 実際に移動させるべき量（補正済み）
                real_delta_x = clamped_cx - cur_cx
                real_delta_y = clamped_cy - cur_cy

            # 6. 移動実行
            self.canvas.move(target_tag, real_delta_x, real_delta_y)

            # マウス位置の更新（ここは元のマウス位置を使わないと操作感が悪くなるので注意）
            # ただし、壁に当たったときは「マウス位置の更新」をスキップするか、
            # あるいはそのままにするかで操作感が変わります。
            # シンプルに「マウス座標」は常に更新しつつ、描画だけ止めるのが一般的です。
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
        filename = f"experiment_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Noneを除外（万が一未実施のデータがあっても保存時にエラーにならないように）
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
    # --- デバイス構成 (元のコードの設定を使用) ---
    # ※動作確認用: Simulator
    # link = Simulator("127.0.0.1:8080")
    
    # 本番用設定（ユーザー提供のリスト）
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
        # Simulatorの場合はこちら
        # with Controller.open(devices, Simulator("127.0.0.1:8080")) as autd:

        # 実機の場合はこちら
        with Controller.open(devices, EtherCrab(err_handler=err_handler, option=EtherCrabOption())) as autd:
            autd.send(Silencer.disable())
            
            root = tk.Tk()
            app = TactileMapApp(root, autd)
            root.mainloop()
            
    except Exception as e:
        print(f"Error: {e}")
        # GUIのみテスト用
        # root = tk.Tk()
        # app = TactileMapApp(root, None) # autd=Noneで起動できるよう調整が必要
        # root.mainloop()
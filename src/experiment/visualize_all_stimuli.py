"""
シード値から生成される全18刺激の軌道を可視化するスクリプト
"""

import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt

# シード値ファイルのパス
SEEDS_FILE = os.path.join(os.path.dirname(__file__), "stimuli_seeds.json")


def generate_points(distance, num_points=1000):
    """ランダムウォークでnum_points点の軌道を生成"""
    points = []
    current_point = np.array([random.uniform(0.0, 10.0), random.uniform(0.0, 10.0)])
    points.append(current_point.copy())
    
    for _ in range(num_points - 1):
        while True:
            angle = random.uniform(0, 2 * np.pi)
            next_x = current_point[0] + distance * np.cos(angle)
            next_y = current_point[1] + distance * np.sin(angle)
            if 0.0 <= next_x <= 10.0 and 0.0 <= next_y <= 10.0:
                current_point = np.array([next_x, next_y])
                points.append(current_point.copy())
                break
    return np.array(points)


def visualize_all_stimuli():
    """全18刺激の軌道を可視化"""
    # シード値を読み込む
    with open(SEEDS_FILE, "r") as f:
        seeds_data = json.load(f)
    
    stimuli = seeds_data["stimuli"]
    
    # 6列×3行のグリッドで表示（dist=0.05が上半分、dist=4.0が下半分）
    fig, axes = plt.subplots(2, 9, figsize=(20, 6))
    fig.suptitle("All 18 Stimuli Trajectories (Generated from Seeds)", fontsize=14)
    
    for stim in stimuli:
        idx = stim["id"]
        dist = stim["dist"]
        velo = stim["velo"]
        am = stim["am_freq"]
        seed = stim["seed"]
        
        # シードを設定して軌道生成
        random.seed(seed)
        np.random.seed(seed)
        num_points = 100000 if dist < 1.0 else 1000
        trajectory = generate_points(dist, num_points)
        
        # プロット位置を決定
        row = 0 if dist < 1.0 else 1
        col = idx if dist < 1.0 else idx - 9
        
        ax = axes[row, col]
        
        # 軌跡をプロット（間引いて表示）
        if len(trajectory) > 2000:
            step = len(trajectory) // 2000
            plot_traj = trajectory[::step]
        else:
            plot_traj = trajectory
        
        ax.plot(plot_traj[:, 0], plot_traj[:, 1], 'b-', alpha=0.4, linewidth=0.3)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=30, marker='o', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=30, marker='x', zorder=5)
        
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"ID:{idx}\nv={velo}, am={am}", fontsize=8)
        
        if col == 0:
            ax.set_ylabel(f"dist={dist}mm", fontsize=10)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(os.path.dirname(__file__), "all_stimuli_trajectories.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    visualize_all_stimuli()

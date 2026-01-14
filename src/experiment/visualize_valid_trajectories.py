"""
合格した軌道を可視化するスクリプト
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os

# distに応じたポイント数
DIST_TO_POINTS = {
    0.05: 100000,
    4.0: 1000
}

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


def is_valid_trajectory(points, centroid_threshold=2.0, std_threshold=1.5, range_threshold=6.0):
    """軌道の妥当性をチェック"""
    centroid = np.mean(points, axis=0)
    centroid_dist = np.linalg.norm(centroid - 5.0)
    
    std_x, std_y = np.std(points[:, 0]), np.std(points[:, 1])
    range_x = np.max(points[:, 0]) - np.min(points[:, 0])
    range_y = np.max(points[:, 1]) - np.min(points[:, 1])
    
    stats = {
        "centroid": centroid,
        "centroid_dist": centroid_dist,
        "std_x": std_x,
        "std_y": std_y,
        "range_x": range_x,
        "range_y": range_y
    }
    
    is_valid = (centroid_dist <= centroid_threshold and 
                std_x >= std_threshold and std_y >= std_threshold and
                range_x >= range_threshold and range_y >= range_threshold)
    
    return is_valid, stats


def generate_valid_trajectory(distance, max_attempts=20):
    """条件を満たす軌道を生成"""
    num_points = DIST_TO_POINTS.get(distance, 1000)
    
    for attempt in range(max_attempts):
        trajectory = generate_points(distance, num_points)
        is_valid, stats = is_valid_trajectory(trajectory)
        if is_valid:
            return trajectory, stats, attempt + 1
    
    return trajectory, stats, max_attempts


def visualize_trajectories():
    """合格した軌道を可視化"""
    distances = [0.05, 4.0]
    num_examples = 3  # 各distで3つの例を生成
    
    fig, axes = plt.subplots(len(distances), num_examples, figsize=(15, 10))
    fig.suptitle("Valid Trajectories Visualization", fontsize=16)
    
    for row, dist in enumerate(distances):
        for col in range(num_examples):
            ax = axes[row, col]
            
            # 合格する軌道を生成
            trajectory, stats, attempts = generate_valid_trajectory(dist)
            
            # 軌道をプロット（点の密度を下げて見やすく）
            num_points = len(trajectory)
            if num_points > 2000:
                # 間引いてプロット
                step = num_points // 2000
                plot_traj = trajectory[::step]
            else:
                plot_traj = trajectory
            
            # 軌跡を線で描画
            ax.plot(plot_traj[:, 0], plot_traj[:, 1], 'b-', alpha=0.3, linewidth=0.5)
            
            # 始点と終点をマーク
            ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
            
            # 重心をマーク
            ax.scatter(stats['centroid'][0], stats['centroid'][1], c='orange', s=100, marker='*', label='Centroid', zorder=5)
            
            # 中心点（5, 5）をマーク
            ax.scatter(5.0, 5.0, c='black', s=50, marker='+', label='Center (5,5)', zorder=5)
            
            # 領域の境界を描画
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-0.5, 10.5)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # タイトルに統計情報を表示
            title = f"dist={dist}mm, {len(trajectory)} pts\n"
            title += f"centroid_dist={stats['centroid_dist']:.2f}, "
            title += f"std=({stats['std_x']:.2f}, {stats['std_y']:.2f})\n"
            title += f"range=({stats['range_x']:.2f}, {stats['range_y']:.2f}), attempts={attempts}"
            ax.set_title(title, fontsize=9)
            
            if col == 0:
                ax.set_ylabel(f"dist = {dist} mm", fontsize=12)
            
            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(os.path.dirname(__file__), "trajectory_visualization.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved visualization to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    visualize_trajectories()

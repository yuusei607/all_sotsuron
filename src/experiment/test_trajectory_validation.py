"""
軌道生成条件の厳しさを検証するスクリプト

各距離パラメータ(dist)について、提案した条件をどの程度の確率で満たせるかを検証する。
"""

import numpy as np
import random
import time
from collections import defaultdict

# --- 設定 ---
NUM_TRIALS = 50  # 各条件で何回試行するか（高ポイント数は時間かかるので少なめ）
MAX_ATTEMPTS = 100  # 1つの有効な軌道を得るための最大試行回数

# 検証するパラメータ
DISTANCES = [0.05]  # 問題のdistのみ検証
NUM_POINTS_LIST = [10000, 50000, 100000, 200000]  # ポイント数を増やして検証

def generate_points(distance, num_points=1000):
    """ランダムウォークでnum_points点の軌道を生成（centerは原点基準）"""
    points = []
    current_point = np.array([random.uniform(0.0, 10.0), random.uniform(0.0, 10.0), 0.0])
    points.append(current_point.copy())
    
    for _ in range(num_points - 1):
        while True:
            angle = random.uniform(0, 2 * np.pi)
            next_x = current_point[0] + distance * np.cos(angle)
            next_y = current_point[1] + distance * np.sin(angle)
            if 0.0 <= next_x <= 10.0 and 0.0 <= next_y <= 10.0:
                current_point = np.array([next_x, next_y, 0.0])
                points.append(current_point.copy())
                break
    return np.array(points)


def is_valid_trajectory(points, centroid_threshold=2.0, std_threshold=1.5, range_threshold=6.0):
    """軌道の妥当性をチェック"""
    coords = points[:, :2]
    
    centroid = np.mean(coords, axis=0)
    centroid_dist = np.linalg.norm(centroid - 5.0)
    centroid_ok = centroid_dist <= centroid_threshold
    
    std_x, std_y = np.std(coords[:, 0]), np.std(coords[:, 1])
    std_ok = std_x >= std_threshold and std_y >= std_threshold
    
    range_x = np.max(coords[:, 0]) - np.min(coords[:, 0])
    range_y = np.max(coords[:, 1]) - np.min(coords[:, 1])
    range_ok = range_x >= range_threshold and range_y >= range_threshold
    
    is_valid = centroid_ok and std_ok and range_ok
    
    stats = {
        "centroid_dist": centroid_dist,
        "centroid_ok": centroid_ok,
        "std_x": std_x,
        "std_y": std_y,
        "std_ok": std_ok,
        "range_x": range_x,
        "range_y": range_y,
        "range_ok": range_ok,
        "is_valid": is_valid
    }
    
    return is_valid, stats


def run_validation_test():
    print("=" * 70)
    print("軌道生成条件の検証（ポイント数による比較）")
    print("=" * 70)
    print(f"各条件で {NUM_TRIALS} 回の軌道を生成")
    print()
    
    for dist in DISTANCES:
        for num_points in NUM_POINTS_LIST:
            print(f"\n--- dist = {dist} mm, points = {num_points} ---")
            
            valid_count = 0
            fail_reasons = defaultdict(int)
            all_stats = []
            
            # 計算時間測定
            start_time = time.time()
            
            for i in range(NUM_TRIALS):
                trajectory = generate_points(dist, num_points)
                is_valid, stats = is_valid_trajectory(trajectory)
                all_stats.append(stats)
                
                if is_valid:
                    valid_count += 1
                else:
                    if not stats["centroid_ok"]:
                        fail_reasons["centroid"] += 1
                    if not stats["std_ok"]:
                        fail_reasons["std"] += 1
                    if not stats["range_ok"]:
                        fail_reasons["range"] += 1
            
            elapsed_time = time.time() - start_time
            avg_time_per_trajectory = elapsed_time / NUM_TRIALS * 1000  # ms
            
            success_rate = valid_count / NUM_TRIALS * 100
            
            print(f"成功率: {valid_count}/{NUM_TRIALS} = {success_rate:.1f}%")
            print(f"計算時間: 合計 {elapsed_time:.2f}秒, 1軌道あたり {avg_time_per_trajectory:.2f}ms")
            
            if fail_reasons:
                print(f"失敗理由:")
                print(f"  - 重心: {fail_reasons['centroid']} 回, 標準偏差: {fail_reasons['std']} 回, 範囲: {fail_reasons['range']} 回")
            
            # 統計サマリー
            std_xs = [s["std_x"] for s in all_stats]
            std_ys = [s["std_y"] for s in all_stats]
            range_xs = [s["range_x"] for s in all_stats]
            range_ys = [s["range_y"] for s in all_stats]
            
            print(f"統計: std_x={np.mean(std_xs):.2f}±{np.std(std_xs):.2f}, "
                  f"std_y={np.mean(std_ys):.2f}±{np.std(std_ys):.2f}")
            print(f"      range_x={np.mean(range_xs):.2f}±{np.std(range_xs):.2f}, "
                  f"range_y={np.mean(range_ys):.2f}±{np.std(range_ys):.2f}")
            
            if success_rate > 0:
                expected_attempts = 100 / success_rate
                print(f"有効な軌道を得るまでの期待試行回数: {expected_attempts:.1f} 回")
    
    print("\n" + "=" * 70)
    print("検証完了")
    print("=" * 70)


if __name__ == "__main__":
    run_validation_test()

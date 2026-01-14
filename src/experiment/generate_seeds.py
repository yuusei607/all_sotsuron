"""
有効な軌道のシード値を生成して保存するスクリプト

このスクリプトを一度実行すると、全18刺激に対して有効な軌道を生成できる
シード値が stimuli_seeds.json に保存されます。
実験スクリプトはこのシード値を読み込んで同じ軌道を再現します。
"""

import numpy as np
import random
import json
import os

# 設定
CENTER = np.array([0.0, 0.0, 0.0])  # 相対座標で生成（後でcenterを足す）
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "stimuli_seeds.json")

# 刺激パラメータ
DISTANCES = [0.05, 4.0]
VELOCITIES = [10, 100, 1000]
AM_FREQS = [0, 20, 100]


def generate_points(distance, num_points=1000):
    """ランダムウォークでnum_points点の軌道を生成"""
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
    if centroid_dist > centroid_threshold:
        return False
    
    std_x, std_y = np.std(coords[:, 0]), np.std(coords[:, 1])
    if std_x < std_threshold or std_y < std_threshold:
        return False
    
    range_x = np.max(coords[:, 0]) - np.min(coords[:, 0])
    range_y = np.max(coords[:, 1]) - np.min(coords[:, 1])
    if range_x < range_threshold or range_y < range_threshold:
        return False
    
    return True


def find_valid_seed(distance, max_attempts=100):
    """有効な軌道を生成できるシード値を探す"""
    num_points = 100000 if distance < 1.0 else 1000
    
    for attempt in range(max_attempts):
        seed = random.randint(0, 2**31 - 1)
        random.seed(seed)
        np.random.seed(seed)
        
        trajectory = generate_points(distance, num_points)
        if is_valid_trajectory(trajectory):
            print(f"  Found valid seed {seed} for dist={distance} on attempt {attempt + 1}")
            return seed
    
    print(f"  WARNING: Could not find valid seed for dist={distance} after {max_attempts} attempts")
    return seed  # 最後に試したシードを返す


def generate_all_seeds():
    """全18刺激のシード値を生成"""
    print("Generating valid seeds for all 18 stimuli...")
    print("=" * 60)
    
    seeds_data = {
        "description": "Seed values for reproducible trajectory generation",
        "stimuli": []
    }
    
    idx = 0
    for dist in DISTANCES:
        for velo in VELOCITIES:
            for am in AM_FREQS:
                print(f"Stimulus {idx}: dist={dist}, velo={velo}, am={am}")
                seed = find_valid_seed(dist)
                
                seeds_data["stimuli"].append({
                    "id": idx,
                    "dist": dist,
                    "velo": velo,
                    "am_freq": am,
                    "seed": seed
                })
                idx += 1
    
    # 保存
    with open(OUTPUT_FILE, "w") as f:
        json.dump(seeds_data, f, indent=2)
    
    print("=" * 60)
    print(f"Saved seeds to: {OUTPUT_FILE}")
    print("Run this script again to regenerate seeds if needed.")


if __name__ == "__main__":
    generate_all_seeds()

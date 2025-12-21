import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform

# ==========================================
# ★ここに18刺激実験の結果ファイル名（JSON）を指定してください
JSON_FILE = "experiment_result_20251216_141720.json" 
# ==========================================

def analyze_18stimuli_with_plots(json_path):
    # --- 1. データの読み込み ---
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_items = data["config"]["num_items_total"]
    
    # パラメータ情報の抽出
    # 18刺激の場合、Velo=[10, 100, 1000], AM=[0, 20, 100] などが含まれます
    id_to_params = {}
    for trial in data["trials"]:
        for item in trial["items"]:
            if item["id"] not in id_to_params:
                id_to_params[item["id"]] = item["params"]
        if len(id_to_params) == num_items:
            break

    # --- 2. 重み付き平均によるRDM作成 ---
    sum_weighted_distances = np.zeros((num_items, num_items))
    sum_weights = np.zeros((num_items, num_items))

    print(f"Processing {len(data['trials'])} trials using Weighted Average...")

    for trial in data["trials"]:
        items = trial["items"]
        if len(items) < 2:
            continue
            
        ids = [item["id"] for item in items]
        coords = np.array([[item["x"], item["y"]] for item in items])
        dists_raw = squareform(pdist(coords))
        
        # スケーリング
        max_dist = np.max(dists_raw)
        if max_dist > 0: dists_scaled = dists_raw / max_dist
        else: dists_scaled = dists_raw

        # 重み (距離の2乗)
        weights = dists_raw ** 2

        # 加算
        for i in range(len(ids)):
            for j in range(len(ids)):
                id_i, id_j = ids[i], ids[j]
                sum_weighted_distances[id_i, id_j] += dists_scaled[i, j] * weights[i, j]
                sum_weights[id_i, id_j] += weights[i, j]

    with np.errstate(divide='ignore', invalid='ignore'):
        rdm = np.divide(sum_weighted_distances, sum_weights)
        rdm[np.isnan(rdm)] = 0 
    np.fill_diagonal(rdm, 0)
    
    # RDM保存
    pd.DataFrame(rdm).to_csv("analysis_18stim_rdm.csv")

    # --- 3. 2次元MDSの実行 ---
    print("Calculating MDS in 2D...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    pos_2d = mds.fit_transform(rdm)

    # --- 4. クラスタリング (K-means) ---
    # 最適なKを探索 (18刺激なので K=2〜10 くらいまで探索)
    sil_scores = []
    k_range = range(2, min(10, num_items))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pos_2d)
        sil_scores.append(silhouette_score(pos_2d, labels))
    best_k = k_range[np.argmax(sil_scores)]
    print(f"Best K selected: {best_k}")
    
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(pos_2d)

    # 座標データの保存
    result_rows = []
    for i in range(num_items):
        params = id_to_params.get(i, {})
        result_rows.append({
            "ID": i,
            "Cluster": clusters[i],
            "MDS_Dim1": pos_2d[i, 0],
            "MDS_Dim2": pos_2d[i, 1],
            "Dist": params.get("dist", 0),
            "Velo": params.get("velo", 0),
            "AM_Freq": params.get("am_freq", 0)
        })
    df_coords = pd.DataFrame(result_rows)
    df_coords.to_csv("analysis_18stim_mds_coordinates.csv", index=False)
    print("Saved: analysis_18stim_mds_coordinates.csv")

    # =========================================================
    # パラメータごとに色分けしてプロット＆保存
    # =========================================================
    
    plot_configs = [
        ("Dist", "Distance (mm)", "Distance"),
        ("Velo", "Velocity (mm/s)", "Velocity"),
        ("AM_Freq", "AM Frequency (Hz)", "AM_Freq"),
        ("Cluster", f"K-means Cluster (K={best_k})", "Cluster")
    ]

    print("\nGenerating plots...")

    for col_name, label_text, filename_suffix in plot_configs:
        plt.figure(figsize=(9, 8)) # 点が増えるので少し大きく
        
        # データのユニーク値を取得してソート (例: Veloなら 10, 100, 1000 の3つになる)
        unique_vals = sorted(df_coords[col_name].unique())
        
        # カラーマップ設定 (tab10は10色まで区別可能なので18刺激の3水準なら余裕)
        colors = cm.get_cmap('tab10', len(unique_vals))

        for idx, val in enumerate(unique_vals):
            subset = df_coords[df_coords[col_name] == val]
            plt.scatter(
                subset["MDS_Dim1"], subset["MDS_Dim2"],
                s=180, label=f"{val}", color=colors(idx), edgecolor='black', alpha=0.9
            )
            
            # ID表示
            for _, row in subset.iterrows():
                plt.text(row["MDS_Dim1"]+0.02, row["MDS_Dim2"]+0.02, str(int(row["ID"])), fontsize=9)

        plt.title(f"MDS Map (18 Stimuli) colored by {label_text}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title=col_name, fontsize=10, loc='best')
        
        # 保存
        save_name = f"MDS_18stim_by_{filename_suffix}.png"
        plt.savefig(save_name, dpi=300)
        print(f"Saved plot: {save_name}")
        plt.show()

if __name__ == "__main__":
    try:
        analyze_18stimuli_with_plots(JSON_FILE)
    except FileNotFoundError:
        print(f"Error: File '{JSON_FILE}' not found. Please check the filename.")
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import os

# ==========================================
# ★ここに18刺激実験の結果ファイル名（JSON）を指定してください
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(BASE_DIR, "..", "results", "raw_results", "experiment_result_hidaka_20260114_193502.json")

# アンカー刺激ID（スケーリングに使用）
ANCHOR_IDS = [0, 17]
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
        
        # スケーリング: アンカー刺激間の距離を基準にする
        anchor_dist = None
        if ANCHOR_IDS[0] in ids and ANCHOR_IDS[1] in ids:
            idx_0 = ids.index(ANCHOR_IDS[0])
            idx_17 = ids.index(ANCHOR_IDS[1])
            anchor_dist = dists_raw[idx_0, idx_17]
        
        if anchor_dist is not None and anchor_dist > 0:
            dists_scaled = dists_raw / anchor_dist
        else:
            # フォールバック（アンカーがない試行は最大距離で）
            max_dist = np.max(dists_raw)
            if max_dist > 0:
                dists_scaled = dists_raw / max_dist
            else:
                dists_scaled = dists_raw

        # 重み (距離の2乗)
        weights = dists_raw ** 0

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

    # --- 3. MDSの実行（次元1〜5の正規化ストレス値を計算） ---
    print("\n" + "="*50)
    print("Calculating MDS Kruskal Stress-1 for dimensions 1-5...")
    print("="*50)
    
    def calculate_kruskal_stress1(rdm, mds_coords):
        """
        Kruskal Stress-1 (正規化ストレス) を計算
        Stress-1 = sqrt( sum((d_ij - d̂_ij)²) / sum(d_ij²) )
        
        参考: Kruskal, J.B. (1964). Psychometrika, 29(1), 1-27.
        """
        # MDS配置からの距離行列を計算
        mds_distances = squareform(pdist(mds_coords))
        
        # 上三角部分のみ使用（対角は0なので除外）
        triu_indices = np.triu_indices_from(rdm, k=1)
        d_original = rdm[triu_indices]
        d_mds = mds_distances[triu_indices]
        
        # Kruskal Stress-1
        numerator = np.sum((d_original - d_mds) ** 2)
        denominator = np.sum(d_original ** 2)
        
        if denominator == 0:
            return 0.0
        
        stress1 = np.sqrt(numerator / denominator)
        return stress1
    
    def interpret_stress(stress):
        """Kruskal (1964) の基準に基づくストレス値の解釈"""
        if stress < 0.025:
            return "Near Perfect"
        elif stress < 0.05:
            return "Excellent"
        elif stress < 0.1:
            return "Good"
        elif stress < 0.2:
            return "Fair"
        else:
            return "Poor"
    
    stress_values = []
    mds_positions = {}  # 各次元のMDS結果を保存
    
    print("\n  Kruskal (1964) Stress-1 基準:")
    print("    < 0.025: Near Perfect")
    print("    < 0.05:  Excellent")
    print("    < 0.1:   Good")
    print("    < 0.2:   Fair")
    print("    >= 0.2:  Poor")
    print("-" * 50)
    
    for n_dim in range(1, 6):
        mds = MDS(n_components=n_dim, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        pos = mds.fit_transform(rdm)
        
        # Kruskal Stress-1 を計算
        stress1 = calculate_kruskal_stress1(rdm, pos)
        stress_values.append(stress1)
        mds_positions[n_dim] = pos
        
        interpretation = interpret_stress(stress1)
        print(f"  Dimension {n_dim}: Stress-1 = {stress1:.4f} ({interpretation})")
    
    # 2次元のMDS結果を使用
    pos_2d = mds_positions[2]
    
    # ストレス値のプロット（Scree plot風）
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), stress_values, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='Good threshold (0.1)')
    plt.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='Fair threshold (0.2)')
    plt.xlabel('Number of Dimensions', fontsize=12)
    plt.ylabel('Kruskal Stress-1', fontsize=12)
    plt.title('MDS Kruskal Stress-1 by Number of Dimensions', fontsize=14)
    plt.xticks(range(1, 6))
    plt.ylim(0, max(stress_values) * 1.2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    for i, stress in enumerate(stress_values):
        plt.annotate(f'{stress:.4f}', (i+1, stress), textcoords="offset points", xytext=(0,10), ha='center')
    plt.tight_layout()
    plt.savefig("MDS_stress_by_dimension.png", dpi=300)
    print("\nSaved: MDS_stress_by_dimension.png")
    plt.show()

    # --- 4. クラスタリング (K-means) とシルエットスコア ---
    print("\n" + "="*50)
    print("Calculating Silhouette Scores for K=2 to 10...")
    print("="*50)
    
    sil_scores = []
    k_range = range(2, min(11, num_items))  # K=2〜10
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pos_2d)
        score = silhouette_score(pos_2d, labels)
        sil_scores.append(score)
        print(f"  K={k}: Silhouette Score = {score:.4f}")
    
    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"\n  ★ Best K = {best_k} (Silhouette Score = {max(sil_scores):.4f})")
    
    # シルエットスコアのプロット
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), sil_scores, 'go-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score by Number of Clusters', fontsize=14)
    plt.xticks(list(k_range))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    for i, score in enumerate(sil_scores):
        plt.annotate(f'{score:.3f}', (list(k_range)[i], score), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig("Silhouette_scores_by_K.png", dpi=300)
    print("Saved: Silhouette_scores_by_K.png")
    plt.show()
    
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
        # plt.savefig(save_name, dpi=300)
        print(f"Saved plot: {save_name}")
        plt.show()

if __name__ == "__main__":
    try:
        analyze_18stimuli_with_plots(JSON_FILE)
    except FileNotFoundError:
        print(f"Error: File '{JSON_FILE}' not found. Please check the filename.")
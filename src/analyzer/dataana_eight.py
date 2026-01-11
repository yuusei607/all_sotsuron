import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

# ==========================================
# 設定: 解析したいJSONファイル名を指定してください
JSON_FILE = "experiment_result_8stimuli_20251221_155626.json" 
# ==========================================

def analyze_experiment_2d(json_path):
    # --- 1. データの読み込み ---
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_items = data["config"]["num_items_total"]
    
    # パラメータ情報の抽出
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
        
        # 生の距離 (ピクセル単位)
        dists_raw = squareform(pdist(coords))
        
        # スケーリング (最大距離で割る)
        max_dist = np.max(dists_raw)
        if max_dist > 0:
            dists_scaled = dists_raw / max_dist
        else:
            dists_scaled = dists_raw

        # ★重み = 距離の2乗 (S/N比考慮)
        weights = dists_raw ** 2

        # 加算
        for i in range(len(ids)):
            for j in range(len(ids)):
                id_i, id_j = ids[i], ids[j]
                sum_weighted_distances[id_i, id_j] += dists_scaled[i, j] * weights[i, j]
                sum_weights[id_i, id_j] += weights[i, j]

    # 加重平均計算
    with np.errstate(divide='ignore', invalid='ignore'):
        rdm = np.divide(sum_weighted_distances, sum_weights)
        rdm[np.isnan(rdm)] = 0 
    np.fill_diagonal(rdm, 0)

    # RDM保存
    pd.DataFrame(rdm).to_csv("analysis_rdm.csv")
    print("Saved: analysis_rdm.csv")

    # ヒートマップ表示
    plt.figure(figsize=(6, 5))
    plt.imshow(rdm, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Dissimilarity (0-1)')
    plt.title("Integrated Dissimilarity Matrix (Weighted)")
    plt.show()

    # --- 3. 2次元MDSの実行 ---
    print("\nCalculating MDS in 2D...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    pos_2d = mds.fit_transform(rdm)
    print(f"MDS Stress (2D): {mds.stress_:.4f}")

    # --- 4. シルエット分析 (2D空間) ---
    print("\nCalculating Silhouette Scores...")
    sil_results = []
    k_range = range(2, min(8, num_items)) 
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pos_2d)
        score = silhouette_score(pos_2d, labels)
        sil_results.append({"K": k, "Silhouette_Score": score})

    # 結果保存と表示
    df_sil = pd.DataFrame(sil_results)
    df_sil.to_csv("analysis_silhouette.csv", index=False)
    
    best_k = df_sil.loc[df_sil["Silhouette_Score"].idxmax(), "K"]
    print(f"Suggested Number of Clusters: K = {best_k}")

    plt.figure(figsize=(6, 4))
    plt.bar(df_sil["K"], df_sil["Silhouette_Score"], color='skyblue')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis (2D)")
    plt.axvline(x=best_k, color='red', linestyle='--')
    plt.show()

    # --- 5. 最終結果の保存 (2D座標 + パラメータ) ---
    kmeans_final = KMeans(n_clusters=int(best_k), random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(pos_2d)

    result_rows = []
    for i in range(num_items):
        params = id_to_params.get(i, {})
        row = {
            "ID": i,
            "Cluster": clusters[i],
            # 2次元なのでDim3は削除
            "MDS_Dim1": pos_2d[i, 0],
            "MDS_Dim2": pos_2d[i, 1],
            "Dist": params.get("dist", ""),
            "Velo": params.get("velo", ""),
            "AM_Freq": params.get("am_freq", "")
        }
        result_rows.append(row)

    df_coords = pd.DataFrame(result_rows)
    df_coords.to_csv("analysis_mds_coordinates.csv", index=False)
    print("Saved: analysis_mds_coordinates.csv")

    # --- 6. 2次元散布図の表示 ---
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c=clusters, cmap='tab10', s=150, edgecolor='k')
    
    # ラベル(ID)
    for i in range(num_items):
        plt.text(pos_2d[i, 0]+0.02, pos_2d[i, 1]+0.02, str(i), fontsize=12, fontweight='bold')

    plt.title(f"MDS Map (2D) with K-means (K={best_k})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # (おまけ) デンドログラム
    plt.figure(figsize=(10, 5))
    linked = linkage(squareform(rdm), 'ward')
    dendrogram(linked, labels=list(range(num_items)))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()

if __name__ == "__main__":
    try:
        analyze_experiment_2d(JSON_FILE)
    except FileNotFoundError:
        print(f"Error: File '{JSON_FILE}' not found. Please check the filename.")
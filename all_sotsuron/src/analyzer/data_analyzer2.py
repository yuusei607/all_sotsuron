import json
import numpy as np
import pandas as pd  # データ保存用にpandasを追加
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 解析したいJSONファイル名
JSON_FILE = "experiment_result_20251212_174248.json" 
# ==========================================

def analyze_and_save(json_path):
    # --- 1. データ読み込みとパラメータ抽出 ---
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_items = data["config"]["num_items_total"]
    
    # 後でCSVに物理パラメータも載せるために、IDごとのパラメータ辞書を作っておく
    # (trialsの中から情報を探して埋める)
    id_to_params = {}
    for trial in data["trials"]:
        for item in trial["items"]:
            if item["id"] not in id_to_params:
                id_to_params[item["id"]] = item["params"]
        if len(id_to_params) == num_items:
            break

    # --- 2. RDM (非類似度行列) の作成 ---
    sum_distances = np.zeros((num_items, num_items))
    counts = np.zeros((num_items, num_items))
    print(f"Loading {len(data['trials'])} trials...")

    for trial in data["trials"]:
        items = trial["items"]
        if len(items) < 2:
            continue
        ids = [item["id"] for item in items]
        coords = np.array([[item["x"], item["y"]] for item in items])
        dists = squareform(pdist(coords))
        max_dist = np.max(dists)
        if max_dist > 0: dists_norm = dists / max_dist
        else: dists_norm = dists

        for i in range(len(ids)):
            for j in range(len(ids)):
                id_i, id_j = ids[i], ids[j]
                sum_distances[id_i, id_j] += dists_norm[i, j]
                counts[id_i, id_j] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        rdm = np.divide(sum_distances, counts)
        rdm[np.isnan(rdm)] = 0
    np.fill_diagonal(rdm, 0)

    # ★保存1: RDMをCSVに保存
    df_rdm = pd.DataFrame(rdm)
    df_rdm.to_csv("analysis_rdm.csv", index=True, header=True)
    print("Saved: analysis_rdm.csv")

    # --- 3. 3次元MDS実行 ---
    print("\nCalculating MDS in 3D...")
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    pos_3d = mds.fit_transform(rdm)

    # --- 4. シルエット分析 ---
    print("\nCalculating Silhouette Scores...")
    sil_results = []
    k_range = range(2, 9)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pos_3d)
        score = silhouette_score(pos_3d, labels)
        sil_results.append({"K": k, "Silhouette_Score": score})

    # ★保存2: シルエットスコアをCSVに保存
    df_sil = pd.DataFrame(sil_results)
    df_sil.to_csv("analysis_silhouette.csv", index=False)
    print("Saved: analysis_silhouette.csv")

    # 最適なKを決定
    best_k = df_sil.loc[df_sil["Silhouette_Score"].idxmax(), "K"]
    print(f"Best K based on Silhouette: {best_k}")

    # --- 5. 最終クラスタリングと詳細データの保存 ---
    kmeans_final = KMeans(n_clusters=int(best_k), random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(pos_3d)

    # データフレーム作成 (ID, 座標, クラスタ, パラメータ)
    result_rows = []
    for i in range(num_items):
        params = id_to_params.get(i, {})
        row = {
            "ID": i,
            "Cluster": clusters[i],
            "MDS_Dim1": pos_3d[i, 0],
            "MDS_Dim2": pos_3d[i, 1],
            "MDS_Dim3": pos_3d[i, 2],
            "Dist": params.get("dist", ""),
            "Velo": params.get("velo", ""),
            "AM_Freq": params.get("am_freq", "")
        }
        result_rows.append(row)

    # ★保存3: 詳細データをCSVに保存 (これが一番使えます！)
    df_coords = pd.DataFrame(result_rows)
    df_coords.to_csv("analysis_mds_coordinates.csv", index=False)
    print("Saved: analysis_mds_coordinates.csv")

    # --- 6. 3次元プロット表示 (確認用) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2],
        c=clusters, cmap='tab10',
    )
    for i in range(num_items):
        ax.text(pos_3d[i, 0], pos_3d[i, 1], pos_3d[i, 2], str(i), fontsize=10)

    ax.set_title(f"3D MDS Map (K={best_k})")
    ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.set_zlabel('Dim 3')
    plt.show()

if __name__ == "__main__":
    try:
        analyze_and_save(JSON_FILE)
    except FileNotFoundError:
        print(f"Error: File '{JSON_FILE}' not found.")
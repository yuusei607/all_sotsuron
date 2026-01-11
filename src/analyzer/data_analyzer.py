import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D  # 3D描画用

# ==========================================
# 解析したいJSONファイル名を指定してください
JSON_FILE = "experiment_result_20251212_174248.json" 
# ==========================================

def analyze_experiment_3d(json_path):
    # --- 1. データの読み込みと統合 (RDM作成) ---
    # ※ここは前回と同じ処理です
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_items = data["config"]["num_items_total"]
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
        if max_dist > 0:
            dists_norm = dists / max_dist
        else:
            dists_norm = dists

        for i in range(len(ids)):
            for j in range(len(ids)):
                id_i, id_j = ids[i], ids[j]
                sum_distances[id_i, id_j] += dists_norm[i, j]
                counts[id_i, id_j] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        rdm = np.divide(sum_distances, counts)
        rdm[np.isnan(rdm)] = 0
    np.fill_diagonal(rdm, 0)

    # --- 2. MDSを「3次元」で実行 ---
    print("\nCalculating MDS in 3D...")
    # n_components を 3 に変更
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    pos_3d = mds.fit_transform(rdm)
    print(f"MDS Stress (3D): {mds.stress_}")

    # --- 3. クラスタ数の検討 (3D空間でのシルエット分析) ---
    print("\nCalculating Silhouette Scores for K=2 to 8 (in 3D space)...")
    sil_scores = []
    k_range = range(2, 9)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pos_3d) # 3次元データに対してクラスタリング
        score = silhouette_score(pos_3d, labels)
        sil_scores.append(score)

    best_k = k_range[np.argmax(sil_scores)]
    print(f"Suggested Number of Clusters: K = {best_k}")

    # シルエットスコアのプロット
    plt.figure(figsize=(6, 4))
    plt.bar(k_range, sil_scores, color='lightgreen')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Analysis (Calculated in 3D)")
    plt.axvline(x=best_k, color='red', linestyle='--')
    plt.show()

    # --- 4. 3次元散布図での可視化 ---
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(pos_3d)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d') # 3D axesを作成

    # 散布図
    scatter = ax.scatter(
        pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2],
        c=clusters, cmap='tab10', s=100, edgecolor='k', alpha=0.8
    )

    # ラベル(ID)をつける
    for i in range(num_items):
        ax.text(pos_3d[i, 0], pos_3d[i, 1], pos_3d[i, 2], str(i), fontsize=10)

    # 軸ラベル
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(f"3D MDS Map with K-means (K={best_k})")
    
    # 視点の初期設定（見やすい角度）
    ax.view_init(elev=20, azim=45)

    plt.show()
    # ※ 表示されたウィンドウ内でマウスドラッグすると回転できます

if __name__ == "__main__":
    try:
        analyze_experiment_3d(JSON_FILE)
    except FileNotFoundError:
        print(f"Error: File '{JSON_FILE}' not found.")
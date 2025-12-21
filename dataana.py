import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

# ==========================================
# 設定: 解析したいJSONファイル名を指定してください
JSON_FILE = "experiment_result_20251216_141720.json" 
# ==========================================

def analyze_experiment(json_path):
    # --- 1. データの読み込みと統合 (RDM作成) ---
    with open(json_path, 'r') as f:
        data = json.load(f)

    num_items = data["config"]["num_items_total"]
    
    # 距離の蓄積用 (分子: 距離の合計, 分母: 回数)
    sum_distances = np.zeros((num_items, num_items))
    counts = np.zeros((num_items, num_items))

    print(f"Loading {len(data['trials'])} trials...")

    for trial in data["trials"]:
        items = trial["items"]
        if len(items) < 2:
            continue
            
        # トライアル内の各点の座標を取得
        ids = [item["id"] for item in items]
        coords = np.array([[item["x"], item["y"]] for item in items])
        
        # 距離行列を計算
        dists = squareform(pdist(coords))
        
        # 正規化: このトライアル内の「最大距離」で割って 0~1 にする
        # (Zoom-inの影響を補正するため)
        max_dist = np.max(dists)
        if max_dist > 0:
            dists_norm = dists / max_dist
        else:
            dists_norm = dists

        # 全体行列に加算
        for i in range(len(ids)):
            for j in range(len(ids)):
                id_i, id_j = ids[i], ids[j]
                sum_distances[id_i, id_j] += dists_norm[i, j]
                counts[id_i, id_j] += 1

    # 平均をとって非類似度行列(RDM)を完成させる
    # (回数が0の場所は0のままにする)
    with np.errstate(divide='ignore', invalid='ignore'):
        rdm = np.divide(sum_distances, counts)
        rdm[np.isnan(rdm)] = 0

    # 対角成分を0にする念押し
    np.fill_diagonal(rdm, 0)

    # ヒートマップ表示
    plt.figure(figsize=(6, 5))
    plt.imshow(rdm, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Dissimilarity')
    plt.title("Integrated Dissimilarity Matrix (RDM)")
    plt.xlabel("Stimulus ID")
    plt.ylabel("Stimulus ID")
    plt.show()

    # --- 2. MDSの次元数検討 (Scree Plot) ---
    print("\nCalculating MDS Stress for dimensions 1 to 5...")
    stresses = []
    dims = range(1, 6)
    
    for k in dims:
        mds_check = MDS(n_components=k, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        mds_check.fit(rdm)
        stresses.append(mds_check.stress_)

    plt.figure(figsize=(6, 4))
    plt.plot(dims, stresses, 'bo-')
    plt.xlabel("Dimensions")
    plt.ylabel("Stress")
    plt.title("MDS Stress Plot (Elbow Method)")
    plt.grid(True)
    plt.show()
    
    # ここでは可視化のために2次元を採用
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    pos = mds.fit_transform(rdm)

    # --- 3. クラスタ数の検討 (Silhouette Analysis) ---
    print("\nCalculating Silhouette Scores for K=2 to 8...")
    sil_scores = []
    k_range = range(2, 9)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pos) # MDS空間上でクラスタリング
        score = silhouette_score(pos, labels)
        sil_scores.append(score)

    # 最適なKを見つける
    best_k = k_range[np.argmax(sil_scores)]
    print(f"Suggested Number of Clusters (Max Silhouette): K = {best_k}")

    plt.figure(figsize=(6, 4))
    plt.bar(k_range, sil_scores, color='skyblue')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
    plt.legend()
    plt.show()

    # --- 4. 最終結果の可視化 (Best Kでプロット) ---
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans_final.fit_predict(pos)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(pos[:, 0], pos[:, 1], c=clusters, cmap='tab10', s=100, edgecolor='k')
    
    # ラベル(ID)をつける
    for i in range(num_items):
        plt.text(pos[i, 0]+0.02, pos[i, 1]+0.02, str(i), fontsize=12)

    plt.title(f"MDS Map with K-means Clustering (K={best_k})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

    # (おまけ) 階層化クラスタリングのデンドログラム
    # データの構造が木構造で見えるので論文に載せやすい
    plt.figure(figsize=(10, 5))
    linked = linkage(squareform(rdm), 'ward')
    dendrogram(linked, labels=list(range(num_items)))
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Stimulus ID")
    plt.ylabel("Distance")
    plt.show()

if __name__ == "__main__":
    try:
        analyze_experiment(JSON_FILE)
    except FileNotFoundError:
        print(f"Error: File '{JSON_FILE}' not found. Please check the filename.")
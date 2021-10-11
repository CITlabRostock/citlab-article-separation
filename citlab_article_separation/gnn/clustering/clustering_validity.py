import numpy as np
from sklearn.preprocessing import LabelEncoder


def calinkski_harabasz_score(distances, labels, is_squared=True):
    # https://stats.stackexchange.com/questions/237792/sums-of-squares-total-between-within-how-to-compute-them-from-a-distance-ma
    # Check labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_labels = len(le.classes_)
    labels_one_hot = np.eye(n_labels)[labels]
    # Check distances
    dists = np.array(distances) if is_squared else np.array(distances) ** 2
    n_samples, _ = distances.shape
    # SST (sum-squares-total)
    sst = np.sum(dists) / (2 * n_samples)
    # SSW (sum-squares-within, intra-cluster-variance)
    ssw = np.diag(np.matmul(np.matmul(labels_one_hot.transpose(), dists), labels_one_hot))
    freq = np.sum(labels_one_hot, axis=0)
    ssw = np.sum(ssw / (2 * freq.transpose()))
    # SSB (sum-squares-between, inter-cluster-variance)
    ssb = sst - ssw
    # Calinkski Harabasz score
    if ssw == 0.0:
        return ssw, ssb, sst, None
    else:
        norm_factor = ((n_samples - n_labels) / (n_labels - 1))
        ch_score = (ssb / ssw) * norm_factor
    return ssw, ssb, sst, ch_score


def c_index_score(distances, labels, is_squared=True):
    # https://pypi.org/project/c-index/
    # Check labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_labels = len(le.classes_)
    labels_one_hot = np.eye(n_labels)[labels]
    # Check distances
    dists = np.array(distances) if is_squared else np.array(distances) ** 2
    # Sum distances within clusters
    sw = np.diag(np.matmul(np.matmul(labels_one_hot.transpose(), dists), labels_one_hot))
    sw = np.sum(sw / 2)  # since we counted dists twice
    # Total Number of pairs of observations belonging to same cluster
    _, counts = np.unique(labels, return_counts=True)
    nw = np.sum(counts * (counts - 1) / 2, dtype=np.int32)
    # Sum of Nw smallest distances between all pairs of points
    # Sum of Nw largest distances between all pairs of points
    cond_indices = np.triu_indices_from(dists, k=1)
    cond_dists = dists[cond_indices]
    cond_dists = np.sort(cond_dists)
    s_min = np.sum(cond_dists[:nw])
    s_max = np.sum(cond_dists[-nw:])
    # Calculate C-Index
    cindex = (sw - s_min) / (s_max - s_min)
    return cindex


def connectivity_score(distances, labels, num_neighbors):
    # https://epub.ub.uni-muenchen.de/12797/1/DA_Scherl.pdf
    # build connectivity matrix
    label_arr = np.tile(labels, (len(labels), 1))
    # 0 for same cluster, 1 for different cluster
    label_mismatch = (label_arr != label_arr.transpose()).astype(np.float32)
    # Remove diagonal (to not consider pairs of same points)
    label_mismatch = label_mismatch[~np.eye(len(label_mismatch), dtype=bool)].reshape(len(label_mismatch), -1)
    dists = distances[~np.eye(len(distances), dtype=bool)].reshape(len(distances), -1)
    # Sort by distance matrix
    sorted_indices = np.argsort(dists)
    label_mismatch = label_mismatch[np.arange(len(dists))[:, None], sorted_indices]
    # Scale by index
    index_scaling = np.arange(1, len(labels))
    connectivity_matrix = label_mismatch / index_scaling
    # Sum up top-k scores for each element
    score = np.sum(connectivity_matrix[:, :num_neighbors])
    return score
def threshold_clusters(matrix, threshold=0.8):
    clusters = []
    visited = set()
    labels = matrix.index.tolist()

    for i in range(len(labels)):
        if i in visited:
            continue

        cluster = {labels[i]}
        for j in range(len(labels)):
            if i != j and matrix.iloc[i, j] >= threshold:
                cluster.add(labels[j])

        visited.update([labels.index(c) for c in cluster])
        clusters.append(cluster)

    return clusters

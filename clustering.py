from sklearn.cluster import KMeans, BisectingKMeans, AgglomerativeClustering


def clustering_kmeans(training_text_features, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, max_iter=100, n_init=10, init="k-means++")
    results = kmeans.fit(training_text_features)
    return results, kmeans


def clustering_bisecting_kmeans(training_text_features, num_clusters):
    bisecting_kmeans = BisectingKMeans(n_clusters=num_clusters, max_iter=100, n_init=10, init="k-means++")
    results = bisecting_kmeans.fit(training_text_features)
    return results, bisecting_kmeans


def clustering_hierarchical(training_text_features, num_clusters):
    hierarchical_clustering = AgglomerativeClustering(n_clusters=num_clusters)
    results = hierarchical_clustering.fit(training_text_features)
    return results, hierarchical_clustering

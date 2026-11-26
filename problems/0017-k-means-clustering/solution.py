def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    def distance_squared(point, center):
        return sum((a - b) ** 2 for a, b in zip(point, center))

    centroids = initial_centroids

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]

        for point in points:
            nearest = min(
                range(k),
                key=lambda i: distance_squared(point, centroids[i]),
            )
            clusters[nearest].append(point)

        new_centroids = []
        for center, cluster in zip(centroids, clusters):
            if cluster:
                center = tuple(
                    sum(values) / len(cluster)
                    for values in zip(*cluster)
                )
            new_centroids.append(center)

        centroids = new_centroids

    return [tuple(round(value, 4) for value in center) for center in centroids]
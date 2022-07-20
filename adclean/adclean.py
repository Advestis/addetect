def detect_outliers():
    pass


def replace_outliers():
    pass


def mean():
    pass


def standard_deviation():
    """
    Points within 3 standard deviations of the mean constitute only about 1% of the distribution.
    These points are atypical of the majority of the other points and are likely to be outliers.
    """
    pass


def gaussian_mixture_model():
    """
    A Gaussian mixture model is a probabilistic model that assumes that all data points are generated
    from a mixture of a finite number of Gaussian distributions with unknown parameters.
    """
    pass


def extreme_value_analysis():
    """
    Estimation of the probability of the rarest events compared to those previously compared.
    """
    pass


def local_outlier_factor():
    """
    In anomaly detection, the local outlier factor is an algorithm that finds anomalous data points
    by measuring the local deviation of a given data point from its neighbours.
    """
    pass


def connectivity_based_outlier_detection():
    """
    The connectivity-based outlier detection is a technique for detecting outliers. It is an improved version of the
    local outlier factor (LOF) technique. The idea of the connectivity-based outlier algorithm is to assign a degree
    of outlier to each data point. This degree of outlier is called the connectivity-based outlier factor (COF) of
    the data point. A high COF of a data point represents a high probability of being an outlier.
    """
    pass


def angular_based_outlier_detection():
    """
    The approach called ABOD (Angle-based Outlier Detection) evaluates the degree of outlier on the variance of
    angles (VOA) between a point and all other pairs of points in the data set.
    """
    pass


def dbscan_clustering():
    """
    Clustering is a way to group a set of data points in a way that similar data points are grouped together.
    Therefore, clustering algorithms look for similarities or dissimilarities among data points. Clustering is an
    unsupervised learning method so there is no label associated with data points. The algorithm tries to find the
    underlying structure of the data.
    """
    pass


def kmeans_clustering():
    """
    Groups data points into k clusters based on their feature values. The scores of each data point within a cluster
    are calculated as the distance to its centroid. Data points that are far from the centroid of their clusters are
    labelled as anomalies.
    """
    pass


def knearest_neighbor():
    """
    For each data point, the whole set of data points is examined to extract the k items that have the most similar
    feature values: these are the k nearest neighbors (NN). Then, the data point is classified as anomalous if the
    majority of NN was previously classified as anomalous.
    """
    pass


def mahalanobis_distance():
    """
    Calculates the distance to the barycentre taking into account the shape of all data points. In an area of high
    density, a point that deviates from the others (its immediate neighbours) should raise more questions than when
    it is located in a less dense area
    """
    pass


def isolation_forest():
    pass


def robust_random_cut_forest():
    pass


def support_vector_machine():
    pass

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def kmeans(data,n):
    """train a model
    
    """

    model = KMeans(n_clusters=n)

    model.fit(data)
    return model


def db_scan(data,n=5):
    """train
    """
    
    model = DBSCAN(eps = 0.5, min_samples = 10).fit(data)

    return model
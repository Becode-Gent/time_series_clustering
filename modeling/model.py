from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def kmeans(data,n):
    """kmeans train a model 

    :param data: the dataset for the training
    :param n: number of clusters
    :return: model
    """
    model = KMeans(n_clusters=n)
    model.fit(data)
    return model


def db_scan(data,n=5):
    """db_scan train a model 

    :param data: the dataset for the training
    :param n: number of clusters
    :return: model
    """
    model = DBSCAN(eps = 0.5, min_samples = 10).fit(data)
    return model
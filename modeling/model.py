from sklearn.cluster import KMeans

def kmeans(data,n):
    """train a model
    
    """

    model = KMeans(n_clusters=n)

    model.fit(data)
    return model


def db_scan(data,n):
    """train
    """
    model = KMeans(n_clusters=n)

    model.fit(data)
    return model
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans


def determine_n_clusters(data, features):
    n_cluster = 5
    k_max = 10
    from sklearn.metrics import silhouette_score

    sil = []  
    threshold = 0.7
   # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in range(2, k_max+1):
        kmeans = KMeans(n_clusters = k).fit(data[features])
        labels = kmeans.labels_
        sil.append(silhouette_score(data[features], labels, metric = 'euclidean'))

    return sil


def apply_pca(data,features,n):
    """apply pca to selected features
    
    """
    
    pca = PCA(n_components= n)
    data_input = data[features]
    features_name = ['pca_'+ str(i) for i in range(n)]
    df_output = pd.DataFrame(pca.fit_transform(data_input), columns = features_name)
    data = pd.concat([data,df_output],axis=1)
    data = data.dropna()  
    

    
    return data, features_name
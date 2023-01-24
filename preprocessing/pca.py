from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
import time


def determine_n_clusters(data, features):
    """determine_n_cluster determine the number of clusters

    :param data: the input dataframe 
    :param features: features is the selected features for modeling
    :return: df output of the function, which is a dataframe with k and silhoute score(note the higher the score the better) columns
    """
    n_cluster = 5
    k_max = 10
    sil = []  
    threshold = 0.7
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    my_bar = st.progress(0)
    scale = round(100/(k_max-1))
    val =0
    for k in range(2, k_max+1):
        kmeans = KMeans(n_clusters = k).fit(data[features])
        labels = kmeans.labels_
        sil.append(silhouette_score(data[features], labels, metric = 'euclidean'))
        val = val + scale
        my_bar.progress(val) 

    my_bar.progress(100)
    k_values= [*range(2, len(sil)+2)]
    zipped = list(zip(k_values, sil))
    df =  pd.DataFrame(zipped, columns= ['k','Silhouette Score'])
    return df

################################################################################


def apply_pca(data,features,n_components):
    """apply_pca reduce the dimension of the data by applying Principal Component Analysis(PCA)

    :param data: the input dataframe 
    :param features: features is the column that we want to reduce the size of
    :return: data the dataframe, features_name is the column names of o/p of PCA
    """
    pca = PCA(n_components= n_components)
    data_input = data[features]
    features_name = ['pca_'+ str(i) for i in range(n_components)]
    df_output = pd.DataFrame(pca.fit_transform(data_input), columns = features_name)
    data = pd.concat([data,df_output],axis=1)
    data = data.dropna()  
    return data, features_name
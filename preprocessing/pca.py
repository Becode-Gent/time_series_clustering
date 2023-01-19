from sklearn.decomposition import PCA
import pandas as pd

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
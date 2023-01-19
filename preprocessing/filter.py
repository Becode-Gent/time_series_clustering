# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import plotly.express as px
from pywt import wavedec
from pywt import waverec
from scipy.signal import hilbert
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pywt import *
import plotly.express as px


def apply_hilbert_filter(df,features):
    """hilbert filter is 
    data:
    """
    hilbert_features = []
    for feature in features:
        if feature =='acc_x_n':
            df[feature] -= 1 #shift the values vertically down by 1

        feature_name = feature + '_hilbert'
        hilbert_features.append(feature_name)
        df[feature_name] = np.abs(hilbert(df[feature]))
    
    return df, hilbert_features


#######################################################################


def apply_haar_filter(data,features):
    """haar filter
    """

    data_haar = data
    haar_features = []

    for feature in features:

        feature_name = feature + '_haar'
        haar_features.append(feature_name)
        # construct wavelet
        wavelet_type='haar' 
        coeffs = wavedec(data[feature].values, wavelet_type)
        # coeffs = wavedec(data.values, 'haar')

        # threshold
        threshold = 1.0

        # perform the thresholding
        coeffs_thresholded = [np.maximum(np.abs(c) - threshold, 0) * np.sign(c) for c in coeffs]

        # reconstruct time domain
        time_series_filtered = waverec(coeffs_thresholded, wavelet_type)
        data_haar[feature_name] = np.delete(time_series_filtered,-1)
    
    return data_haar, haar_features







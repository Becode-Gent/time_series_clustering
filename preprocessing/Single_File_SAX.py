import scipy
import numpy as np
import pandas as pd
import matplotlib.lines as mlines
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import norm

import numpy

import os

from sklearn.preprocessing import StandardScaler # for standardizing the Data
from tslearn.generators import random_walks
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, \
    OneD_SymbolicAggregateApproximation

# labelled_data = '/home/sammich/Documents/BeCode/becode_projects/ML_Project/data/labelled_movement_scription'
# filename1 = '/IVH1-HS1-COMBINED-201-1-1-6-47-25-1671121115-v1.4.2+lucina-buzz.csv'
# #'TP1-NC1-COMBINED-201-1-1-6-53-36-1668684025-v1.4.3+lucina-buzz.csv'
# data_file1 = labelled_data + '/'+ filename1

def single_df_SAX(single_df,features, n_symbols = 8, n_segments=100, *proportion_segments):
   
    #df = pd.read_csv(single_df)
    X = single_df
    X = X[features]
    #scale the data
    std_slc = StandardScaler()
    X = std_slc.fit_transform(X)

    X = np.resize(X, (X.shape[1],X.shape[0]))

    #scale the data using time series mean variance(online example)
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
    dataset = scaler.fit_transform(X)
    #if proportion is given use it to determine number of segements
    if proportion_segments:
        n_segments = int((single_df.index[-1])*proportion_segments)
    # SAX transform
    sax = SymbolicAggregateApproximation(n_segments = n_segments,
                                        alphabet_size_avg=n_symbols)
    sax_fit_trans_array = sax.fit_transform(dataset)
    sax_transform_df = pd.DataFrame(sax_fit_trans_array.squeeze())

    
    sax_dataset_inv = sax.inverse_transform(sax.fit_transform(dataset))
    #make df for sax_dataset_inv
    sax_dataset_inv = sax_dataset_inv.squeeze()
    sax_dataset_inv = np.resize(sax_dataset_inv, (sax_dataset_inv.shape[1],sax_dataset_inv.shape[0]))

    df_sax_inv = pd.DataFrame(sax_dataset_inv)
    df_sax_inv = df_sax_inv.rename(columns = {df_sax_inv.columns[0] : 'SAX_vector'})

    #join with original df
    df_sax_inv = single_df.join(df_sax_inv['SAX_vector'], how='left')

    return df_sax_inv, ['SAX_vector']





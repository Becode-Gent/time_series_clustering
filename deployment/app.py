
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
import random
import sys
import warnings
import streamlit as st
from PIL import Image

# import modules
sys.path.insert(0, "D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\preprocessing")
sys.path.insert(0, "D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\modeling")
from filter import apply_hilbert_filter
from filter import apply_haar_filter
from pca import apply_pca
from pca import determine_n_clusters
from segmentation import apply_segemntation
from sax import apply_symbolic_aggregation
# from viz import viz_data
# from viz import viz_output
# from viz import viz_bar
# from viz import viz_score
from get_data import load_file
from model import kmeans
from model import db_scan
from Single_File_SAX import single_df_SAX  
from get_path import get_file_name
warnings.filterwarnings('ignore')


st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    global data
    global data_copy
    global number_cluster
    global y_label # lebel for y label visualization of output
    global features_copy
    global features
    state = True

    st.title("SPINEWISE")
    menu = ['Home','Cluster model', 'Classification model']
    choice = st.sidebar.selectbox("Menu", menu)

    def viz_data(data,features,title):
        fig_0 = px.scatter(data, x ='ts_n', y=features, title=title)
        st.plotly_chart(fig_0, theme="streamlit", use_container_width=True)
        

    def viz_k(data,title):
        fig_1 = px.line(data, x='k', y='Silhouette Score',title=title)
        st.plotly_chart(fig_1, theme="streamlit", use_container_width=True)
        
    def viz_output(data, color,title,y):
        fig_2 = px.scatter(data, x='ts_n', y= y, color=color)
        fig_2.update_layout(title_text=title, title_x=0.5)
        st.plotly_chart(fig_2, theme="streamlit", use_container_width=True)
        

    def viz_bar(data,title):
        fig_3 = px.bar(data.groupby(['label', 'cluster']).size().unstack(level=1),width=600, height=400)
        fig_3.update_layout(title_text=title, title_x=0.5)
        st.plotly_chart(fig_3, theme="streamlit", use_container_width=True)
        
    def start_preprocessing(data, preprocesses, features):
        print("++++++ data preprocessing started +++++++++++++++++")
        viz_data(data_copy, features, "selected features vs ts_n")
        st.write(data_copy.head(2))

        for  preprocess in preprocesses:
            if preprocess == "filter":
                data, features = apply_haar_filter(data, features)# add function to drop ts_haar
                print("+++++++++ haar_filters applied +++++++++++++++")
                viz_data(data, features,'Output of Haar Filter')
                st.write(data.head(2))

            elif preprocess == "sax":
                data, features = single_df_SAX(data, features)
                st.write("I was in sax")
                print("+++++++++ symbolic aggregation applied ++++++++")
                viz_output(data,'SAX_vector','Output of SAX', y_label)
                st.write(data.head(2))
                
            elif preprocess == "pca":
                data, features = apply_pca(data,features, 2)
                print("++++++++++++++ pca applied ++++++++++++++++++++")
                viz_data(data, features,'Output of PCA')
                st.write(data.head(2))

            else:
                continue
        return data, features



    if choice == 'Home':
        image = Image.open("logo.png")
        st.image(image, caption="SpineWise Belgium")
        st.sidebar.image(image)

    elif choice == 'Cluster model':
        #st.subheader("Cluster model")
        file_type = st.sidebar.radio("File type", ("labeled","unlabeled"))
        file_name = st.sidebar.selectbox("File name", set(get_file_name(file_type)))
        
        data = load_file(file_name, file_type)
        col =data.columns.values.tolist()
        print("file name:", file_name)
        print("+++++++++++++++ data loaded +++++++++++++++++")
        print(f"size of data: {data.shape}")

        features = st.sidebar.multiselect("Model features(X)", set(col))
        y_label = st.sidebar.selectbox("y_label",set(features))
        preprocesses = st.sidebar.multiselect("Preprocessing steps", ('filter','pca','sax'))
        model = st.sidebar.selectbox("Model", ('DBSCAN', 'Kmeans'))
        

        if len(features) > 0:
            features_copy = features.copy()
            features_copy.append('ts_n')
            data_copy = data.copy()
            data = data[features_copy]
        
        if len(features) > 0:
            if st.sidebar.button("Plot k", key='plot_k'):
                data_k = determine_n_clusters(data,features)
                viz_k(data_k, "K vs Sil")

        number_cluster = st.sidebar.number_input("K values", 2, 12, step=1, key='number_clusters')

        if st.sidebar.button("Cluster ⏸", key='cluster'):
            if len(features) > 0:
                state = False
                data, features =  start_preprocessing(data, preprocesses, features)

                if model == 'Kmeans':
                    cluster = kmeans(data[features], number_cluster)
                    print("+++++++++++++ kmeans cluster trained ++++++++++")

                elif model == 'DBSCAN':
                    cluster = db_scan(data[features], None)
                    print("++++++++++++ db_scan cluster trained ++++++++++")
                    
                data['cluster'] = cluster.labels_
                data['ts_n'] =data_copy['ts_n']
                viz_output(data,'cluster','Output of Clusterng', y_label)
                st.write(data.head(2))

                if file_type == "labeled":
                    viz_output(data_copy,'label','Labeled Clusters',y_label)
                    data['label'] = data_copy['label']
                    viz_bar(data, 'Clusters vs Labels')
            else:
                st.warning('First select features', icon="⚠️")
                            
    elif choice == 'Classification model':
        st.subheader("Classification model")  

    


if __name__ == '__main__':
    main()




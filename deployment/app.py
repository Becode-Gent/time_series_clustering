
# import libraries
from Single_File_SAX import single_df_SAX
from get_path import get_file_name
from model import db_scan
from model import kmeans
from get_data import load_file
from pca import determine_n_clusters
from pca import apply_pca
from filter import apply_haar_filter
from filter import apply_hilbert_filter
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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from pywt import *
import random
import sys
import warnings
import streamlit as st
from PIL import Image

# import modules
sys.path.insert(0, "..\preprocessing")
sys.path.insert(0, "..\modeling")
# from viz import viz_data
# from viz import viz_output
# from viz import viz_bar
# from viz import viz_score

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():

    global data
    global data_copy
    global number_cluster
    global y_label  # lebel for y label visualization of output
    global features_copy
    global features
    state = True

    st.title("👷🏼‍♀️  UNLASSIFIED CLUSTERING 🧑🏾‍💻")
    menu = ['Home', 'Analysis', 'Cluster model', 'Classification model']
    choice = st.sidebar.selectbox("Menu", menu)

    def viz_data(data, features, title):
        fig_0 = px.scatter(data, x='ts_n', y=features, title=title)
        st.plotly_chart(fig_0, theme="streamlit", use_container_width=True)

    def viz_k(data, title):
        fig_1 = px.line(data, x='k', y='Silhouette Score', title=title)
        st.plotly_chart(fig_1, theme="streamlit", use_container_width=True)

    def viz_output(data, color, title, y):
        fig_2 = px.scatter(data, x='ts_n', y=y, color=color)
        fig_2.update_layout(title_text=title, title_x=0.5)
        st.plotly_chart(fig_2, theme="streamlit", use_container_width=True)

    def viz_bar(data, title):
        fig_3 = px.bar(data.groupby(['label', 'cluster']).size().unstack(
            level=1), width=600, height=400)
        fig_3.update_layout(title_text=title, title_x=0.5)
        st.plotly_chart(fig_3, theme="streamlit", use_container_width=True)

    def start_preprocessing(data, preprocesses, features):
        print("++++++ data preprocessing started +++++++++++++++++")
        viz_data(data_copy, features, "selected features vs ts_n")
        st.write(data_copy.head(2))

        for preprocess in preprocesses:
            if preprocess == "filter":
                data, features = apply_haar_filter(
                    data, features)  # add function to drop ts_haar
                print("+++++++++ haar_filters applied +++++++++++++++")
                viz_data(data, features, 'Output of Haar Filter')
                st.write(data.head(2))

            elif preprocess == "sax":
                data, features = single_df_SAX(data, features)
                print("+++++++++ symbolic aggregation applied ++++++++")
                viz_output(data, 'SAX_vector', 'Output of SAX', y_label)
                st.write(data.head(2))

            elif preprocess == "pca":
                data, features = apply_pca(data, features, n_components)
                print("++++++++++++++ pca applied ++++++++++++++++++++")
                viz_data(data, features, 'Output of PCA')
                st.write(data.head(2))

            else:
                continue
        return data, features

    if choice == 'Cluster model':
        file_type = st.sidebar.radio("File type", ("labeled", "unlabeled"))
        file_name = st.sidebar.selectbox(
            "File name", set(get_file_name(file_type)))
        data = load_file(file_name, file_type)
        col = data.columns.values.tolist()
        print("file name:", file_name)
        print("+++++++++++++++ data loaded +++++++++++++++++")
        print(f"size of data: {data.shape}")
        features = st.sidebar.multiselect("select features(X)", set(col))
        y_label = st.sidebar.selectbox("y_label", set(features))
        preprocesses = st.sidebar.multiselect(
            "Preprocessing steps", ('filter', 'pca', 'sax'))

        if 'pca' in preprocesses:
            n_components = st.sidebar.number_input(
                "n_components(pca)", 1, len(features)-1, step=1, key='n_components')

        model = st.sidebar.selectbox("Model", ('DBSCAN', 'Kmeans'))
        if len(features) > 0:
            features_copy = features.copy()
            features_copy.append('ts_n')
            data_copy = data.copy()
            data = data[features_copy]

        if len(features) > 0:
            if st.sidebar.button("Plot k", key='plot_k'):
                data_k = determine_n_clusters(data, features)
                viz_k(data_k, "K vs Sil")
                st.write(data_k.head())

        if model == 'Kmeans':
            number_cluster = st.sidebar.number_input(
                "K values", 2, 12, step=1, key='number_clusters')

        if st.sidebar.button("Cluster ⏸", key='cluster'):
            if len(features) > 0:
                state = False
                data, features = start_preprocessing(
                    data, preprocesses, features)

                if model == 'Kmeans':
                    cluster = kmeans(data[features], number_cluster)
                    print("+++++++++++++ kmeans cluster trained ++++++++++")

                elif model == 'DBSCAN':
                    cluster = db_scan(data[features], None)
                    print("++++++++++++ db_scan cluster trained ++++++++++")

                data['cluster'] = cluster.labels_
                data['ts_n'] = data_copy['ts_n']
                viz_output(data, 'cluster', 'Output of Clusterng', y_label)
                st.write(data.head(2))

                if file_type == "labeled":
                    viz_output(data_copy, 'label', 'Labeled Clusters', y_label)
                    data['label'] = data_copy['label']
                    viz_bar(data, 'Clusters vs Labels')
            else:
                st.warning('First select features', icon="⚠️")

    ##############################################################################

    elif choice == 'Classification model':
        global class_names
        st.subheader("Classification model")
        st.sidebar.subheader("Load files")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your input CSV file", type=["csv"])

        def plot_metrics(metrics_list):
            if 'Confusion Matrix' in metrics_list:
                st.subheader("Confusion Matrix")
                plot_confusion_matrix(
                    model, x_test, y_test, display_labels=class_names)
                st.pyplot()

            if 'ROC Curve' in metrics_list:
                st.subheader("ROC Curve")
                plot_roc_curve(model, x_test, y_test)
                st.pyplot()

            if 'Precision-Recall Curve' in metrics_list:
                st.subheader('Precision-Recall Curve')
                plot_precision_recall_curve(model, x_test, y_test)
                st.pyplot()

        def start_class_preprocessing(data, class_preprocesses, class_features_copy, target):
            for class_preprocess in class_preprocesses:
                if class_preprocess == 'standard scaling':
                    standard_scaler = StandardScaler()
                    for class_feature in class_features_copy:
                        st.write("prep", data.head())
                        st.write('prep', data[class_feature])
                        data[class_feature] = standard_scaler.fit_transform(
                            data[class_feature]).reshape(-1, 1)

                elif class_preprocess == "split":
                    y = data[target]
                    x = data.drop(columns=[target])
                    x_train, x_test, y_train, y_test = train_test_split(
                        x, y, test_size=0.3, random_state=0)

            return x_train, x_test, y_train, y_test

        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            col = data.columns.values.tolist()
            class_features = list(
                st.sidebar.multiselect("Features(X)", set(col)))
            class_features_copy = class_features.copy()
            target = st.sidebar.selectbox("Target(y)", set(col), key='target')
            st.write('class features', class_features)
            st.write("target", target)
            if len(target) > 0:
                class_names = data[target].unique().tolist()
                st.write("class_names", class_names)
                st.write("shape of data : ", data.shape)
                st.sidebar.subheader("Preprocesses")
                class_preprocesses = st.sidebar.multiselect(
                    "Basic preprocesses", ("standard scaling", "label encoding", "split"), key='class_preprocesses')
                st.write('class_preprocesses', class_preprocesses)

                if len(class_preprocesses) > 0:
                    if 'split' in class_preprocesses:
                        st.write('data before s', data.head())
                        class_features.append(target)
                        st.write("class_features", class_features)
                        st.write("class_features_copy", class_features_copy)
                        data = data[class_features]
                        st.write(data.head())
                        x_train, x_test, y_train, y_test = start_class_preprocessing(
                            data, class_preprocesses, class_features_copy, target)

        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox(
            "Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input(
                "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
            kernel = st.sidebar.radio(
                "Kernel", ("rbf", "linear"), key='kernel')
            gamma = st.sidebar.radio(
                "Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
            metrics = st.sidebar.multiselect(
                "What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Support Vector Machine (SVM) Results")
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(
                    y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(
                    y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)

        #*************************************************#

        if classifier == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input(
                "C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
            max_iter = st.sidebar.slider(
                "Maximum number of iterations", 100, 500, key='max_iter')

            metrics = st.sidebar.multiselect(
                "What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(
                    C=C, penalty='l2', max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(
                    y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(
                    y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)

         #*************************************************#

        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.number_input(
                "The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
            max_depth = st.sidebar.number_input(
                "The maximum depth of the tree", 1, 20, step=1, key='max_depth')
            bootstrap = st.sidebar.radio(
                "Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
            metrics = st.sidebar.multiselect(
                "What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(
                    y_test, y_pred, labels=class_names).round(2))
                st.write("Recall: ", recall_score(
                    y_test, y_pred, labels=class_names).round(2))
                plot_metrics(metrics)

    elif choice == 'Home':
        st.subheader("Pipeline of the project")
        st.image('pipe.png')
        st.sidebar.image('logo.png')

    elif choice == 'Analysis':
        st.sidebar.subheader("Load files")
        uploaded_file = st.sidebar.file_uploader(
            "Upload your input CSV file", type=["csv"])

        if uploaded_file:
            data_analysis = pd.read_csv(uploaded_file)
            st.write(data_analysis.head())

            if st.sidebar.button("show summary", key='a_sw'):  # todo, add more summary
                st.write('Columns :', data_analysis.columns)
                st.write('Shape   :', data_analysis.shape)

            st.sidebar.subheader("Preprocesses")
            if data_analysis.isnull().values.any():
                st.warning('There are NaN values', icon="⚠️")
                null_methods = st.sidebar.selectbox(
                    "What to do with NaN?", ('drop', 'fill forward', 'fill backward'))

                if len(null_methods) > 0:
                    # todo, add methods for 'fill forwar' and others
                    if st.sidebar.button("handle", key='handle_st'):
                        if null_methods == 'drop':
                            data_analysis = data_analysis.dropna()
                            st.write("New dataframe with handled nans")
                            st.write(data_analysis.head())

            preprocessing_steps = st.sidebar.multiselect(
                "basic preprocessing", ('standard scaling', 'label encoding', 'remove outlier'))
            # todo, methods for preprocessing
            if len(preprocessing_steps) > 0:
                if st.sidebar.button("apply", key='an_ap'):
                    st.write("******* preprocessing started ********")

            st.sidebar.subheader("Visualization")
            x_analysis = st.sidebar.multiselect(
                "select  x", set(data_analysis.columns.values.tolist()))
            y_analysis = st.sidebar.multiselect(
                "select  y", set(data_analysis.columns.values.tolist()))

            # todo, methods for visualization
            if st.sidebar.button("Visualize", key='bt_vis'):
                st.write("***** visualization started ******")


if __name__ == '__main__':
    main()

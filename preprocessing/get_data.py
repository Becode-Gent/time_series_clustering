import pandas as pd
import numpy as np

def load_file(file_name, file_type):
    """load_file load the csv file into pandas dataframe

    :param file_name: the name of the file
    :param file_type: type of the file labeled|unlabeled
    :return: df_main, dataframe 
    """
    
    if file_type == "labeled":
        path = r"D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\data\labeled\movement_scription" + "\\" + file_name
        # change the file path to your directory
    elif file_type == "unlabeled":
        path = r"D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\data\unlabeled" + "\\" + file_name
   
    df_main= pd.read_csv(path)
    df_main = df_main.iloc[75:]
    df_main.reset_index(drop=True, inplace=True)
    df_main = df_main.dropna()
    return df_main


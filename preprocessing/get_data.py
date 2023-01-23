import pandas as pd
import numpy as np

def load_file(file_name, file_type):
    """this load  a file """


    if file_type == "labeled":
        path = r"D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\data\labeled\movement_scription" + "\\" + file_name
        

    elif file_type == "unlabeled":
        path = r"D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\data\unlabeled" + "\\" + file_name
    
    
    df_main= pd.read_csv(path)
    df_main = df_main.iloc[75:]
    df_main.reset_index(drop=True, inplace=True)
    df_main = df_main.dropna()
    
    return df_main


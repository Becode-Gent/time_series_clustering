import pandas as pd
import sys

# change the path to your directory
sys.path.insert(0, "D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\preprocessing")
from get_data import load_file
from get_path import get_file_name


def concat_csv(file_type):
    """concatnate csv files in folder vertically

    :param file_type: type of the file labeled|unlabeled
    :return: df_all, merged dataframe 
    """

    labeled_files = get_file_name('labeled')
    data_all = pd.DataFrame()
    for labeled_file in labeled_files:
        data= load_file(labeled_file, file_type)
        data_all = pd.concat([data_all, data],   # Combine vertically
                            ignore_index = True,
                            sort = False)
    return data_all

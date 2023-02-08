import os


def get_file_name(file_type):
    """get_file_name get the list of file names with .csv extension in the directory

    :param file_type: the type of the file labeled|unlabeled
    :return: filename_list,  the list of files in the directory
    """

    if file_type == "labeled":
        path = r"D:\Arai4_Projects\unclassified_clustering_project\unclassified_clustering\ML_Project\data\labeled\movement_scription"
    else:
        path = r"D:\Arai4_Projects\unclassified_clustering_project\unclassified_clustering\ML_Project\data\unlabeled"

    filename_list = []
    ext = ('.csv')
    for file in os.listdir(path):
        filename_list.append(file)

    return filename_list

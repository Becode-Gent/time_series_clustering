import os

def get_file_name(file_type):
    """ get list of file names in a directory """

    
    if file_type == "labeled":
        path = r"D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\data\labeled\movement_scription"
    else:
        path = r"D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\data\unlabeled"

        
    #path_of_the_directory = "D:\Arai4_Projects\spinwise_project\spinewise_amanuel\ML_Project\data\"
    filename_list = []
    ext = ('.csv')
    for file in os.listdir(path):
        
        filename_list.append(file) 
       

    
    return filename_list
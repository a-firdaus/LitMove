import os
import shutil

# Related to activities of editing directory path; copy/ delete files within directory

def splitall(path):
    """
    Splitting path into its individual components of each sub-/folder.

    Args:
        path (str): The input path to be split.

    Returns:
        list: A list containing the components of the path.

    Source: 
        https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
    """
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def copy_rename_single_file(destination_directory, source_directory, filename, prefix):
    """
    Copy a file from the source directory to the destination directory with an optional filename prefix.

    Args:
        destination_directory (str): The directory where the file should be copied.
        source_directory (str): The directory from which the file should be copied.
        filename (str): The name of the file to be copied.
        prefix (str): An optional prefix to be added to the new filename.

    Returns:
        None
    """
    # Generate the new filename
    if prefix == None:
        new_filename = f"{filename}"
    else:
        new_filename = f"{filename}_{prefix}"
    
    # Get the source file path and destination file path
    destination_path = os.path.join(destination_directory, new_filename)
    
    # Copy the file to the destination directory with the new name
    source_path = os.path.join(source_directory, filename)
    shutil.copy2(source_path, destination_path)
    # print(f"File copied and renamed: {filename} -> {new_filename}")


def copy_rename_files(dataframe, destination_directory, filename, prefix, savedir):
    """
    Copy and rename multiple files based on the contents of a DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame containing file information.
        destination_directory (str): The directory where files should be copied.
        filename (str): The base name of the files.
        prefix (str): An optional prefix to be added to the new filenames.
        savedir (bool): If True, save the new file paths to the DataFrame.

    Returns:
        None
    """
    if savedir == True:
        col_subdir_copiedrenamed_files = f"subdir_{filename}"

        dataframe[col_subdir_copiedrenamed_files] = None

    elif savedir == False:
        pass

    for index in range(dataframe["geometry"].size):
        # Generate the new filename
        if prefix == None:
            new_filename = f"{int(dataframe['geometry'][index])}_{int(dataframe['path'][index])}_{filename}"
        else:
            new_filename = f"{int(dataframe['geometry'][index])}_{int(dataframe['path'][index])}_{filename}_{prefix}"


        # Get the source file path and destination file path
        destination_path = os.path.join(destination_directory, new_filename)
        
        # Copy the file to the destination directory with the new name
        shutil.copy2(dataframe['subdir_new_system'][index], destination_path)
        # print(f"File copied and renamed: {filename} -> {new_filename}")
    
        if savedir == True:
            dataframe.at[int(index), col_subdir_copiedrenamed_files] = destination_path
        elif savedir == False:
            pass


# def copy_rename_single_file_and_delete_elements(destination_directory, source_directory, filename, prefix, line_ranges, line_numbers_edit, new_contents):
#     # Generate the new filename
#     new_filename = f"{filename}_{prefix}"
    
#     # Get the source file path and destination file path
#     destination_path = os.path.join(destination_directory, new_filename)
    
#     # Copy the file to the destination directory with the new name
#     source_path = os.path.join(source_directory, filename)
#     shutil.copy2(source_path, destination_path)
#     print(f"File copied and renamed: {filename} -> {new_filename}")

#     File.delete_elements(destination_path, line_ranges, line_numbers_edit, new_contents)


# def copy_rename_files_and_delete_elements(file_loc, destination_directory, filename, index, prefix, line_ranges, line_numbers_edit, new_contents):
#     # Generate the new filename
#     new_filename = f"{int(file_loc['geometry'][index])}_{int(file_loc['path'][index])}_{filename}_{prefix}"
    
#     # Get the source file path and destination file path
#     destination_path = os.path.join(destination_directory, new_filename)
    
#     # Copy the file to the destination directory with the new name
#     shutil.copy2(file_loc['subdir_new_system'][index], destination_path)
#     print(f"File copied and renamed: {filename} -> {new_filename}")

#     File.delete_elements(destination_path, line_ranges, line_numbers_edit, new_contents)


def check_folder_existance(folder_name, empty_folder):
    """
    Check if a folder exists, create it if not, and optionally empty it.

    Args:
        folder_name (str): The name of the folder to check or create.
        empty_folder (bool): If True, empty the folder if it already exists.

    Returns:
        None
    """
    # Check if the folder exists
    if not os.path.exists(folder_name):
        # Create the folder if it doesn't exist
        os.makedirs(folder_name)
        # print(f"Folder '{folder_name}' created.")
    else:
        if empty_folder == True:
            empty_folder(folder_name)
            # print(f"Folder '{folder_name}' already exists. Emptying it.")
        elif empty_folder == False:
            pass


def empty_folder(folder_name):
    """
    Empty the contents of a folder.

    Args:
        folder_name (str): The name of the folder to be emptied.

    Returns:
        None
    """
    files_inside_folder = os.listdir(folder_name)
    for i in files_inside_folder:
        path_to_files = folder_name + i 
        os.remove(path_to_files)


def delete_files(dataframe, folder_name, file_name_w_format):
    """
    Delete files based in the specified folder.

    Args:
        dataframe (pd.DataFrame): A DataFrame containing information about files ('geometry' and 'path').
        folder_name (str): The directory where the files are located.
        file_name_w_format (str): The base name of the files with format information.

    Returns:
        None

    Raises:
        OSError: If an error occurs during file deletion, an exception is caught,
                and an error message is printed.
    """
    for idx in range(dataframe["geometry"].size):
        filename_to_delete = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{file_name_w_format}"
        filename_to_delete_path = os.path.join(folder_name, filename_to_delete)
        
        try:
            # Attempt to delete the file
            os.remove(filename_to_delete_path)
            # print(f"{filename_to_delete_path} has been deleted.")
        except OSError as e:
            # Handle any errors that occur during file deletion
            print(f"Error deleting {filename_to_delete_path}: {e}")

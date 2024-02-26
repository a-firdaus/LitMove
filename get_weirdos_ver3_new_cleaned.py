import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os, sys
import shutil
from itertools import islice
from itertools import repeat
from addict import Dict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import math
from collections import defaultdict
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from chart_studio import plotly
from adjustText import adjust_text

import plotly.offline as pyoff
import re
import mplcursors
import mpldatacursor

# pymatgen libraries
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.inputs import Poscar


def replace(i):
    """
    Replace a string with NaN if it cannot be converted to a float.

    Args:
        i (str): The input string.

    Returns:
        float or np.nan: The converted float or NaN.

    Source:
        https://stackoverflow.com/questions/57048617/how-do-i-replace-all-string-values-with-nan-dynamically
    """
    try:
        float(i)
        return float(i)
    except:
        return np.nan


class FileOperations:
    @staticmethod
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


    @staticmethod
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


    @staticmethod
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


    # @staticmethod
    # def copy_rename_single_file_and_delete_elements(destination_directory, source_directory, filename, prefix, line_ranges, line_numbers_edit, new_contents):
    #     # Generate the new filename
    #     new_filename = f"{filename}_{prefix}"
        
    #     # Get the source file path and destination file path
    #     destination_path = os.path.join(destination_directory, new_filename)
        
    #     # Copy the file to the destination directory with the new name
    #     source_path = os.path.join(source_directory, filename)
    #     shutil.copy2(source_path, destination_path)
    #     print(f"File copied and renamed: {filename} -> {new_filename}")

    #     delete_elements(destination_path, line_ranges, line_numbers_edit, new_contents)


    # @staticmethod
    # def copy_rename_files_and_delete_elements(file_loc, destination_directory, filename, index, prefix, line_ranges, line_numbers_edit, new_contents):
    #     # Generate the new filename
    #     new_filename = f"{int(file_loc['geometry'][index])}_{int(file_loc['path'][index])}_{filename}_{prefix}"
        
    #     # Get the source file path and destination file path
    #     destination_path = os.path.join(destination_directory, new_filename)
        
    #     # Copy the file to the destination directory with the new name
    #     shutil.copy2(file_loc['subdir_new_system'][index], destination_path)
    #     print(f"File copied and renamed: {filename} -> {new_filename}")

    #     delete_elements(destination_path, line_ranges, line_numbers_edit, new_contents)


    @staticmethod
    def delete_lines(file_path, line_ranges):
        """
        Delete specified lines from a file.

        Args:
            file_path (str): The path to the file to be modified.
            line_ranges (list): A list of tuples representing the ranges of lines to be deleted.

        Returns:
            None
        """
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        lines_to_delete = set()
        for line_range in line_ranges:
            start, end = line_range
            if 1 <= start <= len(lines) and start <= end <= len(lines):
                lines_to_delete.update(range(start - 1, end))

        modified_lines = [line for i, line in enumerate(lines) if i not in lines_to_delete]

        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        # print(f"Lines deleted successfully in file: {file_path}")


    @staticmethod
    def delete_elements(file_path, line_ranges, line_numbers_edit, new_contents):
        """
        Delete specified lines and edit others in a file.

        Args:
            file_path (str): The path to the file to be modified.
            line_ranges (list): A list of tuples representing the ranges of lines to be deleted.
            line_numbers_edit (list): A list of line numbers to be edited.
            new_contents (list): A list of new contents for the edited lines.

        Returns:
            None
        """
        FileOperations.delete_lines(file_path, line_ranges)
        FileOperations.edit_lines(file_path, line_numbers_edit, new_contents)


    @staticmethod
    def edit_lines(file_path, line_numbers, new_contents):
        """
        Edit specified lines in a file.

        Args:
            file_path (str): The path to the file to be modified.
            line_numbers (list): A list of line numbers to be edited.
            new_contents (list): A list of new contents for the edited lines.

        Returns:
            None
        """
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify the line content
        for line_number, new_content in zip(line_numbers, new_contents):
            if 1 <= line_number <= len(lines):
                lines[line_number - 1] = new_content + '\n'

            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)
                # # print(f"Line edited successfully.")
        # # else:
        # # #     print(f"Invalid line number: {line_number}")


    @staticmethod
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
                FileOperations.empty_folder(folder_name)
                # print(f"Folder '{folder_name}' already exists. Emptying it.")
            elif empty_folder == False:
                pass


    @staticmethod
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


    @staticmethod
    def delete_files(dataframe, folder_name, file_name_w_format):
        """
        Delete files based on information in a DataFrame.

        This method iterates over the rows of a DataFrame and constructs filenames
        based on specified columns ('geometry' and 'path'). It then attempts to
        delete each file from the specified folder.

        Args:
            dataframe (pd.DataFrame): A DataFrame containing information about files.
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


# class Transformation:
def get_structure_with_library(dataframe, destination_directory, filename, structure_reference, var_name, prefix):
    for idx in range(dataframe["geometry"].size):
        if prefix == None: 
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}"
        else:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}_{prefix}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)
        structure = Structure.from_file(filename_to_transform_path)
        
        # StructureMatcher can accept different tolerances for judging equivalence
        matcher = StructureMatcher(primitive_cell=False)
        # first, we can verify these lattices are equivalent
        matcher_verify = matcher.fit(structure_reference, structure)  # returns True
        dataframe['verify_w_lib'][idx] = matcher_verify
        # # df['verify_w_lib'][idx] = matcher_verify
        # # # print(f"verify_w_lib: {matcher_verify}")
        if matcher_verify == False:
            print(f"Matcher doesn't match.")

        transformed_structure = matcher.get_s2_like_s1(structure_reference, structure, include_ignored_species=True)
        cif = CifWriter(transformed_structure)
        if prefix == None: 
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        else:
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}_{prefix}.cif"
        destination_path = os.path.join(destination_directory, cif_filename)
        cif.write_file(destination_path)


# corrected or at least attempted to
def get_structure_with_linalg(dataframe, destination_directory, filename, structure_reference, var_name, prefix):
    for idx in range(dataframe["geometry"].size):
        if prefix == None: 
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}"
        else:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}_{prefix}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)
        structure = Structure.from_file(filename_to_transform_path)

        # StructureMatcher can accept different tolerances for judging equivalence
        matcher = StructureMatcher(primitive_cell=False) # don't work if it's True
        # first, we can verify these lattices are equivalent
        matcher_verify = matcher.fit(structure_reference, structure)  # returns True
        dataframe['verify_w_linalg'][idx] = matcher_verify
        if matcher_verify == False:
            print(f"Matcher doesn't match.")

        # Transform desired structure into structure_reference
            # output of transformation:
                # 3x3 matrix of supercell transformation;
                # 1x3 vector of fractional translation;
                # 1x4 mapping to transform struct2 to be similar to struct1
        transformation = matcher.get_transformation(structure_reference, structure)
        if transformation is None:
            return None
        # if prefix == None:
        #     dataframe['trf_matrix'][idx] = transformation[0]
        # else: 
        #     dataframe['trf_matrix_P'][idx] = transformation[0]
        scaling, translation, mapping = transformation
        # print(f"scaling: {scaling}")
        # print(f"scaling type: {type(scaling)}")

        if prefix == None:
            dataframe.at[idx, 'scaling'] = scaling
            dataframe.at[idx, 'translation'] = translation
            dataframe.at[idx, 'mapping'] = mapping
        else:
            dataframe.at[idx, 'transformation_P'] = transformation

        # Apply scaling
        scaled_coords = np.dot(structure.frac_coords, scaling.T)
        # scaled_coords = np.round(scaled_coords, decimals=16)
        # print(scaled_coords)
        # apply translation
        translated_coords = scaled_coords + translation
        # apply mapping
        mapped_coords = translated_coords[mapping]
        # # # long story short
        # # transformed_coords = np.dot(structure.frac_coords[mapping], scaling.T) + translation

        # Create a new structure with the transformed coordinates
        transformed_structure = Structure(structure.lattice, structure.species, mapped_coords)
        # transformed_structure = Structure.from_sites(mapped_coords) # similar to above but different input
        
        cif = CifWriter(transformed_structure)
        if prefix == None: 
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        else:
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}_{prefix}.cif"
        destination_path = os.path.join(destination_directory, cif_filename)
        cif.write_file(destination_path)


# for sanity check
def get_structure_with_linalg_combinded_with_library(dataframe, destination_directory, filename, structure_reference, var_name, prefix):
    for idx in range(dataframe["geometry"].size):
        if prefix == None: 
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}"
        else:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}_{prefix}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)
        structure = Structure.from_file(filename_to_transform_path)

        # StructureMatcher can accept different tolerances for judging equivalence
        matcher = StructureMatcher(primitive_cell=False)
        # first, we can verify these lattices are equivalent
        matcher_verify = matcher.fit(structure_reference, structure)  # returns True
        dataframe['verify_w_linalg'][idx] = matcher_verify
        if matcher_verify == False:
            print(f"Matcher doesn't match.")

        # Transform desired structure into structure_reference
            # output of transformation:
                # 3x3 matrix of supercell transformation;
                # 1x3 vector of fractional translation;
                # 1x4 mapping to transform struct2 to be similar to struct1
        transformation = matcher.get_transformation(structure_reference, structure)
        if transformation is None:
            return None
        
        scaling, translation, mapping = transformation

        if prefix == None:
            dataframe.at[idx, 'transformation'] = transformation
        else:
            dataframe.at[idx, f'transformation_{prefix}'] = transformation

        sites = list(structure)
        # Append the ignored sites at the end.
        # # sites.extend([site for site in struct2 if site not in s2])
        temp = Structure.from_sites(sites)

        # Apply scaling
        temp.make_supercell(scaling)
        # apply translation
        temp.translate_sites(list(range(len(temp))), translation)

        # Apply some modification from library
        for i, j in enumerate(mapping[: len(structure_reference)]):
            if j is not None:
                vec = np.round(structure_reference[i].frac_coords - temp[j].frac_coords)
                temp.translate_sites(j, vec, to_unit_cell=False)
        sites = [temp.sites[i] for i in mapping if i is not None]
        # if include_ignored_species:
        #     start = int(round(len(temp) / len(struct2) * len(s2)))
        #     sites.extend(temp.sites[start:])
        transformed_structure = Structure.from_sites(sites)
        # transformed_structure = Structure(structure.lattice, structure.species, mapped_coords) # similar to above but different input
        
        cif = CifWriter(transformed_structure)
        if prefix == None: 
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        else:
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}_{prefix}.cif"
        destination_path = os.path.join(destination_directory, cif_filename)
        cif.write_file(destination_path)


def get_structure_with_linalg_orientated(dataframe, destination_directory, filename, var_name):
    ## POSCAR file is also created
    dataframe['subdir_orientated'] = None
    for idx in range(dataframe["geometry"].size):
        filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)
        structure = Structure.from_file(filename_to_transform_path)

        # Transform desired structure into structure_reference
        scaling = dataframe["scaling"][idx]
        translation = dataframe["translation"][idx]
        mapping = dataframe["mapping"][idx]

        # Apply scaling
        scaled_coords = np.dot(structure.frac_coords, scaling.T)
        # apply translation
        translated_coords = scaled_coords + translation
        # apply mapping (no mapping here)
        mapped_coords = translated_coords[mapping]
        # # # long story short
        # # transformed_coords = np.dot(structure.frac_coords[mapping], scaling.T) + translation

        # Create a new structure with the transformed coordinates
        transformed_structure = Structure(structure.lattice, structure.species, mapped_coords)
        # transformed_structure = Structure.from_sites(mapped_coords) # similar to above but different input
        
        cif = CifWriter(transformed_structure)
        cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        destination_path = os.path.join(destination_directory, cif_filename)
        # dataframe
        cif.write_file(destination_path)

        poscar = Poscar(transformed_structure)
        poscar_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_POSCAR_{var_name}"
        destination_path_poscar = os.path.join(destination_directory, poscar_filename)
        poscar.write_file(destination_path_poscar)
        
        dataframe['subdir_orientated'][idx] = destination_path


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

def get_orientated_cif_positive(dataframe, destination_directory, cif_line_nr_start, cif_columns, var_name_in, var_name_out):
    dataframe['subdir_orientated_positive'] = None
    for idx in range(dataframe["geometry"].size):
        # lines = []
        # print(idx)
        # print(idx)
        filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name_in}.cif"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)

        with open(filename_to_transform_path, 'r') as file:
            lines = file.readlines()
        data = lines[cif_line_nr_start:]

        # Split each string by space and create the DataFrame
        df = pd.DataFrame([string.strip().split() for string in data])

        # Optional: Rename the columns
        df.columns = cif_columns

        df_positive_val = df
        for idx_a, coord_x in enumerate(df_positive_val['coord_x']):
            if float(coord_x) < 0:
                coord_x = float(coord_x) + 1
                df_positive_val['coord_x'][idx_a] = '{:.{width}f}'.format(coord_x, width=8)
            else:
                df_positive_val['coord_x'][idx_a] = coord_x

        for idx_a, coord_y in enumerate(df_positive_val['coord_y']):
            if float(coord_y) < 0:
                coord_y = float(coord_y) + 1
                df_positive_val['coord_y'][idx_a] = '{:.{width}f}'.format(coord_y, width=8)
            else:
                df_positive_val['coord_y'][idx_a] = coord_y

        for idx_a, coord_z in enumerate(df_positive_val['coord_z']):
            if float(coord_z) < 0:
                coord_z = float(coord_z) + 1
                df_positive_val['coord_z'][idx_a] = '{:.{width}f}'.format(coord_z, width=8)
            else:
                df_positive_val['coord_z'][idx_a] = coord_z

        row_list = df_positive_val.to_string(index=False, header=False).split('\n')
        row_list_space = ['  '.join(string.split()) for string in row_list] # 2 spaces of distance
        row_list_w_beginning = ['  ' + row for row in row_list_space]       # 2 spaces in the beginning
        absolute_correct_list = '\n'.join(row_list_w_beginning).splitlines()        

        line_append_list = []
        for idx_c, line in enumerate(absolute_correct_list):
            line_new_line = str(line) + '\n'
            line_append_list.append(line_new_line)

        file_list = lines[:cif_line_nr_start] + line_append_list

        
        # print(cif_filename_positive)
        
        # print(f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}")
        cif_filename_positive = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name_out}.cif"
        destination_path = os.path.join(destination_directory, cif_filename_positive)

        with open(destination_path, 'w') as fp:
            for item in file_list:
                fp.write(item)

        dataframe['subdir_orientated_positive'][idx] = destination_path


def get_orientated_positive_cif(dataframe, destination_directory, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal):
    ## Convert new cif file of orientated structure into only positive value
    dataframe['subdir_orientated_positive_cif'] = None

    for idx in range(dataframe["geometry"].size):
        filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name_in}.cif"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)

        with open(filename_to_transform_path, 'r') as file:
            lines = file.readlines()
        data = lines[cif_line_nr_start:]

        # Split each string by space and create the DataFrame
        df = pd.DataFrame([string.strip().split() for string in data])

        # Optional: Rename the columns
        df.columns = cif_columns

        df_positive_val = df
        for idx_a, coord_x in enumerate(df_positive_val['coord_x']):
            while float(coord_x) < 0:
                coord_x = float(coord_x) + 1
            df_positive_val['coord_x'][idx_a] = '{:.{width}f}'.format(float(coord_x), width=n_decimal)

        for idx_a, coord_y in enumerate(df_positive_val['coord_y']):
            while float(coord_y) < 0:
                coord_y = float(coord_y) + 1
            df_positive_val['coord_y'][idx_a] = '{:.{width}f}'.format(float(coord_y), width=n_decimal)

        for idx_a, coord_z in enumerate(df_positive_val['coord_z']):
            while float(coord_z) < 0:
                coord_z = float(coord_z) + 1
            df_positive_val['coord_z'][idx_a] = '{:.{width}f}'.format(float(coord_z), width=n_decimal)

        row_list = df_positive_val.to_string(index=False, header=False).split('\n')
        row_list_space = ['  '.join(string.split()) for string in row_list] # 2 spaces of distance
        row_list_w_beginning = ['  ' + row for row in row_list_space]       # 2 spaces in the beginning
        absolute_correct_list = '\n'.join(row_list_w_beginning).splitlines()        

        line_append_list = []
        for idx_c, line in enumerate(absolute_correct_list):
            line_new_line = str(line) + '\n'
            line_append_list.append(line_new_line)

        file_list = lines[:cif_line_nr_start] + line_append_list

        
        # print(cif_filename_positive)
        
        # print(f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}")
        cif_filename_positive = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name_out}.cif"
        destination_path = os.path.join(destination_directory, cif_filename_positive)

        with open(destination_path, 'w') as fp:
            for item in file_list:
                fp.write(item)

        dataframe['subdir_orientated_positive_cif'][idx] = destination_path


# def get_orientated_positive_lessthan1_cif(dataframe, destination_directory, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal):
#     dataframe['subdir_orientated_positive_lessthan1_cif'] = None
#     for idx in range(dataframe["geometry"].size):
#         filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name_in}.cif"
#         filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)

#         with open(filename_to_transform_path, 'r') as file:
#             lines = file.readlines()
#         data = lines[cif_line_nr_start:]

#         # Split each string by space and create the DataFrame
#         df = pd.DataFrame([string.strip().split() for string in data])

#         # Optional: Rename the columns
#         df.columns = cif_columns

#         df_positive_val = df
#         for idx_a, coord_x in enumerate(df_positive_val['coord_x']):
#             while float(coord_x) < 0:
#                 coord_x = float(coord_x) + 1
#             while float(coord_x) > 1:
#                 coord_x = float(coord_x) - 1
#             df_positive_val['coord_x'][idx_a] = '{:.{width}f}'.format(float(coord_x), width=n_decimal)

#         for idx_a, coord_y in enumerate(df_positive_val['coord_y']):
#             while float(coord_y) < 0:
#                 coord_y = float(coord_y) + 1
#             while float(coord_y) > 1:
#                 coord_y = float(coord_y) - 1
#             df_positive_val['coord_y'][idx_a] = '{:.{width}f}'.format(float(coord_y), width=n_decimal)

#         for idx_a, coord_z in enumerate(df_positive_val['coord_z']):
#             while float(coord_z) < 0:
#                 coord_z = float(coord_z) + 1
#             while float(coord_z) > 1:
#                 coord_z = float(coord_z) - 1
#             df_positive_val['coord_z'][idx_a] = '{:.{width}f}'.format(float(coord_z), width=n_decimal)

#         row_list = df_positive_val.to_string(index=False, header=False).split('\n')
#         row_list_space = ['  '.join(string.split()) for string in row_list] # 2 spaces of distance
#         row_list_w_beginning = ['  ' + row for row in row_list_space]       # 2 spaces in the beginning
#         absolute_correct_list = '\n'.join(row_list_w_beginning).splitlines()        

#         line_append_list = []
#         for idx_c, line in enumerate(absolute_correct_list):
#             line_new_line = str(line) + '\n'
#             line_append_list.append(line_new_line)

#         file_list = lines[:cif_line_nr_start] + line_append_list

        
#         # print(cif_filename_positive)
        
#         # print(f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}")
#         cif_filename_positive = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name_out}.cif"
#         destination_path = os.path.join(destination_directory, cif_filename_positive)

#         with open(destination_path, 'w') as fp:
#             for item in file_list:
#                 fp.write(item)

#         dataframe['subdir_orientated_positive_lessthan1_cif'][idx] = destination_path


def get_CONTCAR_normal_elements(dataframe, destination_directory, filename, prefix = None):
    for index in range(dataframe["geometry"].size):
        # Generate the new filename
        if prefix == None:
            new_filename = f"{int(dataframe['geometry'][index])}_{int(dataframe['path'][index])}_{filename}"
        else:
            new_filename = f"{int(dataframe['geometry'][index])}_{int(dataframe['path'][index])}_{filename}_{prefix}"


        # Get the source file path and destination file path
        destination_path = os.path.join(destination_directory, new_filename)
        
        # # Define the pattern to search for
        # pattern = '  Li_sv_GW/24a6a  P_GW/715c28f22  S_GW/357db9cfb  Cl_GW/3ef3b316\n              24               4              20               4'

        # # Define the replacement string
        # replacement = '   Li   P    S    Cl\n'

        # Read CONTCAR file
        with open(destination_path, 'r') as contcar_file:
            contcar_lines = contcar_file.readlines()
        
        contcar_lines[5] = "   Li   P    S    Cl\n"
        contcar_lines[6] = "    24     4    20     4\n"

        # # Find the number of configurations
        # # occurrences = int(contcar_lines[1])
        # occurrences = contcar_lines.count(contcar_lines[0])

        # Iterate through each line and replace if the pattern is found
        # for i in range(len(contcar_lines)):
        #     if pattern in contcar_lines[i]:
        #         contcar_lines[i] = replacement

        # # Print the modified lines
        # for line in contcar_lines:
        #     print(line.strip())  # .strip() is used to remove leading/trailing whitespaces

        # # Loop through each configuration
        # for occurrence in range(occurrences):
        #     # Define the starting and ending lines for each configuration
        #     # start_line = 8 + occurrence * (3 + sum([int(x) for x in contcar_lines[6].split()]))
        #     # end_line = start_line + 3 + sum([int(x) for x in contcar_lines[6].split()])
        #     start_line = (occurrence * line_length)
        #     end_line = start_line + line_length
        #     print(f"start_line: {start_line}, end_line: {end_line}")

        #     # Extract configuration lines
        #     config_lines = contcar_lines[start_line:end_line]

        #     new_dir = f"{dir_XDATCAR}/{occurrence}"
        #     os.makedirs(new_dir, exist_ok=True)

        # Create a new CONTCAR file for each configuration
        with open(destination_path, 'w') as contcar_file:
            contcar_file.writelines(contcar_lines)


def get_positive_lessthan1_poscarcontcar(dataframe, destination_directory, poscarcontcar_line_nr_start, poscarcontcar_line_nr_end, poscarcontcar_columns_type2, file_type, var_name_in, var_name_out, n_decimal):
    col_subdir_positive_file = f"subdir_positive_{file_type}"
    
    dataframe[col_subdir_positive_file] = None
    
    for idx in range(dataframe["geometry"].size):
        if var_name_in == None:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{file_type}"
        else:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{file_type}_{var_name_in}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)

        with open(filename_to_transform_path, 'r') as file:
            lines = file.readlines()
        data = lines[poscarcontcar_line_nr_start:poscarcontcar_line_nr_end]

        # Split each string by space and create the DataFrame
        df = pd.DataFrame([string.strip().split() for string in data])

        # Optional: Rename the columns
        df.columns = poscarcontcar_columns_type2

        df_positive_val = df[['coord_x', 'coord_y', 'coord_z']]
        for idx_a, coord_x in enumerate(df_positive_val['coord_x']):
            while float(coord_x) < 0.0:
                coord_x = float(coord_x) + 1.0
            while float(coord_x) > 1:
                coord_x = float(coord_x) - 1
            df_positive_val['coord_x'][idx_a] = '{:.{width}f}'.format(float(coord_x), width=n_decimal)

        for idx_a, coord_y in enumerate(df_positive_val['coord_y']):
            while float(coord_y) < 0.0:
                coord_y = float(coord_y) + 1.0
            while float(coord_y) > 1:
                coord_y = float(coord_y) - 1
            df_positive_val['coord_y'][idx_a] = '{:.{width}f}'.format(float(coord_y), width=n_decimal)

        for idx_a, coord_z in enumerate(df_positive_val['coord_z']):
            while float(coord_z) < 0.0:
                coord_z = float(coord_z) + 1.0
            while float(coord_z) > 1:
                coord_z = float(coord_z) - 1
            df_positive_val['coord_z'][idx_a] = '{:.{width}f}'.format(float(coord_z), width=n_decimal)

        row_list = df_positive_val.to_string(index=False, header=False).split('\n')
        row_list_space = ['  '.join(string.split()) for string in row_list] # 2 spaces of distance
        row_list_w_beginning = ['  ' + row for row in row_list_space]       # 2 spaces in the beginning
        absolute_correct_list = '\n'.join(row_list_w_beginning).splitlines()        

        line_append_list = []
        for idx_c, line in enumerate(absolute_correct_list):
            line_new_line = str(line) + '\n'
            line_append_list.append(line_new_line)

        file_list = lines[:poscarcontcar_line_nr_start] + line_append_list

        poscarcontcar_filename_positive = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_CONTCAR_{var_name_out}"
        destination_path = os.path.join(destination_directory, poscarcontcar_filename_positive)
        
        with open(destination_path, "w") as poscarcontcar_positive_file:
            for item in file_list:
                poscarcontcar_positive_file.writelines(item)

        dataframe[col_subdir_positive_file][idx] = destination_path


def get_coor_dict_structure(structure):
    coor_origin_Li_init = []; coor_origin_P_init = []; coor_origin_S_init = []; coor_origin_Cl_init = []
    coor_structure_init_dict = {}
    
    for idx, coor in enumerate(structure):
        if coor.species_string == "Li":
            coor_origin_Li_init.append(coor.frac_coords) 
        if coor.species_string == "P":
            coor_origin_P_init.append(coor.frac_coords) 
        if coor.species_string == "S":
            coor_origin_S_init.append(coor.frac_coords) 
        if coor.species_string == "Cl":
            coor_origin_Cl_init.append(coor.frac_coords) 
        
    coor_structure_init_dict["Li"] = coor_origin_Li_init
    coor_structure_init_dict["P"] = coor_origin_P_init
    coor_structure_init_dict["S"] = coor_origin_S_init
    coor_structure_init_dict["Cl"] = coor_origin_Cl_init

    return coor_structure_init_dict


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def get_coor_structure24_dict_iterated(dataframe, mapping):
    col_coor_structure_init_dict = "coor_structure_init_dict"

    dataframe[col_coor_structure_init_dict] = None

    for idx in range(dataframe["geometry"].size):
        print(f"idx: {idx}")
        coor_origin_Li_init = []; coor_origin_P_init = []; coor_origin_S_init = []; coor_origin_Cl_init = []
        coor_structure_init_dict = {}

        if mapping == "False":
            new_structure = Structure.from_file(dataframe['subdir_positive_CONTCAR'][idx]) # use this instead if no mapping is done
        else:
            new_structure = Structure.from_file(dataframe['subdir_orientated_positive_poscar'][idx]) # or we use this
            # new_structure = Structure.from_file(dataframe['subdir_orientated_positive'][idx])

        for idx_24, coor24 in enumerate(new_structure):
            if coor24.species_string == "Li":
                coor_origin_Li_init.append(coor24.frac_coords) 
            if coor24.species_string == "P":
                coor_origin_P_init.append(coor24.frac_coords)
            if coor24.species_string == "S":
                coor_origin_S_init.append(coor24.frac_coords)  
            if coor24.species_string == "Cl":
                coor_origin_Cl_init.append(coor24.frac_coords) 
            
        coor_structure_init_dict["Li"] = coor_origin_Li_init
        coor_structure_init_dict["P"] = coor_origin_P_init
        coor_structure_init_dict["S"] = coor_origin_S_init
        coor_structure_init_dict["Cl"] = coor_origin_Cl_init
    
        dataframe.at[idx, col_coor_structure_init_dict] = coor_structure_init_dict


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


# def eucledian_distance(coor1, coor2):
#     distance = math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(coor1, coor2)))
#     return distance


def apply_pbc(value):
    if abs(value) > 0.5:
        return 1 - abs(value)
    return value


def mic_eucledian_distance(coor1, coor2):
    x_coor1, y_coor1, z_coor1 = coor1
    x_coor2, y_coor2, z_coor2 = coor2
    
    delta_x = x_coor1 - x_coor2
    delta_y = y_coor1 - y_coor2
    delta_z = z_coor1 - z_coor2

    distance = math.sqrt(sum([(apply_pbc(delta_x))**2, (apply_pbc(delta_y))**2, (apply_pbc(delta_z))**2]))
    # distance = math.sqrt(sum([apply_pbc(delta_x)**2, apply_pbc(delta_y)**2, apply_pbc(delta_z)]**2))
    # [apply_pbc(delta_x), apply_pbc(delta_y), apply_pbc(delta_z)]
    # delta_coor = ((x1 - x2) for x1, x2 in zip(coor1, coor2))
    # distance = math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(coor1, coor2)))
    return distance


def apply_pbc_cartesian(value, length):
    # angle is ignored
    if abs(value) > 0.5 * length:
        return length - abs(value)
    return value


def mic_eucledian_distance_cartesian(coor1, coor2, a, b, c):
    x_coor1, y_coor1, z_coor1 = coor1
    x_coor2, y_coor2, z_coor2 = coor2
    
    delta_x = x_coor1 - x_coor2
    delta_y = y_coor1 - y_coor2
    delta_z = z_coor1 - z_coor2

    distance = math.sqrt(sum([(apply_pbc_cartesian(delta_x, a))**2, (apply_pbc_cartesian(delta_y, b))**2, (apply_pbc_cartesian(delta_z, c))**2]))
    return distance


# def get_duplicate_values_in_dict(dict):
#     # dict is atom_mapping_el

#     seen_values = set()
#     duplicate_values = []

#     for value in dict.values():
#         if value in seen_values:
#             duplicate_values.append(value)
#         else:
#             seen_values.add(value)
    
#     return duplicate_values


def get_duplicate_closest24_w_data(dict):
    duplicate_closest24 = {}
    for coor120, values in dict.items():
        for entry in values:
            closest24 = entry["closest24"]
            dist = entry["dist"]

        if closest24 in duplicate_closest24:
            duplicate_closest24[closest24].append({"coor120": coor120, "dist": dist})
        else:
            duplicate_closest24[closest24] = [{"coor120": coor120, "dist": dist}]

    duplicate_closest24_w_data = {}
    for closest24, coor120s_dists in duplicate_closest24.items():
        if len(coor120s_dists) > 1:
            duplicate_closest24_w_data[f"Duplicate closest24: {closest24}"] = [{"coor120s and dists": coor120s_dists}]

    return duplicate_closest24_w_data


def get_atom_mapping_el_w_dist_closestduplicate(dict):
    filtered_data = {}
    for coor120, values in dict.items():
        for entry in values:
            closest24 = entry["closest24"]
            dist = entry["dist"]
            
        if closest24 in filtered_data:
            if dist < filtered_data[closest24]["dist"]:
                filtered_data[closest24] = {"coor120": coor120, "dist": dist}
        else:
            filtered_data[closest24] = {"coor120": coor120, "dist": dist}

    atom_mapping_el_w_dist_closestduplicate = {entry["coor120"]: {"closest24": key, "dist": entry["dist"]} for key, entry in filtered_data.items()}
    return atom_mapping_el_w_dist_closestduplicate


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def merge_dictionaries(dict1, dict2):
    merged_dict = defaultdict(list)

    for d in (dict1, dict2): # you can list as many input dicts as you want here
        for key, value in d.items():
            merged_dict[key].append(value)
    
    return merged_dict


def check_duplicate_values(dictionary):
    # seen_values = set()
    # duplicate_values = set()
    seen_values = []
    duplicate_values = []

    for value in dictionary.values():
        # value_tuple = tuple(value) 
        # if value_tuple in seen_values:
        if value in seen_values:
            duplicate_values.append(value)
            # duplicate_values.add(value_tuple)
        else:
            seen_values.append(value)
            # seen_values.add(value_tuple)

    return duplicate_values


def get_flag_map_weirdos_el(dataframe, coor_structure_init_dict, el, max_mapping_radius):
    coor_origin120_el_init = coor_structure_init_dict[el]
    col_coor_structure_init_dict = "coor_structure_init_dict"

    # col_atom_mapping_el = f"atom_mapping_{el}"
    # col_atom_mapping_el_w_dist = f"atom_mapping_{el}_w_dist"
    # col_coor_weirdos_el_dict = f"coor_weirdos_{el}_dict"            # just added
    # col_coor_reduced120_el = f"coor_reduced120_{el}"
    # col_sum_mapped_el = f"sum_mapped_{el}"
    # col_sum_sanitycheck_el = f"sum_sanitycheck_{el}"
    col_flag_el = f"flag_{el}"
    col_coor_weirdos_el = f"coor_weirdos_{el}"
    col_sum_weirdos_el = f"sum_weirdos_{el}"
    col_duplicate_closest24_w_data_el = f"duplicate_closest24_w_data_{el}"
    col_coor_reduced120_el_closestduplicate = f"coor_reduced120_{el}_closestduplicate"
    col_sum_mapped_el_closestduplicate = f"sum_mapped_{el}_closestduplicate"
    col_sum_sanitycheck_el_closestduplicate = f"sum_sanitycheck_{el}_closestduplicate"
    col_atom_mapping_el_closestduplicate = f"atom_mapping_{el}_closestduplicate"
    col_atom_mapping_el_w_dist_closestduplicate = f"atom_mapping_{el}_w_dist_closestduplicate"

    # dataframe[col_atom_mapping_el] = [{} for _ in range(len(dataframe.index))] 
    # dataframe[col_atom_mapping_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_weirdos_el_dict] = [{el: []} for _ in range(len(dataframe.index))]                       # just added
    # dataframe[col_coor_reduced120_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_sum_mapped_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_sum_sanitycheck_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_flag_el] = "False"
    dataframe[col_coor_weirdos_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_duplicate_closest24_w_data_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reduced120_el_closestduplicate] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_atom_mapping_el_closestduplicate] = [{} for _ in range(len(dataframe.index))] 
    dataframe[col_atom_mapping_el_w_dist_closestduplicate] = [{} for _ in range(len(dataframe.index))]

    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist = {} 
        atom_mapping_el_closestduplicate = {} 
        coor_weirdos_el = []
        # coor_weirdos_el_dict = {}

        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_init_dict][el]     # dataframe['subdir_orientated_positive']
                                                                                        # or without orientation
                                                                                        # dataframe['subdir_CONTCAR']
        
        coor_reduced120_el = coor_origin120_el_init.copy()
        coor_weirdos_el = coor_origin24_el_init.copy()    

        for idx120, coor120 in enumerate(coor_origin120_el_init):        
            counter = 0
            atom_mapping_w_dist_dict = {}
            atom_mapping_el_w_dist_closestduplicate = {}
            distance_prev = float("inf")
            closest24 = None

            for idx24, coor24 in enumerate(coor_origin24_el_init):
                distance = mic_eucledian_distance(coor120, coor24)

                if distance < max_mapping_radius:
                    counter = counter + 1
                    if distance < distance_prev:
                        distance_prev = distance
                        closest24 = coor24
            
                if counter > 1:
                    dataframe.at[idx, col_flag_el] = "True"

                    # if tuple(coor120) in atom_mapping_el_w_dist:
                    #     atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_w_dist_dict)
                    # else:
                    #     atom_mapping_el_w_dist.setdefault(tuple(coor120),[atom_mapping_w_dist_dict])
                    
            
            # if closest24 is not None:
            #     if tuple(coor120) in atom_mapping_el:
            #         atom_mapping_el[tuple(coor120)].append(closest24)
            #     else:
            #         atom_mapping_el[tuple(coor120)] = tuple(closest24)

            if closest24 is not None:
                atom_mapping_w_dist_dict['closest24'] = tuple(closest24)
                atom_mapping_w_dist_dict['dist'] = distance_prev
                
                if tuple(coor120) in atom_mapping_el_w_dist:
                    new_entry = atom_mapping_el_w_dist[tuple(coor120)].copy()
                    new_entry.append(atom_mapping_w_dist_dict)
                    atom_mapping_el_w_dist[tuple(coor120)] = new_entry
                else:
                    atom_mapping_el_w_dist[tuple(coor120)] = [atom_mapping_w_dist_dict.copy()]
 
                coor_weirdos_el = [arr for arr in coor_weirdos_el if not np.array_equal(arr, closest24)]

            if counter == 0:
                coor_reduced120_el = [arr for arr in coor_reduced120_el if not np.array_equal(arr, coor120)]

        duplicate_closest24_w_data = get_duplicate_closest24_w_data(atom_mapping_el_w_dist)

        # get the new reduced coor120, based on the closest distance if it has multiple close coor120 within the radius
        if len(duplicate_closest24_w_data) > 0:
            atom_mapping_el_w_dist_closestduplicate = get_atom_mapping_el_w_dist_closestduplicate(atom_mapping_el_w_dist)
            coor_reduced120_el_closestduplicate = [list(key) for key in atom_mapping_el_w_dist_closestduplicate.keys()]
        else:
            atom_mapping_el_w_dist_closestduplicate = atom_mapping_el_w_dist.copy()
            coor_reduced120_el_closestduplicate = coor_reduced120_el.copy()
        
        # if atom_mapping_el_w_dist_closestduplicate != {}:
        #    for key, values in atom_mapping_el_w_dist_closestduplicate.items():
        #        # atom_mapping_el_closestduplicate[key] = [entry['closest24'] for entry in values]
        #        atom_mapping_el_closestduplicate[key] = values['closest24']
        
        if atom_mapping_el_w_dist_closestduplicate != {}:
            for key, values_list in atom_mapping_el_w_dist_closestduplicate.items():
                closest24_values = []

                if isinstance(values_list, list):
                    # If it's a list, iterate over the dictionaries in the list
                    for entry in values_list:
                        closest24_values.append(entry['closest24'])
                elif isinstance(values_list, dict):
                    # If it's a dictionary, directly access 'closest24'
                    closest24_values.append(values_list['closest24'])

                atom_mapping_el_closestduplicate[key] = closest24_values

        # coor_weirdos_el_dict[el] = coor_weirdos_el

        sum_weirdos_el = len(coor_weirdos_el)
        # sum_mapped_el = len(coor_reduced120_el)
        sum_mapped_el_closestduplicate = len(coor_reduced120_el_closestduplicate)

        # dataframe.at[idx, col_atom_mapping_el] = atom_mapping_el
        # dataframe.at[idx, col_atom_mapping_el_w_dist] = atom_mapping_el_w_dist
        # dataframe.at[idx, col_coor_weirdos_el_dict] = coor_weirdos_el_dict          # just added
        # dataframe.at[idx, col_coor_reduced120_el] = np.array(coor_reduced120_el)
        # dataframe.at[idx, col_sum_mapped_el] = sum_mapped_el
        # dataframe.at[idx, col_sum_sanitycheck_el] = sum_weirdos_el + sum_mapped_el
        dataframe.at[idx, col_coor_weirdos_el] = coor_weirdos_el
        dataframe.at[idx, col_sum_weirdos_el] = sum_weirdos_el
        dataframe.at[idx, col_duplicate_closest24_w_data_el] = duplicate_closest24_w_data
        dataframe.at[idx, col_coor_reduced120_el_closestduplicate] = np.array(coor_reduced120_el_closestduplicate)
        dataframe.at[idx, col_sum_mapped_el_closestduplicate] = sum_mapped_el_closestduplicate
        dataframe.at[idx, col_sum_sanitycheck_el_closestduplicate] = sum_mapped_el_closestduplicate + sum_weirdos_el
        dataframe.at[idx, col_atom_mapping_el_closestduplicate] = atom_mapping_el_closestduplicate
        dataframe.at[idx, col_atom_mapping_el_w_dist_closestduplicate] = atom_mapping_el_w_dist_closestduplicate


def get_flag_map_weirdos_48htype2_el(dataframe, coor_structure_init_dict, el, max_mapping_radius_48htype2, activate_radius):
    coor_origin120_el_init = coor_structure_init_dict[el]         
    if activate_radius == 3:              
        col_coor_structure_48htype2_init_el = f"coor_weirdos_48htype1_48htype2_{el}"               # here is the difference
    elif activate_radius == 2:
        col_coor_structure_48htype2_init_el = f"coor_weirdos_{el}"               # here is the difference
    else:
        print("activate_radius is wrongly given")

    # col_atom_mapping_48htype2_el = f"atom_mapping_48htype2_{el}"
    # col_atom_mapping_48htype2_el_w_dist = f"atom_mapping_48htype2_{el}_w_dist"
    # col_coor_reduced120_48htype2_el = f"coor_reduced120_48htype2_{el}"
    # col_sum_mapped_48htype2_el = f"sum_mapped_48htype2_{el}"
    # col_sum_sanitycheck_48htype2_el = f"sum_sanitycheck_48htype2_{el}"
    col_flag_48htype2_el = f"flag_48htype2_{el}"
    col_coor_weirdos_48htype2_el = f"coor_weirdos_48htype2_{el}"
    col_sum_weirdos_48htype2_el = f"sum_weirdos_48htype2_{el}"
    col_duplicate_closest24_w_data_48htype2_el = f"duplicate_closest24_w_data_48htype2_{el}"
    col_coor_reduced120_48htype2_el_closestduplicate = f"coor_reduced120_48htype2_{el}_closestduplicate"
    col_sum_mapped_48htype2_el_closestduplicate = f"sum_mapped_48htype2_{el}_closestduplicate"
    col_sum_sanitycheck_48htype2_el_closestduplicate = f"sum_sanitycheck_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htype2_el_closestduplicate = f"atom_mapping_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htype2_el_w_dist_closestduplicate = f"atom_mapping_48htype2_{el}_w_dist_closestduplicate"

    # dataframe[col_atom_mapping_48htype2_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_48htype2_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_reduced120_48htype2_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_sum_mapped_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_sum_sanitycheck_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_flag_48htype2_el] = "False"
    dataframe[col_coor_weirdos_48htype2_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_weirdos_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_duplicate_closest24_w_data_48htype2_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reduced120_48htype2_el_closestduplicate] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_48htype2_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_48htype2_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_atom_mapping_48htype2_el_closestduplicate] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_atom_mapping_48htype2_el_w_dist_closestduplicate] = [{} for _ in range(len(dataframe.index))]

    coor_li48htype1_ref = coor_origin120_el_init[0:48]
    coor_li48htype2_ref = coor_origin120_el_init[48:96]
    coor_li24g_ref = coor_origin120_el_init[96:120]

    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist = {} 
        atom_mapping_el_closestduplicate = {} 
        atom_mapping_el_w_dist_closestduplicate = {}
        coor_weirdos_el = []

        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_48htype2_init_el]#[el]             # dataframe['subdir_orientated_positive']
                                                                                        # or without orientation
                                                                                        # dataframe['subdir_CONTCAR']
        
        if len(coor_origin24_el_init) > 0:
            coor_reduced120_el = coor_li48htype2_ref.copy()
            coor_weirdos_el = coor_origin24_el_init.copy()    

            for idx120, coor120 in enumerate(coor_li48htype2_ref):        
                counter = 0
                atom_mapping_w_dist_dict = {}
                distance_prev = float("inf")
                closest24 = None

                for idx24, coor24 in enumerate(coor_origin24_el_init):
                    distance = mic_eucledian_distance(coor120, coor24)
                    
                    if distance < max_mapping_radius_48htype2:
                        counter = counter + 1
                        if distance < distance_prev:
                            distance_prev = distance
                            closest24 = coor24
                
                    if counter > 1:
                        dataframe.at[idx, col_flag_48htype2_el] = "True"
                    
                if closest24 is not None:
                    atom_mapping_w_dist_dict['closest24'] = tuple(closest24)
                    atom_mapping_w_dist_dict['dist'] = distance_prev

                    # if tuple(coor120) in atom_mapping_el_w_dist:
                    #     atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_w_dist_dict)
                    # else:
                    #     atom_mapping_el_w_dist[tuple(coor120)] = atom_mapping_w_dist_dict
                        
                    # if tuple(coor120) in atom_mapping_el:
                    #     atom_mapping_el[tuple(coor120)].append(closest24)
                    # else:
                    #     atom_mapping_el[tuple(coor120)] = tuple(closest24)

                    if tuple(coor120) in atom_mapping_el_w_dist:
                        new_entry = atom_mapping_el_w_dist[tuple(coor120)].copy()
                        new_entry.append(atom_mapping_w_dist_dict)
                        atom_mapping_el_w_dist[tuple(coor120)] = new_entry
                    else:
                        atom_mapping_el_w_dist[tuple(coor120)] = [atom_mapping_w_dist_dict.copy()]

                    coor_weirdos_el = [arr for arr in coor_weirdos_el if not np.array_equal(arr, closest24)]

                if counter == 0:
                    coor_reduced120_el = [arr for arr in coor_reduced120_el if not np.array_equal(arr, coor120)]

            duplicate_closest24_w_data = get_duplicate_closest24_w_data(atom_mapping_el_w_dist)

            # get atom_mapping_el_closestduplicate
            # if duplicate_closest24_w_data != {}:
            if len(duplicate_closest24_w_data) > 0:
                atom_mapping_el_w_dist_closestduplicate = get_atom_mapping_el_w_dist_closestduplicate(atom_mapping_el_w_dist)
                coor_reduced120_el_closestduplicate = [list(key) for key in atom_mapping_el_w_dist_closestduplicate.keys()]
            else:
                atom_mapping_el_w_dist_closestduplicate = atom_mapping_el_w_dist.copy()
                coor_reduced120_el_closestduplicate = coor_reduced120_el.copy()

            
            # if atom_mapping_el_w_dist_closestduplicate != {}:
            #    for key, value in atom_mapping_el_w_dist_closestduplicate.items():
            #        atom_mapping_el_closestduplicate[key] = [entry['closest24'] for entry in value]

            if atom_mapping_el_w_dist_closestduplicate != {}:
                for key, values_list in atom_mapping_el_w_dist_closestduplicate.items():
                    closest24_values = []

                    if isinstance(values_list, list):
                        # If it's a list, iterate over the dictionaries in the list
                        for entry in values_list:
                            closest24_values.append(entry['closest24'])
                    elif isinstance(values_list, dict):
                        # If it's a dictionary, directly access 'closest24'
                        closest24_values.append(values_list['closest24'])

                    atom_mapping_el_closestduplicate[key] = closest24_values

            sum_weirdos_el = len(coor_weirdos_el)
            # sum_mapped_el = len(coor_reduced120_el)
            sum_mapped_el_closestduplicate = len(coor_reduced120_el_closestduplicate)

            # dataframe.at[idx, col_atom_mapping_48htype2_el] = atom_mapping_el
            # dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist] = atom_mapping_el_w_dist
            # dataframe.at[idx, col_coor_reduced120_48htype2_el] = np.array(coor_reduced120_el)
            # dataframe.at[idx, col_sum_mapped_48htype2_el] = sum_mapped_el
            # dataframe.at[idx, col_sum_sanitycheck_48htype2_el] = sum_mapped_el + sum_weirdos_el 
            dataframe.at[idx, col_coor_weirdos_48htype2_el] = coor_weirdos_el
            dataframe.at[idx, col_sum_weirdos_48htype2_el] = sum_weirdos_el
            dataframe.at[idx, col_duplicate_closest24_w_data_48htype2_el] = duplicate_closest24_w_data
            dataframe.at[idx, col_coor_reduced120_48htype2_el_closestduplicate] = np.array(coor_reduced120_el_closestduplicate)
            dataframe.at[idx, col_sum_mapped_48htype2_el_closestduplicate] = sum_mapped_el_closestduplicate
            dataframe.at[idx, col_sum_sanitycheck_48htype2_el_closestduplicate] = sum_mapped_el_closestduplicate + sum_weirdos_el
            dataframe.at[idx, col_atom_mapping_48htype2_el_closestduplicate] = atom_mapping_el_closestduplicate
            dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist_closestduplicate] = atom_mapping_el_w_dist_closestduplicate

        # elif coor_origin24_el_init == []:
        #     dataframe.at[idx, col_atom_mapping_48htype2_el] = {} 
        #     dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist] = {}
        #     dataframe.at[idx, col_sum_weirdos_48htype2_el] = 0


def get_flag_map_weirdos_48htype1_48htype2_el(dataframe, coor_structure_init_dict, el, max_mapping_radius_48htype1_48htype2):
    coor_origin120_el_init = coor_structure_init_dict[el]                       
    col_coor_structure_48htype1_48htype2_init_el = f"coor_weirdos_{el}"               # here is the difference

    # col_atom_mapping_48htype1_48htype2_el = f"atom_mapping_48htype1_48htype2_{el}"
    # col_atom_mapping_48htype1_48htype2_el_w_dist = f"atom_mapping_48htype1_48htype2_{el}_w_dist"
    # col_coor_weirdos_48htype1_48htype2_el_dict = f"coor_weirdos_48htype1_48htype2_{el}_dict"            # just added
    # col_coor_reduced120_48htype1_48htype2_el = f"coor_reduced120_48htype1_48htype2_{el}"
    # col_sum_mapped_48htype1_48htype2_el = f"sum_mapped_48htype1_48htype2_{el}"
    # col_sum_sanitycheck_48htype1_48htype2_el = f"sum_sanitycheck_48htype1_48htype2_{el}"
    col_flag_48htype1_48htype2_el = f"flag_48htype1_48htype2_{el}"
    col_coor_weirdos_48htype1_48htype2_el = f"coor_weirdos_48htype1_48htype2_{el}"
    col_sum_weirdos_48htype1_48htype2_el = f"sum_weirdos_48htype1_48htype2_{el}"
    col_duplicate_closest24_w_data_48htype1_48htype2_el = f"duplicate_closest24_w_data_48htype1_48htype2_{el}"
    col_coor_reduced120_48htype1_48htype2_el_closestduplicate = f"coor_reduced120_48htype1_48htype2_{el}_closestduplicate"
    col_sum_mapped_48htype1_48htype2_el_closestduplicate = f"sum_mapped_48htype1_48htype2_{el}_closestduplicate"
    col_sum_sanitycheck_48htype1_48htype2_el_closestduplicate = f"sum_sanitycheck_48htype1_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htype1_48htype2_el_closestduplicate = f"atom_mapping_48htype1_48htype2_{el}_closestduplicate"

    # dataframe[col_atom_mapping_48htype1_48htype2_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_48htype1_48htype2_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_reduced120_48htype1_48htype2_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_coor_weirdos_48htype1_48htype2_el_dict] = [{el: []} for _ in range(len(dataframe.index))]
    # dataframe[col_sum_mapped_48htype1_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_sum_sanitycheck_48htype1_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_flag_48htype1_48htype2_el] = "False"
    dataframe[col_coor_weirdos_48htype1_48htype2_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_weirdos_48htype1_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_duplicate_closest24_w_data_48htype1_48htype2_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reduced120_48htype1_48htype2_el_closestduplicate] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_48htype1_48htype2_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_48htype1_48htype2_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_atom_mapping_48htype1_48htype2_el_closestduplicate] = [{} for _ in range(len(dataframe.index))] 

    coor_li48htype1_li48htype2_ref = coor_origin120_el_init[0:96]
    coor_li24g_ref = coor_origin120_el_init[96:120]

    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist = {} 
        atom_mapping_el_closestduplicate = {} 
        atom_mapping_el_w_dist_closestduplicate = {}
        coor_weirdos_el = []
        # coor_weirdos_el_dict = {}

        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_48htype1_48htype2_init_el]#[el]             # dataframe['subdir_orientated_positive']
                                                                                        # or without orientation
                                                                                        # dataframe['subdir_CONTCAR']
        
        if len(coor_origin24_el_init) > 0: # need this for the mic_eucledian_distance()
            coor_reduced120_el = coor_li48htype1_li48htype2_ref.copy()
            coor_weirdos_el = coor_origin24_el_init.copy()    

            for idx120, coor120 in enumerate(coor_li48htype1_li48htype2_ref):        
                counter = 0
                atom_mapping_w_dist_dict = {}
                distance_prev = float("inf")
                closest24 = None

                for idx24, coor24 in enumerate(coor_origin24_el_init):
                    distance = mic_eucledian_distance(coor120, coor24)
                    
                    if distance < max_mapping_radius_48htype1_48htype2:
                        counter = counter + 1
                        if distance < distance_prev:
                            distance_prev = distance
                            closest24 = coor24
                
                    if counter > 1:
                        dataframe.at[idx, col_flag_48htype1_48htype2_el] = "True"
                    
                if closest24 is not None:
                    atom_mapping_w_dist_dict['closest24'] = tuple(closest24)
                    atom_mapping_w_dist_dict['dist'] = distance_prev

                    # if tuple(coor120) in atom_mapping_el_w_dist:
                    #     atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_w_dist_dict)
                    # else:
                    #     atom_mapping_el_w_dist[tuple(coor120)] = atom_mapping_w_dist_dict
                        
                    # if tuple(coor120) in atom_mapping_el:
                    #     atom_mapping_el[tuple(coor120)].append(closest24)
                    # else:
                    #     atom_mapping_el[tuple(coor120)] = tuple(closest24)

                    if tuple(coor120) in atom_mapping_el_w_dist:
                        new_entry = atom_mapping_el_w_dist[tuple(coor120)].copy()
                        new_entry.append(atom_mapping_w_dist_dict)
                        atom_mapping_el_w_dist[tuple(coor120)] = new_entry
                    else:
                        atom_mapping_el_w_dist[tuple(coor120)] = [atom_mapping_w_dist_dict.copy()]
    
                    coor_weirdos_el = [arr for arr in coor_weirdos_el if not np.array_equal(arr, closest24)]

                if counter == 0:
                    coor_reduced120_el = [arr for arr in coor_reduced120_el if not np.array_equal(arr, coor120)]

            duplicate_closest24_w_data = get_duplicate_closest24_w_data(atom_mapping_el_w_dist)

            # get atom_mapping_el_closestduplicate
            # if duplicate_closest24_w_data != {}:
            if len(duplicate_closest24_w_data) > 0:
                atom_mapping_el_w_dist_closestduplicate = get_atom_mapping_el_w_dist_closestduplicate(atom_mapping_el_w_dist)
                coor_reduced120_el_closestduplicate = [list(key) for key in atom_mapping_el_w_dist_closestduplicate.keys()]
            else:
                atom_mapping_el_w_dist_closestduplicate = atom_mapping_el_w_dist.copy()
                coor_reduced120_el_closestduplicate = coor_reduced120_el.copy()

            # if atom_mapping_el_w_dist_closestduplicate != {}:
            #    for key, values in atom_mapping_el_w_dist_closestduplicate.items():
            #        # atom_mapping_el_closestduplicate[key] = [entry['closest24'] for entry in values]
            #        atom_mapping_el_closestduplicate[key] = values['closest24']

            if atom_mapping_el_w_dist_closestduplicate != {}:
                for key, values_list in atom_mapping_el_w_dist_closestduplicate.items():
                    closest24_values = []

                    if isinstance(values_list, list):
                        # If it's a list, iterate over the dictionaries in the list
                        for entry in values_list:
                            closest24_values.append(entry['closest24'])
                    elif isinstance(values_list, dict):
                        # If it's a dictionary, directly access 'closest24'
                        closest24_values.append(values_list['closest24'])

                    atom_mapping_el_closestduplicate[key] = closest24_values

            # coor_weirdos_el_dict[el] = coor_weirdos_el

            sum_weirdos_el = len(coor_weirdos_el)
            # sum_mapped_el = len(coor_reduced120_el)
            sum_mapped_el_closestduplicate = len(coor_reduced120_el_closestduplicate)

            # dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el] = atom_mapping_el
            # dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_w_dist] = atom_mapping_el_w_dist
            # dataframe.at[idx, col_coor_weirdos_48htype1_48htype2_el_dict] = coor_weirdos_el_dict
            # dataframe.at[idx, col_coor_reduced120_48htype1_48htype2_el] = np.array(coor_reduced120_el)
            # dataframe.at[idx, col_sum_mapped_48htype1_48htype2_el] = sum_mapped_el
            # dataframe.at[idx, col_sum_sanitycheck_48htype1_48htype2_el] = sum_mapped_el + sum_weirdos_el 
            dataframe.at[idx, col_coor_weirdos_48htype1_48htype2_el] = coor_weirdos_el
            dataframe.at[idx, col_sum_weirdos_48htype1_48htype2_el] = sum_weirdos_el
            dataframe.at[idx, col_duplicate_closest24_w_data_48htype1_48htype2_el] = duplicate_closest24_w_data
            dataframe.at[idx, col_coor_reduced120_48htype1_48htype2_el_closestduplicate] = np.array(coor_reduced120_el_closestduplicate)
            dataframe.at[idx, col_sum_mapped_48htype1_48htype2_el_closestduplicate] = sum_mapped_el_closestduplicate
            dataframe.at[idx, col_sum_sanitycheck_48htype1_48htype2_el_closestduplicate] = sum_mapped_el_closestduplicate + sum_weirdos_el
            dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_closestduplicate] = atom_mapping_el_closestduplicate

        # elif coor_origin24_el_init == []:
        #     dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el] = {} 
        #     dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_w_dist] = {}
        #     dataframe.at[idx, col_sum_weirdos_48htype1_48htype2_el] = 0


def get_flag_map_weirdos_48htypesmerged_level1_el(dataframe, el):
    # # col_flag_el = f"flag_{el}"
    # col_coor_weirdos_el = f"coor_weirdos_{el}"
    # col_coor_weirdos_el_dict = f"coor_weirdos_{el}_dict"            # just added
    # col_atom_mapping_el_w_dist = f"atom_mapping_{el}_w_dist"
    # col_coor_reduced120_el = f"coor_reduced120_{el}_closestduplicate"
    col_atom_mapping_el_closestduplicate = f"atom_mapping_{el}_closestduplicate"
    col_coor_reduced120_el_closestduplicate = f"coor_reduced120_{el}_closestduplicate"


    # # col_flag_48htype1_48htype2_el = f"flag_48htype1_48htype2_{el}"
    # col_atom_mapping_48htype1_48htype2_el_w_dist = f"atom_mapping_48htype1_48htype2_{el}_w_dist"
    # col_coor_reduced120_48htype1_48htype2_el = f"coor_reduced120_48htype1_48htype2_{el}_closestduplicate"
    col_sum_weirdos_48htype1_48htype2_el = f"sum_weirdos_48htype1_48htype2_{el}"
    col_atom_mapping_48htype1_48htype2_el_closestduplicate = f"atom_mapping_48htype1_48htype2_{el}_closestduplicate"
    col_coor_reduced120_48htype1_48htype2_el_closestduplicate = f"coor_reduced120_48htype1_48htype2_{el}_closestduplicate"


    col_flag_48htypesmerged_level1_el = f"flag_48htypesmerged_level1_{el}"
    col_atom_mapping_48htypesmerged_level1_el = f"atom_mapping_48htypesmerged_level1_{el}"
    # col_atom_mapping_48htypesmerged_level1_el_w_dist = f"atom_mapping_48htypesmerged_level1_{el}_w_dist"
    col_coor_reduced120_48htypesmerged_level1_el = f"coor_reduced120_48htypesmerged_level1_{el}"
    col_sum_mapped_48htypesmerged_level1_el = f"sum_mapped_48htypesmerged_level1_{el}"
    col_sum_sanitycheck_48htypesmerged_level1_el = f"sum_sanitycheck_48htypesmerged_level1_{el}"
    col_ndim_coor_reduced120_level1_el_closestduplicate = f"ndim_coor_reduced120_level1_{el}_closestduplicate"
    col_ndim_coor_reduced120_48htype2_level1_el_closestduplicate = f"ndim_coor_reduced120_48htype2_level1_{el}_closestduplicate"

    dataframe[col_flag_48htypesmerged_level1_el] = "False"
    dataframe[col_atom_mapping_48htypesmerged_level1_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_48htypesmerged_level1_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reduced120_48htypesmerged_level1_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_48htypesmerged_level1_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_48htypesmerged_level1_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_reduced120_level1_el_closestduplicate] = None
    dataframe[col_ndim_coor_reduced120_48htype2_level1_el_closestduplicate] = None

    for idx in range(dataframe["geometry"].size):
        # print(f"idx_48htypesmerged: {idx}")
        atom_mapping_el_closestduplicate = dataframe.at[idx, col_atom_mapping_el_closestduplicate]
        atom_mapping_48htype1_48htype2_el_closestduplicate = dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_closestduplicate]
        # atom_mapping_el_w_dist = dataframe.at[idx, col_atom_mapping_el_w_dist]
        # atom_mapping_48htype1_48htype2_el_w_dist = dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_w_dist]
        # coor_reduced120_el = dataframe.at[idx, col_coor_reduced120_el]
        # coor_reduced120_48htype1_48htype2_el = dataframe.at[idx, col_coor_reduced120_48htype1_48htype2_el]
        # _closestduplicate
        coor_reduced120_el_closestduplicate = dataframe.at[idx, col_coor_reduced120_el_closestduplicate]
        coor_reduced120_48htype1_48htype2_el_closestduplicate = dataframe.at[idx, col_coor_reduced120_48htype1_48htype2_el_closestduplicate]
        # # these are just copying
        # coor_weirdos_48htypesmerged_level1_el = dataframe.at[idx, col_coor_weirdos_48htype1_48htype2_el]
        sum_weirdos_48htypesmerged_level1_el = dataframe.at[idx, col_sum_weirdos_48htype1_48htype2_el]
        # duplicate_closest24_w_data_48htypesmerged = dataframe.at[idx, col_duplicate_closest24_w_data_48htype1_48htype2_el]
        
        # # _closestduplicate
        # coor_reduced120_48htypesmerged_level1_el_closestduplicate = dataframe.at[idx, col_coor_reduced120_48htype1_48htype2_el_closestduplicate]
        # coor_reduced120_48htypesmerged_level1_el = []

        atom_mapping_48htypesmerged_level1_el = merge_dictionaries(atom_mapping_el_closestduplicate, atom_mapping_48htype1_48htype2_el_closestduplicate)
        duplicate_coor24s = check_duplicate_values(atom_mapping_48htypesmerged_level1_el)
        if len(duplicate_coor24s) > 1:        
            dataframe.at[idx, col_flag_48htypesmerged_level1_el] = "True"        
        # for coor120 in atom_mapping_48htypesmerged_level1_el:
        #     values24 = atom_mapping_48htypesmerged_level1_el[coor120]   # get value for the key
        #     if len(values24) > 1:        
        #         dataframe.at[idx, col_flag_48htypesmerged_level1_el] = "True"
        
        # atom_mapping_48htypesmerged_level1_el_w_dist = merge_dictionaries(atom_mapping_el_w_dist, atom_mapping_48htype1_48htype2_el_w_dist)

        # # if coor_reduced120_48htype1_48htype2_el != None:
        # # if coor_reduced120_48htype1_48htype2_el.size > 0:
        # if coor_reduced120_48htype1_48htype2_el.ndim == 2:
        #     coor_reduced120_48htypesmerged_level1_el = np.concatenate((coor_reduced120_el, coor_reduced120_48htype1_48htype2_el), axis=0)
        # elif coor_reduced120_48htype1_48htype2_el.ndim == 1:
        #     coor_reduced120_48htypesmerged_level1_el = np.array(coor_reduced120_el.copy())
        # else:
        #     break

        ## we use _closestduplicate here since it's the corrected one wrt distance
        # if coor_reduced120_48htype1_48htype2_el_closestduplicate != None:
        # if coor_reduced120_48htype1_48htype2_el_closestduplicate.size > 0:
        # # # if coor_reduced120_48htype1_48htype2_el_closestduplicate.ndim == 2:
        # # #     if coor_reduced120_el_closestduplicate.ndim == 2:
        # # #         coor_reduced120_48htypesmerged_level1_el_closestduplicate = np.concatenate((coor_reduced120_el_closestduplicate, coor_reduced120_48htype1_48htype2_el_closestduplicate), axis=0)
        # # #     else:
        # # #         print(f"coor_reduced120_el_closestduplicate has no correct dimension at idx: {idx}, dimension: {coor_reduced120_el_closestduplicate.ndim}")
        # # #         pass
        # # #         # print(f"coor_reduced120_el_closestduplicate: \n {coor_reduced120_el_closestduplicate}")
        # # # elif coor_reduced120_48htype1_48htype2_el_closestduplicate.ndim == 1:
        # # #     coor_reduced120_48htypesmerged_level1_el_closestduplicate = np.array(coor_reduced120_el_closestduplicate.copy())
        # # # else:
        # # #     print(f"coor_reduced120_48htype1_48htype2_el_closestduplicate has no correct dimension at idx: {idx}, dimension: {coor_reduced120_48htypesmerged_level1_el_closestduplicate.ndim}")
        # # #     # break

        if coor_reduced120_48htype1_48htype2_el_closestduplicate.ndim == coor_reduced120_el_closestduplicate.ndim:
            coor_reduced120_48htypesmerged_level1_el = np.concatenate((coor_reduced120_el_closestduplicate, coor_reduced120_48htype1_48htype2_el_closestduplicate), axis=0)
            # else:
            #     print(f"coor_reduced120_el_closestduplicate has no correct dimension at idx: {idx}")
            #     continue
                # print(f"coor_reduced120_el_closestduplicate: \n {coor_reduced120_el_closestduplicate}")
        elif coor_reduced120_48htype1_48htype2_el_closestduplicate.ndim == 1:
            coor_reduced120_48htypesmerged_level1_el = np.array(coor_reduced120_el_closestduplicate.copy())
        elif coor_reduced120_el_closestduplicate.ndim == 1:
            coor_reduced120_48htypesmerged_level1_el = np.array(coor_reduced120_48htype1_48htype2_el_closestduplicate.copy())
        else:
            print(f"coor_reduced120_48htype1_48htype2_el_closestduplicate or coor_reduced120_el_closestduplicate has no correct dimension at idx: {idx}")
            # break

        # sum_mapped_48htypesmerged_level1_el = len(atom_mapping_48htypesmerged_level1_el)
        sum_mapped_48htypesmerged_level1_el = len(coor_reduced120_48htypesmerged_level1_el)

        ndim_coor_reduced120_el_closestduplicate = coor_reduced120_el_closestduplicate.ndim
        ndim_coor_reduced120_48htype1_48htype2_el_closestduplicate = coor_reduced120_48htype1_48htype2_el_closestduplicate.ndim

        dataframe.at[idx, col_atom_mapping_48htypesmerged_level1_el] = atom_mapping_48htypesmerged_level1_el
        # dataframe.at[idx, col_atom_mapping_48htypesmerged_level1_el_w_dist] = atom_mapping_48htypesmerged_level1_el_w_dist
        # dataframe.at[idx, col_coor_weirdos_48htypesmerged_level1_el] = coor_weirdos_48htypesmerged_level1_el        # these are just copying
        # dataframe.at[idx, col_coor_reduced120_48htypesmerged_level1_el] = coor_reduced120_48htypesmerged_level1_el  
        # dataframe.at[idx, col_sum_weirdos_48htypesmerged_level1_el] = sum_weirdos_48htypesmerged_level1_el          # these are just copying
        # dataframe.at[idx, col_sum_mapped_48htypesmerged_level1_el] = sum_mapped_48htypesmerged_level1_el
        # dataframe.at[idx, col_sum_sanitycheck_48htypesmerged_level1_el] = sum_weirdos_48htypesmerged_level1_el + sum_mapped_48htypesmerged_level1_el
        # dataframe.at[idx, col_duplicate_closest24_w_data_48htypesmerged_level1_el] = duplicate_closest24_w_data_48htypesmerged     # these are just copying
        dataframe.at[idx, col_coor_reduced120_48htypesmerged_level1_el] = coor_reduced120_48htypesmerged_level1_el
        dataframe.at[idx, col_sum_mapped_48htypesmerged_level1_el] = sum_mapped_48htypesmerged_level1_el 
        dataframe.at[idx, col_sum_sanitycheck_48htypesmerged_level1_el] = sum_mapped_48htypesmerged_level1_el + sum_weirdos_48htypesmerged_level1_el

        dataframe.at[idx, col_ndim_coor_reduced120_level1_el_closestduplicate] = ndim_coor_reduced120_el_closestduplicate
        dataframe.at[idx, col_ndim_coor_reduced120_48htype2_level1_el_closestduplicate] = ndim_coor_reduced120_48htype1_48htype2_el_closestduplicate


def get_flag_map_48htypesmerged_el(dataframe, el, activate_radius):
    if activate_radius == 3:
        # # col_flag_48htype1_48htype2_el = f"flag_48htype1_48htype2_{el}"
        # col_atom_mapping_el_w_dist = f"atom_mapping_48htype1_48htype2_{el}_w_dist"
        # col_coor_reduced120_el = f"coor_reduced120_48htype1_48htype2_{el}"
        # col_atom_mapping_el_closestduplicate = f"atom_mapping_48htype1_48htype2_{el}_closestduplicate"
        # col_coor_reduced120_el_closestduplicate = f"coor_reduced120_48htype1_48htype2_{el}_closestduplicate"
        # # col_flag_48htypesmerged_level1_el = f"flag_48htypesmerged_level1_{el}"
        col_atom_mapping_el_closestduplicate = f"atom_mapping_48htypesmerged_level1_{el}"
        # col_atom_mapping_el_w_dist = f"atom_mapping_48htypesmerged_level1_{el}_w_dist"
        col_coor_reduced120_el_closestduplicate = f"coor_reduced120_48htypesmerged_level1_{el}"

    elif activate_radius == 2:
        # # col_flag_el = f"flag_{el}"
        # col_coor_weirdos_el = f"coor_weirdos_{el}"
        # col_coor_weirdos_el_dict = f"coor_weirdos_{el}_dict"            # just added
        # col_atom_mapping_el_w_dist = f"atom_mapping_{el}_w_dist"
        # col_coor_reduced120_el = f"coor_reduced120_{el}"
        col_atom_mapping_el_closestduplicate = f"atom_mapping_{el}_closestduplicate"
        col_coor_reduced120_el_closestduplicate = f"coor_reduced120_{el}_closestduplicate"
        col_atom_mapping_el_w_dist_closestduplicate = f"atom_mapping_{el}_w_dist_closestduplicate"
        col_sum_weirdos_el = f"sum_weirdos_{el}"
        col_sum_mapped_el = f"sum_mapped_{el}"
        col_sum_sanitycheck_el = f"sum_sanitycheck_{el}"
        col_duplicate_closest24_w_data_el = f"duplicate_closest24_w_data_{el}"
        col_sum_mapped_el_closestduplicate = f"sum_mapped_{el}_closestduplicate"
        col_sum_sanitycheck_el_closestduplicate = f"sum_sanitycheck_{el}_closestduplicate"
    else:
        print("activate_radius is not correct")

    # # col_flag_48htype2_el = f"flag_48htype2_{el}"
    # col_coor_reduced120_48htype2_el = f"coor_reduced120_48htype2_{el}"
    col_atom_mapping_48htype2_el_closestduplicate = f"atom_mapping_48htype2_{el}_closestduplicate"
    col_sum_weirdos_48htype2_el = f"sum_weirdos_48htype2_{el}"
    col_coor_reduced120_48htype2_el_closestduplicate = f"coor_reduced120_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htype2_el_w_dist_closestduplicate = f"atom_mapping_48htype2_{el}_w_dist_closestduplicate"
    # col_atom_mapping_48htype2_el_w_dist = f"atom_mapping_48htype2_{el}_w_dist"
    # col_coor_weirdos_48htype2_el = f"coor_weirdos_48htype2_{el}"
    # col_sum_mapped_48htype2_el = f"sum_mapped_48htype2_{el}"
    # col_sum_sanitycheck_48htype2_el = f"sum_sanitycheck_48htype2_{el}"
    # col_duplicate_closest24_w_data_48htype2_el = f"duplicate_closest24_w_data_48htype2_{el}"
    # col_sum_mapped_48htype2_el_closestduplicate = f"sum_mapped_48htype2_{el}_closestduplicate"
    # cocommand:cellOutput.enableScrolling?23798be1-d942-4554-a3c6-774194ee7e7el_sum_sanitycheck_48htype2_el_closestduplicate = f"sum_sanitycheck_48htype2_{el}_closestduplicate"

    # col_coor_reduced120_48htypesmerged_el = f"coor_reduced120_48htypesmerged_{el}"
    # col_sum_mapped_48htypesmerged_el = f"sum_mapped_48htypesmerged_{el}"
    # col_sum_sanitycheck_48htypesmerged_el = f"sum_sanitycheck_48htypesmerged_{el}"
    col_flag_48htypesmerged_el = f"flag_48htypesmerged_{el}"
    col_atom_mapping_48htypesmerged_el = f"atom_mapping_48htypesmerged_{el}"
    # col_coor_weirdos_48htypesmerged_el = f"coor_weirdos_48htypesmerged_{el}"
    # col_sum_weirdos_48htypesmerged_el = f"sum_weirdos_48htypesmerged_{el}"
    # col_duplicate_closest24_w_data_48htypesmerged_el = f"duplicate_closest24_w_data_48htypesmerged_{el}"
    col_coor_reduced120_48htypesmerged_el = f"coor_reduced120_48htypesmerged_{el}"
    col_sum_mapped_48htypesmerged_el = f"sum_mapped_48htypesmerged_{el}"
    col_sum_sanitycheck_48htypesmerged_el = f"sum_sanitycheck_48htypesmerged_{el}"
    col_ndim_coor_reduced120_el_closestduplicate = f"ndim_coor_reduced120_{el}_closestduplicate"
    col_ndim_coor_reduced120_48htype2_el_closestduplicate = f"ndim_coor_reduced120_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htypesmerged_el_w_dist = f"atom_mapping_48htypesmerged_{el}_w_dist"

    # dataframe[col_coor_reduced120_48htypesmerged_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_sum_mapped_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_sum_sanitycheck_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_flag_48htypesmerged_el] = "False"
    dataframe[col_atom_mapping_48htypesmerged_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_48htypesmerged_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_weirdos_48htypesmerged_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_sum_weirdos_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_duplicate_closest24_w_data_48htypesmerged_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reduced120_48htypesmerged_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_reduced120_el_closestduplicate] = None
    dataframe[col_ndim_coor_reduced120_48htype2_el_closestduplicate] = None
    dataframe[col_atom_mapping_48htypesmerged_el_w_dist] = [{} for _ in range(len(dataframe.index))]

    for idx in range(dataframe["geometry"].size):
        # print(f"idx_48htypesmerged: {idx}")
        # atom_mapping_el_w_dist = dataframe.at[idx, col_atom_mapping_el_w_dist]
        # atom_mapping_48htype2_el_w_dist = dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist]
        atom_mapping_el_closestduplicate = dataframe.at[idx, col_atom_mapping_el_closestduplicate]
        atom_mapping_48htype2_el_closestduplicate = dataframe.at[idx, col_atom_mapping_48htype2_el_closestduplicate]
        # coor_reduced120_el = dataframe.at[idx, col_coor_reduced120_el]
        # coor_reduced120_48htype2_el = dataframe.at[idx, col_coor_reduced120_48htype2_el]
        # _closestduplicate
        coor_reduced120_el_closestduplicate = dataframe.at[idx, col_coor_reduced120_el_closestduplicate]
        coor_reduced120_48htype2_el_closestduplicate = dataframe.at[idx, col_coor_reduced120_48htype2_el_closestduplicate]
        atom_mapping_el_w_dist_closestduplicate = dataframe.at[idx, col_atom_mapping_el_w_dist_closestduplicate]
        atom_mapping_48htype2_el_w_dist_closestduplicate = dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist_closestduplicate]
        # these are just copying
        # coor_weirdos_48htypesmerged_el = dataframe.at[idx, col_coor_weirdos_48htype2_el]
        sum_weirdos_48htypesmerged_el = dataframe.at[idx, col_sum_weirdos_48htype2_el]
        # duplicate_closest24_w_data_48htypesmerged = dataframe.at[idx, col_duplicate_closest24_w_data_48htype2_el]
        
        # # _closestduplicate
        # coor_reduced120_48htypesmerged_el_closestduplicate = dataframe.at[idx, col_coor_reduced120_48htype2_el_closestduplicate]
        coor_reduced120_48htypesmerged_el = []

        atom_mapping_48htypesmerged_el = merge_dictionaries(atom_mapping_el_closestduplicate, atom_mapping_48htype2_el_closestduplicate)
        duplicate_coor24s = check_duplicate_values(atom_mapping_48htypesmerged_el)
        if len(duplicate_coor24s) > 1:        
            dataframe.at[idx, col_flag_48htypesmerged_el] = "True"        
        # for coor120 in atom_mapping_48htypesmerged_el:
        #     values24 = atom_mapping_48htypesmerged_el[coor120]   # get value for the key
        #     if len(values24) > 1:        
        #         dataframe.at[idx, col_flag_48htypesmerged_el] = "True"
        
        # atom_mapping_48htypesmerged_el_w_dist = merge_dictionaries(atom_mapping_el_w_dist, atom_mapping_48htype2_el_w_dist)

        # # if coor_reduced120_48htype2_el != None:
        # # if coor_reduced120_48htype2_el.size > 0:
        # if coor_reduced120_48htype2_el.ndim == 2:
        #     coor_reduced120_48htypesmerged_el = np.concatenate((coor_reduced120_el, coor_reduced120_48htype2_el), axis=0)
        # elif coor_reduced120_48htype2_el.ndim == 1:
        #     coor_reduced120_48htypesmerged_el = np.array(coor_reduced120_el.copy())
        # else:
        #     break

        ## we use _closestduplicate here since it's the corrected one wrt distance
        # if coor_reduced120_48htype2_el_closestduplicate != None:
        # if coor_reduced120_48htype2_el_closestduplicate.size > 0:
        # # # if coor_reduced120_48htype2_el_closestduplicate.ndim == 2:
        # # #     if coor_reduced120_el_closestduplicate.ndim == 2:
        # # #         coor_reduced120_48htypesmerged_el_closestduplicate = np.concatenate((coor_reduced120_el_closestduplicate, coor_reduced120_48htype2_el_closestduplicate), axis=0)
        # # #     else:
        # # #         print(f"coor_reduced120_el_closestduplicate has no correct dimension at idx: {idx}, dimension: {coor_reduced120_el_closestduplicate.ndim}")
        # # #         pass
        # # #         # print(f"coor_reduced120_el_closestduplicate: \n {coor_reduced120_el_closestduplicate}")
        # # # elif coor_reduced120_48htype2_el_closestduplicate.ndim == 1:
        # # #     coor_reduced120_48htypesmerged_el_closestduplicate = np.array(coor_reduced120_el_closestduplicate.copy())
        # # # else:
        # # #     print(f"coor_reduced120_48htype2_el_closestduplicate has no correct dimension at idx: {idx}, dimension: {coor_reduced120_48htypesmerged_el_closestduplicate.ndim}")
        # # #     # break

        if coor_reduced120_48htype2_el_closestduplicate.ndim == coor_reduced120_el_closestduplicate.ndim:
            coor_reduced120_48htypesmerged_el = np.concatenate((coor_reduced120_el_closestduplicate, coor_reduced120_48htype2_el_closestduplicate), axis=0)
            # else:
            #     print(f"coor_reduced120_el_closestduplicate has no correct dimension at idx: {idx}")
            #     continue
                # print(f"coor_reduced120_el_closestduplicate: \n {coor_reduced120_el_closestduplicate}")
        elif coor_reduced120_48htype2_el_closestduplicate.ndim == 1:
            coor_reduced120_48htypesmerged_el = np.array(coor_reduced120_el_closestduplicate.copy())
        elif coor_reduced120_el_closestduplicate.ndim == 1:
            coor_reduced120_48htypesmerged_el = np.array(coor_reduced120_48htype2_el_closestduplicate.copy())
        else:
            print(f"coor_reduced120_48htype2_el_closestduplicate or coor_reduced120_el_closestduplicate has no correct dimension at idx: {idx}")
            # break

        # sum_mapped_48htypesmerged_el = len(atom_mapping_48htypesmerged_el)
        sum_mapped_48htypesmerged_el = len(coor_reduced120_48htypesmerged_el)

        ndim_coor_reduced120_el_closestduplicate = coor_reduced120_el_closestduplicate.ndim
        ndim_coor_reduced120_48htype2_el_closestduplicate = coor_reduced120_48htype2_el_closestduplicate.ndim

        atom_mapping_48htypesmerged_el_w_dist = atom_mapping_el_w_dist_closestduplicate | atom_mapping_48htype2_el_w_dist_closestduplicate

        # dataframe.at[idx, col_coor_weirdos_48htypesmerged_el] = coor_weirdos_48htypesmerged_el        # these are just copying
        # dataframe.at[idx, col_coor_reduced120_48htypesmerged_el] = coor_reduced120_48htypesmerged_el  
        # dataframe.at[idx, col_sum_weirdos_48htypesmerged_el] = sum_weirdos_48htypesmerged_el          # these are just copying
        # dataframe.at[idx, col_sum_mapped_48htypesmerged_el] = sum_mapped_48htypesmerged_el
        # dataframe.at[idx, col_sum_sanitycheck_48htypesmerged_el] = sum_weirdos_48htypesmerged_el + sum_mapped_48htypesmerged_el
        # dataframe.at[idx, col_duplicate_closest24_w_data_48htypesmerged_el] = duplicate_closest24_w_data_48htypesmerged     # these are just copying
        # dataframe.at[idx, col_atom_mapping_48htypesmerged_el_w_dist] = atom_mapping_48htypesmerged_el_w_dist
        dataframe.at[idx, col_atom_mapping_48htypesmerged_el] = atom_mapping_48htypesmerged_el
        dataframe.at[idx, col_coor_reduced120_48htypesmerged_el] = coor_reduced120_48htypesmerged_el
        dataframe.at[idx, col_sum_mapped_48htypesmerged_el] = sum_mapped_48htypesmerged_el 
        dataframe.at[idx, col_sum_sanitycheck_48htypesmerged_el] = sum_mapped_48htypesmerged_el + sum_weirdos_48htypesmerged_el
        dataframe.at[idx, col_ndim_coor_reduced120_el_closestduplicate] = ndim_coor_reduced120_el_closestduplicate
        dataframe.at[idx, col_ndim_coor_reduced120_48htype2_el_closestduplicate] = ndim_coor_reduced120_48htype2_el_closestduplicate
        dataframe.at[idx, col_atom_mapping_48htypesmerged_el_w_dist] = atom_mapping_48htypesmerged_el_w_dist


def get_idx_weirdos_el(dataframe, el, activate_radius):
    col_coor_structure_init_dict = "coor_structure_init_dict"

    if activate_radius == 2 or activate_radius == 3:
        col_coor_weirdos_el = f"coor_weirdos_48htype2_{el}"
    elif activate_radius == 1:
        col_coor_weirdos_el = f"coor_weirdos_{el}"

    col_idx0_weirdos_el = f"idx0_weirdos_{el}"
    col_idx1_weirdos_el = f"idx1_weirdos_{el}"
    col_nr_of_weirdos_el = f"#weirdos_{el}"
    col_idx_coor_weirdos_el = f"idx_coor_weirdos_{el}"

    dataframe[col_idx0_weirdos_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_idx1_weirdos_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_nr_of_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor_weirdos_el] = [{} for _ in range(len(dataframe.index))]

    for idx in range(dataframe["geometry"].size):
        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_init_dict][el]
        coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]
        
        idx_weirdos_el = []
        idx_coor_weirdos_el = {}

        for i, given_arr in enumerate(coor_origin24_el_init):
            
            # coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]
            if len(coor_weirdos_el) > 0:
                for arr in coor_weirdos_el:
                    if (arr == given_arr).all():
                        idx_weirdos_el.append(i)

                        idx_coor_weirdos_el[i] = arr

        dataframe.at[idx, col_idx0_weirdos_el] = idx_weirdos_el
        dataframe.at[idx, col_idx1_weirdos_el] = np.array([value for index, value in enumerate(idx_weirdos_el)]) + 1
        dataframe.at[idx, col_nr_of_weirdos_el] = len(idx_weirdos_el)
        dataframe.at[idx, col_idx_coor_weirdos_el] = idx_coor_weirdos_el


def idx_correcting_mapped_el(dataframe, el, activate_radius):
    col_coor_structure_init_dict = "coor_structure_init_dict"

    if activate_radius == 2 or activate_radius == 3:
        col_coor_reduced120_el = f"coor_reduced120_48htypesmerged_{el}"
    elif activate_radius == 1:
        col_coor_reduced120_el = f"coor_reduced120_{el}_closestduplicate"

    col_idx_correcting_el = f"idx_correcting_{el}"
    # col_atom_mapping_el_w_dist_idx24 = f"atom_mapping_{el}_w_dist_idx24"
    # col_coor_reduced120_closestduplicate_el = f"coor_reduced120_closestduplicate_{el}"
    col_coor_reduced120_sorted_el = f"coor_reduced120_sorted_{el}"

    dataframe[col_idx_correcting_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_el_w_dist_idx24] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_reduced120_closestduplicate_el] = None
    dataframe[col_coor_reduced120_sorted_el] = [np.array([]) for _ in range(len(dataframe.index))]

    for idx in range(dataframe["geometry"].size):
        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_init_dict][el] 
        coor_reduced120_el = dataframe.at[idx, col_coor_reduced120_el]

        idx_correcting_el = []
        # atom_mapping_el_w_dist_idx24 = {} 
        idx_coor_el = {}

        for idx120, coor120 in enumerate(coor_reduced120_el):
            distance_prev = float("inf")
            closest24 = None
            idx_closest24 = None
            # atom_mapping_w_dist_idx24_dict = {}

            # atom_mapping_el_w_dist_idx24[tuple(coor120)] = []

            for idx24, coor24 in enumerate(coor_origin24_el_init):
                distance = mic_eucledian_distance(coor120, coor24)

                # if distance != 0:
                if distance < distance_prev:
                    distance_prev = distance
                    closest24 = coor24
                    idx_closest24 = idx24
            
            idx_correcting_el.append(idx_closest24)

            # if closest24 is not None:
            #     atom_mapping_w_dist_idx24_dict['closest24'] = tuple(closest24)
            #     atom_mapping_w_dist_idx24_dict['dist'] = distance_prev
            #     atom_mapping_w_dist_idx24_dict['idx_closest24'] = idx_closest24

            #     # if tuple(coor120) in atom_mapping_el_w_dist_idx24:
            #     #     atom_mapping_el_w_dist_idx24[tuple(coor120)].append(atom_mapping_w_dist_idx24_dict)
            #     # else:
            #     #     atom_mapping_el_w_dist_idx24[tuple(coor120)] = atom_mapping_w_dist_idx24_dict
    
            #     if tuple(coor120) in atom_mapping_el_w_dist_idx24:
            #         new_entry = atom_mapping_el_w_dist_idx24[tuple(coor120)].copy()
            #         new_entry.append(atom_mapping_w_dist_idx24_dict)
            #         atom_mapping_el_w_dist_idx24[tuple(coor120)] = new_entry
            #     else:
            #         atom_mapping_el_w_dist_idx24[tuple(coor120)] = [atom_mapping_w_dist_idx24_dict.copy()]

        # coor_reduced120_closestduplicate_el = [coor_reduced120_el[i] for i in idx_correcting_el]
        # # coor_reduced120_closestduplicate_el_closestduplicate = [coor_reduced120_el_closestduplicate[i] for i in idx_correcting_el]

        for i in range(len(idx_correcting_el)):
            idx_coor_el[idx_correcting_el[i]] = coor_reduced120_el[i]

        sorted_idx_coor_el = {key: val for key, val in sorted(idx_coor_el.items())}
        sorted_coor = list(sorted_idx_coor_el.values())

        dataframe.at[idx, col_idx_correcting_el] = idx_correcting_el
        # dataframe.at[idx, col_atom_mapping_el_w_dist_idx24] = atom_mapping_el_w_dist_idx24
        # dataframe.at[idx, col_coor_reduced120_closestduplicate_el] = coor_reduced120_closestduplicate_el
        # # dataframe.at[idx, col_coor_reduced120_closestduplicate_el_closestduplicate] = coor_reduced120_closestduplicate_el_closestduplicate
        dataframe.at[idx, col_coor_reduced120_sorted_el] = sorted_coor


def get_distance_weirdos_label_el(dataframe, coor_structure_init_dict, el, litype):
    # to do: add idx of weirdo and coor120
    coor_origin120_el_init = coor_structure_init_dict[el]
    col_idx_coor_weirdos_el = f"idx_coor_weirdos_{el}"

    col_top3_sorted_idxweirdo_dist_label_el = f'top3_sorted_idxweirdo_dist_label_{el}'
    col_top3_sorted_idxweirdo_dist_el = f'top3_sorted_idxweirdo_dist_{el}'
    col_top3_sorted_idxweirdo_label_el = f'top3_sorted_idxweirdo_label_{el}'
    col_top1_sorted_idxweirdo_dist_label_el = f'top1_sorted_idxweirdo_dist_label_{el}'
    col_top1_sorted_idxweirdo_dist_el = f'top1_sorted_idxweirdo_dist_{el}'
    col_top1_sorted_idxweirdo_label_el = f'top1_sorted_idxweirdo_label_{el}'
    col_top1_sorted_idxweirdo_coor_el = f'top1_sorted_idxweirdo_coor_{el}'
    col_top1_sorted_idxweirdo_file_el = f'top1_sorted_idxweirdo_file_{el}'

    col_sum_closest_24g_el      = f'#closest_24g_{el}'
    dataframe[col_sum_closest_24g_el] = [0 for _ in range(len(dataframe.index))]
    if litype == 1:
        col_sum_closest_48htype1_el = f'#closest_48htype1_{el}'
        dataframe[col_sum_closest_48htype1_el] = [0 for _ in range(len(dataframe.index))]
    elif litype == 2:
        col_sum_closest_48htype1_el = f'#closest_48htype1_{el}'
        col_sum_closest_48htype2_el = f'#closest_48htype2_{el}'
        dataframe[col_sum_closest_48htype1_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    elif litype == 3:
        col_sum_closest_48htype1_el = f'#closest_48htype1_{el}'
        col_sum_closest_48htype2_el = f'#closest_48htype2_{el}'
        col_sum_closest_48htype3_el = f'#closest_48htype3_{el}'
        dataframe[col_sum_closest_48htype1_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype2_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype3_el] = [0 for _ in range(len(dataframe.index))]
    elif litype == 4:
        col_sum_closest_48htype1_el = f'#closest_48htype1_{el}'
        col_sum_closest_48htype2_el = f'#closest_48htype2_{el}'
        col_sum_closest_48htype3_el = f'#closest_48htype3_{el}'
        col_sum_closest_48htype4_el = f'#closest_48htype4_{el}'
        dataframe[col_sum_closest_48htype1_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype2_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype3_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype4_el] = [0 for _ in range(len(dataframe.index))]
    elif litype == 5:
        col_sum_closest_48htype1_el = f'#closest_48htype1_{el}'
        col_sum_closest_48htype2_el = f'#closest_48htype2_{el}'
        col_sum_closest_48htype3_el = f'#closest_48htype3_{el}'
        col_sum_closest_48htype4_el = f'#closest_48htype4_{el}'
        col_sum_closest_48htype5_el = f'#closest_48htype5_{el}'
        dataframe[col_sum_closest_48htype1_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype2_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype3_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype4_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype5_el] = [0 for _ in range(len(dataframe.index))]
    elif litype == 6:
        col_sum_closest_48htype1_el = f'#closest_48htype1_{el}'
        col_sum_closest_48htype2_el = f'#closest_48htype2_{el}'
        col_sum_closest_48htype3_el = f'#closest_48htype3_{el}'
        col_sum_closest_48htype4_el = f'#closest_48htype4_{el}'
        col_sum_closest_48htype5_el = f'#closest_48htype5_{el}'
        col_sum_closest_48htype6_el = f'#closest_48htype6_{el}'
        dataframe[col_sum_closest_48htype1_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype2_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype3_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype4_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype5_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype6_el] = [0 for _ in range(len(dataframe.index))]
    elif litype == 7:
        col_sum_closest_48htype1_el = f'#closest_48htype1_{el}'
        col_sum_closest_48htype2_el = f'#closest_48htype2_{el}'
        col_sum_closest_48htype3_el = f'#closest_48htype3_{el}'
        col_sum_closest_48htype4_el = f'#closest_48htype4_{el}'
        col_sum_closest_48htype5_el = f'#closest_48htype5_{el}'
        col_sum_closest_48htype6_el = f'#closest_48htype6_{el}'
        col_sum_closest_48htype7_el = f'#closest_48htype7_{el}'
        dataframe[col_sum_closest_48htype1_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype2_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype3_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype4_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype5_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype6_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype7_el] = [0 for _ in range(len(dataframe.index))]
    elif litype == 8:
        col_sum_closest_48htype1_el = f'#closest_48htype1_{el}'
        col_sum_closest_48htype2_el = f'#closest_48htype2_{el}'
        col_sum_closest_48htype3_el = f'#closest_48htype3_{el}'
        col_sum_closest_48htype4_el = f'#closest_48htype4_{el}'
        col_sum_closest_48htype5_el = f'#closest_48htype5_{el}'
        col_sum_closest_48htype6_el = f'#closest_48htype6_{el}'
        col_sum_closest_48htype7_el = f'#closest_48htype7_{el}'
        col_sum_closest_48htype8_el = f'#closest_48htype8_{el}'
        dataframe[col_sum_closest_48htype1_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype2_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype3_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype4_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype5_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype6_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype7_el] = [0 for _ in range(len(dataframe.index))]
        dataframe[col_sum_closest_48htype8_el] = [0 for _ in range(len(dataframe.index))]


    # dataframe[col_top3_dist_weirdos_array_el] = None
    # dataframe[col_top3_coorweirdo_dist_label_coor120_idxweirdo_idx120_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_dist_weirdos_atom120_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_dist_weirdos_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_coorweirdo_dist_label_coor120_idxweirdo_idx120_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top3_sorted_idxweirdo_dist_label_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top3_sorted_idxweirdo_dist_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top3_sorted_idxweirdo_label_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_dist_label_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_dist_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_label_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_coor_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_file_el] = [{} for _ in range(len(dataframe.index))]
    
    coor_li24g_ref      = coor_origin120_el_init[0:24]
    if litype == 1:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coordinate_lists    = [coor_li48htype1_ref]
        labels              = ["48htype1"]
    elif litype == 2:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref]
        labels              = ["48htype1", "48htype2"]
    elif litype == 3:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref]
        labels              = ["48htype1", "48htype2", "48htype3"]
    elif litype == 4:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4"]
    elif litype == 5:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5"]
    elif litype == 6:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6"]
    elif litype == 7:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
        coor_li48htype7_ref = coor_origin120_el_init[312:360]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref, coor_li48htype7_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7"]
    elif litype == 8:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
        coor_li48htype7_ref = coor_origin120_el_init[312:360]
        coor_li48htype8_ref = coor_origin120_el_init[360:408]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref, coor_li48htype7_ref, coor_li48htype8_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7", "48htype8"]


    for idx in range(dataframe["geometry"].size):
        dist_weirdos_atom120_el = []
        dist_weirdos_el = []
        coorweirdo_dist_label_coor120_idxweirdo_idx120_el = {}
        top3_sorted_idxweirdo_dist_label_el = {}
        top3_sorted_idxweirdo_dist_el = {}
        top3_sorted_idxweirdo_label_el = {}
        top1_sorted_idxweirdo_dist_label_el = {}
        top1_sorted_idxweirdo_label_el = {}
        top1_sorted_idxweirdo_dist_el = {}
        top1_sorted_idxweirdo_coor_el = {}
        top1_sorted_idxweirdo_file_el = {}

        idx_coor_weirdos_el = dataframe.at[idx, col_idx_coor_weirdos_el]
        idx_weirdos_el = list(idx_coor_weirdos_el.keys())

        if len(idx_weirdos_el) > 0:
            for idx_weirdo in idx_weirdos_el:
                coor_weirdo = idx_coor_weirdos_el[idx_weirdo]
                distance_weirdo_prev = float('inf')

                coorweirdo_dist_label_coor120_idxweirdo_idx120_el[idx_weirdo] = []
                
                for idx120, coor120 in enumerate(coor_origin120_el_init):
                    coorweirdo_dist_label_coor120_val_el = {}
            
                    distance_weirdo = mic_eucledian_distance(coor120, coor_weirdo)

                    coorweirdo_dist_label_coor120_val_el['dist'] = distance_weirdo

                    for idx_24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
                        if (coor120 == coor_li24g_ref_temp).all():
                            coorweirdo_dist_label_coor120_val_el["coor"] = tuple(coor_weirdo)
                            coorweirdo_dist_label_coor120_val_el["label"] = "24g"
                            coorweirdo_dist_label_coor120_val_el["coor120"] = tuple(coor120)
                            coorweirdo_dist_label_coor120_val_el["file"] = f"{int(dataframe.at[idx, 'geometry'])}_{int(dataframe.at[idx, 'path'])}"
                            if idx_weirdo in coorweirdo_dist_label_coor120_idxweirdo_idx120_el:
                                coorweirdo_dist_label_coor120_idxweirdo_idx120_el[idx_weirdo].append(coorweirdo_dist_label_coor120_val_el)
                            else:
                                coorweirdo_dist_label_coor120_idxweirdo_idx120_el[idx_weirdo] = coorweirdo_dist_label_coor120_val_el

                    for i in range(1, litype + 1):
                        for j, coor_ref_temp in enumerate(coordinate_lists[i - 1]):
                            if (coor120 == coor_ref_temp).all():
                                coorweirdo_dist_label_coor120_val_el["coor"] = tuple(coor_weirdo)
                                coorweirdo_dist_label_coor120_val_el["label"] = labels[i - 1]
                                coorweirdo_dist_label_coor120_val_el["coor120"] = tuple(coor120)
                                coorweirdo_dist_label_coor120_val_el["file"] = f"{int(dataframe.at[idx, 'geometry'])}_{int(dataframe.at[idx, 'path'])}"

                                if idx_weirdo in coorweirdo_dist_label_coor120_idxweirdo_idx120_el:
                                    coorweirdo_dist_label_coor120_idxweirdo_idx120_el[idx_weirdo].append(coorweirdo_dist_label_coor120_val_el)
                                else:
                                    coorweirdo_dist_label_coor120_idxweirdo_idx120_el[idx_weirdo] = [coorweirdo_dist_label_coor120_val_el]

                    if distance_weirdo < distance_weirdo_prev:
                        distance_weirdo_prev = distance_weirdo
                        closest120 = coor120

                dist_weirdos_atom120_el_array = [distance_weirdo, tuple(coor_weirdo), tuple(closest120)]
                dist_weirdos_el_array = [distance_weirdo]
                dist_weirdos_atom120_el.append(dist_weirdos_atom120_el_array)
                dist_weirdos_el.append(dist_weirdos_el_array)
                # float_dist_weirdos_el = np.append(float_dist_weirdos_el, [distance_weirdo_prev])

                # sorted_dist_weirdos_array_el = sorted(set(dist_weirdos_array_el))
                # top3_dist_weirdos_array_el = sorted_dist_weirdos_array_el[0:3]

                # coorweirdo_dist_label_coor120_idxweirdo_idx120_el['top3_dist'] = top3_dist_weirdos_array_el

                # if tuple(coor_weirdo) in top3_dist_weirdos_el:
                #     top3_dist_weirdos_el[tuple(coor_weirdo)].append(coorweirdo_dist_label_coor120_idxweirdo_idx120_el)
                # else:
                #     top3_dist_weirdos_el[tuple(coor_weirdo)] = coorweirdo_dist_label_coor120_idxweirdo_idx120_el

                # for key_temp1, val_temp1 in coorweirdo_dist_label_coor120_idxweirdo_idx120_el.items():
                #     sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_el = {key_temp1: sorted(val_temp1, key=lambda x: x['dist'])}
                sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_el = {
                                                                    k: sorted(v, key=lambda x: x['dist'])
                                                                    for k, v in coorweirdo_dist_label_coor120_idxweirdo_idx120_el.items()
                                                                }
                
                top3_sorted_coorweirdo_dist_label_coor120_el = {k: v[0:3] for k, v in sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_el.items()}
                for key, values_list in top3_sorted_coorweirdo_dist_label_coor120_el.items():
                    selected_values = [{'dist': entry['dist'], "label": entry["label"]} for entry in values_list]
                    top3_sorted_idxweirdo_dist_label_el[key] = selected_values
                    selected_dists = [entry['dist'] for entry in values_list]
                    top3_sorted_idxweirdo_dist_el[key] = selected_dists
                    selected_types = [entry["label"] for entry in values_list]
                    top3_sorted_idxweirdo_label_el[key] = selected_types

                top1_sorted_coorweirdo_dist_label_coor120_el = {k: v[0:1] for k, v in sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_el.items()}
                for key, values_list in top1_sorted_coorweirdo_dist_label_coor120_el.items():
                    top1_selected_values = [{'dist': entry['dist'], "label": entry["label"]} for entry in values_list]
                    top1_sorted_idxweirdo_dist_label_el[key] = top1_selected_values
                    selected_types = [entry["label"] for entry in values_list]
                    top1_sorted_idxweirdo_label_el[key] = selected_types
                    selected_dists = [entry['dist'] for entry in values_list]
                    top1_sorted_idxweirdo_dist_el[key] = selected_dists
                    selected_coors = [entry['coor'] for entry in values_list]
                    top1_sorted_idxweirdo_coor_el[key] = selected_coors
                    selected_files = [entry['file'] for entry in values_list]
                    top1_sorted_idxweirdo_file_el[key] = selected_files

                # Types to count
                types_to_count = [f'48htype{i}' for i in range(litype, 0, -1)] + ['24g']

                # Initialize counts
                type_counts = {t: 0 for t in types_to_count}

                # Count occurrences
                for closest_type in top1_sorted_idxweirdo_label_el.values():
                    for value in closest_type:
                        if value in type_counts:
                            type_counts[value] += 1

            # # dataframe.at[idx, col_dist_weirdos_atom120_el] = sorted(dist_weirdos_atom120_el, coor_weirdo=lambda x: x[0]) 
            # dataframe.at[idx, col_dist_weirdos_el] = np.array([coor120[0] for index, coor120 in enumerate(dist_weirdos_atom120_el)])
            # # dataframe.at[idx, col_dist_weirdos_el] = sorted(dist_weirdos_el, coor_weirdo=lambda x: x[0]) 
            # dataframe.at[idx, col_coorweirdo_dist_label_coor120_idxweirdo_idx120_el] = coorweirdo_dist_label_coor120_idxweirdo_idx120_el
            # dataframe.at[idx, col_sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_el] = sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_el
            dataframe.at[idx, col_top3_sorted_idxweirdo_dist_label_el] = top3_sorted_idxweirdo_dist_label_el
            dataframe.at[idx, col_top3_sorted_idxweirdo_dist_el] = top3_sorted_idxweirdo_dist_el
            dataframe.at[idx, col_top3_sorted_idxweirdo_label_el] = top3_sorted_idxweirdo_label_el
            dataframe.at[idx, col_top1_sorted_idxweirdo_dist_label_el] = top1_sorted_idxweirdo_dist_label_el
            dataframe.at[idx, col_top1_sorted_idxweirdo_dist_el] = top1_sorted_idxweirdo_dist_el
            dataframe.at[idx, col_top1_sorted_idxweirdo_label_el] = top1_sorted_idxweirdo_label_el
            dataframe.at[idx, col_top1_sorted_idxweirdo_coor_el] = top1_sorted_idxweirdo_coor_el
            dataframe.at[idx, col_top1_sorted_idxweirdo_file_el] = top1_sorted_idxweirdo_file_el
            
            dataframe.at[idx, col_sum_closest_24g_el] = type_counts['24g']
            if litype == 1:
                dataframe.at[idx, col_sum_closest_48htype1_el] = type_counts['48htype1']
            elif litype == 2:
                dataframe.at[idx, col_sum_closest_48htype1_el] = type_counts['48htype1']
                dataframe.at[idx, col_sum_closest_48htype2_el] = type_counts['48htype2']
            elif litype == 3:
                dataframe.at[idx, col_sum_closest_48htype1_el] = type_counts['48htype1']
                dataframe.at[idx, col_sum_closest_48htype2_el] = type_counts['48htype2']
                dataframe.at[idx, col_sum_closest_48htype3_el] = type_counts['48htype3']
            elif litype == 4:
                dataframe.at[idx, col_sum_closest_48htype1_el] = type_counts['48htype1']
                dataframe.at[idx, col_sum_closest_48htype2_el] = type_counts['48htype2']
                dataframe.at[idx, col_sum_closest_48htype3_el] = type_counts['48htype3']
                dataframe.at[idx, col_sum_closest_48htype4_el] = type_counts['48htype4']
            elif litype == 5:
                dataframe.at[idx, col_sum_closest_48htype1_el] = type_counts['48htype1']
                dataframe.at[idx, col_sum_closest_48htype2_el] = type_counts['48htype2']
                dataframe.at[idx, col_sum_closest_48htype3_el] = type_counts['48htype3']
                dataframe.at[idx, col_sum_closest_48htype4_el] = type_counts['48htype4']
                dataframe.at[idx, col_sum_closest_48htype5_el] = type_counts['48htype5']
            elif litype == 6:
                dataframe.at[idx, col_sum_closest_48htype1_el] = type_counts['48htype1']
                dataframe.at[idx, col_sum_closest_48htype2_el] = type_counts['48htype2']
                dataframe.at[idx, col_sum_closest_48htype3_el] = type_counts['48htype3']
                dataframe.at[idx, col_sum_closest_48htype4_el] = type_counts['48htype4']
                dataframe.at[idx, col_sum_closest_48htype5_el] = type_counts['48htype5']
                dataframe.at[idx, col_sum_closest_48htype6_el] = type_counts['48htype6']
            elif litype == 7:
                dataframe.at[idx, col_sum_closest_48htype1_el] = type_counts['48htype1']
                dataframe.at[idx, col_sum_closest_48htype2_el] = type_counts['48htype2']
                dataframe.at[idx, col_sum_closest_48htype3_el] = type_counts['48htype3']
                dataframe.at[idx, col_sum_closest_48htype4_el] = type_counts['48htype4']
                dataframe.at[idx, col_sum_closest_48htype5_el] = type_counts['48htype5']
                dataframe.at[idx, col_sum_closest_48htype6_el] = type_counts['48htype6']
                dataframe.at[idx, col_sum_closest_48htype7_el] = type_counts['48htype7']
            elif litype == 8:
                dataframe.at[idx, col_sum_closest_48htype1_el] = type_counts['48htype1']
                dataframe.at[idx, col_sum_closest_48htype2_el] = type_counts['48htype2']
                dataframe.at[idx, col_sum_closest_48htype3_el] = type_counts['48htype3']
                dataframe.at[idx, col_sum_closest_48htype4_el] = type_counts['48htype4']
                dataframe.at[idx, col_sum_closest_48htype5_el] = type_counts['48htype5']
                dataframe.at[idx, col_sum_closest_48htype6_el] = type_counts['48htype6']
                dataframe.at[idx, col_sum_closest_48htype7_el] = type_counts['48htype7']
                dataframe.at[idx, col_sum_closest_48htype8_el] = type_counts['48htype8']

            # dataframe.at[idx, col_top3_coorweirdo_dist_label_coor120_idxweirdo_idx120_el] = top3_dist_weirdos_el
            # print(coorweirdo_dist_label_coor120_idxweirdo_idx120_el)
        # else:
        #     dataframe.at[idx, col_dist_weirdos_atom120_el] = {}
        #     # dataframe.at[idx, col_dist_weirdos_el] = np.array([coor120[0] for index, coor120 in enumerate(dist_weirdos_atom120_el)])
        #     dataframe.at[idx, col_dist_weirdos_el] = []
        #     dataframe.at[idx, col_top3_coorweirdo_dist_label_coor120_idxweirdo_idx120_el] = []


def get_label_mapping(dataframe, coor_structure_init_dict, el, activate_radius, litype):
    # TO DO: split into elementwise

    coor_origin120_el_init = coor_structure_init_dict[el]

    if activate_radius == 1:
        col_atom_mapping_el_w_dist = f"atom_mapping_{el}_w_dist_closestduplicate"
    elif activate_radius == 2:
        col_atom_mapping_el_w_dist = f"atom_mapping_48htypesmerged_{el}_w_dist"
    
    col_atom_mapping_el_w_dist_label = f"atom_mapping_{el}_w_dist_label"

    dataframe[col_atom_mapping_el_w_dist_label] = [{} for _ in range(len(dataframe.index))]

    coor_li24g_ref      = coor_origin120_el_init[0:24]
    if litype == 1:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coordinate_lists    = [coor_li48htype1_ref]
        labels              = ["48htype1"]
    elif litype == 2:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref]
        labels              = ["48htype1", "48htype2"]
    elif litype == 3:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref]
        labels              = ["48htype1", "48htype2", "48htype3"]
    elif litype == 4:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4"]
    elif litype == 5:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5"]
    elif litype == 6:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6"]
    elif litype == 7:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
        coor_li48htype7_ref = coor_origin120_el_init[312:360]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref, coor_li48htype7_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7"]
    elif litype == 8:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
        coor_li48htype7_ref = coor_origin120_el_init[312:360]
        coor_li48htype8_ref = coor_origin120_el_init[360:408]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref, coor_li48htype7_ref, coor_li48htype8_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7", "48htype8"]
    
    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist = dataframe.at[idx, col_atom_mapping_el_w_dist]

        atom_mapping_el_w_dist_label = {}

        for coor120 in atom_mapping_el_w_dist.keys():
            value = atom_mapping_el_w_dist[tuple(coor120)]

            if isinstance(value, list):
                # Handle the case where the value is a list
                atom_mapping_el_w_dist_label_val = {'closest24': value[0]['closest24'], 'dist': value[0]['dist']}
            else:
                # Handle the case where the value is a dictionary
                atom_mapping_el_w_dist_label_val = {'closest24': value['closest24'], 'dist': value['dist']}

            # # atom_mapping_el_w_dist_label_val = {}
            # # atom_mapping_el_w_dist_label[tuple(coor120)] = []
            # atom_mapping_el_w_dist_label_val = {'closest24': atom_mapping_el_w_dist[tuple(coor120)][0]['closest24'], 'dist': atom_mapping_el_w_dist[tuple(coor120)][0]['dist']}

            for idx_li24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
                if (coor120 == coor_li24g_ref_temp).all():
                    atom_mapping_el_w_dist_label_val["label"] = "24g"

                    atom_mapping_el_w_dist_label[tuple(coor120)] = atom_mapping_el_w_dist_label_val
                    # atom_mapping_el_w_dist_label[tuple(coor120)].append(atom_mapping_el_w_dist_label_val)

            for i in range(1, litype+1):
                coor_li48htype_ref = locals()[f"coor_li48htype{i}_ref"]
                label = f"48htype{i}"

                for idx_temp, coor_ref_temp in enumerate(coor_li48htype_ref):
                    if (coor120 == coor_ref_temp).all():
                        atom_mapping_el_w_dist_label_val["label"] = label
                        atom_mapping_el_w_dist_label[tuple(coor120)] = atom_mapping_el_w_dist_label_val
                        # atom_mapping_el_w_dist_label[tuple(coor120)].append(atom_mapping_el_w_dist_label_val)


            # # for idx_li48htype1_temp, coor_li48htype1_ref_temp in enumerate(coor_li48htype1_ref):
            # #     if (coor120 == coor_li48htype1_ref_temp).all():
            # #         atom_mapping_el_w_dist_label_val["label"] = "48htype1"

            # #         atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_el_w_dist_label_val)

            # # for idx_li48htype2_temp, coor_li48htype2_ref_temp in enumerate(coor_li48htype2_ref):
            # #     if (coor120 == coor_li48htype2_ref_temp).all():
            # #         atom_mapping_el_w_dist_label_val["label"] = "48htype2"

            # #         atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_el_w_dist_label_val)

            # # for idx_li24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
            # #     if (coor120 == coor_li24g_ref_temp).all():
            # #         atom_mapping_el_w_dist_label_val["label"] = "24g"

            # #         atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_el_w_dist_label_val)

            # # if litype == 3:
            # #     for idx_li48htype3_temp, coor_li48htype3_ref_temp in enumerate(coor_li48htype3_ref):
            # #         if (coor120 == coor_li48htype3_ref_temp).all():
            # #             atom_mapping_el_w_dist_label_val["label"] = "48htype3"

            # #             atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_el_w_dist_label_val)

            # # elif litype == 4:
            # #     for idx_li48htype3_temp, coor_li48htype3_ref_temp in enumerate(coor_li48htype3_ref):
            # #         if (coor120 == coor_li48htype3_ref_temp).all():
            # #             atom_mapping_el_w_dist_label_val["label"] = "48htype3"

            # #             atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_el_w_dist_label_val)

            # #     for idx_li48htype4_temp, coor_li48htype4_ref_temp in enumerate(coor_li48htype4_ref):
            # #         if (coor120 == coor_li48htype4_ref_temp).all():
            # #             atom_mapping_el_w_dist_label_val["label"] = "48htype4"

            # #             atom_mapping_el_w_dist[tuple(coor120)].append(atom_mapping_el_w_dist_label_val)
    
        # # dataframe.at[idx, col_atom_mapping_el_w_dist_label] = atom_mapping_el_w_dist
        dataframe.at[idx, col_atom_mapping_el_w_dist_label] = atom_mapping_el_w_dist_label


def create_combine_structure(dataframe, destination_directory, amount_Li, amount_P, amount_S, activate_radius, var_savefilename):
    ## TO DO: under maintenance for disambled into el
    if activate_radius == 2 or activate_radius == 3:
        col_coor_reduced120_el = f"coor_reduced120_48htypesmerged_Li"
    elif activate_radius == 1:
        col_coor_reduced120_el = f"coor_reduced120_Li_closestduplicate"
                    
    # col_coor_reduced120_closestduplicate_Li_closestduplicate = f"coor_reduced120_closestduplicate_Li_closestduplicate" # !!!!!
    # col_coor_reduced120_closestduplicate_Li = f"coor_reduced120_closestduplicate_Li" # !!!!!
    col_coor_structure_init_dict = "coor_structure_init_dict"

    for idx in range(dataframe["geometry"].size):
        coor_combined = []

        # new_structure = Structure.from_file(dataframe['subdir_orientated_positive'][idx])
        # new_structure = Structure.from_file(dataframe['subdir_orientated_positive_poscar'][idx])
        new_structure = Structure.from_file(dataframe['subdir_positive_CONTCAR'][idx])
        coor_origin24_init = dataframe.at[idx, col_coor_structure_init_dict]
        # coor_reduced120_Li = dataframe.at[idx, col_coor_reduced120_closestduplicate_Li]
        # coor_reduced120_Li = dataframe.at[idx, col_coor_reduced120_closestduplicate_Li_closestduplicate]
        coor_reduced120_Li = dataframe.at[idx, col_coor_reduced120_el]

        coor_structure_init_P = coor_origin24_init["P"]
        coor_structure_init_S = coor_origin24_init["S"]
        coor_structure_init_Cl = coor_origin24_init["Cl"]

        coor_mapped_Li = np.array(coor_reduced120_Li)
        coor_origin_P = np.array(coor_structure_init_P)
        coor_origin_S = np.array(coor_structure_init_S)
        coor_origin_Cl = np.array(coor_structure_init_Cl)
    
        ## get the combined coordinate of the mapped Li with all other original elements
        for m in coor_mapped_Li:
            coor_combined.append(np.array(m))
        for n in coor_origin_P:
            coor_combined.append(np.array(n))
        for o in coor_origin_S:
            coor_combined.append(np.array(o))
        for p in coor_origin_Cl:
            coor_combined.append(np.array(p))
        
        coor_combined_array = [arr.tolist() for arr in coor_combined]

        ## getting the index
        amount_Li_temp = len(coor_reduced120_Li)
        amount_P_temp = len(coor_structure_init_P)
        amount_S_temp = len(coor_structure_init_S)
        amount_Cl_temp = len(coor_structure_init_Cl)

        # TO DO: write manually to change the line with Li{i}, where i = #of index based on the orientated positive
        idx_mapped_Li = np.arange(amount_Li_temp)
        idx_origin_P = np.arange(amount_P_temp) + amount_Li
        idx_origin_S = np.arange(amount_S_temp) + amount_Li + amount_P
        idx_origin_Cl = np.arange(amount_Cl_temp) + amount_Li + amount_P + amount_S

        idx_species_combined_idx0 = np.concatenate((idx_mapped_Li, idx_origin_P, idx_origin_S, idx_origin_Cl))
        idx_species_combined_idx0_int = idx_species_combined_idx0.astype(int).tolist()

        ## creating the structure file of combined elements
        selected_species_combined = [new_structure.species[i] for i in idx_species_combined_idx0_int]
        structure_combined = Structure(new_structure.lattice, selected_species_combined, coor_combined_array)
        cif_combined = CifWriter(structure_combined)
        source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename}.cif"
        source_filename_path = os.path.join(destination_directory, source_filename)
        cif_combined.write_file(source_filename_path)

def reindex_P_S_Cl(lines, idx_Li_start, idx_without_weirdos, idx_P_S_Cl_line_new_start, amount_Li, amount_P, amount_S, amount_Cl):
    old_text_P_S_Cl = lines[len(idx_without_weirdos) + idx_Li_start :]

    idx_P_S_Cl_line_new_end      = idx_P_S_Cl_line_new_start + len(old_text_P_S_Cl)
    lines[idx_P_S_Cl_line_new_start : idx_P_S_Cl_line_new_end] = old_text_P_S_Cl

    # re-write the index of P_S_Cl accordingly
    new_text_P_S_Cl = []
    for i in range(amount_P):
        idx_line_P = idx_P_S_Cl_line_new_start + i
        idx_P_new = amount_Li + i
        if lines[idx_line_P].strip().startswith("P"):
            new_label = f"P{idx_P_new}"
            # file_operations_instance = FileOperations()
            # modified_line = file_operations_instance.replace(lines[idx_line_P].split()[1], new_label)
            modified_line = lines[idx_line_P].replace(lines[idx_line_P].split()[1], new_label)
            new_text_P_S_Cl.append(modified_line)
    for i in range(amount_S):
        idx_line_S = idx_P_S_Cl_line_new_start + amount_P + i
        idx_S_new = amount_Li + amount_P + i
        if lines[idx_line_S].strip().startswith("S"):
            new_label = f"S{idx_S_new}"
            # file_operations_instance = FileOperations()
            # modified_line = file_operations_instance.replace(lines[idx_line_S].split()[1], new_label)            
            modified_line = lines[idx_line_S].replace(lines[idx_line_S].split()[1], new_label)
            new_text_P_S_Cl.append(modified_line)
    for i in range(amount_Cl):
        idx_line_Cl = idx_P_S_Cl_line_new_start + amount_P + amount_S + i
        idx_Cl_new = amount_Li + amount_P + amount_S + i
        if lines[idx_line_Cl].strip().startswith("Cl"):
            new_label = f"Cl{idx_Cl_new}"
            # file_operations_instance = FileOperations()
            # modified_line = file_operations_instance.replace(lines[idx_line_Cl].split()[1], new_label)     
            modified_line = lines[idx_line_Cl].replace(lines[idx_line_Cl].split()[1], new_label)
            new_text_P_S_Cl.append(modified_line)

    lines[idx_P_S_Cl_line_new_start : amount_P + amount_S + amount_Cl + idx_P_S_Cl_line_new_start] = new_text_P_S_Cl

    return lines


def get_idx_coor_limapped_weirdos_dict(dataframe, coor_structure_init_dict, activate_radius, el):
    coor_origin120_el_init = coor_structure_init_dict[el]

    col_idx_without_weirdos = "idx_without_weirdos"
    col_idx_coor_weirdos_el = f"idx_coor_weirdos_{el}"
    col_idx0_weirdos_Li = "idx0_weirdos_Li"
    col_sum_of_weirdos_Li = f"#weirdos_Li"
    if activate_radius == 2 or activate_radius == 3:
        #col_coor_reduced120_Li = "coor_reduced120_Li"
        col_coor_reduced120_Li = f"coor_reduced120_48htypesmerged_{el}"
        # col_coor_weirdos_48htypesmerged_Li = "coor_weirdos_48htypesmerged_Li"
        # col_coor_weirdos_el = f"coor_weirdos_48htype2_{el}"
        col_sum_sanitycheck_Li = "sum_sanitycheck_48htypesmerged_Li"
    elif activate_radius == 1:
        col_coor_reduced120_Li = f"coor_reduced120_{el}_closestduplicate"
        col_sum_sanitycheck_Li = f"sum_sanitycheck_{el}_closestduplicate"


    col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"
    col_sum_label_and_weirdo_flag = "#label_and_#weirdo_flag"
    col_amount_types_and_weirdo = "amount_types_and_weirdo"
    col_ratio_48htype1_Li = "ratio_48htype1_Li"
    col_ratio_48htype2_Li = "ratio_48htype2_Li"
    col_ratio_24g_Li = "ratio_24g_Li"
    col_ratio_weirdo_Li = "ratio_weirdo_Li"
    col_sum_amount = "sum_amount"
    col_idx_coor_limapped_weirdos_dict_init = "idx_coor_limapped_weirdos_dict_init"
    col_ndim_coor_reduced120_Li = "ndim_coor_reduced120_Li"
    col_ndim_coor_weirdos_el = "ndim_coor_weirdos_el"
    col_len_coor_weirdos_el = "len_coor_weirdos_el"
    col_len_coor_reduced120_Li = "len_coor_reduced120_Li"
    col_len_idx0_weirdos_Li = "len_idx0_weirdos_Li"
    col_len_idx_without_weirdos = "len_idx_without_weirdos"
    col_ndim_flag_coor = "ndim_flag_coor"

    dataframe[col_idx_coor_limapped_weirdos_dict] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_sum_label_and_weirdo_flag] = "False"
    dataframe[col_amount_types_and_weirdo] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_amount] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor_limapped_weirdos_dict_init] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_reduced120_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_coor_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_coor_reduced120_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_idx0_weirdos_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_idx_without_weirdos] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_flag_coor] = "False"

    coor_li48htype1_ref = coor_origin120_el_init[0:48]
    coor_li48htype2_ref = coor_origin120_el_init[48:96]
    coor_li24g_ref = coor_origin120_el_init[96:120]

    for idx in range(dataframe["geometry"].size):
        coor_limapped_weirdos = []
        idx0_limapped_weirdos = []
        idx_coor_limapped_weirdos_dict_init = {}
        idx_coor_limapped_weirdos_dict = {}

        idx_without_weirdos = dataframe.at[idx, col_idx_without_weirdos]
        coor_reduced120_Li = np.array(dataframe.at[idx, col_coor_reduced120_Li])
        idx_coor_weirdos_el = dataframe.at[idx, col_idx_coor_weirdos_el]
        # coor_weirdos_48htype2_el = np.array(dataframe.at[idx, col_coor_weirdos_48htype2_el])
        coor_weirdos_el = np.array(list(idx_coor_weirdos_el.values()))
        idx0_weirdos_Li = dataframe.at[idx, col_idx0_weirdos_Li]
        nr_of_weirdos_Li = dataframe.at[idx, col_sum_of_weirdos_Li]
        sum_sanitycheck_48htypesmerged_Li = dataframe.at[idx, col_sum_sanitycheck_Li]

        ndim_coor_reduced120_Li = coor_reduced120_Li.ndim
        ndim_coor_weirdos_el = coor_weirdos_el.ndim
        len_coor_weirdos_el = len(coor_weirdos_el)
        len_coor_reduced120_Li = len(coor_reduced120_Li)
        len_idx0_weirdos_Li = len(idx0_weirdos_Li)
        len_idx_without_weirdos = len(idx_without_weirdos)

        # # # # if coor_weirdos_el.ndim == 2:
        # # # if len(coor_weirdos_el) > 0:
        # # #     coor_limapped_weirdos = np.concatenate((coor_reduced120_Li, coor_weirdos_el), axis=0)
        # # # # elif coor_weirdos_el.ndim == 1:
        # # # elif len(coor_weirdos_el) == 0:
        # # #     coor_limapped_weirdos = np.array(coor_reduced120_Li.copy())
        # # # else:
        # # #     print(f"coor_weirdos_el has no correct dimension at idx: {idx}")
        # # #     # break

        # # # if len(idx0_weirdos_Li) > 0:
        # # #     idx0_limapped_weirdos = np.concatenate((idx_without_weirdos, idx0_weirdos_Li), axis=0)
        # # # elif len(idx0_weirdos_Li) == 0:
        # # #     idx0_limapped_weirdos = np.array(idx_without_weirdos.copy())
        # # # else:
        # # #     print(f"idx0_weirdos_Li has no correct len at idx: {idx}")
        # # #     # break
    
        # if coor_weirdos_el.ndim == 2:
        if ndim_coor_reduced120_Li == ndim_coor_weirdos_el & ndim_coor_weirdos_el == 2:
            coor_limapped_weirdos = np.concatenate((coor_reduced120_Li, coor_weirdos_el), axis=0)
            dataframe.at[idx, col_ndim_flag_coor] = "True"
        # elif coor_weirdos_el.ndim == 1:
        elif ndim_coor_weirdos_el == 1:
            coor_limapped_weirdos = np.array(coor_reduced120_Li.copy())
        elif ndim_coor_reduced120_Li == 1:
            coor_limapped_weirdos = np.array(coor_weirdos_el.copy())
        else:
            print(f"coor_weirdos_el or coor_reduced120_Li has no correct dimension at idx: {idx}")
            # break

        if len(idx0_weirdos_Li) > 0:
            idx0_limapped_weirdos = np.concatenate((idx_without_weirdos, idx0_weirdos_Li), axis=0)
        elif len(idx0_weirdos_Li) == 0:
            idx0_limapped_weirdos = np.array(idx_without_weirdos.copy())
        elif len(idx_without_weirdos) == 0:
            idx0_limapped_weirdos = np.array(idx0_weirdos_Li.copy())
        else:
            print(f"idx0_weirdos_Li or idx_without_weirdos has no correct len at idx: {idx}")
            # break

        idx_coor_limapped_weirdos_dict_init = dict(zip(idx0_limapped_weirdos, coor_limapped_weirdos))
        # if idx == 52:
        #     # print(f"type = {type(idx_coor_limapped_weirdos_dict_init)}")
        #     # print(f"idx_coor_limapped_weirdos_dict_init = {idx_coor_limapped_weirdos_dict_init}")

        #     coor_48htype1_Li = []; coor_48htype2_Li = []; coor_24g_Li = []; coor_weirdo_Li = []
        #     for key, value in idx_coor_limapped_weirdos_dict_init.items():
        #         print(f"key: {key}, value: {value}")
        #         if (value == coor_li48htype1_ref).all():

        coor_48htype1_Li = []; coor_48htype2_Li = []; coor_24g_Li = []; coor_weirdo_Li = []
        for key, value in idx_coor_limapped_weirdos_dict_init.items():
            idx_coor_limapped_weirdos_dict_val = {}

            idx_coor_limapped_weirdos_dict_val['coor'] = tuple(value)

            for idx_li48htype1_temp, coor_li48htype1_ref_temp in enumerate(coor_li48htype1_ref):
                if (value == coor_li48htype1_ref_temp).all():
                    idx_coor_limapped_weirdos_dict_val["label"] = "48htype1"
                    coor_48htype1_Li.append(np.array(list(value)))
                    if int(key) in idx_coor_limapped_weirdos_dict:
                        idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                    else:
                        idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val
            for idx_li48htype2_temp, coor_li48htype2_ref_temp in enumerate(coor_li48htype2_ref):
                if (value == coor_li48htype2_ref_temp).all():
                    idx_coor_limapped_weirdos_dict_val["label"] = "48htype2"
                    coor_48htype2_Li.append(np.array(list(value)))
                    if int(key) in idx_coor_limapped_weirdos_dict:
                        idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                    else:
                        idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val
            for idx_24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
                if (value == coor_li24g_ref_temp).all():
                    idx_coor_limapped_weirdos_dict_val["label"] = "24g"
                    coor_24g_Li.append(np.array(list(value)))
                    if int(key) in idx_coor_limapped_weirdos_dict:
                        idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                    else:
                        idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val
            for idx_weirdos_temp, coor_weirdos_ref_temp in enumerate(coor_weirdos_el):
                if (value == coor_weirdos_ref_temp).all():
                    idx_coor_limapped_weirdos_dict_val["label"] = "weirdos"
                    coor_weirdo_Li.append(np.array(list(value)))
                    if int(key) in idx_coor_limapped_weirdos_dict:
                        idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                    else:
                        idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val

            # if int(key) in idx_coor_limapped_weirdos_dict:
            #     idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
            # else:
            #     idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val

        # amount of each type
        amount_48htype1_Li = len(coor_48htype1_Li)
        amount_48htype2_Li = len(coor_48htype2_Li)
        amount_24g_Li = len(coor_24g_Li)
        amount_weirdo = len(coor_weirdo_Li)
        sum_amount = amount_48htype1_Li + amount_48htype2_Li + amount_24g_Li + amount_weirdo

        # sanity check for the amount
        # if amount_weirdo == nr_of_weirdos_Li & sum_amount == sum_sanitycheck_48htypesmerged_Li:
        # if int(amount_weirdo) == int(nr_of_weirdos_Li) & int(sum_amount) == int(sum_sanitycheck_48htypesmerged_Li):
        if int(amount_weirdo) == int(nr_of_weirdos_Li):
            if int(sum_amount) == int(sum_sanitycheck_48htypesmerged_Li):
                dataframe.at[idx, col_sum_label_and_weirdo_flag] = "True"

        amount_types_and_weirdo = f"48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 24g: {amount_24g_Li}, weirdo: {amount_weirdo}"

        # ratio_48htype1_Li = amount_48htype1_Li / sum_amount
        # ratio_48htype2_Li = amount_48htype2_Li / sum_amount
        # ratio_24g_Li = amount_24g_Li / sum_amount
        # ratio_weirdo_Li = amount_weirdo / sum_amount

        dataframe.at[idx, col_idx_coor_limapped_weirdos_dict] = idx_coor_limapped_weirdos_dict
        dataframe.at[idx, col_amount_types_and_weirdo] = amount_types_and_weirdo
        # dataframe.at[idx, col_ratio_48htype1_Li] = ratio_48htype1_Li
        # dataframe.at[idx, col_ratio_48htype2_Li] = ratio_48htype2_Li
        # dataframe.at[idx, col_ratio_24g_Li] = ratio_24g_Li
        # dataframe.at[idx, col_ratio_weirdo_Li] = ratio_weirdo_Li
        dataframe.at[idx, col_sum_amount] = sum_amount
        dataframe.at[idx, col_idx_coor_limapped_weirdos_dict_init] = idx_coor_limapped_weirdos_dict_init
        dataframe.at[idx, col_ndim_coor_reduced120_Li] = ndim_coor_reduced120_Li
        dataframe.at[idx, col_ndim_coor_weirdos_el] = ndim_coor_weirdos_el
        dataframe.at[idx, col_len_coor_weirdos_el] = len_coor_weirdos_el
        dataframe.at[idx, col_len_coor_reduced120_Li] = len_coor_reduced120_Li
        dataframe.at[idx, col_len_idx0_weirdos_Li] = len_idx0_weirdos_Li
        dataframe.at[idx, col_len_idx_without_weirdos] = len_idx_without_weirdos


def get_idx_coor_limapped_weirdos_dict_litype(dataframe, coor_structure_init_dict, activate_radius, litype, el):
    coor_origin120_el_init = coor_structure_init_dict[el]

    col_idx_without_weirdos = "idx_without_weirdos"
    col_idx_coor_weirdos_el = f"idx_coor_weirdos_{el}"
    col_idx0_weirdos_Li = "idx0_weirdos_Li"
    col_sum_of_weirdos_Li = f"#weirdos_Li"
    if activate_radius == 2 or activate_radius == 3:
        #col_coor_reduced120_Li = "coor_reduced120_Li"
        col_coor_reduced120_Li = f"coor_reduced120_48htypesmerged_{el}"
        # col_coor_weirdos_48htypesmerged_Li = "coor_weirdos_48htypesmerged_Li"
        # col_coor_weirdos_el = f"coor_weirdos_48htype2_{el}"
        col_sum_sanitycheck_Li = "sum_sanitycheck_48htypesmerged_Li"
    elif activate_radius == 1:
        col_coor_reduced120_Li = f"coor_reduced120_{el}_closestduplicate"
        col_sum_sanitycheck_Li = f"sum_sanitycheck_{el}_closestduplicate"


    col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"
    col_sum_label_and_weirdo_flag = "#label_and_#weirdo_flag"
    col_amount_types_and_weirdo = "amount_types_and_weirdo"
    col_ratio_48htype1_Li = "ratio_48htype1_Li"
    col_ratio_48htype2_Li = "ratio_48htype2_Li"
    col_ratio_24g_Li = "ratio_24g_Li"
    col_ratio_weirdo_Li = "ratio_weirdo_Li"
    col_sum_amount = "sum_amount"
    col_idx_coor_limapped_weirdos_dict_init = "idx_coor_limapped_weirdos_dict_init"
    col_ndim_coor_reduced120_Li = "ndim_coor_reduced120_Li"
    col_ndim_coor_weirdos_el = "ndim_coor_weirdos_el"
    col_len_coor_weirdos_el = "len_coor_weirdos_el"
    col_len_coor_reduced120_Li = "len_coor_reduced120_Li"
    col_len_idx0_weirdos_Li = "len_idx0_weirdos_Li"
    col_len_idx_without_weirdos = "len_idx_without_weirdos"
    col_ndim_flag_coor = "ndim_flag_coor"

    dataframe[col_idx_coor_limapped_weirdos_dict] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_sum_label_and_weirdo_flag] = "False"
    dataframe[col_amount_types_and_weirdo] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_amount] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor_limapped_weirdos_dict_init] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_reduced120_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_coor_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_coor_reduced120_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_idx0_weirdos_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_idx_without_weirdos] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_flag_coor] = "False"

    coor_li24g_ref      = coor_origin120_el_init[0:24]
    if litype == 1:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
    elif litype == 2:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
    elif litype == 3:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
    elif litype == 4:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
    elif litype == 5:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
    elif litype == 6:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
    elif litype == 7:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
        coor_li48htype7_ref = coor_origin120_el_init[312:360]
    elif litype == 8:
        coor_li48htype1_ref = coor_origin120_el_init[24:72]
        coor_li48htype2_ref = coor_origin120_el_init[72:120]
        coor_li48htype3_ref = coor_origin120_el_init[120:168]
        coor_li48htype4_ref = coor_origin120_el_init[168:216]
        coor_li48htype5_ref = coor_origin120_el_init[216:264]
        coor_li48htype6_ref = coor_origin120_el_init[264:312]
        coor_li48htype7_ref = coor_origin120_el_init[312:360]
        coor_li48htype8_ref = coor_origin120_el_init[360:408]

    for idx in range(dataframe["geometry"].size):
        coor_limapped_weirdos = []
        idx0_limapped_weirdos = []
        idx_coor_limapped_weirdos_dict_init = {}
        idx_coor_limapped_weirdos_dict = {}

        idx_without_weirdos = dataframe.at[idx, col_idx_without_weirdos]
        coor_reduced120_Li = np.array(dataframe.at[idx, col_coor_reduced120_Li])
        idx_coor_weirdos_el = dataframe.at[idx, col_idx_coor_weirdos_el]
        # coor_weirdos_48htype2_el = np.array(dataframe.at[idx, col_coor_weirdos_48htype2_el])
        coor_weirdos_el = np.array(list(idx_coor_weirdos_el.values()))
        idx0_weirdos_Li = dataframe.at[idx, col_idx0_weirdos_Li]
        nr_of_weirdos_Li = dataframe.at[idx, col_sum_of_weirdos_Li]
        sum_sanitycheck_48htypesmerged_Li = dataframe.at[idx, col_sum_sanitycheck_Li]

        ndim_coor_reduced120_Li = coor_reduced120_Li.ndim
        ndim_coor_weirdos_el = coor_weirdos_el.ndim
        len_coor_weirdos_el = len(coor_weirdos_el)
        len_coor_reduced120_Li = len(coor_reduced120_Li)
        len_idx0_weirdos_Li = len(idx0_weirdos_Li)
        len_idx_without_weirdos = len(idx_without_weirdos)

        # if coor_weirdos_el.ndim == 2:
        if ndim_coor_reduced120_Li == ndim_coor_weirdos_el & ndim_coor_weirdos_el == 2:
            coor_limapped_weirdos = np.concatenate((coor_reduced120_Li, coor_weirdos_el), axis=0)
            dataframe.at[idx, col_ndim_flag_coor] = "True"
        # elif coor_weirdos_el.ndim == 1:
        elif ndim_coor_weirdos_el == 1:
            coor_limapped_weirdos = np.array(coor_reduced120_Li.copy())
        elif ndim_coor_reduced120_Li == 1:
            coor_limapped_weirdos = np.array(coor_weirdos_el.copy())
        else:
            print(f"coor_weirdos_el or coor_reduced120_Li has no correct dimension at idx: {idx}")
            # break

        if len(idx0_weirdos_Li) > 0:
            idx0_limapped_weirdos = np.concatenate((idx_without_weirdos, idx0_weirdos_Li), axis=0)
        elif len(idx0_weirdos_Li) == 0:
            idx0_limapped_weirdos = np.array(idx_without_weirdos.copy())
        elif len(idx_without_weirdos) == 0:
            idx0_limapped_weirdos = np.array(idx0_weirdos_Li.copy())
        else:
            print(f"idx0_weirdos_Li or idx_without_weirdos has no correct len at idx: {idx}")
            # break

        idx_coor_limapped_weirdos_dict_init = dict(zip(idx0_limapped_weirdos, coor_limapped_weirdos))

        coor_24g_Li = []; coor_weirdo_Li = []
        coor_48htype1_Li = []; coor_48htype2_Li = []; coor_48htype3_Li = []; coor_48htype4_Li = []; coor_48htype5_Li = []; coor_48htype6_Li = []; coor_48htype7_Li = []; coor_48htype8_Li = []   
        
        for key, value in idx_coor_limapped_weirdos_dict_init.items():
            idx_coor_limapped_weirdos_dict_val = {}

            idx_coor_limapped_weirdos_dict_val['coor'] = tuple(value)

            for idx_24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
                if (value == coor_li24g_ref_temp).all():
                    idx_coor_limapped_weirdos_dict_val["label"] = "24g"
                    coor_24g_Li.append(np.array(list(value)))
                    if int(key) in idx_coor_limapped_weirdos_dict:
                        idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                    else:
                        idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val
            for idx_weirdos_temp, coor_weirdos_ref_temp in enumerate(coor_weirdos_el):
                if (value == coor_weirdos_ref_temp).all():
                    idx_coor_limapped_weirdos_dict_val["label"] = "weirdos"
                    coor_weirdo_Li.append(np.array(list(value)))
                    if int(key) in idx_coor_limapped_weirdos_dict:
                        idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                    else:
                        idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val

            for i in range(1, litype+1):
                coor_li48htype_ref = locals()[f"coor_li48htype{i}_ref"]
                label = f"48htype{i}"

                for idx_temp, coor_ref_temp in enumerate(coor_li48htype_ref):
                    if (value == coor_ref_temp).all():
                        idx_coor_limapped_weirdos_dict_val["label"] = label
                        locals()[f"coor_48htype{i}_Li"].append(np.array(list(value)))
                        if int(key) in idx_coor_limapped_weirdos_dict:
                            idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                        else:
                            idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val
                    

            # if int(key) in idx_coor_limapped_weirdos_dict:
            #     idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
            # else:
            #     idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val

        # amount of each type
        amount_24g_Li = len(coor_24g_Li)
        amount_weirdo = len(coor_weirdo_Li)
        if litype == 0:
            sum_amount = amount_24g_Li + amount_weirdo
        elif litype == 1:
            amount_48htype1_Li = len(coor_48htype1_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li
        elif litype == 2:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li
        elif litype == 3:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li
        elif litype == 4:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li
        elif litype == 5:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            amount_48htype5_Li = len(coor_48htype5_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li + amount_48htype5_Li
        elif litype == 6:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            amount_48htype5_Li = len(coor_48htype5_Li)
            amount_48htype6_Li = len(coor_48htype6_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li + amount_48htype5_Li + amount_48htype6_Li
        elif litype == 7:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            amount_48htype5_Li = len(coor_48htype5_Li)
            amount_48htype6_Li = len(coor_48htype6_Li)
            amount_48htype7_Li = len(coor_48htype7_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li + amount_48htype5_Li + amount_48htype6_Li + amount_48htype7_Li
        elif litype == 8:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            amount_48htype5_Li = len(coor_48htype5_Li)
            amount_48htype6_Li = len(coor_48htype6_Li)
            amount_48htype7_Li = len(coor_48htype7_Li)
            amount_48htype8_Li = len(coor_48htype8_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li + amount_48htype5_Li + amount_48htype6_Li + amount_48htype7_Li + amount_48htype8_Li


        # sanity check for the amount
        # if amount_weirdo == nr_of_weirdos_Li & sum_amount == sum_sanitycheck_48htypesmerged_Li:
        # if int(amount_weirdo) == int(nr_of_weirdos_Li) & int(sum_amount) == int(sum_sanitycheck_48htypesmerged_Li):
        if int(amount_weirdo) == int(nr_of_weirdos_Li):
            if int(sum_amount) == int(sum_sanitycheck_48htypesmerged_Li):
                dataframe.at[idx, col_sum_label_and_weirdo_flag] = "True"

        if litype == 0:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}"
        elif litype == 1:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}"
        elif litype == 2:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}"
        elif litype == 3:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}"
        elif litype == 4:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}"
        elif litype == 5:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}, 48htype5: {amount_48htype5_Li}"
        elif litype == 6:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}, 48htype5: {amount_48htype5_Li}, 48htype6: {amount_48htype6_Li}"
        elif litype == 7:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}, 48htype5: {amount_48htype5_Li}, 48htype6: {amount_48htype6_Li}, 48htype7: {amount_48htype7_Li}"
        elif litype == 8:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}, 48htype5: {amount_48htype5_Li}, 48htype6: {amount_48htype6_Li}, 48htype7: {amount_48htype7_Li}, 48htype8: {amount_48htype8_Li}"

        # ratio_48htype1_Li = amount_48htype1_Li / sum_amount
        # ratio_48htype2_Li = amount_48htype2_Li / sum_amount
        # ratio_24g_Li = amount_24g_Li / sum_amount
        # ratio_weirdo_Li = amount_weirdo / sum_amount

        dataframe.at[idx, col_idx_coor_limapped_weirdos_dict] = idx_coor_limapped_weirdos_dict
        dataframe.at[idx, col_amount_types_and_weirdo] = amount_types_and_weirdo
        # dataframe.at[idx, col_ratio_48htype1_Li] = ratio_48htype1_Li
        # dataframe.at[idx, col_ratio_48htype2_Li] = ratio_48htype2_Li
        # dataframe.at[idx, col_ratio_24g_Li] = ratio_24g_Li
        # dataframe.at[idx, col_ratio_weirdo_Li] = ratio_weirdo_Li
        dataframe.at[idx, col_sum_amount] = sum_amount
        dataframe.at[idx, col_idx_coor_limapped_weirdos_dict_init] = idx_coor_limapped_weirdos_dict_init
        dataframe.at[idx, col_ndim_coor_reduced120_Li] = ndim_coor_reduced120_Li
        dataframe.at[idx, col_ndim_coor_weirdos_el] = ndim_coor_weirdos_el
        dataframe.at[idx, col_len_coor_weirdos_el] = len_coor_weirdos_el
        dataframe.at[idx, col_len_coor_reduced120_Li] = len_coor_reduced120_Li
        dataframe.at[idx, col_len_idx0_weirdos_Li] = len_idx0_weirdos_Li
        dataframe.at[idx, col_len_idx_without_weirdos] = len_idx_without_weirdos



def rewrite_cif_w_correct_Li_idx(dataframe, destination_directory, amount_Li, amount_P, amount_S, amount_Cl, var_savefilename_init, var_savefilename_new):
    col_idx_without_weirdos = "idx_without_weirdos"

    dataframe[col_idx_without_weirdos] = [np.array([]) for _ in range(len(dataframe.index))]
    
    search_string = "Li  Li0"

    for idx in range(dataframe["geometry"].size):
        idx0_weirdos_Li = dataframe["idx0_weirdos_Li"][idx]
        source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_init}.cif"
        source_filename_path = os.path.join(destination_directory, source_filename)

        source_filename_filtered = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_new}.cif"
        destination_path_combined_new = os.path.join(destination_directory, source_filename_filtered)

        with open(source_filename_path, "r") as f:
            lines = f.readlines()

        for idx_line, line in enumerate(lines):
            if search_string in line:
                idx_Li_start = idx_line
                break

        idx_without_weirdos = [i for i in range(amount_Li) if i not in idx0_weirdos_Li]

        new_text = []
        for i in range(len(idx_without_weirdos)):
            idx_line = idx_Li_start + i
            if lines[idx_line].strip().startswith("Li"):
                new_label = f"Li{idx_without_weirdos[i]}"
                # file_operations_instance = FileOperations()
                # modified_line = lines[idx_line].file_operations_instance.replace(lines[idx_line].split()[1], new_label)     
                modified_line = lines[idx_line].replace(lines[idx_line].split()[1], new_label)
                new_text.append(modified_line)
                
        lines[idx_Li_start : len(idx_without_weirdos) + idx_Li_start] = new_text

        # idx_weirdo_line_start   = len(idx_without_weirdos) + idx_Li_start
        # idx_weirdo_line_end     = idx_weirdo_line_start + len(idx0_weirdos_Li)
        # lines[idx_weirdo_line_start : idx_weirdo_line_end] = weirdos_text

        idx_P_S_Cl_line_new_start    = len(idx_without_weirdos) + idx_Li_start
        reindex_P_S_Cl(lines, idx_Li_start, idx_without_weirdos, idx_P_S_Cl_line_new_start, amount_Li, amount_P, amount_S, amount_Cl)

        dataframe.at[idx, col_idx_without_weirdos] = idx_without_weirdos

        with open(destination_path_combined_new, "w") as f:
            f.write("\n".join(line.strip() for line in lines))


# new needed variables: amount_P, amount_S, amount_Cl
def rewrite_cif_w_correct_Li_idx_weirdos_appended(dataframe, destination_directory, amount_Li, amount_P, amount_S, amount_Cl, activate_radius, var_savefilename_init, var_savefilename_new):
    if activate_radius == 2 or activate_radius == 3:
        col_coor_weirdos_el = f"coor_weirdos_48htype2_Li"
    elif activate_radius == 1:
        col_coor_weirdos_el = f"coor_weirdos_Li"

    # col_subdir_cif_w_correct_Li_idx_weirdos_appended = "subdir_cif_w_correct_Li_idx_weirdos_appended"

    # dataframe[col_subdir_cif_w_correct_Li_idx_weirdos_appended] = None
    
    search_string = "Li  Li0"

    for idx in range(dataframe["geometry"].size):
        idx0_weirdos_Li = dataframe["idx0_weirdos_Li"][idx]
        source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_init}.cif"
        source_filename_path = os.path.join(destination_directory, source_filename)

        source_filename_filtered = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_new}.cif"
        destination_path_combined_new = os.path.join(destination_directory, source_filename_filtered)

        with open(source_filename_path, "r") as f:
            lines = f.readlines()

        for idx_line, line in enumerate(lines):
            if search_string in line:
                idx_Li_start = idx_line
                break

        idx_without_weirdos = [i for i in range(amount_Li) if i not in idx0_weirdos_Li]

        new_text = []
        for i in range(len(idx_without_weirdos)):
            idx_line = idx_Li_start + i
            if lines[idx_line].strip().startswith("Li"):
                new_label = f"Li{idx_without_weirdos[i]}"
                # file_operations_instance = FileOperations()
                # modified_line = file_operations_instance.replace(lines[idx_line].split()[1], new_label)     
                modified_line = lines[idx_line].replace(lines[idx_line].split()[1], new_label)
                new_text.append(modified_line)

        lines[idx_Li_start : len(idx_without_weirdos) + idx_Li_start] = new_text

        # now appending weirdos below existing lines of Li
        coor_weirdos = dataframe[col_coor_weirdos_el][idx] # coor_weirdos = dataframe["coor_weirdos_Li"][idx]

        weirdos_text = []
        for i in range(len(idx0_weirdos_Li)):
            coor_weirdo_x = coor_weirdos[i][0]
            coor_weirdo_y = coor_weirdos[i][1]
            coor_weirdo_z = coor_weirdos[i][2]
            idx_weirdo = idx0_weirdos_Li[i]
            new_line_weirdo = f"Li  Li{idx_weirdo}  1  {coor_weirdo_x:.8f}  {coor_weirdo_y:.8f}  {coor_weirdo_z:.8f}  1"  # manually created
            weirdos_text.append(new_line_weirdo)
            # idx_line_weirdos = idx_Li_start + len(idx_without_weirdos)
            
        old_text_P_S_Cl = lines[len(idx_without_weirdos) + idx_Li_start :]

        idx_weirdo_line_start   = len(idx_without_weirdos) + idx_Li_start
        idx_weirdo_line_end     = idx_weirdo_line_start + len(idx0_weirdos_Li)
        lines[idx_weirdo_line_start : idx_weirdo_line_end] = weirdos_text

        idx_P_S_Cl_line_new_start    = idx_weirdo_line_end

        # !!!: for the moment not using the function because P is gone
        # reindex_P_S_Cl(lines, idx_Li_start, idx_without_weirdos, idx_P_S_Cl_line_new_start, amount_P, amount_S, amount_Cl)
        
        idx_P_S_Cl_line_new_end      = idx_P_S_Cl_line_new_start + len(old_text_P_S_Cl)
        lines[idx_P_S_Cl_line_new_start : idx_P_S_Cl_line_new_end] = old_text_P_S_Cl
 
        # re-write the index of P_S_Cl accordingly
        new_text_P_S_Cl = []
        for i in range(amount_P):
            idx_line_P = idx_P_S_Cl_line_new_start + i
            idx_P_new = amount_Li + i
            if lines[idx_line_P].strip().startswith("P"):
                new_label = f"P{idx_P_new}"
                # file_operations_instance = FileOperations()
                # modified_line = file_operations_instance.replace(lines[idx_line_P].split()[1], new_label)     
                modified_line = lines[idx_line_P].replace(lines[idx_line_P].split()[1], new_label)
                new_text_P_S_Cl.append(modified_line)
        for i in range(amount_S):
            idx_line_S = idx_P_S_Cl_line_new_start + amount_P + i
            idx_S_new = amount_Li + amount_P + i
            if lines[idx_line_S].strip().startswith("S"):
                new_label = f"S{idx_S_new}"
                # file_operations_instance = FileOperations()
                # modified_line = file_operations_instance.replace(lines[idx_line_S].split()[1], new_label)     
                modified_line = lines[idx_line_S].replace(lines[idx_line_S].split()[1], new_label)
                new_text_P_S_Cl.append(modified_line)
        for i in range(amount_Cl):
            idx_line_Cl = idx_P_S_Cl_line_new_start + amount_P + amount_S + i
            idx_Cl_new = amount_Li + amount_P + amount_S + i
            if lines[idx_line_Cl].strip().startswith("Cl"):
                new_label = f"Cl{idx_Cl_new}"
                # file_operations_instance = FileOperations()
                # modified_line = file_operations_instance.replace(lines[idx_line_Cl].split()[1], new_label)     
                modified_line = lines[idx_line_Cl].replace(lines[idx_line_Cl].split()[1], new_label)
                new_text_P_S_Cl.append(modified_line)

        lines[idx_P_S_Cl_line_new_start : amount_P + amount_S + amount_Cl + idx_P_S_Cl_line_new_start] = new_text_P_S_Cl

        # dataframe.at[idx, col_subdir_cif_w_correct_Li_idx_weirdos_appended] = destination_path_combined_new

        # Write the modified lines back to the file
        with open(destination_path_combined_new, "w") as f:
            f.write("\n".join(line.strip() for line in lines))



def format_spacing_cif(dataframe, destination_directory, var_savefilename_init, var_savefilename_new):
    for idx in range(dataframe["geometry"].size):
        source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_init}.cif"
        source_filename_path = os.path.join(destination_directory, source_filename)

        source_filename_filtered = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_new}.cif"
        destination_path_combined_new = os.path.join(destination_directory, source_filename_filtered)

        # Read the input file and split it into lines
        with open(source_filename_path, "r") as f:
            lines = f.readlines()

        # Initialize variables to store the indices of loop_ occurrences
        loop_indices = []

        # Find the indices of the loop_ occurrences
        for i, line in enumerate(lines):
            if line.strip() == "loop_":
                loop_indices.append(i)

        # add last index of lines
        loop_indices.append(len(lines))

        # mostly hardcoded
        for i in range(len(loop_indices) - 1):
            i1 = loop_indices[i] + 1
            i2 = loop_indices[i+1] - 1
            for j in range(i1, i2 + 1):
                if lines[j].strip() and not lines[j].startswith(" "):
                    lines[j] = " " + lines[j]
                    if lines[j].strip() == "1  'x, y, z'":
                        lines[j] = " " + lines[j]
                    if lines[j].strip().startswith("Li"):
                        lines[j] = " " + lines[j]
                    if lines[j].strip().startswith("P"):
                        lines[j] = " " + lines[j]
                    if lines[j].strip().startswith("S"):
                        lines[j] = " " + lines[j]
                    if lines[j].strip().startswith("Cl"):
                        lines[j] = " " + lines[j]

        # Write the modified lines back to the file
        with open(destination_path_combined_new, "w") as f:
            f.write("".join(lines))


def create_cif_pymatgen(dataframe, destination_directory, file_restructure, var_name):
    for idx in range(dataframe["geometry"].size):
        source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{file_restructure}"
        source_filename_path = os.path.join(destination_directory, source_filename)

        output_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        output_filename_path = os.path.join(destination_directory, output_filename)

        structure = Structure.from_file(source_filename_path)
        frac_coor = structure.frac_coords
        cif_structure = Structure(structure.lattice, structure.species, frac_coor)
        cif = CifWriter(cif_structure)
        cif.write_file(output_filename_path)


def ascending_Li(dataframe, destination_directory, var_filename_init, var_savefilename_new):
    search_string = "  Li  "

    for idx in range(dataframe["geometry"].size):
        source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename_init}.cif"
        source_filename_path = os.path.join(destination_directory, source_filename)

        filename_new = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_new}.cif"
        filename_path_new = os.path.join(destination_directory, filename_new)

        with open(source_filename_path, "r") as f:
            lines = f.readlines()

        # Find the index where the "Li" atom data starts
        for idx, line in enumerate(lines):
            if search_string in line:
                idx_Li_start = idx
                break

        # Extract and sort the "  Li  " lines based on their indices
        li_lines = [line.strip() for line in lines[idx_Li_start :] if line.strip().startswith("Li")]
        sorted_li_lines = sorted(li_lines, key=lambda line: int(line.split()[1][2:]))

        # replace the original "Li" lines with the sorted lines
        lines[idx_Li_start :idx_Li_start  + len(sorted_li_lines)] = sorted_li_lines

        # Write the modified lines back to the file
        with open(filename_path_new, "w") as f:
            f.write("\n".join(line.strip() for line in lines))


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def diagonalizing_latticeconstantsmatrix(dataframe, destination_directory, latticeconstantsmatrix_line_nr_start, var_name_in, var_name_out, n_decimal):
    # # dataframe['subdir_orientated_positive_poscar'] = None
    for idx in range(dataframe["geometry"].size):
        filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_POSCAR_{var_name_in}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)

        with open(filename_to_transform_path, 'r') as file:
            lines = file.readlines()
        data = lines[latticeconstantsmatrix_line_nr_start:latticeconstantsmatrix_line_nr_start+3]

        df = pd.DataFrame([string.strip().split() for string in data]).astype(float)

        df_diagonal = np.diag(df.values)

        max_val = 0
        for i in df_diagonal:
            if i > max_val:
                max_val = i

        min_val = float('inf')
        for j in df_diagonal:
            if j < min_val:
                min_val = j

        # replace all diagonal of df with the max_val and let off-diagonal as 0
        for row in range(len(df)):
            for col in range(len(df)):
                if row == col:
                    if col == 0:
                        df[row][col] = '{:.{width}f}'.format(float(max_val), width=n_decimal)
                    else:
                        df[row][col] = '  ' + '{:.{width}f}'.format(float(max_val), width=n_decimal)
                else:
                    if col == 0: 
                        df[row][col] = '{:.{width}f}'.format(float(0), width=n_decimal)
                    else:
                        df[row][col] = '  ' + '{:.{width}f}'.format(float(0), width=n_decimal)

        row_list = df.to_string(index=False, header=False).split('\n')
        # row_list_space = ['  '.join(string.split()) for string in row_list] # 2 spaces of distance
        row_list_w_beginning = [' ' + row for row in row_list]       # 1 space in the beginning
        absolute_correct_list = '\n'.join(row_list_w_beginning).splitlines()        

        line_append_list = []
        for idx_c, line in enumerate(absolute_correct_list):
            line_new_line = str(line) + '\n'
            line_append_list.append(line_new_line)

        file_list = lines[:latticeconstantsmatrix_line_nr_start] + line_append_list + lines[latticeconstantsmatrix_line_nr_start+3:]

        poscar_filename_positive = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_POSCAR_{var_name_out}"
        destination_path = os.path.join(destination_directory, poscar_filename_positive)
        
        with open(destination_path, "w") as poscar_positive_file:
            for item in file_list:
                poscar_positive_file.writelines(item)

        # # dataframe['subdir_orientated_positive_poscar'][idx] = destination_path


def get_latticeconstant_structure_dict_iterated(dataframe, destination_directory, proceed_XDATCAR, var_filename):
    col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"
    col_latticeconstant_structure_dict_flag = f"latticeconstant_structure_dict_{var_filename}_flag"

    dataframe[col_latticeconstant_structure_dict] = None
    dataframe[col_latticeconstant_structure_dict_flag] = "False"

    for idx in range(dataframe["geometry"].size):
        latticeconstant_structure_dict = {}

        if var_filename == "CONTCAR":
            source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}"
        else:
            source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}.cif"
            
        source_filename_path = os.path.join(destination_directory, source_filename)

        new_structure = Structure.from_file(source_filename_path)

        a, b, c = new_structure.lattice.abc
        alpha, beta, gamma = new_structure.lattice.angles

        latticeconstant_structure_dict["a"] = a
        latticeconstant_structure_dict["b"] = b
        latticeconstant_structure_dict["c"] = c

        latticeconstant_structure_dict["alpha"] = alpha
        latticeconstant_structure_dict["beta"] = beta
        latticeconstant_structure_dict["gamma"] = gamma

        if a == b == c:
            if alpha == beta == gamma:
                if alpha == 90:
                    latticeconstant_structure_dict_flag = "True"
        else:
            if proceed_XDATCAR == "True":
                latticeconstant_structure_dict_flag = "False"


        dataframe.at[idx, col_latticeconstant_structure_dict] = latticeconstant_structure_dict
        dataframe.at[idx, col_latticeconstant_structure_dict_flag] = latticeconstant_structure_dict_flag


def plot_energy_vs_latticeconstant(dataframe, var_filename):
    col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"
    col_toten = "toten [eV]"

    for idx in range(dataframe["geometry"].size):
        latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict]
        toten = dataframe.at[idx, col_toten]
        
        a = latticeconstant_structure_dict["a"]

        plt.scatter(a, toten)
    
    plt.title("Lattice constant vs Total energy")
    plt.xlabel("Lattice constant [Å]")
    plt.ylabel("Total energy [eV]")
    plt.show()
        

# def get_fractional_to_cartesian_matrix(a, b, c, alpha, beta, gamma, angle_in_degrees=True):
def get_fractional_to_cartesian_matrix(dataframe, var_filename, angle_in_degrees=True): 
    # source: https://gist.github.com/Bismarrck/a68da01f19b39320f78a
    col_fractional_to_cartesian_matrix = f"fractional_to_cartesian_matrix_{var_filename}"

    col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"

    dataframe[col_fractional_to_cartesian_matrix] = None

    for idx in range(dataframe["geometry"].size):
        latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict] 

        a = latticeconstant_structure_dict["a"]
        b = latticeconstant_structure_dict["b"]
        c = latticeconstant_structure_dict["c"]

        alpha = latticeconstant_structure_dict["alpha"]
        beta = latticeconstant_structure_dict["beta"]
        gamma = latticeconstant_structure_dict["gamma"]
        
        if angle_in_degrees:
            alpha = np.deg2rad(alpha)
            beta = np.deg2rad(beta)
            gamma = np.deg2rad(gamma)

        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        cosg = np.cos(gamma)
        sing = np.sin(gamma)
        volume = 1.0 - cosa**2.0 - cosb**2.0 - cosg**2.0 + 2.0 * cosa * cosb * cosg
        volume = np.sqrt(volume)
        r = np.zeros((3, 3))
        r[0, 0] = a
        r[0, 1] = b * cosg
        r[0, 2] = c * cosb
        r[1, 1] = b * sing
        r[1, 2] = c * (cosa - cosb * cosg) / sing
        r[2, 2] = c * volume / sing

        # print(f"type_a_alpha_r: {type(a), type(alpha), type(r)}")
        # print(f"a_alpha_r: {a, alpha, r}")
        dataframe.at[idx, col_fractional_to_cartesian_matrix] = r


def get_fractional_to_cartesian_coor(dataframe, destination_directory, var_filename):
    col_coor_structure_dict_cartesian = f"coor_structure_dict_cartesian_{var_filename}"

    col_fractional_to_cartesian_matrix = f"fractional_to_cartesian_matrix_{var_filename}"

    dataframe[col_coor_structure_dict_cartesian] = None

    for idx in range(dataframe["geometry"].size):
        coor_origin_Li_init_cartesian = []; coor_origin_P_init_cartesian = []; coor_origin_S_init_cartesian = []; coor_origin_Cl_init_cartesian = []
        coor_structure_dict_cartesian = {}

        if var_filename == "CONTCAR":
            source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}"
        else:
            source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}.cif"
        source_filename_path = os.path.join(destination_directory, source_filename)

        new_structure = Structure.from_file(source_filename_path)

        r = dataframe.at[idx, col_fractional_to_cartesian_matrix]

        for idx24, coor24 in enumerate(new_structure):
            if coor24.species_string == "Li":
                coor_origin_Li_init_cartesian.append(np.dot(coor24.frac_coords, r.T)) 
            if coor24.species_string == "P":
                coor_origin_P_init_cartesian.append(np.dot(coor24.frac_coords, r.T))
            if coor24.species_string == "S":
                coor_origin_S_init_cartesian.append(np.dot(coor24.frac_coords, r.T))  
            if coor24.species_string == "Cl":
                coor_origin_Cl_init_cartesian.append(np.dot(coor24.frac_coords, r.T)) 
           
        coor_structure_dict_cartesian["Li"] = coor_origin_Li_init_cartesian
        coor_structure_dict_cartesian["P"] = coor_origin_P_init_cartesian
        coor_structure_dict_cartesian["S"] = coor_origin_S_init_cartesian
        coor_structure_dict_cartesian["Cl"] = coor_origin_Cl_init_cartesian
    
        dataframe.at[idx, col_coor_structure_dict_cartesian] = coor_structure_dict_cartesian



##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def get_closest_neighbors_el_cartesian_coor(dataframe, max_neighbors_radius, el, var_filename):
    col_distance_el = f"distance_cartesian_{var_filename}_{el}"
    col_closest_neighbors_w_dist_el = f"closest_neighbors_w_dist_{var_filename}_{el}"

    col_coor_structure_dict_cartesian = f"coor_structure_dict_cartesian_{var_filename}"
    col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"

    dataframe[col_distance_el] = None
    dataframe[col_closest_neighbors_w_dist_el] = None
    
    for idx in range(dataframe["geometry"].size):

        distance_el = {} 
        closest_neighbors_w_dist_el = {}

        coor_cartesion_el = dataframe.at[idx, col_coor_structure_dict_cartesian][el]

        latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict] 

        a = latticeconstant_structure_dict["a"]
        b = latticeconstant_structure_dict["b"]
        c = latticeconstant_structure_dict["c"]

        for idx24_temp1, coor24_temp1 in enumerate(coor_cartesion_el):        
            closest_neighbors_w_dist_dict = {}
            distance_array = []

            for idx24_temp2, coor24_temp2 in enumerate(coor_cartesion_el):
                distance = mic_eucledian_distance_cartesian(coor24_temp1, coor24_temp2, a, b, c)
                
                if distance < max_neighbors_radius:
                    distance_array.append(distance)

                    closest_neighbors_w_dist_dict['neighbor'] = tuple(coor24_temp2)
                    closest_neighbors_w_dist_dict['dist'] = distance

                    # Get the list of neighbors for the current coordinate, or create one if it doesn't exist
                    neighbors_list = closest_neighbors_w_dist_el.setdefault(tuple(coor24_temp1), [])
                    neighbors_list.append(closest_neighbors_w_dist_dict)

                    # if tuple(coor24_temp1) in closest_neighbors_w_dist_el:
                    #     closest_neighbors_w_dist_el[tuple(coor24_temp1)].append(closest_neighbors_w_dist_dict)
                    # else:
                    #     closest_neighbors_w_dist_el[tuple(coor24_temp1)] = closest_neighbors_w_dist_dict
            
            distance_array_sorted = sorted(set(distance_array))
            if tuple(coor24_temp1) in distance_el:
                distance_el[tuple(coor24_temp1)].append(distance_array_sorted)
            else:
                distance_el[tuple(coor24_temp1)] = distance_array_sorted        

        dataframe.at[idx, col_distance_el] = distance_el
        dataframe.at[idx, col_closest_neighbors_w_dist_el] = closest_neighbors_w_dist_el


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def get_orientation(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation):
    if orientation == "True":
        file_loc_ori_notdeleted = file_loc.copy()

        file_loc_mask_1 = file_loc_ori_notdeleted.loc[file_loc_ori_notdeleted['p_s_mask'].apply(lambda x: x != 0)]
        file_loc_mask_1 = file_loc_mask_1.reset_index()

        # # file_loc_mask_0 = file_loc_ori_notdeleted.loc[file_loc_ori_notdeleted['p_s_mask'].apply(lambda x: x != 1)]
        # # file_loc_mask_0 = file_loc_mask_0.reset_index()
        
        # file_loc_mask_1.to_csv(r'test_save_contcar_directory.txt', header=None, index=None, sep=' ', mode='a')

        # # just refreshing folder
        # check_folder_existance(direc_restructure_destination)

        copy_rename_files(file_loc_mask_1, direc_restructure_destination, file_restructure, prefix=None, savedir = True)

        file_loc_mask_1['verify_w_lib'] = None
        file_loc_mask_1['verify_w_linalg'] = None
        file_loc_mask_1['scaling'] = None
        file_loc_mask_1['translation'] = None
        file_loc_mask_1['mapping'] = None
        file_loc_mask_1['transformation'] = None

        structure_reference = Structure.from_file(path_perfect_poscar_24)
        
        # for all elements
        var_lib = "trf_w_lib"
        get_structure_with_library(file_loc_mask_1, direc_restructure_destination, file_restructure, structure_reference, var_lib, prefix=None)

        # for all elements
        var_linalg = "trf_w_linalg"
        get_structure_with_linalg(file_loc_mask_1, direc_restructure_destination, file_restructure, structure_reference, var_linalg, prefix=None)

        var_linalg_n_lib = "trf_w_linalg_n_lib"
        get_structure_with_linalg_combinded_with_library(file_loc_mask_1, direc_restructure_destination, file_restructure, structure_reference, var_linalg_n_lib, prefix = None)

        # Now Processing with other folders that are with mask = 0 (not perfect system)
        #### copy the data of scaling and translation to the file_loc as initial data
        file_loc['scaling'] = None
        file_loc['translation'] = None
        file_loc['mapping'] = None
        file_loc['index'] = None
        file_loc_important_cols = file_loc[["geometry", "path", "subdir_new_system", "p_s_mask", "scaling", "translation", "mapping", "index", col_excel_toten]]
        file_loc_mask_1_important_cols = file_loc_mask_1[["geometry", "path", "subdir_new_system", "p_s_mask", "scaling", "translation", "mapping", "index", col_excel_toten]]

        idx_row = file_loc_mask_1['index'].values.astype(int)

        # copy scaling, translation, mapping of the path 0
        for i in idx_row:
            scaling = np.array(file_loc_mask_1_important_cols["scaling"][file_loc_mask_1_important_cols['index'] == i])
            translation = np.array(file_loc_mask_1_important_cols["translation"][file_loc_mask_1_important_cols['index'] == i])
            mapping = np.array(file_loc_mask_1_important_cols["mapping"][file_loc_mask_1_important_cols['index'] == i])
            file_loc_important_cols.at[i, 'scaling'] = scaling[0]
            file_loc_important_cols.at[i, 'translation'] = translation[0]
            file_loc_important_cols.at[i, 'mapping'] = mapping[0]

        file_loc_important_cols = file_loc_important_cols[["geometry", "path", "subdir_new_system", "p_s_mask", "scaling", "translation", "mapping", col_excel_toten]]

        idx_mask = np.where(file_loc_important_cols["p_s_mask"] == 1)[0]

        mask = np.append(0, idx_mask)

        file_loc_important_cols["scaling"][0] = file_loc_important_cols["scaling"][3]   # hardcode for initial part
        file_loc_important_cols["translation"][0] = file_loc_important_cols["translation"][3]
        file_loc_important_cols["mapping"][0] = file_loc_important_cols["mapping"][3]

        for i in range(idx_mask.size):
            i1 = mask[i]+1
            i2 = mask[i+1]
            # # print("i1 = "+str(i1) + "; i2 = "+str(i2))
            for j in range(i1,i2+1):
                # # print("j = "+str(j))
                file_loc_important_cols["scaling"][j] = file_loc_important_cols["scaling"][i2]
                file_loc_important_cols["translation"][j] = file_loc_important_cols["translation"][i2]
                file_loc_important_cols["mapping"][j] = file_loc_important_cols["mapping"][i2]

    else:
        file_loc_ori_notdeleted = file_loc.copy()

        file_loc_mask_1 = file_loc_ori_notdeleted.loc[file_loc_ori_notdeleted['p_s_mask'].apply(lambda x: x != 0)]
        file_loc_mask_1 = file_loc_mask_1.reset_index()

        file_loc_important_cols = file_loc.copy()

    return file_loc_mask_1, file_loc_important_cols


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def get_dx_dz_init(file_path, litype):
    dictio = {}

    if litype == 0:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])

    elif litype == 1:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])

    elif litype == 2:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])
                if line.startswith("Li3"):
                    parts = line.split()
                    dictio["dx1_48h_type2_init"] = float(parts[4])
                    dictio["dx2_48h_type2_init"] = float(parts[5])
                    dictio["dz_48h_type2_init "]= float(parts[6])

        # return dx1_48h_type1_init, dx2_48h_type1_init, dz_48h_type1_init, dx1_48h_type2_init, dx2_48h_type2_init, dz_48h_type2_init, dx_24g_init, dz1_24g_init, dz2_24g_init

    elif litype == 3:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])
                if line.startswith("Li3"):
                    parts = line.split()
                    dictio["dx1_48h_type2_init"] = float(parts[4])
                    dictio["dx2_48h_type2_init"] = float(parts[5])
                    dictio["dz_48h_type2_init "]= float(parts[6])
                if line.startswith("Li4"):
                    parts = line.split()
                    dictio["dx1_48h_type3_init"] = float(parts[4])
                    dictio["dx2_48h_type3_init"] = float(parts[5])
                    dictio["dz_48h_type3_init"] = float(parts[6])

        # return dx1_48h_type1_init, dx2_48h_type1_init, dz_48h_type1_init, dx1_48h_type2_init, dx2_48h_type2_init, dz_48h_type2_init, dx_24g_init, dz1_24g_init, dz2_24g_init, dx1_48h_type3_init, dx2_48h_type3_init, dz_48h_type3_init

    elif litype == 4:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])
                if line.startswith("Li3"):
                    parts = line.split()
                    dictio["dx1_48h_type2_init"] = float(parts[4])
                    dictio["dx2_48h_type2_init"] = float(parts[5])
                    dictio["dz_48h_type2_init "]= float(parts[6])
                if line.startswith("Li4"):
                    parts = line.split()
                    dictio["dx1_48h_type3_init"] = float(parts[4])
                    dictio["dx2_48h_type3_init"] = float(parts[5])
                    dictio["dz_48h_type3_init"] = float(parts[6])
                if line.startswith("Li5"):
                    parts = line.split()
                    dictio["dx1_48h_type4_init"] = float(parts[4])
                    dictio["dx2_48h_type4_init"] = float(parts[5])
                    dictio["dz_48h_type4_init"] = float(parts[6])

    elif litype == 5:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])
                if line.startswith("Li3"):
                    parts = line.split()
                    dictio["dx1_48h_type2_init"] = float(parts[4])
                    dictio["dx2_48h_type2_init"] = float(parts[5])
                    dictio["dz_48h_type2_init "]= float(parts[6])
                if line.startswith("Li4"):
                    parts = line.split()
                    dictio["dx1_48h_type3_init"] = float(parts[4])
                    dictio["dx2_48h_type3_init"] = float(parts[5])
                    dictio["dz_48h_type3_init"] = float(parts[6])
                if line.startswith("Li5"):
                    parts = line.split()
                    dictio["dx1_48h_type4_init"] = float(parts[4])
                    dictio["dx2_48h_type4_init"] = float(parts[5])
                    dictio["dz_48h_type4_init"] = float(parts[6])
                if line.startswith("Li6"):
                    parts = line.split()
                    dictio["dx1_48h_type5_init"] = float(parts[4])
                    dictio["dx2_48h_type5_init"] = float(parts[5])
                    dictio["dz_48h_type5_init"] = float(parts[6])
    elif litype == 6:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])
                if line.startswith("Li3"):
                    parts = line.split()
                    dictio["dx1_48h_type2_init"] = float(parts[4])
                    dictio["dx2_48h_type2_init"] = float(parts[5])
                    dictio["dz_48h_type2_init "]= float(parts[6])
                if line.startswith("Li4"):
                    parts = line.split()
                    dictio["dx1_48h_type3_init"] = float(parts[4])
                    dictio["dx2_48h_type3_init"] = float(parts[5])
                    dictio["dz_48h_type3_init"] = float(parts[6])
                if line.startswith("Li5"):
                    parts = line.split()
                    dictio["dx1_48h_type4_init"] = float(parts[4])
                    dictio["dx2_48h_type4_init"] = float(parts[5])
                    dictio["dz_48h_type4_init"] = float(parts[6])
                if line.startswith("Li6"):
                    parts = line.split()
                    dictio["dx1_48h_type5_init"] = float(parts[4])
                    dictio["dx2_48h_type5_init"] = float(parts[5])
                    dictio["dz_48h_type5_init"] = float(parts[6])
                if line.startswith("Li7"):
                    parts = line.split()
                    dictio["dx1_48h_type6_init"] = float(parts[4])
                    dictio["dx2_48h_type6_init"] = float(parts[5])
                    dictio["dz_48h_type6_init"] = float(parts[6])
    elif litype == 7:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])
                if line.startswith("Li3"):
                    parts = line.split()
                    dictio["dx1_48h_type2_init"] = float(parts[4])
                    dictio["dx2_48h_type2_init"] = float(parts[5])
                    dictio["dz_48h_type2_init "]= float(parts[6])
                if line.startswith("Li4"):
                    parts = line.split()
                    dictio["dx1_48h_type3_init"] = float(parts[4])
                    dictio["dx2_48h_type3_init"] = float(parts[5])
                    dictio["dz_48h_type3_init"] = float(parts[6])
                if line.startswith("Li5"):
                    parts = line.split()
                    dictio["dx1_48h_type4_init"] = float(parts[4])
                    dictio["dx2_48h_type4_init"] = float(parts[5])
                    dictio["dz_48h_type4_init"] = float(parts[6])
                if line.startswith("Li6"):
                    parts = line.split()
                    dictio["dx1_48h_type5_init"] = float(parts[4])
                    dictio["dx2_48h_type5_init"] = float(parts[5])
                    dictio["dz_48h_type5_init"] = float(parts[6])
                if line.startswith("Li7"):
                    parts = line.split()
                    dictio["dx1_48h_type6_init"] = float(parts[4])
                    dictio["dx2_48h_type6_init"] = float(parts[5])
                    dictio["dz_48h_type6_init"] = float(parts[6])
                if line.startswith("Li8"):
                    parts = line.split()
                    dictio["dx1_48h_type7_init"] = float(parts[4])
                    dictio["dx2_48h_type7_init"] = float(parts[5])
                    dictio["dz_48h_type7_init"] = float(parts[6])
    elif litype == 8:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])
                if line.startswith("Li3"):
                    parts = line.split()
                    dictio["dx1_48h_type2_init"] = float(parts[4])
                    dictio["dx2_48h_type2_init"] = float(parts[5])
                    dictio["dz_48h_type2_init "]= float(parts[6])
                if line.startswith("Li4"):
                    parts = line.split()
                    dictio["dx1_48h_type3_init"] = float(parts[4])
                    dictio["dx2_48h_type3_init"] = float(parts[5])
                    dictio["dz_48h_type3_init"] = float(parts[6])
                if line.startswith("Li5"):
                    parts = line.split()
                    dictio["dx1_48h_type4_init"] = float(parts[4])
                    dictio["dx2_48h_type4_init"] = float(parts[5])
                    dictio["dz_48h_type4_init"] = float(parts[6])
                if line.startswith("Li6"):
                    parts = line.split()
                    dictio["dx1_48h_type5_init"] = float(parts[4])
                    dictio["dx2_48h_type5_init"] = float(parts[5])
                    dictio["dz_48h_type5_init"] = float(parts[6])
                if line.startswith("Li7"):
                    parts = line.split()
                    dictio["dx1_48h_type6_init"] = float(parts[4])
                    dictio["dx2_48h_type6_init"] = float(parts[5])
                    dictio["dz_48h_type6_init"] = float(parts[6])
                if line.startswith("Li8"):
                    parts = line.split()
                    dictio["dx1_48h_type7_init"] = float(parts[4])
                    dictio["dx2_48h_type7_init"] = float(parts[5])
                    dictio["dz_48h_type7_init"] = float(parts[6])
                if line.startswith("Li9"):
                    parts = line.split()
                    dictio["dx1_48h_type8_init"] = float(parts[4])
                    dictio["dx2_48h_type8_init"] = float(parts[5])
                    dictio["dz_48h_type8_init"] = float(parts[6])

    elif litype == 5:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    dictio["dx1_48h_type1_init"] = float(parts[4])
                    dictio["dx2_48h_type1_init"] = float(parts[5])
                    dictio["dz_48h_type1_init"] = float(parts[6])
                if line.startswith("Li2"):
                    parts = line.split()
                    dictio["dx1_48h_type2_init"] = float(parts[4])
                    dictio["dx2_48h_type2_init"] = float(parts[5])
                    dictio["dz_48h_type2_init "]= float(parts[6])
                if line.startswith("Li3"):
                    parts = line.split()
                    dictio["dx_24g_init"] = float(parts[4])
                    dictio["dz1_24g_init"] = float(parts[5])
                    dictio["dz2_24g_init"] = float(parts[6])
                if line.startswith("Li4"):
                    parts = line.split()
                    dictio["dx1_48h_type3_init"] = float(parts[4])
                    dictio["dx2_48h_type3_init"] = float(parts[5])
                    dictio["dz_48h_type3_init"] = float(parts[6])
                if line.startswith("Li5"):
                    parts = line.split()
                    dictio["dx1_48h_type4_init"] = float(parts[4])
                    dictio["dx2_48h_type4_init"] = float(parts[5])
                    dictio["dz_48h_type4_init"] = float(parts[6])
                if line.startswith("Li5"):
                    parts = line.split()
                    dictio["dx1_48h_type5_init"] = float(parts[4])
                    dictio["dx2_48h_type5_init"] = float(parts[5])
                    dictio["dz_48h_type5_init"] = float(parts[6])

        # return dx1_48h_type1_init, dx2_48h_type1_init, dz_48h_type1_init, dx1_48h_type2_init, dx2_48h_type2_init, dz_48h_type2_init, dx_24g_init, dz1_24g_init, dz2_24g_init, dx1_48h_type3_init, dx2_48h_type3_init, dz_48h_type3_init, dx1_48h_type4_init, dx2_48h_type4_init, dz_48h_type4_init 
    
    # return dx1_48h_type1_init, dx2_48h_type1_init, dz_48h_type1_init, dx1_48h_type2_init, dx2_48h_type2_init, dz_48h_type2_init, dx_24g_init, dz1_24g_init, dz2_24g_init, dx1_48h_type3_init, dx2_48h_type3_init, dz_48h_type3_init
    return tuple(dictio[key] for key in dictio.keys())


def format_float(number):
    # # basically nothing is formatted here
    # if number < 0:
    #     # return f'{(number*-1):.5f}0'
    #     return f'{number:.5f}'
    # else:
    #     return f'{number:.5f}'
    return number


def create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype):
    formatted_positions = [format_float(pos) for pos in ref_positions_array]
    formatted_positions_str = list(map(str, formatted_positions))
    return os.path.join(direc_perfect_poscar, f"Li6PS5Cl_{'_'.join(formatted_positions_str)}_{var_optitype}.cif")


def modif_dx_dz_get_filepath(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, ref_positions_array_filename, litype, var_optitype, modif_all_litype):
    file_path_new = create_file_name(direc_perfect_poscar, ref_positions_array_filename, var_optitype)
    if modif_all_litype == True:
        change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)
    elif modif_all_litype == False:
        change_dx_dz_specificlitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)
    elif modif_all_litype == None:
        change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

    return file_path_new


# def modif_dx_dz_cif_alllitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, litype, var_optitype):
#     file_path_new = create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype)
#     change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

#     return file_path_new


# def modif_dx_dz_cif_specificlitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, litype, var_optitype):
#     file_path_new = create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype)
#     change_dx_dz_specificlitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

#     return file_path_new


def change_dx_dz_alllitype(file_path, file_path_new, ref_positions_array, litype):
    # old_name = change_dx_dz
    # ref_positions_array = ALL values in this array

    formatted_positions = [format_float(pos) for pos in ref_positions_array]
    print(f"formatted_positions: {formatted_positions}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    if litype == 0:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 1:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 2:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 3:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 4:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 5:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
        new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li6"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)
    elif litype == 6:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
        new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
        new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li6"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li7"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)
    elif litype == 7:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
        new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
        new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]
        new_dx1_48h_type7, new_dx2_48h_type7, new_dz_48h_type7 = formatted_positions[21:24]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li6"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li7"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li8"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type7}",f"{new_dx2_48h_type7}",f"{new_dz_48h_type7}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)
    elif litype == 8:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
        new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
        new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]
        new_dx1_48h_type7, new_dx2_48h_type7, new_dz_48h_type7 = formatted_positions[21:24]
        new_dx1_48h_type8, new_dx2_48h_type8, new_dz_48h_type8 = formatted_positions[24:27]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li6"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li7"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li8"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type7}",f"{new_dx2_48h_type7}",f"{new_dz_48h_type7}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li9"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type8}",f"{new_dx2_48h_type8}",f"{new_dz_48h_type8}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

def change_dx_dz_specificlitype(file_path, file_path_new, ref_positions_array, litype):

    formatted_positions = [format_float(pos) for pos in ref_positions_array]

    new_dx1_type, new_dx2_type, new_dz_type = formatted_positions

    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path_new, 'w') as f:
        for line in lines:
            if line.startswith(f"Li{litype+1}"):
                parts = line.split()
                # parts[4:6+1] = [f"{new_dx1_type:.5f}",f"{new_dx2_type:.5f}",f"{new_dz_type:.5f}"]
                parts[4:6+1] = [f"{new_dx1_type}",f"{new_dx2_type}",f"{new_dz_type}"]
                parts[2] = f" {parts[2]}"
                parts[-1] = f"{parts[-1]}\n"
                line = " ".join(parts)
            f.write(line)


# # not yet changed from 3665 - 4444
def get_sum_weirdos_Li_var(max_mapping_radius, max_mapping_radius_48htype2, activate_radius, file_perfect_poscar_24_wo_cif, file_perfect_poscar_48n24_wo_cif, litype, var_optitype, iter_type, foldermapping_namestyle_all, cif_namestyle_all, modif_all_litype, full_calculation):
    """
        iter_type: varying_dx_dz, varying_radius, none
        cif_namestyle_all: True, False, None
        full_calculation: True, False
    """
    
    direc = os.getcwd() # get current working directory

    # # user input
    # max_mapping_radius = 0.043
    # max_mapping_radius_48htype2 = 0.076
    # activate_radius = 2
    lattice_constant = 10.2794980000

    folder_name_init_system = "/Init_System"
    file_new_system = "CONTCAR"
    file_name_toten = "toten_final.ods"
    col_excel_geo = "geometry"
    col_excel_path = "path"
    reference_folder = "_reference_cif"
    results_folder = "_results"

    file_perfect_poscar_24 = f"{file_perfect_poscar_24_wo_cif}.cif"
    file_perfect_poscar_48n24 = f"{file_perfect_poscar_48n24_wo_cif}.cif"

    file_path_ori_ref_48n24 = f"/{reference_folder}/{file_perfect_poscar_48n24}"
    path_ori_ref_48n24 = direc+str(file_path_ori_ref_48n24)
    path_reference_folder = direc+"/"+str(reference_folder)

    ref_positions_array_all = np.array(get_dx_dz_init(path_ori_ref_48n24, litype))

    if litype == 0:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        
        ref_positions_array_singlelitype = ref_positions_array_all[0:3]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g]

    elif litype == 1:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]

        ref_positions_array_singlelitype = ref_positions_array_all[3:6]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1]

    elif litype == 2:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
        dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]

        ref_positions_array_singlelitype = ref_positions_array_all[6:9]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2]

    elif litype == 3:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
        dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
        dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]

        ref_positions_array_singlelitype = ref_positions_array_all[9:12]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3]

    elif litype == 4:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
        dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
        dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
        dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]

        ref_positions_array_singlelitype = ref_positions_array_all[12:15]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4]

    elif litype == 5:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
        dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
        dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
        dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]
        dx1_48h_type5, dx2_48h_type5, dz_48h_type5 = ref_positions_array_all[15:18]

        ref_positions_array_singlelitype = ref_positions_array_all[15:18]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4, dx1_48h_type5, dx2_48h_type5]
    
    elif litype == 6:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
        dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
        dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
        dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]
        dx1_48h_type5, dx2_48h_type5, dz_48h_type5 = ref_positions_array_all[15:18]
        dx1_48h_type6, dx2_48h_type6, dz_48h_type6 = ref_positions_array_all[18:21]

        ref_positions_array_singlelitype = ref_positions_array_all[18:21]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4, dx1_48h_type5, dx2_48h_type5, dx1_48h_type6, dx2_48h_type6]
    
    elif litype == 7:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
        dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
        dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
        dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]
        dx1_48h_type5, dx2_48h_type5, dz_48h_type5 = ref_positions_array_all[15:18]
        dx1_48h_type6, dx2_48h_type6, dz_48h_type6 = ref_positions_array_all[18:21]
        dx1_48h_type7, dx2_48h_type7, dz_48h_type7 = ref_positions_array_all[21:24]

        ref_positions_array_singlelitype = ref_positions_array_all[21:24]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4, dx1_48h_type5, dx2_48h_type5, dx1_48h_type6, dx2_48h_type6, dx1_48h_type7, dx2_48h_type7]

    elif litype == 8:
        dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
        dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
        dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
        dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
        dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]
        dx1_48h_type5, dx2_48h_type5, dz_48h_type5 = ref_positions_array_all[15:18]
        dx1_48h_type6, dx2_48h_type6, dz_48h_type6 = ref_positions_array_all[18:21]
        dx1_48h_type7, dx2_48h_type7, dz_48h_type7 = ref_positions_array_all[21:24]
        dx1_48h_type8, dx2_48h_type8, dz_48h_type8 = ref_positions_array_all[24:27]

        ref_positions_array_singlelitype = ref_positions_array_all[24:27]
        ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4, dx1_48h_type5, dx2_48h_type5, dx1_48h_type6, dx2_48h_type6, dx1_48h_type7, dx2_48h_type7, dx1_48h_type8, dx2_48h_type8]


    # max_mapping_radius_48htype1_48htype2 = (max_mapping_radius + max_mapping_radius_48htype2) / 2
    # file_perfect_poscar_48n24 = "Li6PS5Cl_type2.cif"
    # file_perfect_poscar_24 = "Li6PS5Cl_24_mod_2p27291.cif" # copy this manually to folder_name_perfect_poscar  

    folder_name_iter_type = f"/{results_folder}/_{iter_type}/{file_perfect_poscar_48n24_wo_cif}/"
    path_folder_name_iter_type = direc+str(folder_name_iter_type)
    check_folder_existance(path_folder_name_iter_type, empty_folder=False)


    if foldermapping_namestyle_all == True:
        if activate_radius == 2:
            if litype == 0:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 1:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 2:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 3:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 4:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 5:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 6:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"            
            elif litype == 7:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"            
            elif litype == 8:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
        
        elif activate_radius == 1:
            if litype == 0:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{max_mapping_radius}/"
            elif litype == 1:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}/"
            elif litype == 2:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}/"
            elif litype == 3:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}/"
            elif litype == 4:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}/"
            elif litype == 5:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}/"
            elif litype == 6:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}/"            
            elif litype == 7:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}/"            
            elif litype == 8:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}/"
        
    else:
        if activate_radius == 2:
            if litype == 0:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 1:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 2:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 3:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 4:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 5:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 6:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 7:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
            elif litype == 8:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                
        elif activate_radius == 1:
            if litype == 0:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{max_mapping_radius}/"
            elif litype == 1:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}/"
            elif litype == 2:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}/"
            elif litype == 3:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}/"
            elif litype == 4:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}/"
            elif litype == 5:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}/"
            elif litype == 6:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}/"
            elif litype == 7:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}/"
            elif litype == 8:
                folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}/"


    folder_name_perfect_poscar = "/_reference_cif/cif_matrix/Li1/"
    cif_line_nr_start = 26  # index from 0
    poscar_line_nr_start = 8
    poscar_line_nr_end = 60
    contcar_columns_type2 = ['coord_x', 'coord_y', 'coord_z']
    amount_Li = 24
    col_excel_toten = "toten [eV]" 
    amount_P = 4 
    amount_S = 20 
    amount_Cl = 4
    file_restructure = "CONTCAR" 
    cif_columns = ['species', 'idx_species', 'unkownvar_1', 'coord_x', 'coord_y', 'coord_z', 'unkownvar_2'] 

    direc_restructure_destination = direc+str(folder_name_destination_restructure)
    # direc_perfect_poscar = direc+str(folder_name_iter_type) ### direc+str(folder_name_perfect_poscar)
    path_perfect_poscar_24 = os.path.join(path_folder_name_iter_type, file_perfect_poscar_24)
    direc_init_system = direc+str(folder_name_init_system)

    dtype = {col_excel_geo: float, col_excel_path: float}

    data_toten = pd.read_excel(file_name_toten, dtype=dtype, engine="odf")
    data_toten_ori = data_toten
    data_toten = data_toten.sort_values(by=["geometry","path"],ignore_index=True,ascending=False)

    file_loc = create_file_loc(direc_init_system, data_toten, file_new_system)

    # just refreshing folder
    check_folder_existance(direc_restructure_destination, empty_folder=True)

    # copy ref.cif inside _results/../.. 
    copy_rename_single_file(path_folder_name_iter_type, reference_folder, file_perfect_poscar_48n24, prefix=None)

    copy_rename_files(file_loc, direc_restructure_destination, file_restructure, prefix=None, savedir = False)
    get_positive_lessthan1_poscarcontcar(file_loc, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)

    file_loc_mask_1, file_loc_important_cols = get_orientation(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation="False")
    
    if modif_all_litype == True:
        ref_positions_array = ref_positions_array_all
    elif modif_all_litype == False:
        ref_positions_array = ref_positions_array_singlelitype
    elif modif_all_litype == None:
        ref_positions_array = ref_positions_array_all

    if cif_namestyle_all == True:
        ref_positions_array_filename = ref_positions_array_all_compactform
    elif cif_namestyle_all == False:
        ref_positions_array_filename = ref_positions_array_singlelitype
    # # DUNNO WHAT TO DO HERE
    elif cif_namestyle_all == None:
        ref_positions_array_filename = ref_positions_array_all_compactform

    # path_perfect_poscar_48n24 = modif_dx_dz_cif(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
    # path_perfect_poscar_48n24 = modif_dx_dz_get_filepath(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, ref_positions_array_filename, litype, var_optitype, modif_all_litype)
    path_perfect_poscar_48n24 = modif_dx_dz_get_filepath(path_folder_name_iter_type, path_ori_ref_48n24, ref_positions_array, ref_positions_array_filename, litype, var_optitype, modif_all_litype)

    # just copy file
    # copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_24, prefix=None)
    # !!! had to copy file_perfect_poscar_48n24 into Li1
    # copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_48n24, prefix=None)
    copy_rename_single_file(direc_restructure_destination, path_folder_name_iter_type, file_perfect_poscar_48n24, prefix=None)

    # copy_rename_files(file_loc_important_cols, direc_restructure_destination, file_restructure, prefix=None,  savedir = True)

    # # var_c = "trf_w_linalg_orientated"
    # # get_structure_with_linalg_orientated(file_loc_important_cols, direc_restructure_destination, file_restructure, var_c)


    # # var_name_in = "trf_w_linalg_orientated"
    # # var_name_out = "trf_w_linalg_orientated_positive"
    # # n_decimal = 8
    # # get_orientated_positive_lessthan1_cif(file_loc_important_cols, direc_restructure_destination, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal)

    # max_mapping_radius = 0.05282658993283027
    # max_mapping_radius = 0.045
    # max_mapping_radius = 0.055
    # max_mapping_radius = 0.04197083906
    ref_structure_48n24 = Structure.from_file(path_perfect_poscar_48n24)
    cif_structure = Structure(ref_structure_48n24.lattice, ref_structure_48n24.species, ref_structure_48n24.frac_coords)
    cif = CifWriter(cif_structure)
    cif.write_file(f"{direc_restructure_destination}{file_perfect_poscar_48n24_wo_cif}_expanded.cif")

    coor_structure_init_dict = get_coor_dict_structure(ref_structure_48n24)
    coor_structure_init_dict_expanded = get_coor_dict_structure(Structure.from_file(f"{direc_restructure_destination}{file_perfect_poscar_48n24_wo_cif}_expanded.cif"))

    # get_positive_lessthan1_poscarcontcar(file_loc_important_cols, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)
    get_coor_structure24_dict_iterated(file_loc_important_cols, mapping = "False")

    # if activate_radius == 3:
    #     get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
    #     get_flag_map_weirdos_48htype1_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype1_48htype2)
    #     get_flag_map_weirdos_48htypesmerged_level1_el(file_loc_important_cols, "Li")
    #     get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
    #     get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
    if activate_radius == 2:
        get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
        get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
        get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
    elif activate_radius == 1:
        get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)

    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "P", max_mapping_radius)
    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "S", max_mapping_radius)
    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Cl", max_mapping_radius)

    get_idx_weirdos_el(file_loc_important_cols, "Li", activate_radius)

    # idx_correcting_mapped_el(file_loc_important_cols, el="Li")
    idx_correcting_mapped_el(file_loc_important_cols, "Li", activate_radius)
    # # create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, var_savefilename = "mapLi")
    
    get_distance_weirdos_label_el(file_loc_important_cols, coor_structure_init_dict, "Li", litype)

    # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","coor_weirdos_48htypesmerged_Li","top3_dist_weirdos_dict_Li","idx0_weirdos_Li","#weirdos_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
    # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_weirdos_Li","sum_mapped_48htype2_Li_new","#weirdos_Li","sum_mapped_48htypesmerged_Li_new","sum_sanitycheck_48htypesmerged_Li_new","idx0_weirdos_Li","top3_sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_Li","duplicate_closest24_w_data_Li","duplicate_closest24_w_data_48htype2_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)

    # if activate_radius == 3:
    #     file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","sum_weirdos_Li","sum_mapped_48htype1_48htype2_Li_closestduplicate","sum_weirdos_48htype1_48htype2_Li","sum_mapped_48htype2_Li_closestduplicate","#weirdos_Li","sum_mapped_48htypesmerged_Li","sum_sanitycheck_48htypesmerged_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
    #     file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","sum_weirdos_Li","sum_mapped_48htype1_48htype2_Li_closestduplicate","sum_weirdos_48htype1_48htype2_Li","sum_mapped_48htype2_Li_closestduplicate","#weirdos_Li","sum_mapped_48htypesmerged_Li","sum_sanitycheck_48htypesmerged_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","toten [eV]"]]

    #     sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())

    #     var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type1}_{dx2_48h_type1}_{formatted_dz_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{formatted_dz_48h_type2}_{dx_24g}_{dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}_{max_mapping_radius_48htype1_48htype2}"
    
    if full_calculation == False:
        pass
    elif full_calculation == True:
        create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, activate_radius, var_savefilename = "mapLi")
        rewrite_cif_w_correct_Li_idx(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, amount_Cl, var_savefilename_init = "mapLi", var_savefilename_new = "mapLi_reindexed")
        format_spacing_cif(file_loc_important_cols, direc_restructure_destination, var_savefilename_init = "mapLi_reindexed", var_savefilename_new = "mapLi_reindexed")
        # # # # delete_files(file_loc_important_cols, direc_restructure_destination, file_name_w_format = "mapLi_reindexed.cif")

        rewrite_cif_w_correct_Li_idx_weirdos_appended(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, amount_Cl, activate_radius,var_savefilename_init = "mapLi", var_savefilename_new = "mapLi_reindexed_weirdos_appended")
        format_spacing_cif(file_loc_important_cols, direc_restructure_destination, var_savefilename_init = "mapLi_reindexed_weirdos_appended", var_savefilename_new = "mapLi_reindexed_weirdos_appended")
        # # # delete_files(file_loc_important_cols, direc_restructure_destination, file_name_w_format = "mapLi_reindexed_weirdos_appended.cif")

        create_cif_pymatgen(file_loc_important_cols, direc_restructure_destination, file_restructure = "CONTCAR_positive", var_name = "CONTCAR_positive_pymatgen")

        # # # ascending_Li(file_loc_important_cols, direc_restructure_destination, var_filename_init = "mapLi_reindexed_weirdos_appended", var_savefilename_new = "mapLi_reindexed_weirdos_appended_reordered")
        # # # format_spacing_cif(file_loc_important_cols, direc_restructure_destination, var_savefilename_init = "mapLi_reindexed_weirdos_appended_reordered", var_savefilename_new = "mapLi_reindexed_weirdos_appended_reordered")

        get_idx_coor_limapped_weirdos_dict_litype(file_loc_important_cols, coor_structure_init_dict, activate_radius, litype, el="Li")

        get_latticeconstant_structure_dict_iterated(file_loc_important_cols, direc_restructure_destination, var_filename = "CONTCAR")
        # plot_energy_vs_latticeconstant(file_loc_important_cols, var_filename = "CONTCAR")
        plot_weirdos_directcoor(file_loc_important_cols, activate_radius)

        coor_weirdos_Li = get_coor_weirdos_array(file_loc_important_cols, activate_radius)
        create_POSCAR_weirdos(coor_weirdos_Li, direc_restructure_destination, lattice_constant, filename = "POSCAR_weirdos")

        get_label_mapping(file_loc_important_cols, coor_structure_init_dict, "Li", activate_radius, litype)

    if litype == 0:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 1:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 2:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 3:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 4:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 5:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]]   
    elif litype == 6:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]] 
    elif litype == 7:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]] 
    elif litype == 8:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]]   

        # var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type1}_{dx2_48h_type1}_{formatted_dz_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{formatted_dz_48h_type2}_{dx_24g}_{dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
    
    sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())



    if foldermapping_namestyle_all == True:
        if activate_radius == 2:
            if litype == 0:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 1:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 2:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 3:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 4:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 5:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 6:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 7:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 8:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
        
        elif activate_radius == 1:
            if litype == 0:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{max_mapping_radius}"
            elif litype == 1:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}"
            elif litype == 2:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}"
            elif litype == 3:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}"
            elif litype == 4:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}"
            elif litype == 5:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}"
            elif litype == 6:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}"
            elif litype == 7:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}"
            elif litype == 8:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}"
        
    
    else:
        if activate_radius == 2:
            if litype == 0:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 1:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 2:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 3:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 4:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 5:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 6:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 7:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            elif litype == 8:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}_{max_mapping_radius_48htype2}"

        elif activate_radius == 1:
            if litype == 0:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{max_mapping_radius}"
            elif litype == 1:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}"
            elif litype == 2:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}"
            elif litype == 3:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}"
            elif litype == 4:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}"
            elif litype == 5:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}"
            elif litype == 6:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}"
            elif litype == 7:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}"
            elif litype == 8:
                var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}"


    # path_excel_file = os.path.join(direc_perfect_poscar, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
    path_excel_file = os.path.join(path_folder_name_iter_type, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
    file_loc_important_cols_sorted_toten.to_excel(path_excel_file, index=False)

    if activate_radius == 1:
        file_loc_important_cols.to_pickle(f'{path_folder_name_iter_type}file_loc_important_cols_{max_mapping_radius}_{file_perfect_poscar_48n24_wo_cif}.pkl') 
    elif activate_radius == 2:
        file_loc_important_cols.to_pickle(f'{path_folder_name_iter_type}file_loc_important_cols_{max_mapping_radius}_{max_mapping_radius_48htype2}_{file_perfect_poscar_48n24_wo_cif}.pkl')
    # elif activate_radius == 3:
    #     file_loc_important_cols.to_pickle(f'{path_folder_name_iter_type}file_loc_important_cols_{max_mapping_radius}_{max_mapping_radius_48htype2}_{max_mapping_radius_48htype1_48htype2}_{file_perfect_poscar_48n24_wo_cif}.pkl')

    return sum_weirdos_Li


# def get_sum_weirdos_Li_var_wo_weirdo(dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, max_mapping_radius, max_mapping_radius_48htype2, df_wo_weirdos_selectedcol, activate_radius, file_perfect_poscar_24, file_ori_ref_48n24, litype, var_optitype):

#     file_loc_important_cols = df_wo_weirdos_selectedcol

#     formatted_dx1_48h_type1 = format_float(dx1_48h_type1)
#     formatted_dx2_48h_type1 = format_float(dx2_48h_type1)
#     formatted_dz_48h_type1 = format_float(dz_48h_type1)
#     formatted_dx1_48h_type2 = format_float(dx1_48h_type2)
#     formatted_dx2_48h_type2 = format_float(dx2_48h_type2)
#     formatted_dz_48h_type2 = format_float(dz_48h_type2)
#     formatted_dx_24g = format_float(dx_24g)
#     formatted_dz1_24g = format_float(dz1_24g)
#     formatted_dz2_24g = format_float(dz2_24g)
#     direc = os.getcwd() # get current working directory

#     # # user input
#     # max_mapping_radius = 0.043
#     # max_mapping_radius_48htype2 = 0.076
#     # activate_radius = 2

#     file_path_ori_ref_48n24 = f"./perfect_poscar/cif_matrix/ori/{file_ori_ref_48n24}"
#     # max_mapping_radius_48htype1_48htype2 = (max_mapping_radius + max_mapping_radius_48htype2) / 2
#     # file_ori_ref_48n24 = "Li6PS5Cl_type2.cif"
#     # file_perfect_poscar_24 = "Li6PS5Cl_24_mod_2p27291.cif" # copy this manually to folder_name_perfect_poscar  

#     folder_name_init_system = "/Init_System"
#     file_new_system = "CONTCAR"
#     file_name_toten = "toten_final.ods"
#     col_excel_geo = "geometry"
#     col_excel_path = "path"

#     if activate_radius == 2:
#         folder_name_destination_restructure = f"/restructure_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}_optimizer/"
#     elif activate_radius == 1:
#         folder_name_destination_restructure = f"/restructure_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{max_mapping_radius}_optimizer/"

#     folder_name_perfect_poscar = "/perfect_poscar/cif_matrix/Li1/"
#     cif_line_nr_start = 26  # index from 0
#     poscar_line_nr_start = 8
#     poscar_line_nr_end = 60
#     contcar_columns_type2 = ['coord_x', 'coord_y', 'coord_z']
#     amount_Li = 24
#     col_excel_toten = "toten [eV]" 
#     amount_P = 4 
#     amount_S = 20 
#     file_restructure = "CONTCAR" 
#     cif_columns = ['species', 'idx_species', 'unkownvar_1', 'coord_x', 'coord_y', 'coord_z', 'unkownvar_2'] 

#     direc_restructure_destination = direc+str(folder_name_destination_restructure)
#     direc_perfect_poscar = direc+str(folder_name_perfect_poscar)
#     path_perfect_poscar_24 = os.path.join(direc_perfect_poscar, file_perfect_poscar_24)
#     direc_init_system = direc+str(folder_name_init_system)

#     dtype = {col_excel_geo: float, col_excel_path: float}

#     data_toten = pd.read_excel(file_name_toten, dtype=dtype, engine="odf")
#     data_toten_ori = data_toten
#     data_toten = data_toten.sort_values(by=["geometry","path"],ignore_index=True,ascending=False)

#     # just refreshing folder
#     check_folder_existance(direc_restructure_destination)

#     # path_perfect_poscar_48n24 = modif_dx_dz_cif(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
#     path_perfect_poscar_48n24 = modif_dx_dz_cif_allvariables(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)

#     # just copy file
#     # copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_24, prefix=None)
#     # !!! had to copy file_ori_ref_48n24 into Li1
#     copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_ori_ref_48n24, prefix=None)

#     # file_loc_mask_1, file_loc_important_cols = get_orientation(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation="False")

#     copy_rename_files(file_loc_important_cols, direc_restructure_destination, file_restructure, prefix=None, savedir = True)


#     # # var_c = "trf_w_linalg_orientated"
#     # # get_structure_with_linalg_orientated(file_loc_important_cols, direc_restructure_destination, file_restructure, var_c)


#     # # var_name_in = "trf_w_linalg_orientated"
#     # # var_name_out = "trf_w_linalg_orientated_positive"
#     # # n_decimal = 8
#     # # get_orientated_positive_lessthan1_cif(file_loc_important_cols, direc_restructure_destination, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal)

#     # max_mapping_radius = 0.05282658993283027
#     # max_mapping_radius = 0.045
#     # max_mapping_radius = 0.055
#     # max_mapping_radius = 0.04197083906
#     ref_structure_48n24 = Structure.from_file(path_perfect_poscar_48n24)

#     coor_structure_init_dict = get_coor_dict_structure(ref_structure_48n24)
#     get_positive_lessthan1_poscarcontcar(file_loc_important_cols, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)
#     get_coor_structure24_dict_iterated(file_loc_important_cols, mapping = "False")

#     # if activate_radius == 3:
#     #     get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
#     #     get_flag_map_weirdos_48htype1_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype1_48htype2)
#     #     get_flag_map_weirdos_48htypesmerged_level1_el(file_loc_important_cols, "Li")
#     #     get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
#     #     get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
#     if activate_radius == 2:
#         get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
#         get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
#         get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
#     elif activate_radius == 1:
#         get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)

#     # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "P", max_mapping_radius)
#     # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "S", max_mapping_radius)
#     # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Cl", max_mapping_radius)

#     get_idx_weirdos_el(file_loc_important_cols, "Li", activate_radius)

#     # idx_correcting_mapped_el(file_loc_important_cols, el="Li")
#     idx_correcting_mapped_el(file_loc_important_cols, "Li", activate_radius)
#     # # create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, var_savefilename = "mapLi")
    
#     get_distance_weirdos_label_el(file_loc_important_cols, coor_structure_init_dict, "Li", litype)

#     # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","coor_weirdos_48htypesmerged_Li","top3_dist_weirdos_dict_Li","idx0_weirdos_Li","#weirdos_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#     # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_weirdos_Li","sum_mapped_48htype2_Li_new","#weirdos_Li","sum_mapped_48htypesmerged_Li_new","sum_sanitycheck_48htypesmerged_Li_new","idx0_weirdos_Li","top3_sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_Li","duplicate_closest24_w_data_Li","duplicate_closest24_w_data_48htype2_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)

#     # if activate_radius == 3:
#     #     file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","sum_weirdos_Li","sum_mapped_48htype1_48htype2_Li_closestduplicate","sum_weirdos_48htype1_48htype2_Li","sum_mapped_48htype2_Li_closestduplicate","#weirdos_Li","sum_mapped_48htypesmerged_Li","sum_sanitycheck_48htypesmerged_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#     #     file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","sum_weirdos_Li","sum_mapped_48htype1_48htype2_Li_closestduplicate","sum_weirdos_48htype1_48htype2_Li","sum_mapped_48htype2_Li_closestduplicate","#weirdos_Li","sum_mapped_48htypesmerged_Li","sum_sanitycheck_48htypesmerged_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","toten [eV]"]]

#     #     sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())

#     #     var_excel_file = f"{int(sum_weirdos_Li)}_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dz_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dz_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}_{max_mapping_radius_48htype1_48htype2}"
    
#     if activate_radius == 2:
#         if litype == 0:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 1:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 2:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 3:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 4:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 5:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]]
        
#         sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())

#         var_excel_file = f"{int(sum_weirdos_Li)}_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dz_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dz_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}"

#     elif activate_radius == 1:
#         if litype == 0:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 1:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 2:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 3:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 4:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 5:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]]

#         sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())

#         var_excel_file = f"{int(sum_weirdos_Li)}_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dz_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dz_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}"

#     path_excel_file = os.path.join(direc_perfect_poscar, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
#     file_loc_important_cols_sorted_toten.to_excel(path_excel_file, index=False)

#     return sum_weirdos_Li


def get_sum_weirdos_Li_var_wo_weirdo_litype(ref_positions_array, max_mapping_radius, max_mapping_radius_48htype2, df_wo_weirdos_selectedcol, activate_radius, file_perfect_poscar_24, file_ori_ref_48n24, litype, var_optitype, iter_type):

    file_loc_important_cols = df_wo_weirdos_selectedcol

    formatted_positions = [format_float(pos) for pos in ref_positions_array]
    new_dx1_type, new_dx2_type, new_dz_type = formatted_positions
    
    direc = os.getcwd() # get current working directory

    # # user input
    # max_mapping_radius = 0.043
    # max_mapping_radius_48htype2 = 0.076
    # activate_radius = 2
    results_folder = "_results"
    reference_folder = "_reference_cif"

    file_path_ori_ref_48n24 = f"./{reference_folder}/{file_ori_ref_48n24}"
    # max_mapping_radius_48htype1_48htype2 = (max_mapping_radius + max_mapping_radius_48htype2) / 2
    # file_ori_ref_48n24 = "Li6PS5Cl_type2.cif"
    # file_perfect_poscar_24 = "Li6PS5Cl_24_mod_2p27291.cif" # copy this manually to folder_name_perfect_poscar  

    folder_name_init_system = "/Init_System"
    file_new_system = "CONTCAR"
    file_name_toten = "toten_final.ods"
    col_excel_geo = "geometry"
    col_excel_path = "path"

    folder_name_iter_type = f"/{results_folder}/_{iter_type}/{file_ori_ref_48n24}/"
    path_folder_name_iter_type = direc+str(folder_name_iter_type)
    check_folder_existance(path_folder_name_iter_type, empty_folder=False)

    # copy ref.cif inside _results/../.. 
    copy_rename_single_file(path_folder_name_iter_type, reference_folder, file_ori_ref_48n24, prefix=None)

    if activate_radius == 2:
        folder_name_destination_restructure = f"{path_folder_name_iter_type}restructure_{new_dx1_type}_{new_dx2_type}_{max_mapping_radius}_{max_mapping_radius_48htype2}_optimizer/"
    elif activate_radius == 1:
        folder_name_destination_restructure = f"{path_folder_name_iter_type}restructure_{new_dx1_type}_{new_dx2_type}_{max_mapping_radius}_optimizer/"

    folder_name_perfect_poscar = folder_name_iter_type
    cif_line_nr_start = 26  # index from 0
    poscar_line_nr_start = 8
    poscar_line_nr_end = 60
    contcar_columns_type2 = ['coord_x', 'coord_y', 'coord_z']
    amount_Li = 24
    col_excel_toten = "toten [eV]" 
    amount_P = 4 
    amount_S = 20 
    file_restructure = "CONTCAR" 
    cif_columns = ['species', 'idx_species', 'unkownvar_1', 'coord_x', 'coord_y', 'coord_z', 'unkownvar_2'] 

    direc_restructure_destination = direc+str(folder_name_destination_restructure)
    direc_perfect_poscar = direc+str(folder_name_perfect_poscar)
    path_perfect_poscar_24 = os.path.join(direc_perfect_poscar, file_perfect_poscar_24)
    direc_init_system = direc+str(folder_name_init_system)

    dtype = {col_excel_geo: float, col_excel_path: float}

    data_toten = pd.read_excel(file_name_toten, dtype=dtype, engine="odf")
    data_toten_ori = data_toten
    data_toten = data_toten.sort_values(by=["geometry","path"],ignore_index=True,ascending=False)

    # just refreshing folder
    check_folder_existance(direc_restructure_destination, empty_folder=True)

    # path_perfect_poscar_48n24 = modif_dx_dz_cif(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type, dx2_48h_type, dz_48h_type, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
    # path_perfect_poscar_48n24 = modif_dx_dz_cif_specificlitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
    path_perfect_poscar_48n24 = modif_dx_dz_get_filepath(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, ref_positions_array, litype, var_optitype, modif_all_litype = False)

    # just copy file
    # copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_24, prefix=None)
    # !!! had to copy file_ori_ref_48n24 into Li1
    copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_ori_ref_48n24, prefix=None)

    # file_loc_mask_1, file_loc_important_cols = get_orientation(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation="False")

    copy_rename_files(file_loc_important_cols, direc_restructure_destination, file_restructure, prefix=None, savedir = True)


    # # var_c = "trf_w_linalg_orientated"
    # # get_structure_with_linalg_orientated(file_loc_important_cols, direc_restructure_destination, file_restructure, var_c)


    # # var_name_in = "trf_w_linalg_orientated"
    # # var_name_out = "trf_w_linalg_orientated_positive"
    # # n_decimal = 8
    # # get_orientated_positive_lessthan1_cif(file_loc_important_cols, direc_restructure_destination, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal)

    # max_mapping_radius = 0.05282658993283027
    # max_mapping_radius = 0.045
    # max_mapping_radius = 0.055
    # max_mapping_radius = 0.04197083906
    ref_structure_48n24 = Structure.from_file(path_perfect_poscar_48n24)

    coor_structure_init_dict = get_coor_dict_structure(ref_structure_48n24)
    get_positive_lessthan1_poscarcontcar(file_loc_important_cols, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)
    get_coor_structure24_dict_iterated(file_loc_important_cols, mapping = "False")

    if activate_radius == 2:
        get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
        get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
        get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
    elif activate_radius == 1:
        get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)

    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "P", max_mapping_radius)
    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "S", max_mapping_radius)
    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Cl", max_mapping_radius)

    get_idx_weirdos_el(file_loc_important_cols, "Li", activate_radius)

    # idx_correcting_mapped_el(file_loc_important_cols, el="Li")
    idx_correcting_mapped_el(file_loc_important_cols, "Li", activate_radius)
    # # create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, var_savefilename = "mapLi")
    
    get_distance_weirdos_label_el(file_loc_important_cols, coor_structure_init_dict, "Li", litype)

    # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","coor_weirdos_48htypesmerged_Li","top3_dist_weirdos_dict_Li","idx0_weirdos_Li","#weirdos_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
    # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_weirdos_Li","sum_mapped_48htype2_Li_new","#weirdos_Li","sum_mapped_48htypesmerged_Li_new","sum_sanitycheck_48htypesmerged_Li_new","idx0_weirdos_Li","top3_sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_Li","duplicate_closest24_w_data_Li","duplicate_closest24_w_data_48htype2_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)

    if litype == 0:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 1:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 2:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 3:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 4:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 5:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]]
    elif litype == 6:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]] 
    elif litype == 7:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]] 
    elif litype == 8:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]]   

    sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())
            
    if activate_radius == 2:
        var_excel_file = f"{int(sum_weirdos_Li)}_{new_dx1_type}_{new_dx2_type}_{new_dz_type}_{max_mapping_radius}_{max_mapping_radius_48htype2}"

    elif activate_radius == 1:
        var_excel_file = f"{int(sum_weirdos_Li)}_{new_dx1_type}_{new_dx2_type}_{new_dz_type}_{max_mapping_radius}"

    path_excel_file = os.path.join(direc_perfect_poscar, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
    file_loc_important_cols_sorted_toten.to_excel(path_excel_file, index=False)

    return sum_weirdos_Li


def get_sum_weirdos_Li_var_litype(ref_positions_array, max_mapping_radius, max_mapping_radius_48htype2, activate_radius, file_perfect_poscar_24, file_ori_ref_48n24, litype, var_optitype):
    
    formatted_positions = [format_float(pos) for pos in ref_positions_array]
    new_dx1_type, new_dx2_type, new_dz_type = formatted_positions

    direc = os.getcwd() # get current working directory

    # # user input
    # max_mapping_radius = 0.043
    # max_mapping_radius_48htype2 = 0.076
    # activate_radius = 2

    file_path_ori_ref_48n24 = f"./perfect_poscar/cif_matrix/ori/{file_ori_ref_48n24}"
    # max_mapping_radius_48htype1_48htype2 = (max_mapping_radius + max_mapping_radius_48htype2) / 2
    # file_ori_ref_48n24 = "Li6PS5Cl_type2.cif"
    # file_perfect_poscar_24 = "Li6PS5Cl_24_mod_2p27291.cif" # copy this manually to folder_name_perfect_poscar  

    folder_name_init_system = "/Init_System"
    file_new_system = "CONTCAR"
    file_name_toten = "toten_final.ods"
    col_excel_geo = "geometry"
    col_excel_path = "path"

    if activate_radius == 2:
        folder_name_destination_restructure = f"/restructure_{new_dx1_type}_{new_dx2_type}_{max_mapping_radius}_{max_mapping_radius_48htype2}_optimizer/"
    elif activate_radius == 1:
        folder_name_destination_restructure = f"/restructure_{new_dx1_type}_{new_dx2_type}_{max_mapping_radius}_optimizer/"

    folder_name_perfect_poscar = "/perfect_poscar/cif_matrix/Li1/"
    cif_line_nr_start = 26  # index from 0
    poscar_line_nr_start = 8
    poscar_line_nr_end = 60
    contcar_columns_type2 = ['coord_x', 'coord_y', 'coord_z']
    amount_Li = 24
    col_excel_toten = "toten [eV]" 
    amount_P = 4 
    amount_S = 20 
    file_restructure = "CONTCAR" 
    cif_columns = ['species', 'idx_species', 'unkownvar_1', 'coord_x', 'coord_y', 'coord_z', 'unkownvar_2'] 

    direc_restructure_destination = direc+str(folder_name_destination_restructure)
    direc_perfect_poscar = direc+str(folder_name_perfect_poscar)
    path_perfect_poscar_24 = os.path.join(direc_perfect_poscar, file_perfect_poscar_24)
    direc_init_system = direc+str(folder_name_init_system)

    dtype = {col_excel_geo: float, col_excel_path: float}

    data_toten = pd.read_excel(file_name_toten, dtype=dtype, engine="odf")
    data_toten_ori = data_toten
    data_toten = data_toten.sort_values(by=["geometry","path"],ignore_index=True,ascending=False)

    geometry = np.array([])
    path = np.array([])
    subdir_col = np.array([])
    for subdir, dirs, files in os.walk(direc,topdown=False):
        # source: https://stackoverflow.com/questions/27805919/how-to-only-read-lines-in-a-text-file-after-a-certain-string
        for file in files:
            filepath = subdir + os.sep
            # get directory of CONTCAR
            if os.path.basename(file) == file_new_system:
                geometry_nr = FileOperations.splitall(subdir)[-2]
                path_nr = FileOperations.splitall(subdir)[-1]
                geometry = pd.DataFrame(np.append(geometry, int(geometry_nr)), columns=["geometry"])
                geometry_ori = geometry
                geometry.dropna(axis=1)
                path = pd.DataFrame(np.append(path, int(path_nr)), columns=["path"])#
                path.dropna(axis=1)
                path_sorted = path.sort_values(by="path",ascending=False)
                subdir_file = os.path.join(subdir,file_new_system)
                # # create directory of POSCAR of init system
                subdir_init_system = direc_init_system + os.sep + geometry_nr + os.sep + path_nr
                subdir_col = pd.DataFrame(np.append(subdir_col, subdir_file), columns=["subdir_new_system"])
                file_loc = geometry.join(path)
                file_loc["subdir_new_system"] = subdir_col#
                path_ori = path

    file_loc_ori_notsorted = file_loc.copy()
    file_loc = file_loc.sort_values(by=["geometry","path"],ignore_index=True,ascending=False) # sort descendingly based on path

    file_loc["g+p"] = (file_loc["geometry"] + file_loc["path"]).fillna(0) # replace NaN with 0
    file_loc["g+p+1"] = file_loc["g+p"].shift(1)
    file_loc["g+p+1"][0] = 0 # replace 1st element with 0
    file_loc["g+p-1"] = file_loc["g+p"].shift(-1)
    file_loc["g+p-1"][(file_loc["g+p-1"]).size - 1] = 0.0 # replace last element with 0
    file_loc["perfect_system"] = file_loc["g+p"][(file_loc["g+p+1"] > file_loc["g+p"]) & (file_loc["g+p-1"] > file_loc["g+p"])]
    file_loc["perfect_system"][file_loc["geometry"].size-1] = 0.0 # hardcode the path 0/0
    file_loc["p_s_mask"] = [0 if np.isnan(item) else 1 for item in file_loc["perfect_system"]]



    if data_toten[col_excel_geo].all() == file_loc["geometry"].all() & data_toten[col_excel_path].all() == file_loc["path"].all():
        file_loc[col_excel_toten] = data_toten[col_excel_toten]
    else:
        print("check the compatibility of column geometry and path between data_toten file and file_loc")

    # just refreshing folder
    check_folder_existance(direc_restructure_destination, empty_folder=True)

    # path_perfect_poscar_48n24 = modif_dx_dz_cif(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type, dx2_48h_type, dz_48h_type, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
    path_perfect_poscar_48n24 = modif_dx_dz_cif_specificlitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, var_optitype, litype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)

    # just copy file
    # copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_24, prefix=None)
    # !!! had to copy file_ori_ref_48n24 into Li1
    copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_ori_ref_48n24, prefix=None)

    file_loc_mask_1, file_loc_important_cols = get_orientation(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation="False")

    copy_rename_files(file_loc_important_cols, direc_restructure_destination, file_restructure, prefix=None, savedir = True)


    # # var_c = "trf_w_linalg_orientated"
    # # get_structure_with_linalg_orientated(file_loc_important_cols, direc_restructure_destination, file_restructure, var_c)


    # # var_name_in = "trf_w_linalg_orientated"
    # # var_name_out = "trf_w_linalg_orientated_positive"
    # # n_decimal = 8
    # # get_orientated_positive_lessthan1_cif(file_loc_important_cols, direc_restructure_destination, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal)

    # max_mapping_radius = 0.05282658993283027
    # max_mapping_radius = 0.045
    # max_mapping_radius = 0.055
    # max_mapping_radius = 0.04197083906
    ref_structure_48n24 = Structure.from_file(path_perfect_poscar_48n24)

    coor_structure_init_dict = get_coor_dict_structure(ref_structure_48n24)
    get_positive_lessthan1_poscarcontcar(file_loc_important_cols, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)
    get_coor_structure24_dict_iterated(file_loc_important_cols, mapping = "False")

    if activate_radius == 2:
        get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
        get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
        get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
    elif activate_radius == 1:
        get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)

    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "P", max_mapping_radius)
    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "S", max_mapping_radius)
    # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Cl", max_mapping_radius)

    get_idx_weirdos_el(file_loc_important_cols, "Li", activate_radius)

    # idx_correcting_mapped_el(file_loc_important_cols, el="Li")
    idx_correcting_mapped_el(file_loc_important_cols, "Li", activate_radius)
    # # create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, var_savefilename = "mapLi")
    
    get_distance_weirdos_label_el(file_loc_important_cols, coor_structure_init_dict, "Li", litype)

    # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","coor_weirdos_48htypesmerged_Li","top3_dist_weirdos_dict_Li","idx0_weirdos_Li","#weirdos_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
    # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_weirdos_Li","sum_mapped_48htype2_Li_new","#weirdos_Li","sum_mapped_48htypesmerged_Li_new","sum_sanitycheck_48htypesmerged_Li_new","idx0_weirdos_Li","top3_sorted_coorweirdo_dist_label_coor120_idxweirdo_idx120_Li","duplicate_closest24_w_data_Li","duplicate_closest24_w_data_48htype2_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)

    if litype == 0:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
    elif litype == 1:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
    elif litype == 2:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
    elif litype == 3:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
    elif litype == 4:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_48htype4_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_48htype4_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
    elif litype == 5:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
    elif litype == 6:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]] 
    elif litype == 7:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]] 
    elif litype == 8:
        file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
        file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]]   


    sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())
        
    if activate_radius == 2:
        var_excel_file = f"{int(sum_weirdos_Li)}_{new_dx1_type}_{new_dx2_type}_{new_dz_type}_{max_mapping_radius}_{max_mapping_radius_48htype2}"

    elif activate_radius == 1:
        var_excel_file = f"{int(sum_weirdos_Li)}_{new_dx1_type}_{new_dx2_type}_{new_dz_type}_{max_mapping_radius}"

    path_excel_file = os.path.join(direc_perfect_poscar, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
    file_loc_important_cols_sorted_toten.to_excel(path_excel_file, index=False)

    return sum_weirdos_Li


# def varying_radius_vs_sumweirdosLi(dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, max_mapping_radius, max_mapping_radius_48htype2, delta_radius, n_sample, var_optitype):
    
#     col_radius_type1 = "radius_type1"
#     col_radius_type2 = "radius_type2"
#     col_sumweirdosLi = "sumweirdosLi"

#     radius_sumweirdosLi_df = pd.DataFrame()
#     radius_sumweirdosLi_df[col_radius_type1] = None
#     radius_sumweirdosLi_df[col_radius_type2] = None
#     radius_sumweirdosLi_df[col_sumweirdosLi] = None

#     idx_sumweirdosLi_df = 0
#     forward_max_mapping_radius = max_mapping_radius
#     forward_max_mapping_radius_48htype2 = max_mapping_radius_48htype2
#     backward_max_mapping_radius = max_mapping_radius
#     backward_max_mapping_radius_48htype2 = max_mapping_radius_48htype2

#     for i in range(int(n_sample/2)):
#         forward_max_mapping_radius += i * delta_radius
#         forward_max_mapping_radius_48htype2 += i * delta_radius
    
#         sum_weirdos_Li = get_sum_weirdos_Li_w_radius(dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, forward_max_mapping_radius, forward_max_mapping_radius_48htype2, var_optitype)

#         radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_radius_type1] = forward_max_mapping_radius
#         radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_radius_type2] = forward_max_mapping_radius_48htype2
#         radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_sumweirdosLi] = sum_weirdos_Li
#         print(idx_sumweirdosLi_df)

#     # for j in range(int(n_sample/2)):
#     #     backward_max_mapping_radius -= j * delta_radius
#     #     backward_max_mapping_radius_48htype2 -= j * delta_radius
    
#     #     sum_weirdos_Li = get_sum_weirdos_Li_w_radius(dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, backward_max_mapping_radius, backward_max_mapping_radius_48htype2, var_optitype)

#     #     radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_radius_type1] = backward_max_mapping_radius
#     #     radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_radius_type2] = backward_max_mapping_radius_48htype2
#     #     radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_sumweirdosLi] = sum_weirdos_Li
#     #     print(idx_sumweirdosLi_df)

#     return radius_sumweirdosLi_df


def plot_varying_radius_vs_sumweirdosLi(dataframe):
    col_radius_type1 = "radius_type1"
    col_radius_type2 = "radius_type2"
    col_sumweirdosLi = "sumweirdosLi"

    # # dataframe_to_float = dataframe.copy()
    # dataframe_to_float[col_radius_type1] = dataframe_to_float[col_radius_type1].apply(lambda x: float(x[0]))
    # dataframe_to_float[col_radius_type2] = dataframe_to_float[col_radius_type2].apply(lambda x: float(x[0]))
    # dataframe_to_float[col_sumweirdosLi] = dataframe_to_float[col_sumweirdosLi].apply(lambda x: float(x[0]))

    # # %matplotlib inline
    matrix_map = dataframe.pivot_table(index=col_radius_type2, columns=col_radius_type1,values=col_sumweirdosLi)  
    sns.heatmap(matrix_map)


def plot_weirdos_directcoor(dataframe, activate_radius):
    if activate_radius == 2 or activate_radius == 3:
        col_coor_weirdos_el = f"coor_weirdos_48htype2_Li"
    elif activate_radius == 1:
        col_coor_weirdos_el = f"coor_weirdos_Li"

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx in range(dataframe["geometry"].size):
        coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]

        # Plot each set of coordinates in the loop
        for coordinates in coor_weirdos_el:
            ax.scatter(*coordinates, marker='o')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Weirdos direct coordinate')

    # Show the plot
    plt.show()


def plot_distweirdos(dataframe):
    col_top1_sorted_idxweirdo_dist_el = "top1_sorted_idxweirdo_dist_Li"
    col_top1_sorted_idxweirdo_label_el = "top1_sorted_idxweirdo_label_Li"
    col_top1_sorted_idxweirdo_coor_el = "top1_sorted_idxweirdo_coor_Li"
    col_top1_sorted_idxweirdo_file_el = "top1_sorted_idxweirdo_file_Li"

    val = 0.

    dist_weirdos_el_appendend = []
    y_appended = []
    label_weirdos_el_appended = []
    coor_weirdos_el_appended = []
    file_weirdos_el_appended = []

    for idx in range(dataframe["geometry"].size):
        dist_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_dist_el].values()
        label_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_label_el].values()
        coor_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_coor_el].values()
        file_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_file_el].values()

        for single_dist in dist_weirdos_el:
            dist_weirdos_el_appendend.append(single_dist[0])
            y_appended.append(np.zeros_like(single_dist[0]) + val)

        for single_label in label_weirdos_el:
            label_weirdos_el_appended.append(single_label[0])

        for single_coor in coor_weirdos_el:
            coor_weirdos_el_appended.append(single_coor[0])
        
        for single_file in file_weirdos_el:
            file_weirdos_el_appended.append(single_file[0])

    df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'y': y_appended, 'coor': coor_weirdos_el_appended, 'file': file_weirdos_el_appended})

    # fig = px.scatter(df, x = 'dist', y = 'y', color='label', color_discrete_map={'48htype1': 'red', '48htype2': 'blue', '24g': 'green'})
    fig = px.scatter(df, 
                    x = 'dist', 
                    y = 'y', 
                    color ='label', 
                    hover_data = ['dist', 'label', 'coor', 'file'])

    fig.show(config={'scrollZoom': True})


def get_coor_weirdos_array(dataframe, activate_radius):
    if activate_radius == 2 or activate_radius == 3:
        col_coor_weirdos_el = f"coor_weirdos_48htype2_Li"
    elif activate_radius == 1:
        col_coor_weirdos_el = f"coor_weirdos_Li"

    coor_weirdos_el_appended = []

    for idx in range(dataframe["geometry"].size):
        coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]

        for single_coor in coor_weirdos_el:
            coor_weirdos_el_appended.append(single_coor)

    coor_weirdos_el_appended = np.array(coor_weirdos_el_appended)

    return coor_weirdos_el_appended


def create_POSCAR_weirdos(coor_weirdos, destination_directory, lattice_constant, filename):

    # Define lattice constants (you might need to adjust these based on your actual system)
    lattice_constants_matrix = np.array([
        [lattice_constant, 0.0000000000000000, 0.0000000000000000],
        [0.0000000000000000, lattice_constant, 0.0000000000000000],
        [0.0000000000000000, 0.0000000000000000, lattice_constant]
    ])

    # Define the header and comment lines for the POSCAR file
    header = "Generated POSCAR"
    comment = "1.0"

    # Define filename
    filename_path = os.path.join(destination_directory, filename)

    # Write the POSCAR file
    with open(filename_path, "w") as f:
        # Write the header and comment
        f.write(header + "\n")
        f.write(comment + "\n")

        # Write the lattice constants
        for row in lattice_constants_matrix:
            f.write(" ".join(map(str, row)) + "\n")

        # Write the element symbols (in this case, using 'Li' for lithium)
        f.write("Li\n")

        # Write the number of atoms for each element
        # f.write(" ".join(map(str, np.ones(len(coordinates), dtype=int))) + "\n")
        f.write(str(len(coor_weirdos)) + "\n")  # Number of Li atoms

        # Write the selective dynamics tag (in this case, 'Direct')
        f.write("Direct\n")

        # Write the atomic coordinates
        for coor in coor_weirdos:
            # f.write(" ".join(map(str, coord)) + "\n")
            formatted_coords = [format(x, ".16f") for x in coor]
            f.write(" ".join(formatted_coords) + "\n")

    # print("POSCAR file created successfully.")


def kmeans_cluster_weirdos(coor_weirdos, amount_clusters):
    # source: https://stackoverflow.com/questions/64987810/3d-plotting-of-a-dataset-that-uses-k-means
    kmeans = KMeans(n_clusters=amount_clusters)                # Number of clusters
    kmeans = kmeans.fit(coor_weirdos)                          # Fitting the input data
    labels = kmeans.predict(coor_weirdos)                      # Getting the cluster labels
    centroids = kmeans.cluster_centers_             # Centroid values
    # print("Centroids are:", centroids)              # From sci-kit learn

    fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    # x = np.array(labels==0)
    # y = np.array(labels==1)
    # z = np.array(labels==2)


    # ax.scatter(coor_weirdos[x][:, 0], coor_weirdos[x][:, 1], coor_weirdos[x][:, 2], color='red')
    # ax.scatter(coor_weirdos[y][:, 0], coor_weirdos[y][:, 1], coor_weirdos[y][:, 2], color='blue')
    # ax.scatter(coor_weirdos[z][:, 0], coor_weirdos[z][:, 1], coor_weirdos[z][:, 2], color='green')
    # ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
    #             marker='x', s=169, linewidths=10,
    #             color='black', zorder=50)
    # # ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c="black",s=150,label="Centers",alpha=1) # for dot marker

    # Define a colormap for different clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, amount_clusters))

    for cluster_label in range(amount_clusters):
        cluster_mask = np.array(labels == cluster_label)
        ax.scatter(
            coor_weirdos[cluster_mask][:, 0],
            coor_weirdos[cluster_mask][:, 1],
            coor_weirdos[cluster_mask][:, 2],
            color=colors[cluster_label],
            label=f'Cluster {cluster_label}'
        )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        marker='x',
        s=169,
        linewidths=10,
        color='black',
        zorder=50,
        label='Centroids'
    )

    ax.legend()

    return centroids, labels


def create_POSCAR_weirdos_centroids_appended(coor_weirdos, coor_centroids, destination_directory, lattice_constant, filename):

    # Define lattice constants (you might need to adjust these based on your actual system)
    lattice_constants_matrix = np.array([
        [lattice_constant, 0.0000000000000000, 0.0000000000000000],
        [0.0000000000000000, lattice_constant, 0.0000000000000000],
        [0.0000000000000000, 0.0000000000000000, lattice_constant]
    ])

    # Define the header and comment lines for the POSCAR file
    header = "Generated POSCAR"
    comment = "1.0"

    # Define filename
    filename_path = os.path.join(destination_directory, filename)

    # Write the POSCAR file
    with open(filename_path, "w") as f:
        # Write the header and comment
        f.write(header + "\n")
        f.write(comment + "\n")

        # Write the lattice constants
        for row in lattice_constants_matrix:
            f.write(" ".join(map(str, row)) + "\n")

        # K as mock element for centroids
        f.write("Li K\n")

        # Write the number of atoms for each element
        # f.write(" ".join(map(str, np.ones(len(coordinates), dtype=int))) + "\n")
        f.write(str(len(coor_weirdos)) + " " + str(len(coor_centroids)) + "\n")  # Number of Li atoms

        # Write the selective dynamics tag (in this case, 'Direct')
        f.write("Direct\n")

        # Write the atomic coordinates
        for coor_weirdo in coor_weirdos:
            # f.write(" ".join(map(str, coord)) + "\n")
            formatted_coor_weirdo = [format(x, ".16f") for x in coor_weirdo]
            f.write(" ".join(formatted_coor_weirdo) + "\n")
        for coor_centroid in coor_centroids:
            formatted_coor_centroid = [format(x, ".16f") for x in coor_centroid]
            f.write(" ".join(formatted_coor_centroid) + "\n")

    # print("POSCAR file created successfully.")
            

def create_file_loc(direc_init_system, data_toten, file_new_system):
    direc = os.getcwd()

    col_excel_geo = "geometry"
    col_excel_path = "path"
    col_excel_toten = "toten [eV]"

    geometry = np.array([])
    path = np.array([])
    subdir_col = np.array([])
    subdir_col_init_system = np.array([])
    subdir_col_perfect_poscar = np.array([])
    for subdir, dirs, files in os.walk(direc,topdown=False):
        # source: https://stackoverflow.com/questions/27805919/how-to-only-read-lines-in-a-text-file-after-a-certain-string
        for file in files:
            filepath = subdir + os.sep
            # get directory of CONTCAR
            if os.path.basename(file) == file_new_system:
                geometry_nr = FileOperations.splitall(subdir)[-2]
                path_nr = FileOperations.splitall(subdir)[-1]
                geometry = pd.DataFrame(np.append(geometry, int(geometry_nr)), columns=["geometry"])
                geometry_ori = geometry
                # geometry = geometry.applymap(func=replace)
                geometry.dropna(axis=1)
                path = pd.DataFrame(np.append(path, int(path_nr)), columns=["path"])
                # path = path.applymap(func=replace)
                path.dropna(axis=1)
                path_sorted = path.sort_values(by="path",ascending=False)
                subdir_file = os.path.join(subdir,file_new_system)
                # # create directory of POSCAR of init system
                subdir_init_system = direc_init_system + os.sep + geometry_nr + os.sep + path_nr
                # # subdir_file_init_system = os.path.join(subdir_init_system,file_init_system)
                # subdir_file_perfect_poscar = os.path.join()
                subdir_col = pd.DataFrame(np.append(subdir_col, subdir_file), columns=["subdir_new_system"])
                # # subdir_col_init_system = pd.DataFrame(np.append(subdir_col_init_system, subdir_file_init_system), columns=["subdir_init_system"])
                # # subdir_col_perfect_poscar = pd.DataFrame(np.append(subdir_col_perfect_poscar, direc_perfect_system), columns=["subdir_perfect_poscar"])
                file_loc = geometry.join(path)
                file_loc["subdir_new_system"] = subdir_col
                # # file_loc["subdir_init_system"] = subdir_col_init_system
                # # file_loc["subdir_perfect_poscar"] = subdir_col_perfect_poscar
                path_ori = path

    file_loc_ori_notsorted = file_loc.copy()
    # file_loc_ori_notsorted = file_loc
    file_loc = file_loc.sort_values(by=["geometry","path"],ignore_index=True,ascending=False) # sort descendingly based on path

    file_loc["g+p"] = (file_loc["geometry"] + file_loc["path"]).fillna(0) # replace NaN with 0
    # file_loc["g+p"] = file_loc["geometry"] + file_loc["path"]
    file_loc["g+p+1"] = file_loc["g+p"].shift(1)
    file_loc["g+p+1"][0] = 0 # replace 1st element with 0
    file_loc["g+p-1"] = file_loc["g+p"].shift(-1)
    file_loc["g+p-1"][(file_loc["g+p-1"]).size - 1] = 0.0 # replace last element with 0
    file_loc["perfect_system"] = file_loc["g+p"][(file_loc["g+p+1"] > file_loc["g+p"]) & (file_loc["g+p-1"] > file_loc["g+p"])]
    file_loc["perfect_system"][file_loc["geometry"].size-1] = 0.0 # hardcode the path 0/0
    file_loc["p_s_mask"] = [0 if np.isnan(item) else 1 for item in file_loc["perfect_system"]]
    # # subdir_filtered = file_loc["subdir"] * file_loc["p_s_mask"]


    if data_toten[col_excel_geo].all() == file_loc["geometry"].all() & data_toten[col_excel_path].all() == file_loc["path"].all():
        file_loc[col_excel_toten] = data_toten[col_excel_toten]
    else:
        print("check the compatibility of column geometry and path between data_toten file and file_loc")

    return file_loc


def create_file_loc_compact_demo(direc_init_system, data_toten, file_new_system):
    direc = os.getcwd()

    col_excel_geo = "geometry"
    col_excel_path = "path"
    col_excel_toten = "toten [eV]"

    geometry = np.array([])
    path = np.array([])
    subdir_col = np.array([])
    for subdir, dirs, files in os.walk(direc,topdown=False):
        # source: https://stackoverflow.com/questions/27805919/how-to-only-read-lines-in-a-text-file-after-a-certain-string
        for file in files:
            filepath = subdir + os.sep
            # get directory of CONTCAR
            if os.path.basename(file) == file_new_system:
                geometry_nr = FileOperations.splitall(subdir)[-2]
                path_nr = FileOperations.splitall(subdir)[-1]
                geometry = pd.DataFrame(np.append(geometry, int(geometry_nr)), columns=["geometry"])
                geometry_ori = geometry
                geometry.dropna(axis=1)
                path = pd.DataFrame(np.append(path, int(path_nr)), columns=["path"])#
                path.dropna(axis=1)
                path_sorted = path.sort_values(by="path",ascending=False)
                subdir_file = os.path.join(subdir,file_new_system)
                # # create directory of POSCAR of init system
                subdir_init_system = direc_init_system + os.sep + geometry_nr + os.sep + path_nr
                subdir_col = pd.DataFrame(np.append(subdir_col, subdir_file), columns=["subdir_new_system"])
                file_loc = geometry.join(path)
                file_loc["subdir_new_system"] = subdir_col#
                path_ori = path

    file_loc_ori_notsorted = file_loc.copy()
    file_loc = file_loc.sort_values(by=["geometry","path"],ignore_index=True,ascending=False) # sort descendingly based on path

    file_loc["g+p"] = (file_loc["geometry"] + file_loc["path"]).fillna(0) # replace NaN with 0
    file_loc["g+p+1"] = file_loc["g+p"].shift(1)
    file_loc["g+p+1"][0] = 0 # replace 1st element with 0
    file_loc["g+p-1"] = file_loc["g+p"].shift(-1)
    file_loc["g+p-1"][(file_loc["g+p-1"]).size - 1] = 0.0 # replace last element with 0
    file_loc["perfect_system"] = file_loc["g+p"][(file_loc["g+p+1"] > file_loc["g+p"]) & (file_loc["g+p-1"] > file_loc["g+p"])]
    file_loc["perfect_system"][file_loc["geometry"].size-1] = 0.0 # hardcode the path 0/0
    file_loc["p_s_mask"] = [0 if np.isnan(item) else 1 for item in file_loc["perfect_system"]]



    if data_toten[col_excel_geo].all() == file_loc["geometry"].all() & data_toten[col_excel_path].all() == file_loc["path"].all():
        file_loc[col_excel_toten] = data_toten[col_excel_toten]
    else:
        print("check the compatibility of column geometry and path between data_toten file and file_loc")

    return file_loc



##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def get_triads_movement(destination_directory, geo, var_filename, filename_ref_72):
    # df_coor = pd.DataFrame()
    df_triad = pd.DataFrame()
    df_ratio = pd.DataFrame()
    df_dist = pd.DataFrame()

    df_dist["dist"] = None
    # coor_Li_ref = []

    # col_xyz_coor = "xyz_coor"

    # df_coor[col_xyz_coor] = None

    if geo == 0:
        path_geo = path_geo_0
    elif geo == 1:
        path_geo = path_geo_1
    elif geo == 2:
        path_geo = path_geo_2
    elif geo == 3:
        path_geo = path_geo_3
    elif geo == 4:
        path_geo = path_geo_4
    elif geo == 5:
        path_geo = path_geo_5
    elif geo == 6:
        path_geo = path_geo_6
    elif geo == 7:
        path_geo = path_geo_7
    elif geo == 8:
        path_geo = path_geo_8

    file_ref_24 = f"{geo}_0_{var_filename}.cif"
    file_path_ref_24 = os.path.join(destination_directory, file_ref_24)

    file_ref_72 = f"{filename_ref_72}.cif"
    file_path_ref_72 = os.path.join(destination_directory, file_ref_72)


    idx_coor_Li_dict_ref_24 = get_idx_coor_Li_dict(file_path_ref_24)    # key is the pointer to 24
    idx_coor_Li_dict_ref_72 = get_idx_coor_Li_dict(file_path_ref_72)    # key is the pointer to 24

    idx_coor_Li_dict_ref_triad = get_idx_coor_Li_dict_ref_triad(idx_coor_Li_dict_ref_24, idx_coor_Li_dict_ref_72)
    # idxs_Li_ref_24 = list(idx_coor_Li_dict_ref_24.keys())
    # idxs_Li_ref_72 = list(idx_coor_Li_dict_ref_72.keys())

    for i in path_geo:
        # coor_Li = []
        file = f"{geo}_{i}_{var_filename}.cif"
        file_path = os.path.join(destination_directory, file)

        idx_coor_Li_dict = get_idx_coor_Li_dict(file_path)
        # idxs_Li = list(idx_coor_Li_dict.keys())

        # # idx_coor_Li_triad_belonging_initial = defaultdict(list)
        # # idx_coor_Li_triad_belonging_initial_centroid = defaultdict(list)

        ### does the numeration of Li is important?
        ### 1) check which triad it does belong to initially

        idx_coor_Li_idx_centroid_triad_ref = get_idx_coor_Li_idx_centroid_triad(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_ref_24)
        idx_coor_Li_idx_centroid_triad = get_idx_coor_Li_idx_centroid_triad(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict)
        idxs_Li_dict = [i for i in range(24) if i in idx_coor_Li_idx_centroid_triad.keys()]

        ## get ratio of 24:48
        counter_48 = 0
        for Li_idx, val in idx_coor_Li_idx_centroid_triad.items():
            # print(mic_eucledian_distance(val['coor'], val['centroid_triad']))
            if val['structure'] == 48:
                counter_48 = counter_48 + 1
        # print(f"path {i} has ratio of 48 of: {counter_48/len(idx_coor_Li_idx_centroid_triad)}")
        df_ratio.at[i, "ratio of 48"] = counter_48/len(idx_coor_Li_idx_centroid_triad)

        ## get li-to-li-distance 
        dist_ascending, sorted_coors_Li_dist_structures = get_dist_ascending(idx_coor_Li_idx_centroid_triad)
        # print(dist_ascending)
        df_dist.at[i, "dist"] = dist_ascending[1:6]

        for j in idxs_Li_dict:
            # df_triad.at[i, f"{j}"] = None  

            triad = idx_coor_Li_idx_centroid_triad[j]["idx_triad"]

            df_triad.at[i, f"{j}"] = triad

            if triad == df_triad.at[0, f"{j}"] and i != 0:
                print(f"path: {i}, Li: {j}, triad: {triad}")

    return df_triad, df_ratio, df_dist, sorted_coors_Li_dist_structures



def get_triads_fullness(destination_directory, geo, var_filename, filename_ref_72):
    # df_idx_triad_counts = pd.DataFrame #(np.zeros((24, 1)))
    # df_idx_triad_counts["idx_triad_counts"] = None

    idx_coor_Li_idx_centroid_triad_weirdos_appended_dict = defaultdict(list)

    if geo == 0:
        path_geo = path_geo_0
    elif geo == 1:
        path_geo = path_geo_1
    elif geo == 2:
        path_geo = path_geo_2
    elif geo == 3:
        path_geo = path_geo_3
    elif geo == 4:
        path_geo = path_geo_4
    elif geo == 5:
        path_geo = path_geo_5
    elif geo == 6:
        path_geo = path_geo_6
    elif geo == 7:
        path_geo = path_geo_7
    elif geo == 8:
        path_geo = path_geo_8

    df_idx_triad_counts = pd.DataFrame(np.zeros((24, len(path_geo))))

    file_ref_24 = f"{geo}_0_{var_filename}.cif"
    file_path_ref_24 = os.path.join(destination_directory, file_ref_24)

    file_ref_72 = f"{filename_ref_72}.cif"
    file_path_ref_72 = os.path.join(destination_directory, file_ref_72)

    idx_coor_Li_dict_ref_24 = get_idx_coor_Li_dict(file_path_ref_24)    # key is the pointer to 24
    idx_coor_Li_dict_ref_72 = get_idx_coor_Li_dict(file_path_ref_72)    # key is the pointer to 24

    idx_coor_Li_dict_ref_triad = get_idx_coor_Li_dict_ref_triad(idx_coor_Li_dict_ref_24, idx_coor_Li_dict_ref_72)
    # idxs_Li_ref_24 = list(idx_coor_Li_dict_ref_24.keys())
    # idxs_Li_ref_72 = list(idx_coor_Li_dict_ref_72.keys())

    for i in path_geo:
        # coor_Li = []
        file = f"{geo}_{i}_{var_filename}.cif"
        file_path = os.path.join(destination_directory, file)

        idx_coor_Li_dict = get_idx_coor_Li_dict(file_path)

        file_weirdos_appended = f"{geo}_{i}_{var_filename}_weirdos_appended.cif"
        file_path_weirdos_appended = os.path.join(destination_directory, file_weirdos_appended)

        idx_coor_Li_dict_weirdos_appended = get_idx_coor_Li_dict(file_path_weirdos_appended)
        # idx_coor_Li_dict_ref_triad_weirdos_appended = get_idx_coor_Li_dict_ref_triad(idx_coor_Li_dict_weirdos_appended, idx_coor_Li_dict_ref_72)

        idxs_Li = list(idx_coor_Li_dict.keys())
        idxs_Li_not = sorted(i for i in range(24) if i not in idxs_Li)
        # idxs_Li = list(idx_coor_Li_dict.keys())

        # # idx_coor_Li_triad_belonging_initial = defaultdict(list)
        # # idx_coor_Li_triad_belonging_initial_centroid = defaultdict(list)

        ### does the numeration of Li is important?
        ### 1) check which triad it does belong to initially

        idx_coor_Li_idx_centroid_triad_ref = get_idx_coor_Li_idx_centroid_triad(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_ref_24)
        idx_coor_Li_idx_centroid_triad = get_idx_coor_Li_idx_centroid_triad_w_closest_dist(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict)
        idx_coor_Li_idx_centroid_triad_weirdos_appended = get_idx_coor_Li_idx_centroid_triad_w_closest_dist_weirdos_appended(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_weirdos_appended, idxs_Li_not)
        # idx_coor_Li_idx_centroid_triad_weirdos_appended = get_idx_coor_Li_idx_centroid_triad_w_closest_dist(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_weirdos_appended)
        idxs_Li_dict = [i for i in range(24) if i in idx_coor_Li_idx_centroid_triad.keys()]
        
        idx_triad_array = sorted([val['idx_triad'] for val in idx_coor_Li_idx_centroid_triad.values()])
        idx_triad_array_not = sorted(i for i in range(24) if i not in idx_triad_array)
        # idxs_Li_triad_dict = [i for i in range(24) if i in idx_coor_Li_idx_centroid_triad()]

        # idx_triad_series = pd.Series(idx_triad_array)
        # df_idx_triad_counts[i] = idx_triad_series.value_counts()

        idx_triad_counts = defaultdict(int)
        # Count the occurrences of each idx_triad
        # idx_triad_counts = pd.DataFrame(np.zeros((24, 1)))
        for key, val in idx_coor_Li_idx_centroid_triad.items():
            idx_triad = val['idx_triad']
            idx_triad_counts[idx_triad] += 1
        for j in idx_triad_array_not:
            idx_triad_counts[j] = 0

        # df_idx_triad_counts.at[i, "idx_triad_counts"] = dict(idx_triad_counts)
        # df_idx_triad_counts = pd.DataFrame(np.zeros((24, 1)))
        df_idx_triad_counts[i] = dict(idx_triad_counts)
        # df_idx_triad_counts[i].fillna(0)

        idx_coor_Li_idx_centroid_triad_weirdos_appended_dict[i] = dict(idx_coor_Li_idx_centroid_triad_weirdos_appended)

    return df_idx_triad_counts, idx_coor_Li_idx_centroid_triad_weirdos_appended_dict


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def plot_mapped_label_vs_dist_and_histogram(dataframe, litype, category_data, el):
    """
        category_data: mapping, weirdo
        TO DO: correct the map for histogram
    """
    dist_weirdos_el_appendend = []
    label_weirdos_el_appended = []
    idx_appended = []

    if category_data == "mapping":
        col_dist_label_el = f"atom_mapping_{el}_w_dist_label"
    elif category_data == "weirdo":
        col_dist_label_el = f'top1_sorted_idxweirdo_dist_label_{el}'
        col_top1_sorted_idxweirdo_coor_el = f"top1_sorted_idxweirdo_coor_{el}"
        coor_weirdos_el_appended = []    

    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist_label = dataframe.at[idx, col_dist_label_el]
        if category_data == "weirdo":
            coor_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_coor_el].values()

        for i in atom_mapping_el_w_dist_label.values():
            if category_data == "mapping":
                dist = i['dist']
                label = i['label']
            elif category_data == "weirdo":
                dist = i[0]['dist']
                label = i[0]['label']

            dist_weirdos_el_appendend.append(dist)
            label_weirdos_el_appended.append(label)
            idx_appended.append(idx)

        if category_data == "weirdo":
            for single_coor in coor_weirdos_el:
                coor_weirdos_el_appended.append(single_coor[0])

    if category_data == "mapping":
        df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'idx_nr': idx_appended})
    elif category_data == "weirdo":
        df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'idx_nr': idx_appended, 'coor': coor_weirdos_el_appended})

    fig = px.scatter(df, 
                    x = 'label', 
                    y = 'dist',
                    title = 'Mapped atom type vs its distance'
                    )

    fig.show(config={'scrollZoom': True})

    # if litype == 0:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
    # elif litype == 1:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')
    # elif litype == 2:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')
    # elif litype == 3:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')
    # elif litype == 4:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')
    # elif litype == 5:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')

    #     df_48htype5 = df.loc[df['label'] == '48htype5']
    #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype5')
    # elif litype == 6:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')

    #     df_48htype5 = df.loc[df['label'] == '48htype5']
    #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype5')

    #     df_48htype6 = df.loc[df['label'] == '48htype6']
    #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype6')
    # elif litype == 7:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')

    #     df_48htype5 = df.loc[df['label'] == '48htype5']
    #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype5')

    #     df_48htype6 = df.loc[df['label'] == '48htype6']
    #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype6')

    #     df_48htype7 = df.loc[df['label'] == '48htype7']
    #     plt.hist(df_48htype7['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype7')
    # elif litype == 8:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')

    #     df_48htype5 = df.loc[df['label'] == '48htype5']
    #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype5')

    #     df_48htype6 = df.loc[df['label'] == '48htype6']
    #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype6')

    #     df_48htype7 = df.loc[df['label'] == '48htype7']
    #     plt.hist(df_48htype7['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype7')

    #     df_48htype8 = df.loc[df['label'] == '48htype8']
    #     plt.hist(df_48htype8['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype8')

    return df


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def get_tuple_metainfo(coor_structure_init_dict_expanded, litype, el):
    coor_structure_init_dict_expanded_el = coor_structure_init_dict_expanded[el]
    
    if litype == 1:
        n = 3
    else:
        n = ((litype * 2) - 1)

    tuple_metainfo = {}

    coor_li24g_ref      = coor_structure_init_dict_expanded_el[0:24]
    if litype == 1:
        coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]
    elif litype == 2:
        coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]
        coor_li48htype2_ref = coor_structure_init_dict_expanded_el[72:120]
    elif litype == 3:
        coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]
        coor_li48htype2_ref = coor_structure_init_dict_expanded_el[72:120]
        coor_li48htype3_ref = coor_structure_init_dict_expanded_el[120:168]
    elif litype == 4:
        coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]
        coor_li48htype2_ref = coor_structure_init_dict_expanded_el[72:120]
        coor_li48htype3_ref = coor_structure_init_dict_expanded_el[120:168]
        coor_li48htype4_ref = coor_structure_init_dict_expanded_el[168:216]
    elif litype == 5:
        coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]
        coor_li48htype2_ref = coor_structure_init_dict_expanded_el[72:120]
        coor_li48htype3_ref = coor_structure_init_dict_expanded_el[120:168]
        coor_li48htype4_ref = coor_structure_init_dict_expanded_el[168:216]
        coor_li48htype5_ref = coor_structure_init_dict_expanded_el[216:264]
    elif litype == 6:
        coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]
        coor_li48htype2_ref = coor_structure_init_dict_expanded_el[72:120]
        coor_li48htype3_ref = coor_structure_init_dict_expanded_el[120:168]
        coor_li48htype4_ref = coor_structure_init_dict_expanded_el[168:216]
        coor_li48htype5_ref = coor_structure_init_dict_expanded_el[216:264]
        coor_li48htype6_ref = coor_structure_init_dict_expanded_el[264:312]
    elif litype == 7:
        coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]
        coor_li48htype2_ref = coor_structure_init_dict_expanded_el[72:120]
        coor_li48htype3_ref = coor_structure_init_dict_expanded_el[120:168]
        coor_li48htype4_ref = coor_structure_init_dict_expanded_el[168:216]
        coor_li48htype5_ref = coor_structure_init_dict_expanded_el[216:264]
        coor_li48htype6_ref = coor_structure_init_dict_expanded_el[264:312]
        coor_li48htype7_ref = coor_structure_init_dict_expanded_el[312:360]
    elif litype == 8:
        coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]
        coor_li48htype2_ref = coor_structure_init_dict_expanded_el[72:120]
        coor_li48htype3_ref = coor_structure_init_dict_expanded_el[120:168]
        coor_li48htype4_ref = coor_structure_init_dict_expanded_el[168:216]
        coor_li48htype5_ref = coor_structure_init_dict_expanded_el[216:264]
        coor_li48htype6_ref = coor_structure_init_dict_expanded_el[264:312]
        coor_li48htype7_ref = coor_structure_init_dict_expanded_el[312:360]
        coor_li48htype8_ref = coor_structure_init_dict_expanded_el[360:408]

    tuple_metainfo_all = defaultdict(list)

    for idx_i, i in enumerate(coor_li24g_ref):

        tuple_metainfo_24g_dict =  {'coor': i, 'dist': 0.0, 'type': '24g'}

        tuple_metainfo_all[idx_i].append(tuple_metainfo_24g_dict)

        if litype == 1:
            for j in coor_li48htype1_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
        
        elif litype == 2:
            for j in coor_li48htype1_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype2_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 3:
            for j in coor_li48htype1_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 4:
            for j in coor_li48htype1_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 5:
            for j in coor_li48htype1_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype5_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 6:
            for j in coor_li48htype1_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype5_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype6_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 7:
            for j in coor_li48htype1_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype5_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype6_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype7_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype7'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
        elif litype == 8:
            for j in coor_li48htype1_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype5_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype6_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype7_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype7'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype8_ref:
                distance = mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype8'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)


    sorted_tuple_metainfo_all = {key: sorted(value, key=lambda x: x['dist']) for key, value in tuple_metainfo_all.items()}
    top_n_tuple_metainfo = {k: v[0:n] for k, v in sorted_tuple_metainfo_all.items()}

    for key, values_list in top_n_tuple_metainfo.items():
        selected_values = [{'coor': entry['coor'], "type": entry["type"]} for entry in values_list]
        tuple_metainfo[key] = selected_values
                        
    return tuple_metainfo   


def get_occupancy(dataframe, coor_structure_init_dict_expanded, tuple_metainfo, destination_directory, var_filename, el):
    col_occupancy = "occupancy"
    col_coor24li_tuple_cage_belongin = "coor24li_tuple_cage_belongin"

    dataframe[col_occupancy] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor24li_tuple_cage_belongin] = [{} for _ in range(len(dataframe.index))]

    coor_structure_init_dict_expanded_el = coor_structure_init_dict_expanded[el]
    coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]

    for idx in range(dataframe["geometry"].size):
        coor24li_tuple_cage_belongin = defaultdict(list)

        file_24Li = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}.cif"
        file_path_24Li = os.path.join(destination_directory, file_24Li)

        coor_structure_24Li_dict_el = get_coor_dict_structure(Structure.from_file(file_path_24Li))[el]
        
        # Convert lists of arrays to sets for efficient comparison
        set_coor_structure = set(map(tuple, coor_structure_24Li_dict_el))
        set_coor_li48htype1_ref = set(map(tuple, coor_li48htype1_ref))

        # Find the difference between the two sets
        result_set = set_coor_structure.difference(set_coor_li48htype1_ref)

        # Convert the result back to a list of arrays
        result_list = list(map(np.array, result_set))
        # for idx_triad, val in tuple_metainfo.items():

        for idx_triad, values_list in tuple_metainfo.items():
            coor24li_tuple_cage_belongin[idx_triad] = []
            
            for entry in values_list:
                for i in result_list:
            
                    if (i == entry['coor']).all():
                        # if (tuple(i) == tuple(entry['coor'])).all():
                        # coor24li_tuple_belongin_dict = {'coor': i, 'type':entry['type']}
                        coor24li_tuple_cage_belongin_dict = {'coor': i, 'type':entry['type'], 'idx_cage':entry['idx_cage']}
                        coor24li_tuple_cage_belongin[idx_triad].append(coor24li_tuple_cage_belongin_dict)

        # idx_coor_weirdos_Li_dict = dataframe['idx_coor_weirdos_Li'][idx]

        # for idx_weirdo, values_list in idx_coor_weirdos_Li_dict.items():
        #         coorweirdo_tuple_belongin_dict = {'coor': values_list, 'type':'weirdo'}
        #         coor24li_tuple_cage_belongin['weirdo'].append(coorweirdo_tuple_belongin_dict)
        
        # for key, val in coor24li_tuple_cage_belongin.items():
        #     for i

        len_occupancy = []
        for key, val in coor24li_tuple_cage_belongin.items():
            len_occupancy.append(len(val))


        amount_48htype1 = (len(coor_structure_24Li_dict_el)-len(result_list))
        amount_weirdo = dataframe['#weirdos_Li'][idx]
        occupancy_2 = len_occupancy.count(2)
        occupancy_1 = len_occupancy.count(1)
        occupancy_0 = len_occupancy.count(0) - amount_48htype1 - amount_weirdo

        sanity_check_occupancy = occupancy_2 * 2 + occupancy_1

        # if sanity_check_occupancy != 24:
        #     sys.exit()

        # print(f"idx: {idx}")

        # if sanity_check_occupancy != 24:
        #     sys.exit()

        occupancy = {'2': occupancy_2, '1': occupancy_1, '0': occupancy_0, '48htype1': amount_48htype1,'weirdo': amount_weirdo}

        dataframe.at[idx, col_occupancy] = occupancy
        dataframe.at[idx, col_coor24li_tuple_cage_belongin] = coor24li_tuple_cage_belongin


def get_idx_cage_coor_24g(coor_24g_array, labels, idx_coor_cage_order, amount_clusters):
    idx_cage_coor_24g = defaultdict(list)

    # idx_coor_cage_order = {0_1: np.array([0.97111, 0.25   , 0.25   ]), 3_4: np.array([0.02889, 0.75   , 0.25   ]),
    #                        1_3: np.array([0.02889, 0.25   , 0.75   ]), 2_2: np.array([0.97111, 0.75   , 0.75   ])}

    for idx_cluster in range(amount_clusters):
        idx_cage_coor_24g[idx_cluster] = []

        cluster_mask = np.array(labels == idx_cluster)

        # print(f"idx_cluster:{idx_cluster}\n{coor_24g_array[np.array(labels == idx_cluster)]}")

        coors_cluster = coor_24g_array[cluster_mask]
        # print(coor_cluster)

        for coor_cluster in coors_cluster: 

            # coor_cluster_rounded = tuple(round(coordinate, 5) for coordinate in coor_cluster)
            # # print(coor_cluster_rounded)
            # print(coor_cluster)

            idx_cage_coor_24g[idx_cluster].append(coor_cluster)

    updated_idx_cage_coor_24g = {}

    # Iterate over idx_coor_cage_order
    for new_key, coor in idx_coor_cage_order.items():
        # Find the corresponding key in idx_cage_coor_24g based on coor
        for old_key, coor_list in idx_cage_coor_24g.items():
            if any((coor == c).all() for c in coor_list):
                updated_idx_cage_coor_24g[new_key] = coor_list
                break

    return updated_idx_cage_coor_24g


def get_tuple_cage_metainfo(tuple_metainfo, idx_cage_coor_24g):
    # Good. use further!

    tuple_cage_metainfo = tuple_metainfo.copy()

    # for idx_tuple, value_list in tuple_cage_metainfo.items():
    #     # Iterate over values in idx_cage_coor_24g
    #     for idx_cage, coor_list in idx_cage_coor_24g.items():
    #         # Iterate over coordinate lists
    #         for coor in coor_list:
    #             # Check if coor matches any 'coor' in value_list with 'type': '24g'
    #             for item in value_list:
    #                 if item['type'] == '24g' and (item['coor'] == coor).all():
    #                     # Assign idx_cage to the matching value
    #                     item['idx_cage'] = idx_cage
    #                 elif item['type'] != '24g':
    #                     item['idx_cage'] = idx_cage

    # Iterate over tuple_metainfo
    for key, value_list in tuple_cage_metainfo.items():
        # Initialize idx_cage for the current group
        current_idx_cage = None
        
        # Iterate over values in idx_cage_coor_24g
        for idx_cage, coor_list in idx_cage_coor_24g.items():
            # Iterate over coordinate lists
            for coor in coor_list:
                # Check if coor matches any 'coor' in value_list with 'type': '24g'
                for item in value_list:
                    if item['type'] == '24g' and (item['coor'] == coor).all():
                        # Assign idx_cage to the matching value
                        item['idx_cage'] = idx_cage
                        current_idx_cage = idx_cage
        
        # Assign the same idx_cage to other types in the current group
        if current_idx_cage is not None:
            for item in value_list:
                if item['type'] != '24g':
                    item['idx_cage'] = current_idx_cage

    return tuple_cage_metainfo


def get_complete_closest_tuple(dataframe, tuple_metainfo):
    col_coor24li_tuple_cage_belongin = "coor24li_tuple_cage_belongin"
    col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"

    col_idx_coor24li_tuple_cage_belongin = "idx_coor24li_tuple_cage_belongin"
    col_idx_coor24li_tuple_cage_belongin_complete_closest = "idx_coor24li_tuple_cage_belongin_complete_closest"
    col_top_n_distance_coors = "top_n_distance_coors"

    dataframe[col_idx_coor24li_tuple_cage_belongin] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor24li_tuple_cage_belongin_complete_closest] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top_n_distance_coors] = [{} for _ in range(len(dataframe.index))]
    
    for idx in range(dataframe["geometry"].size):
        idx_coor24li_tuple_cage_belongin = defaultdict(list)

        idx_coor_limapped_weirdos_dict = dataframe[col_idx_coor_limapped_weirdos_dict][idx]
        coor24li_tuple_cage_belongin = dataframe[col_coor24li_tuple_cage_belongin][idx]

        for key_a, val_a in idx_coor_limapped_weirdos_dict.items():
            idx_li = key_a
            coor_li_mapped_a = val_a['coor']
            coor_li_mapped_a_rounded = tuple(round(coordinate, 5) for coordinate in coor_li_mapped_a)
            label_li_a = val_a['label']

            idx_coor24li_tuple_cage_belongin[idx_li] = []
            for key_b, val_b in coor24li_tuple_cage_belongin.items():
                idx_tuple = key_b
                for entry_b in val_b:
                    coor_li_mapped_b = entry_b['coor']
                    coor_li_mapped_b_rounded = tuple(round(coordinate, 5) for coordinate in coor_li_mapped_b)
                    label_li_b = entry_b['type']
                    idx_cage_b = entry_b['idx_cage']

                    if (coor_li_mapped_a_rounded == coor_li_mapped_b_rounded) and (label_li_a == label_li_b):
                        # idx_coor24li_tuple_belongin_val = {'coor': coor_li_mapped_a, 'type':label_li_a, 'idx_tuple':idx_tuple}
                        idx_coor24li_tuple_cage_belongin_val = {'coor': coor_li_mapped_a, 'type':label_li_a, 'idx_tuple':idx_tuple, 'idx_cage':idx_cage_b}
                        idx_coor24li_tuple_cage_belongin[idx_li].append(idx_coor24li_tuple_cage_belongin_val)
        
        dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin] = idx_coor24li_tuple_cage_belongin
                        
        distance_coors_all = defaultdict(list)
        n = 3
        idx_coor_limapped_weirdos_dict = dataframe[col_idx_coor_limapped_weirdos_dict][idx]
        idx_coor24li_tuple_cage_belongin = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin]
        # idx_coor24li_tuple_cage_belongin_complete_closest = idx_coor24li_tuple_cage_belongin.copy()
        idx_coor24li_tuple_cage_belongin_complete_closest = defaultdict(list)

        for key_c, val_c in idx_coor24li_tuple_cage_belongin.items():
            idx_li = key_c
            idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = []

            if val_c == []:
                coor_li_mapped_c = idx_coor_limapped_weirdos_dict[idx_li]['coor']
                label_li_c = idx_coor_limapped_weirdos_dict[idx_li]['label']

                distance_prev = float("inf")
                closest_idx_tuple = None
                closest_idx_cage = None
                
                for key_d, val_d in tuple_metainfo.items():
                    for entry_d in val_d: 
                        idx_tuple = key_d
                        coor_tuple_d = entry_d['coor']
                        label_li_d = entry_d['type']
                        idx_cage_d = entry_d['idx_cage']

                        distance = mic_eucledian_distance(coor_li_mapped_c, coor_tuple_d)

                        # distance_coors_all_val = {'coor_li_mapped': coor_li_mapped_c, 'coor_tuple': coor_tuple_d, 'dist': distance, 'label':label_li_d}

                        distance_coors_all_val = {'coor_tuple': coor_tuple_d, 'dist': distance, 'label':label_li_d, 'idx_tuple':idx_tuple, 'idx_cage':idx_cage_d}

                        distance_coors_all[idx_li].append(distance_coors_all_val)

                        if distance < distance_prev:
                            distance_prev = distance
                            closest_idx_tuple = idx_tuple
                            closest_idx_cage = idx_cage_d

                idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = {'coor': coor_li_mapped_c, 'type': label_li_c, 'idx_tuple': closest_idx_tuple, 'idx_cage': closest_idx_cage}

            elif val_c != []:
                for entry_c in val_c: 
                    coor_li_mapped_c = entry_c['coor']
                    label_li_c = entry_c['type']
                    idx_tuple_c = entry_c['idx_tuple']
                    idx_cage_c = entry_c['idx_cage']

                    idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = {'coor': coor_li_mapped_c, 'type': label_li_c, 'idx_tuple': idx_tuple_c, 'idx_cage': idx_cage_c}

        sorted_distance_coors_all = {key: sorted(value, key=lambda x: x['dist']) for key, value in distance_coors_all.items()}
        top_n_distance_coors = {k: v[0:n] for k, v in sorted_distance_coors_all.items()}
        # !!! assumed there's NO DUPLICATE with the SECOND distance

        dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest] = idx_coor24li_tuple_cage_belongin_complete_closest
        dataframe.at[idx, col_top_n_distance_coors] = top_n_distance_coors


def weighing_movement(dataframe, litype):
    col_idx_coor24li_tuple_cage_belongin_complete_closest = "idx_coor24li_tuple_cage_belongin_complete_closest"
    col_idx_coor24li_tuple_cage_belongin_complete_closest_weight = "idx_coor24li_tuple_cage_belongin_complete_closest_weight"

    dataframe[col_idx_coor24li_tuple_cage_belongin_complete_closest_weight] = [{} for _ in range(len(dataframe.index))]

    multiplicator = litype + 2

    # TO DO: to be refined with different litype
    if litype == 4:
        weight_24g = 0
        weight_48htype4 = 1
        weight_48htype2 = 2
        weight_48htype3 = 3
        weight_48htype1 = 4
        weight_weirdos = 5 

    for idx in range(dataframe["geometry"].size):
        # dict_weighted = defaultdict(list)

        idx_coor24li_tuple_cage_belongin_complete_closest = dataframe[col_idx_coor24li_tuple_cage_belongin_complete_closest][idx]

        idx_coor24li_tuple_cage_belongin_complete_closest_weight = defaultdict(list)

        for key_a, val_a in idx_coor24li_tuple_cage_belongin_complete_closest.items():
            idx_li = key_a
            coor_li_mapped = val_a['coor']
            type = val_a['type']
            idx_tuple = val_a['idx_tuple']
            idx_cage = val_a['idx_cage']

            if type == "24g":
                weighted_type = weight_24g
            elif type == "48htype4":
                weighted_type = weight_48htype4
            elif type == "48htype2":
                weighted_type = weight_48htype2
            elif type == "48htype3":
                weighted_type = weight_48htype3
            elif type == "48htype1":
                weighted_type = weight_48htype1
            elif type == "weirdos":
                weighted_type = weight_weirdos
            else:
                print("wrong type")
            
            weight = idx_tuple * multiplicator + weighted_type
        
            idx_coor24li_tuple_cage_belongin_complete_closest_weight[idx_li] = {'coor': coor_li_mapped, 'type': type, 'idx_tuple': idx_tuple, 'weight':weight, 'idx_cage': idx_cage}

        dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest_weight] = idx_coor24li_tuple_cage_belongin_complete_closest_weight


def plot_movement(dataframe, to_plot):
    """
    to_plot = idx_tuple, type, idx_cage
    """
    col_idx_coor24li_tuple_cage_belongin_complete_closest_weight = "idx_coor24li_tuple_cage_belongin_complete_closest_weight"

    df_to_plot = pd.DataFrame()

    for idx in range(dataframe["geometry"].size):

        idx_coor24li_tuple_cage_belongin_complete_closest_weight = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest_weight]

        for j in range(len(idx_coor24li_tuple_cage_belongin_complete_closest_weight)):
            df_to_plot.at[idx, f"{j}"] = None  

            # coor_Li_ref_mean = np.mean(coor_Li_ref, axis=0)
            # distance = mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

            # dict_weighted[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref_mean}, coor_Li: {coor_Li[j]}'}
            
            for key_b, val_b in idx_coor24li_tuple_cage_belongin_complete_closest_weight.items():
                # for entry_b in val_b: 
                df_to_plot.at[idx, f"{key_b}"] = val_b[f'{to_plot}']

            # diameter_24g48h = max_mapping_radius * 2
            # # if distance < diameter_24g48h and index != idx_ref:
            # if distance > diameter_24g48h and idx != idx_ref:
            #     print(f"path: {idx}, Li: {j}, distance: {distance}")

    return df_to_plot


def plot_occupancy(dataframe, category_labels = None):
    col_occupancy = "occupancy"

    df = pd.DataFrame()
    df['idx_file'] = None
    df['2'] = None
    df['1'] = None
    df['0'] = None
    df['48htype1'] = None
    df['weirdo'] = None

    for idx in range(dataframe["geometry"].size):

        occupancy = dataframe.at[idx, col_occupancy]

        # for key, val in occupancy.items():
        df.at[idx, 'idx_file'] = idx
        df.at[idx, '2'] = occupancy['2']
        df.at[idx, '1'] = occupancy['1']
        df.at[idx, '0'] = occupancy['0']
        df.at[idx, '48htype1'] = occupancy['48htype1']
        df.at[idx, 'weirdo'] = occupancy['weirdo']

    wide_df = pd.DataFrame(df)

    # Convert wide format to long format
    # long_df = pd.melt(wide_df, var_name='Category', value_name='Count')
    long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

    if category_labels:
        long_df['category'] = long_df['category'].replace(category_labels)

    fig = px.bar(long_df, x="idx_file", y="count", color="category", title="Idx of file vs Occupancy")
    fig.show()

    return df


def get_plot_movement_counted(df_movement):
    df = pd.DataFrame()
    df['idx_file'] = None
    df['inTERcage'] = None
    df['intracage'] = None
    df['intratriad'] = None
    df['staying'] = None

    for i in range(len(df_movement)):
        counter_inTERcage = 0
        counter_intracage = 0
        counter_intratriad = 0
        counter_staying = 0

        for j in df_movement.iloc[i]:
            # print(j)
            if j == 'inTERcage':
                counter_inTERcage = counter_inTERcage + 1
            elif j == 'intracage':
                counter_intracage = counter_intracage + 1
            elif j == 'intratriad':
                counter_intratriad = counter_intratriad + 1
            elif j == 'staying':
                counter_staying = counter_staying + 1

        df.at[i, 'idx_file'] = i
        df.at[i, 'inTERcage'] = counter_inTERcage
        df.at[i, 'intracage'] = counter_intracage
        df.at[i, 'intratriad'] = counter_intratriad
        df.at[i, 'staying'] = counter_staying

    wide_df = pd.DataFrame(df)

    # Convert wide format to long format
    # long_df = pd.melt(wide_df, var_name='Category', value_name='Count')
    long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

    long_df['idx_file'] += 0.5

    fig = px.bar(long_df, x="idx_file", y="count", color="category", title="Idx of movement vs Category")
    fig.show()

    return df


def plot_distance_wrtpath0(df_distance, max_mapping_radius, activate_shifting_x, activate_diameter_line, Li_idxs):

    diameter_24g48h = max_mapping_radius * 2

    # x = df_distance.index
    if activate_shifting_x == True:
        x = [xi + 0.5 for xi in range(len(df_distance))]
    else:
        x = range(len(df_distance))

    # # fig = plt.figure()
    # fig = plt.figure(figsize=(800/96, 600/96))  # 800x600 pixels, assuming 96 DPI
    # ax = plt.subplot(111)

    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size in inches

    lines = []

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Example color list

    # for i in df_distance.index:
    for i in range(len(df_distance.columns)):

        line_color = colors[i % len(colors)]  # Cycle through colors list

        if Li_idxs == "all" or i in Li_idxs:
            # # i = i
            # # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
            line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", color=line_color, linewidth=2)  # Set line width to 2 pixels
            lines.append(line)
            # label = f"{i}" if Li_idxs == "all" else None
            # line, = ax.plot(x, df_distance[f"{i}"], label=label)
            # lines.append(line)

        # if type(Li_idxs) == list:
        #     for j in Li_idxs:
        #         if i == j:
        #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
        #             lines.append(line)

    # # ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}')
    if activate_diameter_line == True:
        ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}', linewidth=1)  # Set line width to 1 pixel

    # plt.title(f"Geometry {geo} with d={diameter_24g48h}")
        
    # Explicitly set x-ticks
    # ax.set_xticks(x)
    if activate_shifting_x == True:
        ax.set_xticks([0,1,2,3,4,5,6,7,8])

    # Optionally, if you want to label each tick with the original index before adjustment:
    # # # if activate_shifting_x == True:
    # # #     ax.set_xticklabels([str(int(xi - 0.5)) for xi in x])

    # Shrink current axis's height by 10% on the bottom
        # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)

    # Enable cursor information
    mplcursors.cursor(hover=True)

    # Enable zooming with cursor
    mpldatacursor.datacursor(display='multiple', draggable=True)

    plt.show()



def plot_distance_wrtpath0_sign(df_distance, df_type, df_idx_tuple, max_mapping_radius, amount_Li, category_labels, activate_diameter_line, Li_idxs):

    # df_distance = df_distance.iloc[:,:amount_Li]
    # df_type = df_type.iloc[:,:amount_Li]
    # df_idx_tuple = df_idx_tuple.iloc[:,:amount_Li]

    diameter_24g48h = max_mapping_radius * 2

    x = range(len(df_distance))

    # # fig = plt.figure()
    # fig = plt.figure(figsize=(800/96, 600/96))  # 800x600 pixels, assuming 96 DPI
    # ax = plt.subplot(111)

    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size in inches

    lines = []
    texts = []

    # type_marker_mapping = {
    #     '48htype1': 'o',
    #     '48htype2': 's',
    #     '48htype3': '^',
    #     '48htype4': 'D',
    #     'weirdos': 'X',
    #     '24g': 'v'    
    # }

    type_marker_mapping = {
        '48htype1': ('o', 'r'),  # Example: Circle marker with red color
        '48htype2': ('s', 'g'),  # Square marker with green color
        '48htype3': ('^', 'b'),  # Triangle marker with blue color
        '48htype4': ('D', 'c'),  # Diamond marker with cyan color
        'weirdos': ('X', 'm'),   # X marker with magenta color
        '24g': ('v', 'y')        # Triangle_down marker with yellow color
    }

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Example color list
    # colors = list(mcolors.CSS4_COLORS.values())
    # colors = [color + (0.7,) for color in mcolors.CSS4_COLORS.values()]
    # colors = mcolors
    # names = list(colors)

    # Define offsets for text position
    x_offset = 0.02  # Adjust these values as needed
    y_offset = -0.05  # Adjust these values as needed

    # Track which labels have been added
    added_labels = set()

    # for i in range(24):
    for i in range(len(df_distance.columns)):
        if Li_idxs == "all" or i in Li_idxs:
            column_data = df_distance[f"{i}"]
            column_val = df_type[f"{i}"]
            column_idx_tuple = df_idx_tuple[f"{i}"]
            # type_val = df_type[0, i]
            # print(type_val)

            line_color = colors[i % len(colors)]  # Cycle through colors list
            # # # # # # # line_color = colors[i % len(colors)] if i < len(colors) else 'black'  # Use a default color if the index exceeds available colors

            # # # for j in x:
            for j, (y_val, type_val, idx_tuple_val) in enumerate(zip(column_data, column_val, column_idx_tuple)):
                # type = column_val[j]
                # idx_tuple = column_idx_tuple[j]

                # marker_style = type_marker_mapping.get(column_val, 'o')  # Get the marker style for the type
                # # marker_style = type_marker_mapping.get(type, 'o')  # Get the marker style for the type
                marker_style, marker_color = type_marker_mapping.get(type_val, ('o','k'))  # Get the marker style for the type
                # # # # # # ax.scatter(j, df_distance[f"{i}"][j], label=f"Type: {column_val}", marker=marker_style, s=100)
                # # # # # label = f"{type_val}" if type_val not in added_labels else None
                # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100)
                # # # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
                # # # # # ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
                # # # # # added_labels.add(type_val)
                mapped_label = category_labels.get(type_val, type_val)  # Use the original type_val if it's not found in category_labels
                # Use mapped_label for the label. Only add it if it's not already added.
                label = mapped_label if mapped_label not in added_labels else None
                ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color=marker_color, alpha=0.5)
                if label:  # If a label was added, record it as added
                    added_labels.add(mapped_label)

                # # # # ax.text(j, df_distance.iloc[j, i], str(int(idx_tuple_val)), color=line_color, fontsize=20)
                # # # # # ax.text(j, y_val, str(int(idx_tuple_val)), color=line_color, fontsize=20)
                # Apply offsets to text position
                text_x = j + x_offset * ax.get_xlim()[1]  # Adjust text x-position
                text_y = y_val + y_offset * ax.get_ylim()[1]  # Adjust text y-position
                
                # # # # # # text = ax.text(j+x_offset, y_val+y_offset, str(int(idx_tuple_val)), color=line_color, fontsize=15)
                text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=line_color, fontsize=18)
                texts.append(text)

            # # i = i
            # # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
            # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", linewidth=2, marker=marker_style, markersize=10)  # Set line width to 2 pixels

            line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", color=line_color, linewidth=2)  # Set line width to 2 pixels
            # ax.text(i, value, str(int(idx_value)), color=line_color, fontsize=8)

            lines.append(line)
            # label = f"{i}" if Li_idxs == "all" else None
            # line, = ax.plot(x, df_distance[f"{i}"], label=label)
            # lines.append(line)

        # if type(Li_idxs) == list:
        #     for j in Li_idxs:
        #         if i == j:
        #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
        #             lines.append(line)

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    # # ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}')
    if activate_diameter_line == True:
        ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}', linewidth=1)  # Set line width to 1 pixel

    # Set the y-axis to only show ticks at 0, 1, 2, 3
    plt.yticks([0, 1, 2, 3])

    # plt.title(f"Geometry {geo} with d={diameter_24g48h}")

    # Shrink current axis's height by 10% on the bottom
        # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)

    # Enable cursor information
    mplcursors.cursor(hover=True)

    # Enable zooming with cursor
    mpldatacursor.datacursor(display='multiple', draggable=True)

    plt.show()


def get_df_movement(dataframe):
    col_idx_coor24li_tuple_cage_belongin_complete_closest_weight = "idx_coor24li_tuple_cage_belongin_complete_closest_weight"

    df_to_plot = pd.DataFrame()

    for idx in range(dataframe["geometry"].size - 1):  # CHANGED HERE

        idx_coor24li_tuple_cage_belongin_complete_closest_weight = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest_weight]
        idx_coor24li_tuple_cage_belongin_complete_closest_weight_next = dataframe.at[idx+1, col_idx_coor24li_tuple_cage_belongin_complete_closest_weight]

        for j in range(len(idx_coor24li_tuple_cage_belongin_complete_closest_weight)):
            df_to_plot.at[idx, f"{j}"] = None  

            type = idx_coor24li_tuple_cage_belongin_complete_closest_weight[j]['type']
            idx_tuple = idx_coor24li_tuple_cage_belongin_complete_closest_weight[j]['idx_tuple']
            idx_cage = idx_coor24li_tuple_cage_belongin_complete_closest_weight[j]['idx_cage']

            type_next = idx_coor24li_tuple_cage_belongin_complete_closest_weight_next[j]['type']
            idx_tuple_next = idx_coor24li_tuple_cage_belongin_complete_closest_weight_next[j]['idx_tuple']
            idx_cage_next = idx_coor24li_tuple_cage_belongin_complete_closest_weight_next[j]['idx_cage']

            if idx_cage != idx_cage_next:
                type_movement = 'inTERcage'
            elif idx_cage == idx_cage_next and idx_tuple != idx_tuple_next:
                type_movement = 'intracage'
            elif idx_cage == idx_cage_next and idx_tuple == idx_tuple_next and type != type_next:
                type_movement = 'intratriad'
            elif idx_cage == idx_cage_next and idx_tuple == idx_tuple_next and type == type_next:
                type_movement = 'staying'

            df_to_plot.at[idx, f"{j}"] = type_movement

            # coor_Li_ref_mean = np.mean(coor_Li_ref, axis=0)
            # distance = mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

            # dict_weighted[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref_mean}, coor_Li: {coor_Li[j]}'}
            
            # for key_b, val_b in idx_coor24li_tuple_cage_belongin_complete_closest_weight.items():
            #     type = val_b['type']
            #     idx_tuple = val_b['idx_tuple']
            #     idx_cage = val_b['idx_cage']
                # for entry_b in val_b: 
                # df_to_plot.at[idx, f"{key_b}"] = val_b[f'{to_plot}']

            # diameter_24g48h = max_mapping_radius * 2
            # # if distance < diameter_24g48h and index != idx_ref:
            # if distance > diameter_24g48h and idx != idx_ref:
            #     print(f"path: {idx}, Li: {j}, distance: {distance}")
    return df_to_plot


def get_amount_type(dataframe, litype, el):
    col_atom_mapping_el_w_dist_label = f"atom_mapping_{el}_w_dist_label"
    col_amount_weirdos_el = f'#weirdos_{el}'

    col_amount_type = f"amount_type_{el}"

    dataframe[col_amount_type] = [{} for _ in range(len(dataframe.index))]

    for idx in range(dataframe["geometry"].size):

        atom_mapping_el_w_dist_label = dataframe[col_atom_mapping_el_w_dist_label][idx]
        amount_weirdo = dataframe[col_amount_weirdos_el][idx]

        label_count = {}

        if litype == 0:
            labels = ["24g", "weirdo"]
        elif litype == 1: 
            labels = ["48htype1", "24g", "weirdo"]
        elif litype == 2:
            labels = ["48htype1", "48htype2", "24g", "weirdo"]
        elif litype == 3:
            labels = ["48htype1", "48htype2", "48htype3", "24g", "weirdo"]
        elif litype == 4:
            labels = ["48htype1", "48htype2", "48htype3", "48htype4", "24g", "weirdo"]
        elif litype == 5:
            labels = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "24g", "weirdo"]
        elif litype == 6:
            labels = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "24g", "weirdo"]
        elif litype == 7:
            labels = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7", "24g", "weirdo"]
        elif litype == 8:
            labels = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7", "48htype8", "24g", "weirdo"]

        for i in labels:
            label_count[i] = 0

        for key, value in atom_mapping_el_w_dist_label.items():
            label = value['label']
            label_count[label] = label_count.get(label, 0) + 1
        label_count['weirdo'] = amount_weirdo

        dataframe.at[idx, col_amount_type] = label_count
    

def plot_amount_type(dataframe, litype, el, style, category_labels = None):
    """
        style: scatter, bar
    """
    col_amount_type_el = f"amount_type_{el}"

    df = pd.DataFrame()
    df['idx_file'] = None

    if litype == 0:
        df['24g'] = None; df['weirdo'] = None
    elif litype == 1:
        df['48htype1'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 2:
        df['48htype1'] = None; df['48htype2'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 3:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 4:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 5:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 6:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 7:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['48htype7'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 8:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['48htype7'] = None; df['48htype8'] = None; df['24g'] = None; df['weirdo'] = None

    for idx in range(dataframe["geometry"].size):

        amount_type = dataframe.at[idx, col_amount_type_el]
        df.at[idx, 'idx_file'] = idx

        if litype == 0:
            df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 1:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 2:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 3:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 4:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 5:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 6:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 7:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '48htype7'] = amount_type['48htype7']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 8:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '48htype7'] = amount_type['48htype7']; df.at[idx, '48htype8'] = amount_type['48htype8']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']

    wide_df = pd.DataFrame(df)

    long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

    if category_labels:
        long_df['category'] = long_df['category'].replace(category_labels)

    if style == "bar":
        fig = px.bar(long_df, x="idx_file", y="count", color="category", title="Idx file vs Li type")
    elif style == "scatter":
        fig = px.scatter(long_df, x="idx_file", y="count", color="category", title="Idx file vs Li type")
    fig.show()

    return df


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################





##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


# def get_idxs_val(df, val):
#     idxs = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1]) if df.iloc[row, col] == val]
#     return idxs


# def get_key_Li_idx(dict, path, idx_triad):
#     idxs_li = [key for key, value in dict[path].items() if value.get('idx_triad') == idx_triad]
#     return idxs_li

# def get_idx_coor_Li_dict_ref_triad(idx_coor_Li_dict_ref_24, idx_coor_Li_dict_ref_72):
#     idxs_Li_ref_24 = list(idx_coor_Li_dict_ref_24.keys())
#     idxs_Li_ref_72 = list(idx_coor_Li_dict_ref_72.keys())

#     idx_coor_Li_dict_ref_triad = defaultdict(list)

#     for key_72, coor_72 in idx_coor_Li_dict_ref_72.items():
#         for key_24, coor_24 in idx_coor_Li_dict_ref_24.items():
#             distance = mic_eucledian_distance(coor_72, coor_24)
#             if distance == 0:
#                 idx_coor_Li_dict_ref_triad[key_24].append(coor_72)

#     for key_72, coor_72 in idx_coor_Li_dict_ref_72.items():
#         for key_24, coor_24 in idx_coor_Li_dict_ref_24.items():
#             distance = mic_eucledian_distance(coor_72, coor_24)
#             if distance <= 0.086399 and distance != 0:                      # to edit this number
#                 idx_coor_Li_dict_ref_triad[key_24].append(coor_72)
#     return idx_coor_Li_dict_ref_triad


# def get_idx_coor_Li_idx_centroid_triad_w_closest_dist_weirdos_appended(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_weirdos_appended, idxs_Li_not):
#     idx_coor_Li_idx_centroid_triad_weirdos_appended = defaultdict(list)
#     distance_array = []
#     for key, coor in idx_coor_Li_dict_weirdos_appended.items():
#         # if key in idxs_Li_not:
#         #     idx_coor_Li_idx_centroid_triad_dict = {}    
#         #     for key_triad, coor_triad in idx_coor_Li_dict_ref_triad.items():
#         #         for coor_triad_component in coor_triad:
#         #             distance = mic_eucledian_distance(coor_triad_component, coor)
#         #             distance_array.append(distance)  
#         #     distance_array_sorted = sorted(set(distance_array))
#         #     distance_array_sorted_top3 = distance_array_sorted[0:4]
#         #     idx_coor_Li_idx_centroid_triad_dict['dist_top3'] = distance_array_sorted_top3  
#         #     if key in idx_coor_Li_idx_centroid_triad_weirdos_appended:
#         #         idx_coor_Li_idx_centroid_triad_weirdos_appended[key].append(idx_coor_Li_idx_centroid_triad_dict)
#         #     else:
#         #         idx_coor_Li_idx_centroid_triad_weirdos_appended[key] = idx_coor_Li_idx_centroid_triad_dict         
#         # else:
#         idx_coor_Li_idx_centroid_triad_dict = {}    
#         for key_triad, coor_triad in idx_coor_Li_dict_ref_triad.items():
#             for coor_triad_component in coor_triad:
#                 distance = mic_eucledian_distance(coor_triad_component, coor)
#                 distance_array.append(distance)
#                 if distance == 0:
#                     idx_coor_Li_idx_centroid_triad_dict['coor'] = coor
#                     idx_coor_Li_idx_centroid_triad_dict['idx_triad'] = key_triad
#                     idx_coor_Li_idx_centroid_triad_dict['centroid_triad'] = coor_triad[0]
#                     # check if it's at 24g or 48h
#                     if coor == coor_triad[0]:
#                         idx_coor_Li_idx_centroid_triad_dict['structure'] = 24
#                     else:
#                         idx_coor_Li_idx_centroid_triad_dict['structure'] = 48           
#         distance_array_sorted = sorted(set(distance_array))
#         distance_array_sorted_top3 = distance_array_sorted[0:4]
#         # idx_coor_Li_idx_centroid_triad_dict['dist_top3'] = distance_array_sorted_top3
#         if key in idx_coor_Li_idx_centroid_triad_weirdos_appended:
#             idx_coor_Li_idx_centroid_triad_weirdos_appended[key].append(idx_coor_Li_idx_centroid_triad_dict)
#         else:
#             idx_coor_Li_idx_centroid_triad_weirdos_appended[key] = idx_coor_Li_idx_centroid_triad_dict
#     return idx_coor_Li_idx_centroid_triad_weirdos_appended


# def get_idx_coor_Li_idx_centroid_triad_w_closest_dist(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict):
#     idx_coor_Li_idx_centroid_triad = defaultdict(list)
#     distance_array = []
#     for key, coor in idx_coor_Li_dict.items():
#         idx_coor_Li_idx_centroid_triad_dict = {}    
#         for key_triad, coor_triad in idx_coor_Li_dict_ref_triad.items():
#             for coor_triad_component in coor_triad:
#                 distance = mic_eucledian_distance(coor_triad_component, coor)
#                 distance_array.append(distance)
#                 if distance == 0:
#                     idx_coor_Li_idx_centroid_triad_dict['coor'] = coor
#                     idx_coor_Li_idx_centroid_triad_dict['idx_triad'] = key_triad
#                     idx_coor_Li_idx_centroid_triad_dict['centroid_triad'] = coor_triad[0]
#                     # check if it's at 24g or 48h
#                     if coor == coor_triad[0]:
#                         idx_coor_Li_idx_centroid_triad_dict['structure'] = 24
#                     else:
#                         idx_coor_Li_idx_centroid_triad_dict['structure'] = 48                    
#         distance_array_sorted = sorted(distance_array)
#         distance_array_sorted_top3 = distance_array_sorted[1:4]
#         idx_coor_Li_idx_centroid_triad_dict['dist_top3'] = distance_array_sorted_top3
#         # idx_coor_Li_idx_centroid_triad_dict['dist'] = distance_array_sorted
#         if key in idx_coor_Li_idx_centroid_triad:
#             idx_coor_Li_idx_centroid_triad[key].append(idx_coor_Li_idx_centroid_triad_dict)
#         else:
#             idx_coor_Li_idx_centroid_triad[key] = idx_coor_Li_idx_centroid_triad_dict
#     return idx_coor_Li_idx_centroid_triad


# def get_dist_ascending(idx_coor_Li_idx_centroid_triad):
#     coors_Li_dist_structures = defaultdict(list)

#     for Li_idx_temp1, val_temp1 in idx_coor_Li_idx_centroid_triad.items():
#         coors_Li_dist_structures_dict = {}
#         for Li_idx_temp2, val_temp2 in idx_coor_Li_idx_centroid_triad.items():
#             distance = mic_eucledian_distance(val_temp1['coor'], val_temp2['coor'])
#             coors_Li_dist_structures_dict['coors'] = (val_temp1['coor'], val_temp2['coor'])
#             coors_Li_dist_structures_dict['dist'] = distance
#             coors_Li_dist_structures_dict['structures'] = (val_temp1['structure'], val_temp2['structure'])

#             key = (Li_idx_temp1, Li_idx_temp2)
#             if key in coors_Li_dist_structures:
#                 coors_Li_dist_structures[key].append(coors_Li_dist_structures_dict)
#             else:
#                 coors_Li_dist_structures[key] = coors_Li_dist_structures_dict

#     sorted_coors_Li_dist_structures = dict(sorted(coors_Li_dist_structures.items(), key=lambda item: item[1]['dist']))
#     dist_ascending = list({val['dist'] for idx, val in sorted_coors_Li_dist_structures.items()})

#     return dist_ascending, sorted_coors_Li_dist_structures


# # def get_idx_coor_Li_dict(file_path):
# #     with open(file_path, 'r') as f:
# #         content = f.read()

# #     # Initialize a dictionary to store the data
# #     Li_idx_coor_dict = {}

# #     # Use regular expressions to extract Li indices and coordinates
# #     li_pattern = re.compile(r'Li\s+Li(\d+)\s+1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
# #     matches = li_pattern.findall(content)

# #     # Iterate through the matches and populate the dictionary
# #     for match in matches:
# #         index = int(match[0])
# #         x = float(match[1])
# #         y = float(match[2])
# #         z = float(match[3])
# #         Li_idx_coor_dict[index] = (x, y, z)

# #     return Li_idx_coor_dict


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


def get_distance_litoli(dataframe, max_mapping_radius, destination_directory, idx_file_group, idx_ref, mean_ref, var_filename):
    """
        idx_file_group = [idx_init, idx_end]
    """
    df_distance = pd.DataFrame()
    coor_Li_ref = []
    # df_dist_litoli = pd.DataFrame()

    # df_dist_litoli["dist"] = None

    if 'CONTCAR' in var_filename:
        file_ref = f"{int(dataframe['geometry'][idx_ref])}_{int(dataframe['path'][idx_ref])}_{var_filename}"
    else:
        file_ref = f"{int(dataframe['geometry'][idx_ref])}_{int(dataframe['path'][idx_ref])}_{var_filename}.cif"
    file_path_ref = os.path.join(destination_directory, file_ref)

    structure_ref = Structure.from_file(file_path_ref)

    for idx, coor in enumerate(structure_ref):
        if coor.species_string == "Li":
            coor_Li_ref.append(coor.frac_coords)

    print(f"coor_Li_ref: {coor_Li_ref}")

    # for i in path_geo:
    dataframe_group = dataframe.copy()
    dataframe_group = dataframe_group[idx_file_group[0]:idx_file_group[1]]
    idx_range = list(range(dataframe_group["geometry"].size))
    print(idx_range)
    
    if idx_ref > idx_file_group[1]:
         # dataframe_group = dataframe_group.append(dataframe[idx_ref-1:idx_ref], ignore_index=True)
        dataframe_group = pd.concat([dataframe[idx_ref:idx_ref+1], dataframe[idx_file_group[0]:idx_file_group[1]]], ignore_index=False)
        # idx_range = [idx_ref] + list(range(dataframe_group["geometry"].size - 1))
        idx_range = [idx_ref] + idx_range

    for index in idx_range:
        print(index)
        # for index in [1]:
        coor_Li = []
        dict_distance = defaultdict(list)

        if 'CONTCAR' in var_filename:
            file = f"{int(dataframe_group['geometry'][index])}_{int(dataframe_group['path'][index])}_{var_filename}"
        else:
            file = f"{int(dataframe_group['geometry'][index])}_{int(dataframe_group['path'][index])}_{var_filename}.cif"
        print(file)
        file_path = os.path.join(destination_directory, file)

        structure = Structure.from_file(file_path)
        # frac_coor = structure.frac_coords

        for idx, coor in enumerate(structure):
            if coor.species_string == "Li":
                coor_Li.append(coor.frac_coords)        

        print(f"coor_Li: {coor_Li}")
            
        coors_Li_dist_structures = defaultdict(list)

        if mean_ref == True:
            for j in range(len(coor_Li)):
                df_distance.at[index, f"{j}"] = None  

                coor_Li_ref_mean = np.mean(coor_Li_ref, axis=0)
                distance = mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

                dict_distance[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref_mean}, coor_Li: {coor_Li[j]}'}
                df_distance.at[index, f"{j}"] = distance

                diameter_24g48h = max_mapping_radius * 2
                # if distance < diameter_24g48h and index != idx_ref:
                if distance > diameter_24g48h and index != idx_ref:
                    print(f"path: {index}, Li: {j}, distance: {distance}")

        elif mean_ref == False:
            for j in range(len(coor_Li)):
                df_distance.at[index, f"{j}"] = None  

                distance = mic_eucledian_distance(coor_Li_ref[j], coor_Li[j])

                dict_distance[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref[j]}, coor_Li: {coor_Li[j]}'}
                df_distance.at[index, f"{j}"] = distance

                diameter_24g48h = max_mapping_radius * 2
                # if distance < diameter_24g48h and index != idx_ref:
                if distance > diameter_24g48h and index != idx_ref:
                    print(f"path: {index}, Li: {j}, distance: {distance}")

    #         coors_Li_dist_structures_dict = {}
            
    #         for k in range(len(coor_Li)):

    #             distance_litoli = mic_eucledian_distance(coor_Li[j], coor_Li[k])

    #             coors_Li_dist_structures_dict['coors'] = (coor_Li[j], coor_Li[k])
    #             coors_Li_dist_structures_dict['dist'] = distance_litoli
    #             # coors_Li_dist_structures_dict['structures'] = (val_temp1['structure'], val_temp2['structure'])

    #             key = (j, k)
    #             if key in coors_Li_dist_structures:
    #                 coors_Li_dist_structures[key].append(coors_Li_dist_structures_dict)
    #             else:
    #                 coors_Li_dist_structures[key] = coors_Li_dist_structures_dict               

    #     sorted_coors_Li_dist_structures = dict(sorted(coors_Li_dist_structures.items(), key=lambda item: item[1]['dist']))
    #     dist_ascending = list({val['dist'] for idx, val in sorted_coors_Li_dist_structures.items()})

    #     df_dist_litoli.at[index, "dist"] = dist_ascending[1:6]

    #     # df_coor.at[i, col_xyz_coor] = coor_Li

    # #     # # for j in range(len(coor_Li)):
    # #     # #     # df_distance.at[i, f"{j}"] = None  

    # #     # #     distance = mic_eucledian_distance(coor_Li_ref[j], coor_Li[j])

    # #     # #     df_distance.at[i, f"{j}"] = distance

    # #     # #     diameter_24g48h = max_mapping_radius * 2
    # #     # #     if distance < diameter_24g48h and i != 0:
    # #     # #         print(f"path: {i}, Li: {j}, distance: {distance}")

    return df_distance, dataframe_group



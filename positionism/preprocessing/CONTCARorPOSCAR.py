import os
import pandas as pd
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter


def edit_to_normal_elements(dataframe, destination_directory, filename, prefix=None):
    # rename from: get_CONTCAR_normal_elements
    """
    Modifies CONTCAR files to include standard elemental labels for each site configuration, instead of `Li_sv_GW/24a6a  P_GW/715c28f22  S_GW/357db9cfb  Cl_GW/3ef3b316` from the result of NEB calculation.

    Parameters
    ==========
    dataframe: pd.DataFrame
        DataFrame containing necessary data columns.
    destination_directory: str
        Path to the directory where CONTCAR files are located.
    filename: str
        Base filename for the CONTCAR files.
    prefix: str or None
        Prefix to append to the filenames (optional).

    Returns
    =======
    None
        The function modifies CONTCAR files but does not return any value.
    """
    # Function body goes here
    for index in range(dataframe["geometry"].size):
        # Generate the new filename
        if prefix == None:
            new_filename = f"{int(dataframe['geometry'][index])}_{int(dataframe['path'][index])}_{filename}"
        else:
            new_filename = f"{int(dataframe['geometry'][index])}_{int(dataframe['path'][index])}_{filename}_{prefix}"

        # Get the source file path and destination file path
        destination_path = os.path.join(destination_directory, new_filename)

        # Read CONTCAR file
        with open(destination_path, 'r') as contcar_file:
            contcar_lines = contcar_file.readlines()
        
        contcar_lines[5] = "   Li   P    S    Cl\n"
        contcar_lines[6] = "    24     4    20     4\n"

        # Create a new CONTCAR file for each configuration
        with open(destination_path, 'w') as contcar_file:
            contcar_file.writelines(contcar_lines)


def positive_lessthan1(dataframe, destination_directory, poscarorcontcar_line_nr_start, poscarorcontcar_line_nr_end, poscarorcontcar_columns_type2, file_type, var_name_in, var_name_out, n_decimal):
    # rename from: get_positive_lessthan1_poscarorcontcar
    """
    Creates positive POSCAR/CONTCAR files that have normalized value within the range [0, 1]
    (based on specified data transformations).

    Parameters
    ==========
    dataframe: pd.DataFrame
        DataFrame containing necessary data columns.
    destination_directory: str
        Path to the directory where CONTCAR files will be saved.
    file_type: str
        Type of files to be processed.
    poscarorcontcar_line_nr_start: int
        Line number where relevant data starts in the CONTCAR file.
    poscarorcontcar_line_nr_end: int
        Line number where relevant data ends in the CONTCAR file.
    poscarorcontcar_columns_type2: list
        Column names for the relevant data in the CONTCAR file.
    var_name_in: str or None
        Variable name for input data (optional).
    var_name_out: str
        Variable name for output data.
    n_decimal: int
        Number of decimal places to round the coordinates to (default is 6).

    Returns
    =======
    None
        The function modifies and saves CONTCAR files but does not return any value.
    """
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
        data = lines[poscarorcontcar_line_nr_start:poscarorcontcar_line_nr_end]

        # Split each string by space and create the DataFrame
        df = pd.DataFrame([string.strip().split() for string in data])

        # Optional: Rename the columns
        df.columns = poscarorcontcar_columns_type2

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

        file_list = lines[:poscarorcontcar_line_nr_start] + line_append_list

        poscarorcontcar_filename_positive = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_CONTCAR_{var_name_out}"
        destination_path = os.path.join(destination_directory, poscarorcontcar_filename_positive)
        
        with open(destination_path, "w") as poscarorcontcar_positive_file:
            for item in file_list:
                poscarorcontcar_positive_file.writelines(item)

        dataframe[col_subdir_positive_file][idx] = destination_path


def convert_to_cif_pymatgen(dataframe, destination_directory, file_restructure, var_name):
    # rename from: create_cif_pymatgen
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


def get_latticeconstant_dict(dataframe, destination_directory, proceed_XDATCAR, var_filename):
    # rename from: get_latticeconstant_structure_dict_iterated
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


def diagonalize_latticeconstantsmatrix(dataframe, destination_directory, latticeconstantsmatrix_line_nr_start, var_name_in, var_name_out, n_decimal):
    # rename from: diagonalizing_latticeconstantsmatrix
    
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

        poscar_filename_diagonalized = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_POSCAR_{var_name_out}"
        destination_path = os.path.join(destination_directory, poscar_filename_diagonalized)
        
        with open(destination_path, "w") as poscar_positive_file:
            for item in file_list:
                poscar_positive_file.writelines(item)

        # # dataframe['subdir_orientated_positive_poscar'][idx] = destination_path


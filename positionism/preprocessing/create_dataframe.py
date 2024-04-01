import numpy as np
import pandas as pd
import os

from positionism.functional import func_directory


def base(data_toten, file_name):
    # rename: 
    # create_file_loc = base
    # file_loc = df_file
    """
    Generate a DataFrame with columns information, such as: geometry, path (of the file CONTCARs/ POSCARs), 
    total energy. Those are extracted from input parameters.

    This method walks through a directory structure starting from the current working
    directory, looking for the specified file. For each file found, it generates a geometry
    and path identifier based on the directory structure. It then combines this
    information into a Pandas DataFrame, sorts it, and performs various operations to
    calculate new columns of perfect system.

    Source: https://stackoverflow.com/questions/27805919/how-to-only-read-lines-in-a-text-file-after-a-certain-string

    Args
    ====
    data_toten: pd.DataFrame
        DataFrame containing total energy data to be merged.
    file_name: str
        "CONTCAR", "POSCAR". The filename to search for in the directory walk. 

    Returns
    =======
    df_file: pd.DataFrame
        A DataFrame with columns of: geometries, paths of CONTCARs/ POSCARs, path location of files, and total energy, and calculated columns for further analysis.
        
    Note
    ====
    - The compatibility check with `data_toten` relies on exact matches in 'geometry' and 'path' columns between the newly created DataFrame and `data_toten`.
    """
    direc = os.getcwd()

    # Column names for the DataFrame
    col_excel_geo = "geometry"
    col_excel_path = "path"
    col_excel_toten = "toten [eV]"

    # Initialize arrays for DataFrame construction
    geometry = np.array([])
    path = np.array([])
    subdir_col = np.array([])

    # Walk through the directory structure
    for subdir, dirs, files in os.walk(direc,topdown=False):
        for file in files:
            filepath = subdir + os.sep
            # get directory of CONTCARs/ POSCARs
            if os.path.basename(file) == file_name:
                # Extract geometry and path numbers from the directory structure
                geometry_nr = func_directory.splitall(subdir)[-2]
                path_nr = func_directory.splitall(subdir)[-1]

                # Construct geometry and path DataFrames
                geometry = pd.DataFrame(np.append(geometry, int(geometry_nr)), columns=["geometry"])
                path = pd.DataFrame(np.append(path, int(path_nr)), columns=["path"])

                # Drop NaNs
                geometry.dropna(axis=1)
                path.dropna(axis=1) 

                # Construct full file path location and initialize DataFrame for new system directories
                subdir_file = os.path.join(subdir,file_name)
                subdir_col = pd.DataFrame(np.append(subdir_col, subdir_file), columns=["subdir_new_system"])

                # Join geometry and path DataFrames, add new system directories
                df_file = geometry.join(path)
                df_file["subdir_new_system"] = subdir_col

    # Perform DataFrame sorting based on columns of geometry and path
    df_file = df_file.sort_values(by=["geometry","path"],ignore_index=True,ascending=False) # sort descendingly based on path

    # Additional calculations and DataFrame modifications
    df_file["g+p"] = (df_file["geometry"] + df_file["path"]).fillna(0) # replace NaN with 0
    df_file = columns_perfectsystem(df_file)

    # Merge `data_toten` if compatible
    if data_toten[col_excel_geo].all() == df_file["geometry"].all() & data_toten[col_excel_path].all() == df_file["path"].all():
        df_file[col_excel_toten] = data_toten[col_excel_toten]
    else:
        print("check the compatibility of column geometry and path between data_toten file and df_file")

    return df_file


def columns_perfectsystem(df_file):
    """
    Calculate additional columns for the DataFrame generated in `CreateDataFrame.base`.

    This method adds calculated columns to identify "perfect systems" based on the conditions
    specified in the calculation.

    Args
    ====
    df_file: pd.DataFrame
        The DataFrame to which the calculations are applied.

    Returns
    =======
    df_file: pd.DataFrame)
        The DataFrame with additional calculated columns.
    """
    # Shift operations and perfect system identification
    df_file["g+p+1"] = df_file["g+p"].shift(1)
    df_file["g+p+1"][0] = 0 # replace 1st element with 0
    df_file["g+p-1"] = df_file["g+p"].shift(-1)
    df_file["g+p-1"][(df_file["g+p-1"]).size - 1] = 0.0 # replace last element with 0
    df_file["perfect_system"] = df_file["g+p"][(df_file["g+p+1"] > df_file["g+p"]) & (df_file["g+p-1"] > df_file["g+p"])]
    df_file["perfect_system"][df_file["geometry"].size-1] = 0.0 # hardcode the path 0/0
    df_file["p_s_mask"] = [0 if np.isnan(item) else 1 for item in df_file["perfect_system"]]

    return df_file


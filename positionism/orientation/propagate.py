import numpy as np

from functional import operation
from orientation import perfect_structure

from pymatgen.core.structure import Structure


def calculate(df_file, direc_restructure_destination, file_restructure, path_perfect_poscar_24, orientation):
    # rename from: get_orientation
    """
    Processes the orientation of structures based on a reference structure. 
    It identifies perfect systems (file with path number 0) denoted in DataFrame as `p_s_mask=1` 
    and applies necessary transformations to align them with the reference structure. 
    The transformations are then propagated to other related systems.

    Parameters
    ==========
        df_file: pd.DataFrame
            DataFrame containing the file locations and all other metadata.
        direc_restructure_destination: str
            Directory path where the restructured files will be saved.
        file_restructure: str
            Name of the file after restructuring.
        path_perfect_poscar_24: str
            File path to the reference structure, which the orientation is based on.
        orientation: str
            Flag indicating whether to process the orientation ('True') or skip it ('False').

    Returns
    =======
        df_file_mask_1: pd.DataFrame
            DataFrame with rows where 'p_s_mask' is 1, indicating perfect systems.
        df_file_important_cols: pd.DataFrame
            DataFrame with important columns  and possibly updated with transformations of all systems.
    """
    if orientation == "True":
        df_file_ori_notdeleted = df_file.copy()

        # Filter rows where 'p_s_mask' is not 0, indicating these are perfect systems.
        df_file_mask_1 = df_file_ori_notdeleted.loc[df_file_ori_notdeleted['p_s_mask'].apply(lambda x: x != 0)]
        df_file_mask_1 = df_file_mask_1.reset_index()

        # Copy and rename files for the perfect systems and initiate columns for transformation data.
        operation.File.copy_rename_files(df_file_mask_1, direc_restructure_destination, file_restructure, 
                                            prefix=None, savedir = True)

        # Initialize columns for storing transformation data
        df_file_mask_1['verify_w_lib'] = None
        df_file_mask_1['verify_w_linalg'] = None
        df_file_mask_1['scaling'] = None
        df_file_mask_1['translation'] = None
        df_file_mask_1['mapping'] = None
        df_file_mask_1['transformation'] = None

        # Load the reference structure
        structure_reference = Structure.from_file(path_perfect_poscar_24)
        
        # Apply transformations using different methods and update the DataFrame
        perfect_structure.with_library(df_file_mask_1, direc_restructure_destination, file_restructure,
                                                structure_reference, "trf_w_lib", prefix=None)
        perfect_structure.with_linalg(df_file_mask_1, direc_restructure_destination, file_restructure,
                                                structure_reference, "trf_w_linalg", prefix=None)
        perfect_structure.get_structure_with_linalg_combinded_with_library(df_file_mask_1,
                                                                        direc_restructure_destination,
                                                                        file_restructure, structure_reference,
                                                                        "trf_w_linalg_n_lib", prefix=None)

        # Initialize columns for further processing
        df_file['scaling'] = None
        df_file['translation'] = None
        df_file['mapping'] = None
        df_file['index'] = None
        # Copy important columns from `df_file_mask_1` to the `df_file` as initial data for further processing.
        important_cols = ["geometry", "path", "subdir_new_system", "p_s_mask", "scaling",
                            "translation", "mapping", "index", "toten [eV]"]
        df_file_important_cols = df_file[important_cols]
        df_file_mask_1_important_cols = df_file_mask_1[important_cols]

        # Now Processing with other folders that are with mask = 0 (not perfect system)
        # Propagate scaling, translation, and mapping from perfect systems to related systems.
        idx_row = df_file_mask_1['index'].values.astype(int)
        # copy scaling, translation, mapping of the path 0
        for i in idx_row:
            scaling = np.array(df_file_mask_1_important_cols["scaling"][df_file_mask_1_important_cols['index'] == i])
            translation = np.array(df_file_mask_1_important_cols["translation"][df_file_mask_1_important_cols['index'] == i])
            mapping = np.array(df_file_mask_1_important_cols["mapping"][df_file_mask_1_important_cols['index'] == i])
            df_file_important_cols.at[i, 'scaling'] = scaling[0]
            df_file_important_cols.at[i, 'translation'] = translation[0]
            df_file_important_cols.at[i, 'mapping'] = mapping[0]

        # # df_file_important_cols = df_file_important_cols[["geometry", "path", "subdir_new_system", "p_s_mask", "scaling", 
        # #                                                  "translation", "mapping", "toten [eV]"]]

        # Further processing to ensure all systems are updated appropriately
        idx_mask = np.where(df_file_important_cols["p_s_mask"] == 1)[0]
        mask = np.append(0, idx_mask)
        df_file_important_cols["scaling"][0] = df_file_important_cols["scaling"][3]   # hardcode for initial part
        df_file_important_cols["translation"][0] = df_file_important_cols["translation"][3]
        df_file_important_cols["mapping"][0] = df_file_important_cols["mapping"][3]

        for i in range(idx_mask.size):
            i1 = mask[i]+1
            i2 = mask[i+1]
            for j in range(i1,i2+1):
                df_file_important_cols["scaling"][j] = df_file_important_cols["scaling"][i2]
                df_file_important_cols["translation"][j] = df_file_important_cols["translation"][i2]
                df_file_important_cols["mapping"][j] = df_file_important_cols["mapping"][i2]

    else:
        # If orientation processing is not required, prepare a basic DataFrame for return.
        df_file_ori_notdeleted = df_file.copy()
        df_file_mask_1 = df_file_ori_notdeleted.loc[df_file_ori_notdeleted['p_s_mask'].apply(lambda x: x != 0)]
        df_file_mask_1 = df_file_mask_1.reset_index()

        df_file_important_cols = df_file.copy()

    return df_file_mask_1, df_file_important_cols

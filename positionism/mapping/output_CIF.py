import numpy as np
import os

from functional import func_string

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter


def write_merged_structure(dataframe, destination_directory, amount_Li, amount_P, amount_S, activate_radius, var_savefilename):
    # rename from: create_combine_structure
    """
    Write structure files by merging mapped Li atoms with original P, S, and Cl atoms.

    Args
    ====
    dataframe: pandas.DataFrame
        DataFrame containing coordinate information.
    destination_directory: str
        The directory where the structure files will be saved.
    amount_Li: int
        The amount of Lithium atoms in the structure.
    amount_P: int
        The amount of Phosphorus atoms in the structure.
    amount_S: int
        The amount of Sulfur atoms in the structure.
    activate_radius: int
        The activation radius.
    var_savefilename: str
        The variable save filename.

    Note
    ====
    -  TO DO: under maintenance for disambled into el
    """
    # CHANGED FOR INDEXING
    # # # # # if activate_radius == 2 or activate_radius == 3:
    # # # # #     col_coor_reducedreference_el = f"coor_reducedreference_48htypesmerged_Li"
    # # # # # elif activate_radius == 1:
    # # # # #     col_coor_reducedreference_el = f"coor_reducedreference_Li_closestduplicate"
    col_coor_reducedreference_el = f"coor_reducedreference_sorted_Li"
                    
    # col_coor_reducedreference_closestduplicate_Li_closestduplicate = f"coor_reducedreference_closestduplicate_Li_closestduplicate" # !!!!!
    # col_coor_reducedreference_closestduplicate_Li = f"coor_reducedreference_closestduplicate_Li" # !!!!!
    col_coor_structure_init_dict = "coor_structure_init_dict"

    for idx in range(dataframe["geometry"].size):
        coor_combined = []

        # new_structure = Structure.from_file(dataframe['subdir_orientated_positive'][idx])
        # new_structure = Structure.from_file(dataframe['subdir_orientated_positive_poscar'][idx])
        new_structure = Structure.from_file(dataframe['subdir_positive_CONTCAR'][idx])
        coor_origin24_init = dataframe.at[idx, col_coor_structure_init_dict]
        # coor_reducedreference_Li = dataframe.at[idx, col_coor_reducedreference_closestduplicate_Li]
        # coor_reducedreference_Li = dataframe.at[idx, col_coor_reducedreference_closestduplicate_Li_closestduplicate]
        coor_reducedreference_Li = dataframe.at[idx, col_coor_reducedreference_el]

        coor_structure_init_P = coor_origin24_init["P"]
        coor_structure_init_S = coor_origin24_init["S"]
        coor_structure_init_Cl = coor_origin24_init["Cl"]

        coor_mapped_Li = np.array(coor_reducedreference_Li)
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
        amount_Li_temp = len(coor_reducedreference_Li)
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


def correct_Li_idx(dataframe, destination_directory, amount_Li, amount_P, amount_S, amount_Cl, var_savefilename_init, var_savefilename_new):
    # rename from: rewrite_cif_w_correct_Li_idx
    """
    Rewrite CIF files with correct Li atom indices.

    Parameters
    ==========
    dataframe: pd.DataFrame
        DataFrame containing the necessary data.
    destination_directory: str
        Path to the directory where the rewritten CIF files will be saved.
    amount_Li: int
        Number of Li atoms.
    amount_P: int
        Number of P atoms.
    amount_S: int
        Number of S atoms.
    amount_Cl: int
        Number of Cl atoms.
    var_savefilename_init: str
        Variable to save the initial filename.
    var_savefilename_new: str
        Variable to save the new filename.
    """
    col_idx0_weirdos_Li = "idx0_weirdos_Li"
    
    col_idx_without_weirdos = "idx_without_weirdos"

    dataframe[col_idx_without_weirdos] = [np.array([]) for _ in range(len(dataframe.index))]
    
    search_string = "Li  Li0"

    for idx in range(dataframe["geometry"].size):
        idx0_weirdos_Li = dataframe[col_idx0_weirdos_Li][idx]
        source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_init}.cif"
        source_filename_path = os.path.join(destination_directory, source_filename)

        source_filename_filtered = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_savefilename_new}.cif"
        destination_path_combined_new = os.path.join(destination_directory, source_filename_filtered)

        with open(source_filename_path, "r") as f:
            lines = f.readlines()

        # added: check if the search_string is found
        search_string_found = False
        for idx_line, line in enumerate(lines):
            if search_string in line:
                search_string_found = True
                idx_Li_start = idx_line
                break

        idx_without_weirdos = [i for i in range(amount_Li) if i not in idx0_weirdos_Li]

        if not search_string_found:
            pass
        else:
            # idx_without_weirdos = [i for i in range(amount_Li) if i not in idx0_weirdos_Li]

            new_text = []
            for i in range(len(idx_without_weirdos)):
                idx_line = idx_Li_start + i
                if lines[idx_line].strip().startswith("Li"):
                    new_label = f"Li{idx_without_weirdos[i]}"
                    modified_line = func_string.modify_line(lines[idx_line], lines[idx_line].split()[1], new_label)
                    new_text.append(modified_line)
                    
            lines[idx_Li_start : len(idx_without_weirdos) + idx_Li_start] = new_text

            idx_P_S_Cl_line_new_start    = len(idx_without_weirdos) + idx_Li_start
            Edit.reindex_P_S_Cl(lines, idx_Li_start, idx_without_weirdos, idx_P_S_Cl_line_new_start, amount_Li, amount_P, amount_S, amount_Cl)

        dataframe.at[idx, col_idx_without_weirdos] = idx_without_weirdos

        with open(destination_path_combined_new, "w") as f:
            f.write("\n".join(line.strip() for line in lines))



def correct_Li_idx_weirdos_appended(dataframe, destination_directory, amount_Li, amount_P, amount_S, amount_Cl, activate_radius, var_savefilename_init, var_savefilename_new):
    """
    Rewrite CIF files with correct Li atom indices and weirdos appended

    Parameters
    ==========
    dataframe: pd.DataFrame
        DataFrame containing the necessary data.
    destination_directory: str
        Path to the directory where the rewritten CIF files will be saved.
    amount_Li: int
        Number of Li atoms.
    amount_P: int
        Number of P atoms.
    amount_S: int
        Number of S atoms.
    amount_Cl: int
        Number of Cl atoms.
    activate_radius: int
        Type of radius
    var_savefilename_init: str
        Variable to save the initial filename.
    var_savefilename_new: str
        Variable to save the new filename.
    """
    # rename from: rewrite_cif_w_correct_Li_idx_weirdos_appended
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

        # added: check if the search_string is found
        search_string_found = False
        for idx_line, line in enumerate(lines):
            if search_string in line:
                search_string_found = True
                idx_Li_start = idx_line
                break

        idx_without_weirdos = [i for i in range(amount_Li) if i not in idx0_weirdos_Li]

        if not search_string_found:
            pass
        else:
            new_text = []
            for i in range(len(idx_without_weirdos)):
                idx_line = idx_Li_start + i
                if lines[idx_line].strip().startswith("Li"):
                    new_label = f"Li{idx_without_weirdos[i]}"
                    # file_operations_instance = Operation.File()
                    # modified_line = file_operations_instance.replace(lines[idx_line].split()[1], new_label)     
                    # # modified_line = lines[idx_line].replace(lines[idx_line].split()[1], new_label)
                    modified_line = func_string.modify_line(lines[idx_line], lines[idx_line].split()[1], new_label)
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
            # Edit.reindex_P_S_Cl(lines, idx_Li_start, idx_without_weirdos, idx_P_S_Cl_line_new_start, amount_P, amount_S, amount_Cl)
            
            idx_P_S_Cl_line_new_end      = idx_P_S_Cl_line_new_start + len(old_text_P_S_Cl)
            lines[idx_P_S_Cl_line_new_start : idx_P_S_Cl_line_new_end] = old_text_P_S_Cl
    
            # re-write the index of P_S_Cl accordingly
            new_text_P_S_Cl = []
            for i in range(amount_P):
                idx_line_P = idx_P_S_Cl_line_new_start + i
                idx_P_new = amount_Li + i
                if lines[idx_line_P].strip().startswith("P"):
                    new_label = f"P{idx_P_new}"
                    # file_operations_instance = Operation.File()
                    # modified_line = file_operations_instance.replace(lines[idx_line_P].split()[1], new_label)     
                    # # modified_line = lines[idx_line_P].replace(lines[idx_line_P].split()[1], new_label)
                    modified_line = func_string.modify_line(lines[idx_line_P], lines[idx_line_P].split()[1], new_label)
                    new_text_P_S_Cl.append(modified_line)
            for i in range(amount_S):
                idx_line_S = idx_P_S_Cl_line_new_start + amount_P + i
                idx_S_new = amount_Li + amount_P + i
                if lines[idx_line_S].strip().startswith("S"):
                    new_label = f"S{idx_S_new}"
                    # file_operations_instance = Operation.File()
                    # modified_line = file_operations_instance.replace(lines[idx_line_S].split()[1], new_label)     
                    # # modified_line = lines[idx_line_S].replace(lines[idx_line_S].split()[1], new_label)
                    modified_line = func_string.modify_line(lines[idx_line_S], lines[idx_line_S].split()[1], new_label)
                    new_text_P_S_Cl.append(modified_line)
            for i in range(amount_Cl):
                idx_line_Cl = idx_P_S_Cl_line_new_start + amount_P + amount_S + i
                idx_Cl_new = amount_Li + amount_P + amount_S + i
                if lines[idx_line_Cl].strip().startswith("Cl"):
                    new_label = f"Cl{idx_Cl_new}"
                    # file_operations_instance = Operation.File()
                    # modified_line = file_operations_instance.replace(lines[idx_line_Cl].split()[1], new_label)     
                    # # modified_line = lines[idx_line_Cl].replace(lines[idx_line_Cl].split()[1], new_label)
                    modified_line = func_string.modify_line(lines[idx_line_Cl], lines[idx_line_Cl].split()[1], new_label)
                    new_text_P_S_Cl.append(modified_line)

            lines[idx_P_S_Cl_line_new_start : amount_P + amount_S + amount_Cl + idx_P_S_Cl_line_new_start] = new_text_P_S_Cl

            # dataframe.at[idx, col_subdir_cif_w_correct_Li_idx_weirdos_appended] = destination_path_combined_new

        # Write the modified lines back to the file
        with open(destination_path_combined_new, "w") as f:
            f.write("\n".join(line.strip() for line in lines))


def ascending_Li(dataframe, destination_directory, var_filename_init, var_savefilename_new):
    """
    Rewrite CIF files with ascending Li atom indices (weirdos included).

    Parameters
    ==========
    dataframe: pd.DataFrame
        DataFrame containing the necessary data.
    destination_directory: str
        Path to the directory where the rewritten CIF files will be saved.
    var_filename_init: str
        Variable of the initial filename.
    var_savefilename_new: str
        Variable to save the new filename.
    """    
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


class Edit:
    def correct_P_S_Cl_idx(lines, idx_Li_start, idx_without_weirdos, idx_P_S_Cl_line_new_start, amount_Li, amount_P, amount_S, amount_Cl):
        # rename from: reindex_P_S_Cl
        """
        This function modifies the CIF file lines for P, S, and Cl atoms to reflect updated indices. It ensures that after Li atoms index have been corrected and potentially filtered out, the subsequent atom types (P, S, Cl) receive new, sequentially increasing indices starting immediately after the last Li index.

        Parameters
        ==========
        lines: list of str
            The original lines of the CIF file, where each line corresponds to a string representing a line in the file.
        idx_Li_start: int
            The starting index in `lines` where lithium (Li) atoms begin.
        idx_without_weirdos: list of int
            The indices of Li atoms after removing specific "weirdo" atoms, used to calculate new starting positions for P, S, and Cl.
        idx_P_S_Cl_line_new_start: int
            The new starting index for P, S, and Cl atom lines in the modified `lines` list.
        amount_Li: int
            The total number of Li atoms considered for indexing, influencing the new indices of P, S, and Cl atoms.
        amount_P: int
            The total number of phosphorus (P) atoms in the structure.
        amount_S: int
            The total number of sulfur (S) atoms in the structure.
        amount_Cl: int
            The total number of chlorine (Cl) atoms in the structure.
        """
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
                # file_operations_instance = Operation.File()
                # modified_line = file_operations_instance.replace(lines[idx_line_P].split()[1], new_label)
                # # modified_line = lines[idx_line_P].replace(lines[idx_line_P].split()[1], new_label)
                modified_line = func_string.modify_line(lines[idx_line_P], lines[idx_line_P].split()[1], new_label)
                new_text_P_S_Cl.append(modified_line)
        for i in range(amount_S):
            idx_line_S = idx_P_S_Cl_line_new_start + amount_P + i
            idx_S_new = amount_Li + amount_P + i
            if lines[idx_line_S].strip().startswith("S"):
                new_label = f"S{idx_S_new}"
                # file_operations_instance = Operation.File()
                # modified_line = file_operations_instance.replace(lines[idx_line_S].split()[1], new_label)            
                # # modified_line = lines[idx_line_S].replace(lines[idx_line_S].split()[1], new_label)
                modified_line = func_string.modify_line(lines[idx_line_S], lines[idx_line_S].split()[1], new_label)
                new_text_P_S_Cl.append(modified_line)
        for i in range(amount_Cl):
            idx_line_Cl = idx_P_S_Cl_line_new_start + amount_P + amount_S + i
            idx_Cl_new = amount_Li + amount_P + amount_S + i
            if lines[idx_line_Cl].strip().startswith("Cl"):
                new_label = f"Cl{idx_Cl_new}"
                # file_operations_instance = Operation.File()
                # modified_line = file_operations_instance.replace(lines[idx_line_Cl].split()[1], new_label)     
                # # modified_line = lines[idx_line_Cl].replace(lines[idx_line_Cl].split()[1], new_label)
                modified_line = func_string.modify_line(lines[idx_line_Cl], lines[idx_line_Cl].split()[1], new_label)
                new_text_P_S_Cl.append(modified_line)

        lines[idx_P_S_Cl_line_new_start : amount_P + amount_S + amount_Cl + idx_P_S_Cl_line_new_start] = new_text_P_S_Cl

        return lines


    def format_spacing_cif(dataframe, destination_directory, var_savefilename_init, var_savefilename_new):
        """
        This function reads CIF files specified in a DataFrame and adjusts the spacing of lines starting with atom labels ('Li', 'P', 'S', 'Cl'). 
        The purpose is to ensure that these lines have a consistent formatting, which might be required for compatibility with certain software or for readability.
        
        Parameters
        ==========
        dataframe: pandas.DataFrame
            A pandas DataFrame containing at least the columns 'geometry' and 'path', which are used to construct the filenames of the CIF files to be processed.
        destination_directory: str
            The directory path where the CIF files are located and where the modified files will be saved.
        var_savefilename_init: str
            The filename for source CIF files to be read.
        var_savefilename_new: str
            The filename for new CIF files to be saved.
        """
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

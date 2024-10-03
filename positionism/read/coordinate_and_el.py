from pymatgen.core.structure import Structure


# class ReadStructure:
#     class Coor:
def single_structure(structure):
    # rename from: get_coor_structure_init_dict
    """
    Extracts fractional coordinates of different elements from a given structure
    and organizes them into a dictionary.

    Parameters:
    - structure (pymatgen Structure): The input structure containing atomic coordinates.

    Returns:
    - dict: A dictionary where keys represent element symbols and values are lists
    of fractional coordinates corresponding to each element in the structure.
    """
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


def structures(dataframe, mapping):
    # rename from: get_coor_structure_init_dict_wholedataframe
    """
    Same like function get_coor_structure_init_dict() but goes over 
    all structures stated in the DataFrame and save it in the corresponding column.

    Parameters:
    - dataframe (pandas.DataFrame): DataFrame containing structure file paths.
    - mapping (str): Flag indicating whether the structures have undergone mapping.

    Returns:
    - None: The function updates the DataFrame with dictionaries of fractional coordinates
    for each element in the respective structures.
    """
    col_coor_structure_init_dict = "coor_structure_init_dict"

    dataframe[col_coor_structure_init_dict] = None

    for idx in range(dataframe["geometry"].size):
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


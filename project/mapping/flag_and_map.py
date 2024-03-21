import numpy as np
import sys

from functional import calc_distance, dictionary, operation


def all_atoms_of_el(dataframe, coor_structure_init_dict, el, max_mapping_radius):
    # rename from: get_flag_map_weirdos_el
    """
    This function does:
    - Mapping to the closest reference by the radius given. 
    - In the correct case, no multiple atom should belong to a same reference atom, otherwise flag stated "True". 
    - Get weirdos that don't belong to any closest reference.

    Parameters:
    - dataframe (pandas.DataFrame): DataFrame containing necessary data columns.
    - coor_structure_init_dict (dict): Dictionary containing initial fractional coordinates of elements.
    - el (str): Element symbol for which mapping is performed.
    - max_mapping_radius (float): Maximum mapping radius for identifying nearby atomic positions.

    Returns:
    - flag_el: By default False. True if there's > 1 atom belong to a same reference. 
    - coor_weirdos_el: Coordinate of weirdos
    - sum_weirdos_el: Sum amount of weirdos
    - duplicate_closest24_w_data_el: Dictionary, whose key is coor24 and values are multiple coorreference it belongs to and dist.
    - atom_mapping_el_w_dist_closestduplicate: Dictionary, whose key is coorreference and its value is THE CLOSEST coor24 and dist
    - coor_reducedreference_el_closestduplicate: List of coorreference based on atom_mapping_el_w_dist_closestduplicate, so its the closest only.
    - atom_mapping_el_closestduplicate: Dictionary, key: coorreference, value: coor24
    - sum_mapped_el_closestduplicate: Sum amount of coor_reducedreference_el_closestduplicate
    - sum_sanitycheck_el_closestduplicate: sum_mapped_el_closestduplicate + sum_weirdos_el
    """
    coor_reference_el_init = coor_structure_init_dict[el]
    col_coor_structure_init_dict = "coor_structure_init_dict"

    # col_atom_mapping_el = f"atom_mapping_{el}"
    # col_atom_mapping_el_w_dist = f"atom_mapping_{el}_w_dist"
    # col_coor_weirdos_el_dict = f"coor_weirdos_{el}_dict"            # just added
    # col_coor_reducedreference_el = f"coor_reducedreference_{el}"
    # col_sum_mapped_el = f"sum_mapped_{el}"
    # col_sum_sanitycheck_el = f"sum_sanitycheck_{el}"
    col_flag_el = f"flag_{el}"
    col_coor_weirdos_el = f"coor_weirdos_{el}"
    col_sum_weirdos_el = f"sum_weirdos_{el}"
    col_duplicate_closest24_w_data_el = f"duplicate_closest24_w_data_{el}"
    col_coor_reducedreference_el_closestduplicate = f"coor_reducedreference_{el}_closestduplicate"
    col_sum_mapped_el_closestduplicate = f"sum_mapped_{el}_closestduplicate"
    col_sum_sanitycheck_el_closestduplicate = f"sum_sanitycheck_{el}_closestduplicate"
    col_atom_mapping_el_closestduplicate = f"atom_mapping_{el}_closestduplicate"
    col_atom_mapping_el_w_dist_closestduplicate = f"atom_mapping_{el}_w_dist_closestduplicate"

    # dataframe[col_atom_mapping_el] = [{} for _ in range(len(dataframe.index))] 
    # dataframe[col_atom_mapping_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_weirdos_el_dict] = [{el: []} for _ in range(len(dataframe.index))]                       # just added
    # dataframe[col_coor_reducedreference_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_sum_mapped_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_sum_sanitycheck_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_flag_el] = "False"
    dataframe[col_coor_weirdos_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_duplicate_closest24_w_data_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reducedreference_el_closestduplicate] = [np.array([]) for _ in range(len(dataframe.index))]
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
        
        coor_reducedreference_el = coor_reference_el_init.copy()
        coor_weirdos_el = coor_origin24_el_init.copy()    

        for idxreference, coorreference in enumerate(coor_reference_el_init):        
            counter = 0
            atom_mapping_w_dist_dict = {}
            atom_mapping_el_w_dist_closestduplicate = {}
            distance_prev = float("inf")
            closest24 = None

            for idx24, coor24 in enumerate(coor_origin24_el_init):
                distance = calc_distance.mic_eucledian_distance(coorreference, coor24)

                if distance < max_mapping_radius:
                    counter = counter + 1
                    if distance < distance_prev:
                        distance_prev = distance
                        closest24 = coor24
            
                if counter > 1:
                    dataframe.at[idx, col_flag_el] = "True"

                    # if tuple(coorreference) in atom_mapping_el_w_dist:
                    #     atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_w_dist_dict)
                    # else:
                    #     atom_mapping_el_w_dist.setdefault(tuple(coorreference),[atom_mapping_w_dist_dict])
                    
            
            # if closest24 is not None:
            #     if tuple(coorreference) in atom_mapping_el:
            #         atom_mapping_el[tuple(coorreference)].append(closest24)
            #     else:
            #         atom_mapping_el[tuple(coorreference)] = tuple(closest24)

            if closest24 is not None:
                atom_mapping_w_dist_dict['closest24'] = tuple(closest24)
                atom_mapping_w_dist_dict['dist'] = distance_prev
                
                if tuple(coorreference) in atom_mapping_el_w_dist:
                    new_entry = atom_mapping_el_w_dist[tuple(coorreference)].copy()
                    new_entry.append(atom_mapping_w_dist_dict)
                    atom_mapping_el_w_dist[tuple(coorreference)] = new_entry
                else:
                    atom_mapping_el_w_dist[tuple(coorreference)] = [atom_mapping_w_dist_dict.copy()]

                coor_weirdos_el = [arr for arr in coor_weirdos_el if not np.array_equal(arr, closest24)]

            if counter == 0:
                coor_reducedreference_el = [arr for arr in coor_reducedreference_el if not np.array_equal(arr, coorreference)]

        duplicate_closest24_w_data = dictionary.Mapping.get_duplicate_closest24_w_data(atom_mapping_el_w_dist)

        # get the new reduced coorreference, based on the closest distance if it has multiple close coorreference within the radius
        if len(duplicate_closest24_w_data) > 0:
            atom_mapping_el_w_dist_closestduplicate = dictionary.Mapping.get_atom_mapping_el_w_dist_closestduplicate(atom_mapping_el_w_dist)
            coor_reducedreference_el_closestduplicate = [list(key) for key in atom_mapping_el_w_dist_closestduplicate.keys()]
        else:
            atom_mapping_el_w_dist_closestduplicate = atom_mapping_el_w_dist.copy()
            coor_reducedreference_el_closestduplicate = coor_reducedreference_el.copy()
        
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
        # sum_mapped_el = len(coor_reducedreference_el)
        sum_mapped_el_closestduplicate = len(coor_reducedreference_el_closestduplicate)
        sum_sanitycheck_el_closestduplicate = sum_mapped_el_closestduplicate + sum_weirdos_el

        if sum_sanitycheck_el_closestduplicate != 24:
            print(f'sum of mapped atom and weirdos are not 24 at idx: {idx}')
            sys.exit()

        # dataframe.at[idx, col_atom_mapping_el] = atom_mapping_el
        # dataframe.at[idx, col_atom_mapping_el_w_dist] = atom_mapping_el_w_dist
        # dataframe.at[idx, col_coor_weirdos_el_dict] = coor_weirdos_el_dict          # just added
        # dataframe.at[idx, col_coor_reducedreference_el] = np.array(coor_reducedreference_el)
        # dataframe.at[idx, col_sum_mapped_el] = sum_mapped_el
        # dataframe.at[idx, col_sum_sanitycheck_el] = sum_weirdos_el + sum_mapped_el
        dataframe.at[idx, col_coor_weirdos_el] = coor_weirdos_el
        dataframe.at[idx, col_sum_weirdos_el] = sum_weirdos_el
        dataframe.at[idx, col_duplicate_closest24_w_data_el] = duplicate_closest24_w_data
        dataframe.at[idx, col_atom_mapping_el_w_dist_closestduplicate] = atom_mapping_el_w_dist_closestduplicate
        dataframe.at[idx, col_coor_reducedreference_el_closestduplicate] = np.array(coor_reducedreference_el_closestduplicate)
        dataframe.at[idx, col_atom_mapping_el_closestduplicate] = atom_mapping_el_closestduplicate
        dataframe.at[idx, col_sum_mapped_el_closestduplicate] = sum_mapped_el_closestduplicate
        dataframe.at[idx, col_sum_sanitycheck_el_closestduplicate] = sum_sanitycheck_el_closestduplicate


def li_48htype2(dataframe, coor_structure_init_dict, el, max_mapping_radius_48htype2, activate_radius):
    # rename from: get_flag_map_weirdos_48htype2_el
    coor_reference_el_init = coor_structure_init_dict[el]         
    if activate_radius == 3:              
        col_coor_structure_48htype2_init_el = f"coor_weirdos_48htype1_48htype2_{el}"               # here is the difference
    elif activate_radius == 2:
        col_coor_structure_48htype2_init_el = f"coor_weirdos_{el}"               # here is the difference
    else:
        print("activate_radius is wrongly given")

    # col_atom_mapping_48htype2_el = f"atom_mapping_48htype2_{el}"
    # col_atom_mapping_48htype2_el_w_dist = f"atom_mapping_48htype2_{el}_w_dist"
    # col_coor_reducedreference_48htype2_el = f"coor_reducedreference_48htype2_{el}"
    # col_sum_mapped_48htype2_el = f"sum_mapped_48htype2_{el}"
    # col_sum_sanitycheck_48htype2_el = f"sum_sanitycheck_48htype2_{el}"
    col_flag_48htype2_el = f"flag_48htype2_{el}"
    col_coor_weirdos_48htype2_el = f"coor_weirdos_48htype2_{el}"
    col_sum_weirdos_48htype2_el = f"sum_weirdos_48htype2_{el}"
    col_duplicate_closest24_w_data_48htype2_el = f"duplicate_closest24_w_data_48htype2_{el}"
    col_coor_reducedreference_48htype2_el_closestduplicate = f"coor_reducedreference_48htype2_{el}_closestduplicate"
    col_sum_mapped_48htype2_el_closestduplicate = f"sum_mapped_48htype2_{el}_closestduplicate"
    col_sum_sanitycheck_48htype2_el_closestduplicate = f"sum_sanitycheck_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htype2_el_closestduplicate = f"atom_mapping_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htype2_el_w_dist_closestduplicate = f"atom_mapping_48htype2_{el}_w_dist_closestduplicate"

    # dataframe[col_atom_mapping_48htype2_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_48htype2_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_reducedreference_48htype2_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_sum_mapped_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_sum_sanitycheck_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_flag_48htype2_el] = "False"
    dataframe[col_coor_weirdos_48htype2_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_weirdos_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_duplicate_closest24_w_data_48htype2_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reducedreference_48htype2_el_closestduplicate] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_48htype2_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_48htype2_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_atom_mapping_48htype2_el_closestduplicate] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_atom_mapping_48htype2_el_w_dist_closestduplicate] = [{} for _ in range(len(dataframe.index))]

    coor_li48htype1_ref = coor_reference_el_init[0:48]
    coor_li48htype2_ref = coor_reference_el_init[48:96]
    coor_li24g_ref = coor_reference_el_init[96:120]

    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist = {} 
        atom_mapping_el_closestduplicate = {} 
        atom_mapping_el_w_dist_closestduplicate = {}
        coor_weirdos_el = []

        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_48htype2_init_el]#[el]             # dataframe['subdir_orientated_positive']
                                                                                        # or without orientation
                                                                                        # dataframe['subdir_CONTCAR']
        
        if len(coor_origin24_el_init) > 0:
            coor_reducedreference_el = coor_li48htype2_ref.copy()
            coor_weirdos_el = coor_origin24_el_init.copy()    

            for idxreference, coorreference in enumerate(coor_li48htype2_ref):        
                counter = 0
                atom_mapping_w_dist_dict = {}
                distance_prev = float("inf")
                closest24 = None

                for idx24, coor24 in enumerate(coor_origin24_el_init):
                    distance = calc_distance.mic_eucledian_distance(coorreference, coor24)
                    
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

                    # if tuple(coorreference) in atom_mapping_el_w_dist:
                    #     atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_w_dist_dict)
                    # else:
                    #     atom_mapping_el_w_dist[tuple(coorreference)] = atom_mapping_w_dist_dict
                        
                    # if tuple(coorreference) in atom_mapping_el:
                    #     atom_mapping_el[tuple(coorreference)].append(closest24)
                    # else:
                    #     atom_mapping_el[tuple(coorreference)] = tuple(closest24)

                    if tuple(coorreference) in atom_mapping_el_w_dist:
                        new_entry = atom_mapping_el_w_dist[tuple(coorreference)].copy()
                        new_entry.append(atom_mapping_w_dist_dict)
                        atom_mapping_el_w_dist[tuple(coorreference)] = new_entry
                    else:
                        atom_mapping_el_w_dist[tuple(coorreference)] = [atom_mapping_w_dist_dict.copy()]

                    coor_weirdos_el = [arr for arr in coor_weirdos_el if not np.array_equal(arr, closest24)]

                if counter == 0:
                    coor_reducedreference_el = [arr for arr in coor_reducedreference_el if not np.array_equal(arr, coorreference)]

            duplicate_closest24_w_data = dictionary.Mapping.get_duplicate_closest24_w_data(atom_mapping_el_w_dist)

            # get atom_mapping_el_closestduplicate
            # if duplicate_closest24_w_data != {}:
            if len(duplicate_closest24_w_data) > 0:
                atom_mapping_el_w_dist_closestduplicate = dictionary.Mapping.get_atom_mapping_el_w_dist_closestduplicate(atom_mapping_el_w_dist)
                coor_reducedreference_el_closestduplicate = [list(key) for key in atom_mapping_el_w_dist_closestduplicate.keys()]
            else:
                atom_mapping_el_w_dist_closestduplicate = atom_mapping_el_w_dist.copy()
                coor_reducedreference_el_closestduplicate = coor_reducedreference_el.copy()

            
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
            # sum_mapped_el = len(coor_reducedreference_el)
            sum_mapped_el_closestduplicate = len(coor_reducedreference_el_closestduplicate)

            # dataframe.at[idx, col_atom_mapping_48htype2_el] = atom_mapping_el
            # dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist] = atom_mapping_el_w_dist
            # dataframe.at[idx, col_coor_reducedreference_48htype2_el] = np.array(coor_reducedreference_el)
            # dataframe.at[idx, col_sum_mapped_48htype2_el] = sum_mapped_el
            # dataframe.at[idx, col_sum_sanitycheck_48htype2_el] = sum_mapped_el + sum_weirdos_el 
            dataframe.at[idx, col_coor_weirdos_48htype2_el] = coor_weirdos_el
            dataframe.at[idx, col_sum_weirdos_48htype2_el] = sum_weirdos_el
            dataframe.at[idx, col_duplicate_closest24_w_data_48htype2_el] = duplicate_closest24_w_data
            dataframe.at[idx, col_coor_reducedreference_48htype2_el_closestduplicate] = np.array(coor_reducedreference_el_closestduplicate)
            dataframe.at[idx, col_sum_mapped_48htype2_el_closestduplicate] = sum_mapped_el_closestduplicate
            dataframe.at[idx, col_sum_sanitycheck_48htype2_el_closestduplicate] = sum_mapped_el_closestduplicate + sum_weirdos_el
            dataframe.at[idx, col_atom_mapping_48htype2_el_closestduplicate] = atom_mapping_el_closestduplicate
            dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist_closestduplicate] = atom_mapping_el_w_dist_closestduplicate

        # elif coor_origin24_el_init == []:
        #     dataframe.at[idx, col_atom_mapping_48htype2_el] = {} 
        #     dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist] = {}
        #     dataframe.at[idx, col_sum_weirdos_48htype2_el] = 0


def li_48htype1_48htype2(dataframe, coor_structure_init_dict, el, max_mapping_radius_48htype1_48htype2):
    # rename from: get_flag_map_weirdos_48htype1_48htype2_el
    coor_reference_el_init = coor_structure_init_dict[el]                       
    col_coor_structure_48htype1_48htype2_init_el = f"coor_weirdos_{el}"               # here is the difference

    # col_atom_mapping_48htype1_48htype2_el = f"atom_mapping_48htype1_48htype2_{el}"
    # col_atom_mapping_48htype1_48htype2_el_w_dist = f"atom_mapping_48htype1_48htype2_{el}_w_dist"
    # col_coor_weirdos_48htype1_48htype2_el_dict = f"coor_weirdos_48htype1_48htype2_{el}_dict"            # just added
    # col_coor_reducedreference_48htype1_48htype2_el = f"coor_reducedreference_48htype1_48htype2_{el}"
    # col_sum_mapped_48htype1_48htype2_el = f"sum_mapped_48htype1_48htype2_{el}"
    # col_sum_sanitycheck_48htype1_48htype2_el = f"sum_sanitycheck_48htype1_48htype2_{el}"
    col_flag_48htype1_48htype2_el = f"flag_48htype1_48htype2_{el}"
    col_coor_weirdos_48htype1_48htype2_el = f"coor_weirdos_48htype1_48htype2_{el}"
    col_sum_weirdos_48htype1_48htype2_el = f"sum_weirdos_48htype1_48htype2_{el}"
    col_duplicate_closest24_w_data_48htype1_48htype2_el = f"duplicate_closest24_w_data_48htype1_48htype2_{el}"
    col_coor_reducedreference_48htype1_48htype2_el_closestduplicate = f"coor_reducedreference_48htype1_48htype2_{el}_closestduplicate"
    col_sum_mapped_48htype1_48htype2_el_closestduplicate = f"sum_mapped_48htype1_48htype2_{el}_closestduplicate"
    col_sum_sanitycheck_48htype1_48htype2_el_closestduplicate = f"sum_sanitycheck_48htype1_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htype1_48htype2_el_closestduplicate = f"atom_mapping_48htype1_48htype2_{el}_closestduplicate"

    # dataframe[col_atom_mapping_48htype1_48htype2_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_48htype1_48htype2_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_reducedreference_48htype1_48htype2_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_coor_weirdos_48htype1_48htype2_el_dict] = [{el: []} for _ in range(len(dataframe.index))]
    # dataframe[col_sum_mapped_48htype1_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_sum_sanitycheck_48htype1_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_flag_48htype1_48htype2_el] = "False"
    dataframe[col_coor_weirdos_48htype1_48htype2_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_weirdos_48htype1_48htype2_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_duplicate_closest24_w_data_48htype1_48htype2_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reducedreference_48htype1_48htype2_el_closestduplicate] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_48htype1_48htype2_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_48htype1_48htype2_el_closestduplicate] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_atom_mapping_48htype1_48htype2_el_closestduplicate] = [{} for _ in range(len(dataframe.index))] 

    coor_li48htype1_li48htype2_ref = coor_reference_el_init[0:96]
    coor_li24g_ref = coor_reference_el_init[96:120]

    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist = {} 
        atom_mapping_el_closestduplicate = {} 
        atom_mapping_el_w_dist_closestduplicate = {}
        coor_weirdos_el = []
        # coor_weirdos_el_dict = {}

        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_48htype1_48htype2_init_el]#[el]             # dataframe['subdir_orientated_positive']
                                                                                        # or without orientation
                                                                                        # dataframe['subdir_CONTCAR']
        
        if len(coor_origin24_el_init) > 0: # need this for the Operation.Distance.mic_eucledian_distance()
            coor_reducedreference_el = coor_li48htype1_li48htype2_ref.copy()
            coor_weirdos_el = coor_origin24_el_init.copy()    

            for idxreference, coorreference in enumerate(coor_li48htype1_li48htype2_ref):        
                counter = 0
                atom_mapping_w_dist_dict = {}
                distance_prev = float("inf")
                closest24 = None

                for idx24, coor24 in enumerate(coor_origin24_el_init):
                    distance = calc_distance.mic_eucledian_distance(coorreference, coor24)
                    
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

                    # if tuple(coorreference) in atom_mapping_el_w_dist:
                    #     atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_w_dist_dict)
                    # else:
                    #     atom_mapping_el_w_dist[tuple(coorreference)] = atom_mapping_w_dist_dict
                        
                    # if tuple(coorreference) in atom_mapping_el:
                    #     atom_mapping_el[tuple(coorreference)].append(closest24)
                    # else:
                    #     atom_mapping_el[tuple(coorreference)] = tuple(closest24)

                    if tuple(coorreference) in atom_mapping_el_w_dist:
                        new_entry = atom_mapping_el_w_dist[tuple(coorreference)].copy()
                        new_entry.append(atom_mapping_w_dist_dict)
                        atom_mapping_el_w_dist[tuple(coorreference)] = new_entry
                    else:
                        atom_mapping_el_w_dist[tuple(coorreference)] = [atom_mapping_w_dist_dict.copy()]
    
                    coor_weirdos_el = [arr for arr in coor_weirdos_el if not np.array_equal(arr, closest24)]

                if counter == 0:
                    coor_reducedreference_el = [arr for arr in coor_reducedreference_el if not np.array_equal(arr, coorreference)]

            duplicate_closest24_w_data = dictionary.Mapping.get_duplicate_closest24_w_data(atom_mapping_el_w_dist)

            # get atom_mapping_el_closestduplicate
            # if duplicate_closest24_w_data != {}:
            if len(duplicate_closest24_w_data) > 0:
                atom_mapping_el_w_dist_closestduplicate = dictionary.Mapping.get_atom_mapping_el_w_dist_closestduplicate(atom_mapping_el_w_dist)
                coor_reducedreference_el_closestduplicate = [list(key) for key in atom_mapping_el_w_dist_closestduplicate.keys()]
            else:
                atom_mapping_el_w_dist_closestduplicate = atom_mapping_el_w_dist.copy()
                coor_reducedreference_el_closestduplicate = coor_reducedreference_el.copy()

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
            # sum_mapped_el = len(coor_reducedreference_el)
            sum_mapped_el_closestduplicate = len(coor_reducedreference_el_closestduplicate)

            # dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el] = atom_mapping_el
            # dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_w_dist] = atom_mapping_el_w_dist
            # dataframe.at[idx, col_coor_weirdos_48htype1_48htype2_el_dict] = coor_weirdos_el_dict
            # dataframe.at[idx, col_coor_reducedreference_48htype1_48htype2_el] = np.array(coor_reducedreference_el)
            # dataframe.at[idx, col_sum_mapped_48htype1_48htype2_el] = sum_mapped_el
            # dataframe.at[idx, col_sum_sanitycheck_48htype1_48htype2_el] = sum_mapped_el + sum_weirdos_el 
            dataframe.at[idx, col_coor_weirdos_48htype1_48htype2_el] = coor_weirdos_el
            dataframe.at[idx, col_sum_weirdos_48htype1_48htype2_el] = sum_weirdos_el
            dataframe.at[idx, col_duplicate_closest24_w_data_48htype1_48htype2_el] = duplicate_closest24_w_data
            dataframe.at[idx, col_coor_reducedreference_48htype1_48htype2_el_closestduplicate] = np.array(coor_reducedreference_el_closestduplicate)
            dataframe.at[idx, col_sum_mapped_48htype1_48htype2_el_closestduplicate] = sum_mapped_el_closestduplicate
            dataframe.at[idx, col_sum_sanitycheck_48htype1_48htype2_el_closestduplicate] = sum_mapped_el_closestduplicate + sum_weirdos_el
            dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_closestduplicate] = atom_mapping_el_closestduplicate

        # elif coor_origin24_el_init == []:
        #     dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el] = {} 
        #     dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_w_dist] = {}
        #     dataframe.at[idx, col_sum_weirdos_48htype1_48htype2_el] = 0


def li_48htypesmerged_level1(dataframe, el):
    # rename from: get_flag_map_weirdos_48htypesmerged_level1_el

    # # col_flag_el = f"flag_{el}"
    # col_coor_weirdos_el = f"coor_weirdos_{el}"
    # col_coor_weirdos_el_dict = f"coor_weirdos_{el}_dict"            # just added
    # col_atom_mapping_el_w_dist = f"atom_mapping_{el}_w_dist"
    # col_coor_reducedreference_el = f"coor_reducedreference_{el}_closestduplicate"
    col_atom_mapping_el_closestduplicate = f"atom_mapping_{el}_closestduplicate"
    col_coor_reducedreference_el_closestduplicate = f"coor_reducedreference_{el}_closestduplicate"


    # # col_flag_48htype1_48htype2_el = f"flag_48htype1_48htype2_{el}"
    # col_atom_mapping_48htype1_48htype2_el_w_dist = f"atom_mapping_48htype1_48htype2_{el}_w_dist"
    # col_coor_reducedreference_48htype1_48htype2_el = f"coor_reducedreference_48htype1_48htype2_{el}_closestduplicate"
    col_sum_weirdos_48htype1_48htype2_el = f"sum_weirdos_48htype1_48htype2_{el}"
    col_atom_mapping_48htype1_48htype2_el_closestduplicate = f"atom_mapping_48htype1_48htype2_{el}_closestduplicate"
    col_coor_reducedreference_48htype1_48htype2_el_closestduplicate = f"coor_reducedreference_48htype1_48htype2_{el}_closestduplicate"


    col_flag_48htypesmerged_level1_el = f"flag_48htypesmerged_level1_{el}"
    col_atom_mapping_48htypesmerged_level1_el = f"atom_mapping_48htypesmerged_level1_{el}"
    # col_atom_mapping_48htypesmerged_level1_el_w_dist = f"atom_mapping_48htypesmerged_level1_{el}_w_dist"
    col_coor_reducedreference_48htypesmerged_level1_el = f"coor_reducedreference_48htypesmerged_level1_{el}"
    col_sum_mapped_48htypesmerged_level1_el = f"sum_mapped_48htypesmerged_level1_{el}"
    col_sum_sanitycheck_48htypesmerged_level1_el = f"sum_sanitycheck_48htypesmerged_level1_{el}"
    col_ndim_coor_reducedreference_level1_el_closestduplicate = f"ndim_coor_reducedreference_level1_{el}_closestduplicate"
    col_ndim_coor_reducedreference_48htype2_level1_el_closestduplicate = f"ndim_coor_reducedreference_48htype2_level1_{el}_closestduplicate"

    dataframe[col_flag_48htypesmerged_level1_el] = "False"
    dataframe[col_atom_mapping_48htypesmerged_level1_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_48htypesmerged_level1_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reducedreference_48htypesmerged_level1_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_48htypesmerged_level1_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_48htypesmerged_level1_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_reducedreference_level1_el_closestduplicate] = None
    dataframe[col_ndim_coor_reducedreference_48htype2_level1_el_closestduplicate] = None

    for idx in range(dataframe["geometry"].size):
        # print(f"idx_48htypesmerged: {idx}")
        atom_mapping_el_closestduplicate = dataframe.at[idx, col_atom_mapping_el_closestduplicate]
        atom_mapping_48htype1_48htype2_el_closestduplicate = dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_closestduplicate]
        # atom_mapping_el_w_dist = dataframe.at[idx, col_atom_mapping_el_w_dist]
        # atom_mapping_48htype1_48htype2_el_w_dist = dataframe.at[idx, col_atom_mapping_48htype1_48htype2_el_w_dist]
        # coor_reducedreference_el = dataframe.at[idx, col_coor_reducedreference_el]
        # coor_reducedreference_48htype1_48htype2_el = dataframe.at[idx, col_coor_reducedreference_48htype1_48htype2_el]
        # _closestduplicate
        coor_reducedreference_el_closestduplicate = dataframe.at[idx, col_coor_reducedreference_el_closestduplicate]
        coor_reducedreference_48htype1_48htype2_el_closestduplicate = dataframe.at[idx, col_coor_reducedreference_48htype1_48htype2_el_closestduplicate]
        # # these are just copying
        # coor_weirdos_48htypesmerged_level1_el = dataframe.at[idx, col_coor_weirdos_48htype1_48htype2_el]
        sum_weirdos_48htypesmerged_level1_el = dataframe.at[idx, col_sum_weirdos_48htype1_48htype2_el]
        # duplicate_closest24_w_data_48htypesmerged = dataframe.at[idx, col_duplicate_closest24_w_data_48htype1_48htype2_el]
        
        # # _closestduplicate
        # coor_reducedreference_48htypesmerged_level1_el_closestduplicate = dataframe.at[idx, col_coor_reducedreference_48htype1_48htype2_el_closestduplicate]
        # coor_reducedreference_48htypesmerged_level1_el = []

        atom_mapping_48htypesmerged_level1_el = dictionary.merge_dictionaries(atom_mapping_el_closestduplicate, atom_mapping_48htype1_48htype2_el_closestduplicate)
        duplicate_coor24s = dictionary.get_duplicate_values(atom_mapping_48htypesmerged_level1_el)
        if len(duplicate_coor24s) > 1:        
            dataframe.at[idx, col_flag_48htypesmerged_level1_el] = "True"        
        # for coorreference in atom_mapping_48htypesmerged_level1_el:
        #     values24 = atom_mapping_48htypesmerged_level1_el[coorreference]   # get value for the key
        #     if len(values24) > 1:        
        #         dataframe.at[idx, col_flag_48htypesmerged_level1_el] = "True"
        
        # atom_mapping_48htypesmerged_level1_el_w_dist = Operation.Dict.merge_dictionaries(atom_mapping_el_w_dist, atom_mapping_48htype1_48htype2_el_w_dist)

        # # if coor_reducedreference_48htype1_48htype2_el != None:
        # # if coor_reducedreference_48htype1_48htype2_el.size > 0:
        # if coor_reducedreference_48htype1_48htype2_el.ndim == 2:
        #     coor_reducedreference_48htypesmerged_level1_el = np.concatenate((coor_reducedreference_el, coor_reducedreference_48htype1_48htype2_el), axis=0)
        # elif coor_reducedreference_48htype1_48htype2_el.ndim == 1:
        #     coor_reducedreference_48htypesmerged_level1_el = np.array(coor_reducedreference_el.copy())
        # else:
        #     break

        ## we use _closestduplicate here since it's the corrected one wrt distance
        # if coor_reducedreference_48htype1_48htype2_el_closestduplicate != None:
        # if coor_reducedreference_48htype1_48htype2_el_closestduplicate.size > 0:
        # # # if coor_reducedreference_48htype1_48htype2_el_closestduplicate.ndim == 2:
        # # #     if coor_reducedreference_el_closestduplicate.ndim == 2:
        # # #         coor_reducedreference_48htypesmerged_level1_el_closestduplicate = np.concatenate((coor_reducedreference_el_closestduplicate, coor_reducedreference_48htype1_48htype2_el_closestduplicate), axis=0)
        # # #     else:
        # # #         print(f"coor_reducedreference_el_closestduplicate has no correct dimension at idx: {idx}, dimension: {coor_reducedreference_el_closestduplicate.ndim}")
        # # #         pass
        # # #         # print(f"coor_reducedreference_el_closestduplicate: \n {coor_reducedreference_el_closestduplicate}")
        # # # elif coor_reducedreference_48htype1_48htype2_el_closestduplicate.ndim == 1:
        # # #     coor_reducedreference_48htypesmerged_level1_el_closestduplicate = np.array(coor_reducedreference_el_closestduplicate.copy())
        # # # else:
        # # #     print(f"coor_reducedreference_48htype1_48htype2_el_closestduplicate has no correct dimension at idx: {idx}, dimension: {coor_reducedreference_48htypesmerged_level1_el_closestduplicate.ndim}")
        # # #     # break

        if coor_reducedreference_48htype1_48htype2_el_closestduplicate.ndim == coor_reducedreference_el_closestduplicate.ndim:
            coor_reducedreference_48htypesmerged_level1_el = np.concatenate((coor_reducedreference_el_closestduplicate, coor_reducedreference_48htype1_48htype2_el_closestduplicate), axis=0)
            # else:
            #     print(f"coor_reducedreference_el_closestduplicate has no correct dimension at idx: {idx}")
            #     continue
                # print(f"coor_reducedreference_el_closestduplicate: \n {coor_reducedreference_el_closestduplicate}")
        elif coor_reducedreference_48htype1_48htype2_el_closestduplicate.ndim == 1:
            coor_reducedreference_48htypesmerged_level1_el = np.array(coor_reducedreference_el_closestduplicate.copy())
        elif coor_reducedreference_el_closestduplicate.ndim == 1:
            coor_reducedreference_48htypesmerged_level1_el = np.array(coor_reducedreference_48htype1_48htype2_el_closestduplicate.copy())
        else:
            print(f"coor_reducedreference_48htype1_48htype2_el_closestduplicate or coor_reducedreference_el_closestduplicate has no correct dimension at idx: {idx}")
            # break

        # sum_mapped_48htypesmerged_level1_el = len(atom_mapping_48htypesmerged_level1_el)
        sum_mapped_48htypesmerged_level1_el = len(coor_reducedreference_48htypesmerged_level1_el)

        ndim_coor_reducedreference_el_closestduplicate = coor_reducedreference_el_closestduplicate.ndim
        ndim_coor_reducedreference_48htype1_48htype2_el_closestduplicate = coor_reducedreference_48htype1_48htype2_el_closestduplicate.ndim

        dataframe.at[idx, col_atom_mapping_48htypesmerged_level1_el] = atom_mapping_48htypesmerged_level1_el
        # dataframe.at[idx, col_atom_mapping_48htypesmerged_level1_el_w_dist] = atom_mapping_48htypesmerged_level1_el_w_dist
        # dataframe.at[idx, col_coor_weirdos_48htypesmerged_level1_el] = coor_weirdos_48htypesmerged_level1_el        # these are just copying
        # dataframe.at[idx, col_coor_reducedreference_48htypesmerged_level1_el] = coor_reducedreference_48htypesmerged_level1_el  
        # dataframe.at[idx, col_sum_weirdos_48htypesmerged_level1_el] = sum_weirdos_48htypesmerged_level1_el          # these are just copying
        # dataframe.at[idx, col_sum_mapped_48htypesmerged_level1_el] = sum_mapped_48htypesmerged_level1_el
        # dataframe.at[idx, col_sum_sanitycheck_48htypesmerged_level1_el] = sum_weirdos_48htypesmerged_level1_el + sum_mapped_48htypesmerged_level1_el
        # dataframe.at[idx, col_duplicate_closest24_w_data_48htypesmerged_level1_el] = duplicate_closest24_w_data_48htypesmerged     # these are just copying
        dataframe.at[idx, col_coor_reducedreference_48htypesmerged_level1_el] = coor_reducedreference_48htypesmerged_level1_el
        dataframe.at[idx, col_sum_mapped_48htypesmerged_level1_el] = sum_mapped_48htypesmerged_level1_el 
        dataframe.at[idx, col_sum_sanitycheck_48htypesmerged_level1_el] = sum_mapped_48htypesmerged_level1_el + sum_weirdos_48htypesmerged_level1_el

        dataframe.at[idx, col_ndim_coor_reducedreference_level1_el_closestduplicate] = ndim_coor_reducedreference_el_closestduplicate
        dataframe.at[idx, col_ndim_coor_reducedreference_48htype2_level1_el_closestduplicate] = ndim_coor_reducedreference_48htype1_48htype2_el_closestduplicate


def li_48htypesmerged(dataframe, el, activate_radius):
    # rename from: get_flag_map_48htypesmerged_el
    if activate_radius == 3:
        # # col_flag_48htype1_48htype2_el = f"flag_48htype1_48htype2_{el}"
        # col_atom_mapping_el_w_dist = f"atom_mapping_48htype1_48htype2_{el}_w_dist"
        # col_coor_reducedreference_el = f"coor_reducedreference_48htype1_48htype2_{el}"
        # col_atom_mapping_el_closestduplicate = f"atom_mapping_48htype1_48htype2_{el}_closestduplicate"
        # col_coor_reducedreference_el_closestduplicate = f"coor_reducedreference_48htype1_48htype2_{el}_closestduplicate"
        # # col_flag_48htypesmerged_level1_el = f"flag_48htypesmerged_level1_{el}"
        col_atom_mapping_el_closestduplicate = f"atom_mapping_48htypesmerged_level1_{el}"
        # col_atom_mapping_el_w_dist = f"atom_mapping_48htypesmerged_level1_{el}_w_dist"
        col_coor_reducedreference_el_closestduplicate = f"coor_reducedreference_48htypesmerged_level1_{el}"

    elif activate_radius == 2:
        # # col_flag_el = f"flag_{el}"
        # col_coor_weirdos_el = f"coor_weirdos_{el}"
        # col_coor_weirdos_el_dict = f"coor_weirdos_{el}_dict"            # just added
        # col_atom_mapping_el_w_dist = f"atom_mapping_{el}_w_dist"
        # col_coor_reducedreference_el = f"coor_reducedreference_{el}"
        col_atom_mapping_el_closestduplicate = f"atom_mapping_{el}_closestduplicate"
        col_coor_reducedreference_el_closestduplicate = f"coor_reducedreference_{el}_closestduplicate"
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
    # col_coor_reducedreference_48htype2_el = f"coor_reducedreference_48htype2_{el}"
    col_atom_mapping_48htype2_el_closestduplicate = f"atom_mapping_48htype2_{el}_closestduplicate"
    col_sum_weirdos_48htype2_el = f"sum_weirdos_48htype2_{el}"
    col_coor_reducedreference_48htype2_el_closestduplicate = f"coor_reducedreference_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htype2_el_w_dist_closestduplicate = f"atom_mapping_48htype2_{el}_w_dist_closestduplicate"
    # col_atom_mapping_48htype2_el_w_dist = f"atom_mapping_48htype2_{el}_w_dist"
    # col_coor_weirdos_48htype2_el = f"coor_weirdos_48htype2_{el}"
    # col_sum_mapped_48htype2_el = f"sum_mapped_48htype2_{el}"
    # col_sum_sanitycheck_48htype2_el = f"sum_sanitycheck_48htype2_{el}"
    # col_duplicate_closest24_w_data_48htype2_el = f"duplicate_closest24_w_data_48htype2_{el}"
    # col_sum_mapped_48htype2_el_closestduplicate = f"sum_mapped_48htype2_{el}_closestduplicate"
    # cocommand:cellOutput.enableScrolling?23798be1-d942-4554-a3c6-774194ee7e7el_sum_sanitycheck_48htype2_el_closestduplicate = f"sum_sanitycheck_48htype2_{el}_closestduplicate"

    # col_coor_reducedreference_48htypesmerged_el = f"coor_reducedreference_48htypesmerged_{el}"
    # col_sum_mapped_48htypesmerged_el = f"sum_mapped_48htypesmerged_{el}"
    # col_sum_sanitycheck_48htypesmerged_el = f"sum_sanitycheck_48htypesmerged_{el}"
    col_flag_48htypesmerged_el = f"flag_48htypesmerged_{el}"
    col_atom_mapping_48htypesmerged_el = f"atom_mapping_48htypesmerged_{el}"
    # col_coor_weirdos_48htypesmerged_el = f"coor_weirdos_48htypesmerged_{el}"
    # col_sum_weirdos_48htypesmerged_el = f"sum_weirdos_48htypesmerged_{el}"
    # col_duplicate_closest24_w_data_48htypesmerged_el = f"duplicate_closest24_w_data_48htypesmerged_{el}"
    col_coor_reducedreference_48htypesmerged_el = f"coor_reducedreference_48htypesmerged_{el}"
    col_sum_mapped_48htypesmerged_el = f"sum_mapped_48htypesmerged_{el}"
    col_sum_sanitycheck_48htypesmerged_el = f"sum_sanitycheck_48htypesmerged_{el}"
    col_ndim_coor_reducedreference_el_closestduplicate = f"ndim_coor_reducedreference_{el}_closestduplicate"
    col_ndim_coor_reducedreference_48htype2_el_closestduplicate = f"ndim_coor_reducedreference_48htype2_{el}_closestduplicate"
    col_atom_mapping_48htypesmerged_el_w_dist = f"atom_mapping_48htypesmerged_{el}_w_dist"

    # dataframe[col_coor_reducedreference_48htypesmerged_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_sum_mapped_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_sum_sanitycheck_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_flag_48htypesmerged_el] = "False"
    dataframe[col_atom_mapping_48htypesmerged_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_48htypesmerged_el_w_dist] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_weirdos_48htypesmerged_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_sum_weirdos_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    # dataframe[col_duplicate_closest24_w_data_48htypesmerged_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_coor_reducedreference_48htypesmerged_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_sum_mapped_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_sanitycheck_48htypesmerged_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_reducedreference_el_closestduplicate] = None
    dataframe[col_ndim_coor_reducedreference_48htype2_el_closestduplicate] = None
    dataframe[col_atom_mapping_48htypesmerged_el_w_dist] = [{} for _ in range(len(dataframe.index))]

    for idx in range(dataframe["geometry"].size):
        # print(f"idx_48htypesmerged: {idx}")
        # atom_mapping_el_w_dist = dataframe.at[idx, col_atom_mapping_el_w_dist]
        # atom_mapping_48htype2_el_w_dist = dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist]
        atom_mapping_el_closestduplicate = dataframe.at[idx, col_atom_mapping_el_closestduplicate]
        atom_mapping_48htype2_el_closestduplicate = dataframe.at[idx, col_atom_mapping_48htype2_el_closestduplicate]
        # coor_reducedreference_el = dataframe.at[idx, col_coor_reducedreference_el]
        # coor_reducedreference_48htype2_el = dataframe.at[idx, col_coor_reducedreference_48htype2_el]
        # _closestduplicate
        coor_reducedreference_el_closestduplicate = dataframe.at[idx, col_coor_reducedreference_el_closestduplicate]
        coor_reducedreference_48htype2_el_closestduplicate = dataframe.at[idx, col_coor_reducedreference_48htype2_el_closestduplicate]
        atom_mapping_el_w_dist_closestduplicate = dataframe.at[idx, col_atom_mapping_el_w_dist_closestduplicate]
        atom_mapping_48htype2_el_w_dist_closestduplicate = dataframe.at[idx, col_atom_mapping_48htype2_el_w_dist_closestduplicate]
        # these are just copying
        # coor_weirdos_48htypesmerged_el = dataframe.at[idx, col_coor_weirdos_48htype2_el]
        sum_weirdos_48htypesmerged_el = dataframe.at[idx, col_sum_weirdos_48htype2_el]
        # duplicate_closest24_w_data_48htypesmerged = dataframe.at[idx, col_duplicate_closest24_w_data_48htype2_el]
        
        # # _closestduplicate
        # coor_reducedreference_48htypesmerged_el_closestduplicate = dataframe.at[idx, col_coor_reducedreference_48htype2_el_closestduplicate]
        coor_reducedreference_48htypesmerged_el = []

        atom_mapping_48htypesmerged_el = dictionary.merge_dictionaries(atom_mapping_el_closestduplicate, atom_mapping_48htype2_el_closestduplicate)
        duplicate_coor24s = dictionary.get_duplicate_values(atom_mapping_48htypesmerged_el)
        if len(duplicate_coor24s) > 1:        
            dataframe.at[idx, col_flag_48htypesmerged_el] = "True"        
        # for coorreference in atom_mapping_48htypesmerged_el:
        #     values24 = atom_mapping_48htypesmerged_el[coorreference]   # get value for the key
        #     if len(values24) > 1:        
        #         dataframe.at[idx, col_flag_48htypesmerged_el] = "True"
        
        # atom_mapping_48htypesmerged_el_w_dist = Operation.Dict.merge_dictionaries(atom_mapping_el_w_dist, atom_mapping_48htype2_el_w_dist)

        # # if coor_reducedreference_48htype2_el != None:
        # # if coor_reducedreference_48htype2_el.size > 0:
        # if coor_reducedreference_48htype2_el.ndim == 2:
        #     coor_reducedreference_48htypesmerged_el = np.concatenate((coor_reducedreference_el, coor_reducedreference_48htype2_el), axis=0)
        # elif coor_reducedreference_48htype2_el.ndim == 1:
        #     coor_reducedreference_48htypesmerged_el = np.array(coor_reducedreference_el.copy())
        # else:
        #     break

        ## we use _closestduplicate here since it's the corrected one wrt distance
        # if coor_reducedreference_48htype2_el_closestduplicate != None:
        # if coor_reducedreference_48htype2_el_closestduplicate.size > 0:
        # # # if coor_reducedreference_48htype2_el_closestduplicate.ndim == 2:
        # # #     if coor_reducedreference_el_closestduplicate.ndim == 2:
        # # #         coor_reducedreference_48htypesmerged_el_closestduplicate = np.concatenate((coor_reducedreference_el_closestduplicate, coor_reducedreference_48htype2_el_closestduplicate), axis=0)
        # # #     else:
        # # #         print(f"coor_reducedreference_el_closestduplicate has no correct dimension at idx: {idx}, dimension: {coor_reducedreference_el_closestduplicate.ndim}")
        # # #         pass
        # # #         # print(f"coor_reducedreference_el_closestduplicate: \n {coor_reducedreference_el_closestduplicate}")
        # # # elif coor_reducedreference_48htype2_el_closestduplicate.ndim == 1:
        # # #     coor_reducedreference_48htypesmerged_el_closestduplicate = np.array(coor_reducedreference_el_closestduplicate.copy())
        # # # else:
        # # #     print(f"coor_reducedreference_48htype2_el_closestduplicate has no correct dimension at idx: {idx}, dimension: {coor_reducedreference_48htypesmerged_el_closestduplicate.ndim}")
        # # #     # break

        if coor_reducedreference_48htype2_el_closestduplicate.ndim == coor_reducedreference_el_closestduplicate.ndim:
            coor_reducedreference_48htypesmerged_el = np.concatenate((coor_reducedreference_el_closestduplicate, coor_reducedreference_48htype2_el_closestduplicate), axis=0)
            # else:
            #     print(f"coor_reducedreference_el_closestduplicate has no correct dimension at idx: {idx}")
            #     continue
                # print(f"coor_reducedreference_el_closestduplicate: \n {coor_reducedreference_el_closestduplicate}")
        elif coor_reducedreference_48htype2_el_closestduplicate.ndim == 1:
            coor_reducedreference_48htypesmerged_el = np.array(coor_reducedreference_el_closestduplicate.copy())
        elif coor_reducedreference_el_closestduplicate.ndim == 1:
            coor_reducedreference_48htypesmerged_el = np.array(coor_reducedreference_48htype2_el_closestduplicate.copy())
        else:
            print(f"coor_reducedreference_48htype2_el_closestduplicate or coor_reducedreference_el_closestduplicate has no correct dimension at idx: {idx}")
            # break

        # sum_mapped_48htypesmerged_el = len(atom_mapping_48htypesmerged_el)
        sum_mapped_48htypesmerged_el = len(coor_reducedreference_48htypesmerged_el)

        ndim_coor_reducedreference_el_closestduplicate = coor_reducedreference_el_closestduplicate.ndim
        ndim_coor_reducedreference_48htype2_el_closestduplicate = coor_reducedreference_48htype2_el_closestduplicate.ndim

        atom_mapping_48htypesmerged_el_w_dist = atom_mapping_el_w_dist_closestduplicate | atom_mapping_48htype2_el_w_dist_closestduplicate

        # dataframe.at[idx, col_coor_weirdos_48htypesmerged_el] = coor_weirdos_48htypesmerged_el        # these are just copying
        # dataframe.at[idx, col_coor_reducedreference_48htypesmerged_el] = coor_reducedreference_48htypesmerged_el  
        # dataframe.at[idx, col_sum_weirdos_48htypesmerged_el] = sum_weirdos_48htypesmerged_el          # these are just copying
        # dataframe.at[idx, col_sum_mapped_48htypesmerged_el] = sum_mapped_48htypesmerged_el
        # dataframe.at[idx, col_sum_sanitycheck_48htypesmerged_el] = sum_weirdos_48htypesmerged_el + sum_mapped_48htypesmerged_el
        # dataframe.at[idx, col_duplicate_closest24_w_data_48htypesmerged_el] = duplicate_closest24_w_data_48htypesmerged     # these are just copying
        # dataframe.at[idx, col_atom_mapping_48htypesmerged_el_w_dist] = atom_mapping_48htypesmerged_el_w_dist
        dataframe.at[idx, col_atom_mapping_48htypesmerged_el] = atom_mapping_48htypesmerged_el
        dataframe.at[idx, col_coor_reducedreference_48htypesmerged_el] = coor_reducedreference_48htypesmerged_el
        dataframe.at[idx, col_sum_mapped_48htypesmerged_el] = sum_mapped_48htypesmerged_el 
        dataframe.at[idx, col_sum_sanitycheck_48htypesmerged_el] = sum_mapped_48htypesmerged_el + sum_weirdos_48htypesmerged_el
        dataframe.at[idx, col_ndim_coor_reducedreference_el_closestduplicate] = ndim_coor_reducedreference_el_closestduplicate
        dataframe.at[idx, col_ndim_coor_reducedreference_48htype2_el_closestduplicate] = ndim_coor_reducedreference_48htype2_el_closestduplicate
        dataframe.at[idx, col_atom_mapping_48htypesmerged_el_w_dist] = atom_mapping_48htypesmerged_el_w_dist


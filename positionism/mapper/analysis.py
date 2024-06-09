import pandas as pd
import numpy as np
import os
from collections import defaultdict

from positionism.functional import func_distance

from pymatgen.core.structure import Structure

def relabel_48htype1(dataframe, litype):
    if litype == 1:
        col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"

        label_new = "48htype2"

        for idx in range(dataframe["geometry"].size):
            idx_coor_limapped_weirdos_dict = dataframe[col_idx_coor_limapped_weirdos_dict][idx]

            for key, val in idx_coor_limapped_weirdos_dict.items():
                label = val['label']
                if label == "48htype1":
                    val['label'] = label_new
            
            dataframe.at[idx, col_idx_coor_limapped_weirdos_dict] = idx_coor_limapped_weirdos_dict
    else:
        pass


def get_occupancy(dataframe, coor_structure_init_dict_expanded, tuple_metainfo, el):
    """
    strict_count: True or False
    """

    col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"
    col_sum_of_weirdos_Li = f"#weirdos_Li"

    col_occupancy_strict = "occupancy_strict"
    col_occupancy_notstrict = "occupancy_notstrict"
    col_idx_coor24li_tuple_cage_belongin_empty = "idx_coor24li_tuple_cage_belongin_empty"

    dataframe[col_occupancy_strict] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_occupancy_notstrict] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor24li_tuple_cage_belongin_empty] = [{} for _ in range(len(dataframe.index))]

    coor_structure_init_dict_expanded_el = coor_structure_init_dict_expanded[el]
    coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]

    # for idx in [4]: 
    for idx in range(dataframe["geometry"].size):
        idx_coor24li_tuple_cage_belongin_empty = defaultdict(list)
        temp_idxtuple_coor24li_cage_belongin_empty = defaultdict(list)

        idx_coor_limapped_weirdos_dict = dataframe.at[idx, col_idx_coor_limapped_weirdos_dict]

        # # # file_24Li = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}.cif"
        # # # file_path_24Li = os.path.join(destination_directory, file_24Li)

        # # # coor_structure_24Li_dict_el = ReadStructure.Coor.get_coor_structure_init_dict(Structure.from_file(file_path_24Li))[el]
        
        # # # # Convert lists of arrays to sets for efficient comparison
        # # # set_coor_structure = set(map(tuple, coor_structure_24Li_dict_el))
        # # # set_coor_li48htype1_ref = set(map(tuple, coor_li48htype1_ref))

        # # # # Find the difference between the two sets
        # # # result_set = set_coor_structure.difference(set_coor_li48htype1_ref)

        # # # # Convert the result back to a list of arrays
        # # # result_list = list(map(np.array, result_set))
        # for idx_triad, val in tuple_metainfo.items():

        for idx_triad, values_list in tuple_metainfo.items():
            idx_coor24li_tuple_cage_belongin_empty[idx_triad] = []    # idx_atom as the actual index
            temp_idxtuple_coor24li_cage_belongin_empty[idx_triad] = []

            for entry in values_list:

                coor_metainfo = entry['coor']
                coor_metainfo_rounded = tuple(round(coordinate, 5) for coordinate in coor_metainfo)
                
                # for i in result_list:
                for idx_atom, values_list_atom in idx_coor_limapped_weirdos_dict.items():

                    coor_li_mapped = values_list_atom['coor']
                    coor_li_mapped_rounded = tuple(round(coordinate, 5) for coordinate in coor_li_mapped)
            
                    if (coor_li_mapped_rounded == coor_metainfo_rounded):
                        # if (tuple(i) == tuple(coor_metainfo_rounded)).all():
                        # coor24li_tuple_belongin_dict = {'coor': i, 'type':entry['type']}
                        idx_coor24li_tuple_cage_belongin_empty_dict = {'coor': coor_li_mapped_rounded, 'type':entry['type'], 'idx_tuple':idx_triad, 'idx_cage':entry['idx_cage']}
                        idx_coor24li_tuple_cage_belongin_empty[idx_atom].append(idx_coor24li_tuple_cage_belongin_empty_dict)        # changed into idx_atom

                        temp_idxtuple_coor24li_cage_belongin_empty_dict = {'coor': coor_li_mapped_rounded, 'type':entry['type'], 'idx_cage':entry['idx_cage']}
                        temp_idxtuple_coor24li_cage_belongin_empty[idx_triad].append(temp_idxtuple_coor24li_cage_belongin_empty_dict)        # changed into idx_atom

        # idx_coor_weirdos_Li_dict = dataframe['idx_coor_weirdos_Li'][idx]

        # for idx_weirdo, values_list in idx_coor_weirdos_Li_dict.items():
        #         coorweirdo_tuple_belongin_dict = {'coor': values_list, 'type':'weirdo'}
        #         idx_coor24li_tuple_cage_belongin_empty['weirdo'].append(coorweirdo_tuple_belongin_dict)
        
        # for key, val in idx_coor24li_tuple_cage_belongin_empty.items():
        #     for i

        len_occupancy = []
        for key, val in temp_idxtuple_coor24li_cage_belongin_empty.items():
            len_occupancy.append(len(val))

        # Initialize a counter
        amount_48htype1 = 0
        # # amount_weirdo = 0

        # Iterate through each key and list in the dictionary
        for key, list_of_dicts in idx_coor_limapped_weirdos_dict.items():
            # Iterate through each dictionary in the list
            # # for item in list_of_dicts:
            # Check if the 'type' is '48htype1'
            if list_of_dicts['label'] == '48htype1':
                # Increment the counter
                amount_48htype1 += 1
                    # if amount_48htype1 == 1:
                    #     print('true')
                # # if item['type'] == 'weirdo':
                # #     # Increment the counter
                # #     amount_weirdo += 1

        # print(amount_48htype1)
        # amount_48htype1 = (len(coor_structure_24Li_dict_el)-len(result_list))
        amount_weirdo = dataframe[col_sum_of_weirdos_Li][idx]
        occupancy_2 = len_occupancy.count(2)
        occupancy_1 = len_occupancy.count(1)

        occupancy_0_strict = len_occupancy.count(0)
        occupancy_0_notstrict = len_occupancy.count(0) - amount_48htype1 - amount_weirdo

        for number in len_occupancy:
            if number > 2:
                print("Occupancy greater than 2 detected, breaking the loop.")
                break

        # sanity_check_occupancy = occupancy_2 * 2 + occupancy_1 + amount_48htype1 + amount_weirdo + occupancy_0

        # if sanity_check_occupancy != 24:
        #     print(f'sum of occupancy not achieved at idx {idx}')
        #     sys.exit()

        # print(f"idx: {idx}")

        # if sanity_check_occupancy != 24:
        #     sys.exit()

        occupancy_strict = {'2': occupancy_2, '1': occupancy_1, '0': occupancy_0_strict, '48htype1': amount_48htype1,'weirdo': amount_weirdo}
        occupancy_notstrict = {'2': occupancy_2, '1': occupancy_1, '0': occupancy_0_notstrict, '48htype1': amount_48htype1,'weirdo': amount_weirdo}

        dataframe.at[idx, col_occupancy_strict] = occupancy_strict
        dataframe.at[idx, col_occupancy_notstrict] = occupancy_notstrict
        dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_empty] = idx_coor24li_tuple_cage_belongin_empty


# class Movement:    
#     class Distance:
def get_distance_li(dataframe, max_mapping_radius, destination_directory, idx_file_group, idx_ref, mean_ref, var_filename):
    # rename from: get_distance_litoli
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
                distance = func_distance.mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

                dict_distance[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref_mean}, coor_Li: {coor_Li[j]}'}
                df_distance.at[index, f"{j}"] = distance

                diameter_24g48h = max_mapping_radius * 2
                # if distance < diameter_24g48h and index != idx_ref:
                if distance > diameter_24g48h and index != idx_ref:
                    print(f"path: {index}, Li: {j}, distance: {distance}")

        elif mean_ref == False:
            for j in range(len(coor_Li)):
                df_distance.at[index, f"{j}"] = None  

                distance = func_distance.mic_eucledian_distance(coor_Li_ref[j], coor_Li[j])

                dict_distance[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref[j]}, coor_Li: {coor_Li[j]}'}
                df_distance.at[index, f"{j}"] = distance

                diameter_24g48h = max_mapping_radius * 2
                # if distance < diameter_24g48h and index != idx_ref:
                if distance > diameter_24g48h and index != idx_ref:
                    print(f"path: {index}, Li: {j}, distance: {distance}")

    #         coors_Li_dist_structures_dict = {}
            
    #         for k in range(len(coor_Li)):

    #             distance_litoli = calc_distance.mic_eucledian_distance(coor_Li[j], coor_Li[k])

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

    # #     # #     distance = calc_distance.mic_eucledian_distance(coor_Li_ref[j], coor_Li[j])

    # #     # #     df_distance.at[i, f"{j}"] = distance

    # #     # #     diameter_24g48h = max_mapping_radius * 2
    # #     # #     if distance < diameter_24g48h and i != 0:
    # #     # #         print(f"path: {i}, Li: {j}, distance: {distance}")

    return df_distance, dataframe_group


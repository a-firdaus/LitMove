import pandas as pd
import numpy as np
import os
from collections import defaultdict

from functional import func_distance

from pymatgen.core.structure import Structure


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


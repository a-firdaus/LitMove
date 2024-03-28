import numpy as np
from collections import defaultdict

from functional import calc_distance


# class Metainfo:
def tuple(coor_structure_init_dict_expanded, litype, el):
    # rename from: get_tuple_metainfo
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
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
        
        elif litype == 2:
            for j in coor_li48htype1_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype2_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 3:
            for j in coor_li48htype1_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 4:
            for j in coor_li48htype1_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 5:
            for j in coor_li48htype1_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype5_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 6:
            for j in coor_li48htype1_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype5_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype6_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

        elif litype == 7:
            for j in coor_li48htype1_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype5_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype6_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype7_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype7'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
        elif litype == 8:
            for j in coor_li48htype1_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
            for j in coor_li48htype2_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype3_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype4_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype5_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype6_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype7_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype7'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

            for j in coor_li48htype8_ref:
                distance = calc_distance.mic_eucledian_distance(i, j)

                tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype8'}

                tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)


    sorted_tuple_metainfo_all = {key: sorted(value, key=lambda x: x['dist']) for key, value in tuple_metainfo_all.items()}
    top_n_tuple_metainfo = {k: v[0:n] for k, v in sorted_tuple_metainfo_all.items()}

    for key, values_list in top_n_tuple_metainfo.items():
        selected_values = [{'coor': entry['coor'], "type": entry["type"]} for entry in values_list]
        tuple_metainfo[key] = selected_values
                        
    return tuple_metainfo   


def coor_48htype2(coor_structure_init_dict_expanded, el):
    # rename from: get_coor_48htype1_metainfo
    all_coor_48htype1 = coor_structure_init_dict_expanded[el][24:72]

    coor_48htype1_metainfo = defaultdict(list)

    for id, coor in enumerate(all_coor_48htype1):
        coor_48htype1_metainfo[id] = {'coor': coor}

    return coor_48htype1_metainfo


def idx_cage_coor_24g(coor_24g_array, labels, idx_coor_cage_order, amount_clusters):
    """
    Note
    ====
        - to rename
    """
    # rename from: get_idx_cage_coor_24g ???
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


def tuple_cage(tuple_metainfo, idx_cage_coor_24g):
    # rename from: get_tuple_cage_metainfo
    """
    Note
    ====
        - maybe to rename
    """

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

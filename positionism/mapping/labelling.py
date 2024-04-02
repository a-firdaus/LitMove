from positionism.functional import func_distance


def weirdos_to_top_n_closestcoorref_el(dataframe, coor_structure_init_dict, el, litype):
    # rename from: get_distance_weirdos_label_el
    """
    Calculate WEIRDOS to top 3 or 1 closest coor reference. Get distances, and labels.

    Args
    ====
    dataframe: pandas.DataFrame
        The DataFrame containing the data.
    coor_structure_init_dict: dict
        Dictionary containing the initial coordinate structure.
    el: str
        Element for which distances and labels are calculated.
    litype: int
        Lithium type (int from 0 to 8)

    Note
    ====
    - TO DO: add idx of weirdo and coorreference
    """
    coor_reference_el_init = coor_structure_init_dict[el]
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
    # dataframe[col_top3_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_dist_weirdos_atomreference_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_dist_weirdos_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top3_sorted_idxweirdo_dist_label_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top3_sorted_idxweirdo_dist_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top3_sorted_idxweirdo_label_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_dist_label_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_dist_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_label_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_coor_el] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top1_sorted_idxweirdo_file_el] = [{} for _ in range(len(dataframe.index))]
    
    coor_li24g_ref      = coor_reference_el_init[0:24]
    if litype == 1:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coordinate_lists    = [coor_li48htype1_ref]
        labels              = ["48htype1"]
    elif litype == 2:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref]
        labels              = ["48htype1", "48htype2"]
    elif litype == 3:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref]
        labels              = ["48htype1", "48htype2", "48htype3"]
    elif litype == 4:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4"]
    elif litype == 5:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5"]
    elif litype == 6:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6"]
    elif litype == 7:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
        coor_li48htype7_ref = coor_reference_el_init[312:360]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref, coor_li48htype7_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7"]
    elif litype == 8:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
        coor_li48htype7_ref = coor_reference_el_init[312:360]
        coor_li48htype8_ref = coor_reference_el_init[360:408]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref, coor_li48htype7_ref, coor_li48htype8_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7", "48htype8"]


    for idx in range(dataframe["geometry"].size):
        dist_weirdos_atomreference_el = []
        dist_weirdos_el = []
        coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el = {}
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

                coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el[idx_weirdo] = []
                
                for idxreference, coorreference in enumerate(coor_reference_el_init):
                    coorweirdo_dist_label_coorreference_val_el = {}
            
                    distance_weirdo = func_distance.mic_eucledian_distance(coorreference, coor_weirdo)

                    coorweirdo_dist_label_coorreference_val_el['dist'] = distance_weirdo

                    for idx_24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
                        if (coorreference == coor_li24g_ref_temp).all():
                            coorweirdo_dist_label_coorreference_val_el["coor"] = tuple(coor_weirdo)
                            coorweirdo_dist_label_coorreference_val_el["label"] = "24g"
                            coorweirdo_dist_label_coorreference_val_el["coorreference"] = tuple(coorreference)
                            coorweirdo_dist_label_coorreference_val_el["file"] = f"{int(dataframe.at[idx, 'geometry'])}_{int(dataframe.at[idx, 'path'])}"
                            if idx_weirdo in coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el:
                                coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el[idx_weirdo].append(coorweirdo_dist_label_coorreference_val_el)
                            else:
                                coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el[idx_weirdo] = coorweirdo_dist_label_coorreference_val_el

                    for i in range(1, litype + 1):
                        for j, coor_ref_temp in enumerate(coordinate_lists[i - 1]):
                            if (coorreference == coor_ref_temp).all():
                                coorweirdo_dist_label_coorreference_val_el["coor"] = tuple(coor_weirdo)
                                coorweirdo_dist_label_coorreference_val_el["label"] = labels[i - 1]
                                coorweirdo_dist_label_coorreference_val_el["coorreference"] = tuple(coorreference)
                                coorweirdo_dist_label_coorreference_val_el["file"] = f"{int(dataframe.at[idx, 'geometry'])}_{int(dataframe.at[idx, 'path'])}"

                                if idx_weirdo in coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el:
                                    coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el[idx_weirdo].append(coorweirdo_dist_label_coorreference_val_el)
                                else:
                                    coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el[idx_weirdo] = [coorweirdo_dist_label_coorreference_val_el]

                    if distance_weirdo < distance_weirdo_prev:
                        distance_weirdo_prev = distance_weirdo
                        closestreference = coorreference

                dist_weirdos_atomreference_el_array = [distance_weirdo, tuple(coor_weirdo), tuple(closestreference)]
                dist_weirdos_el_array = [distance_weirdo]
                dist_weirdos_atomreference_el.append(dist_weirdos_atomreference_el_array)
                dist_weirdos_el.append(dist_weirdos_el_array)
                # float_dist_weirdos_el = np.append(float_dist_weirdos_el, [distance_weirdo_prev])

                # sorted_dist_weirdos_array_el = sorted(set(dist_weirdos_array_el))
                # top3_dist_weirdos_array_el = sorted_dist_weirdos_array_el[0:3]

                # coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el['top3_dist'] = top3_dist_weirdos_array_el

                # if tuple(coor_weirdo) in top3_dist_weirdos_el:
                #     top3_dist_weirdos_el[tuple(coor_weirdo)].append(coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el)
                # else:
                #     top3_dist_weirdos_el[tuple(coor_weirdo)] = coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el

                # for key_temp1, val_temp1 in coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el.items():
                #     sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el = {key_temp1: sorted(val_temp1, key=lambda x: x['dist'])}
                sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el = {
                                                                    k: sorted(v, key=lambda x: x['dist'])
                                                                    for k, v in coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el.items()
                                                                }
                
                top3_sorted_coorweirdo_dist_label_coorreference_el = {k: v[0:3] for k, v in sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el.items()}
                for key, values_list in top3_sorted_coorweirdo_dist_label_coorreference_el.items():
                    selected_values = [{'dist': entry['dist'], "label": entry["label"]} for entry in values_list]
                    top3_sorted_idxweirdo_dist_label_el[key] = selected_values
                    selected_dists = [entry['dist'] for entry in values_list]
                    top3_sorted_idxweirdo_dist_el[key] = selected_dists
                    selected_types = [entry["label"] for entry in values_list]
                    top3_sorted_idxweirdo_label_el[key] = selected_types

                top1_sorted_coorweirdo_dist_label_coorreference_el = {k: v[0:1] for k, v in sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el.items()}
                for key, values_list in top1_sorted_coorweirdo_dist_label_coorreference_el.items():
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

            # # dataframe.at[idx, col_dist_weirdos_atomreference_el] = sorted(dist_weirdos_atomreference_el, coor_weirdo=lambda x: x[0]) 
            # dataframe.at[idx, col_dist_weirdos_el] = np.array([coorreference[0] for index, coorreference in enumerate(dist_weirdos_atomreference_el)])
            # # dataframe.at[idx, col_dist_weirdos_el] = sorted(dist_weirdos_el, coor_weirdo=lambda x: x[0]) 
            # dataframe.at[idx, col_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el] = coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el
            # dataframe.at[idx, col_sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el] = sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el
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

            # dataframe.at[idx, col_top3_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el] = top3_dist_weirdos_el
            # print(coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el)
        # else:
        #     dataframe.at[idx, col_dist_weirdos_atomreference_el] = {}
        #     # dataframe.at[idx, col_dist_weirdos_el] = np.array([coorreference[0] for index, coorreference in enumerate(dist_weirdos_atomreference_el)])
        #     dataframe.at[idx, col_dist_weirdos_el] = []
        #     dataframe.at[idx, col_top3_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_el] = []


def get_label_mapping(dataframe, coor_structure_init_dict, el, activate_radius, litype):
    # TO DO: split into elementwise

    coor_reference_el_init = coor_structure_init_dict[el]

    if activate_radius == 1:
        col_atom_mapping_el_w_dist = f"atom_mapping_{el}_w_dist_closestduplicate"
    elif activate_radius == 2:
        col_atom_mapping_el_w_dist = f"atom_mapping_48htypesmerged_{el}_w_dist"
    
    col_atom_mapping_el_w_dist_label = f"atom_mapping_{el}_w_dist_label"

    dataframe[col_atom_mapping_el_w_dist_label] = [{} for _ in range(len(dataframe.index))]

    coor_li24g_ref      = coor_reference_el_init[0:24]
    if litype == 1:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coordinate_lists    = [coor_li48htype1_ref]
        labels              = ["48htype1"]
    elif litype == 2:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref]
        labels              = ["48htype1", "48htype2"]
    elif litype == 3:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref]
        labels              = ["48htype1", "48htype2", "48htype3"]
    elif litype == 4:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4"]
    elif litype == 5:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5"]
    elif litype == 6:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6"]
    elif litype == 7:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
        coor_li48htype7_ref = coor_reference_el_init[312:360]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref, coor_li48htype7_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7"]
    elif litype == 8:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
        coor_li48htype7_ref = coor_reference_el_init[312:360]
        coor_li48htype8_ref = coor_reference_el_init[360:408]
        coordinate_lists    = [coor_li48htype1_ref, coor_li48htype2_ref, coor_li48htype3_ref, coor_li48htype4_ref, coor_li48htype5_ref, coor_li48htype6_ref, coor_li48htype7_ref, coor_li48htype8_ref]
        labels              = ["48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7", "48htype8"]
    
    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist = dataframe.at[idx, col_atom_mapping_el_w_dist]

        atom_mapping_el_w_dist_label = {}

        for coorreference in atom_mapping_el_w_dist.keys():
            value = atom_mapping_el_w_dist[tuple(coorreference)]

            if isinstance(value, list):
                # Handle the case where the value is a list
                atom_mapping_el_w_dist_label_val = {'closest24': value[0]['closest24'], 'dist': value[0]['dist']}
            else:
                # Handle the case where the value is a dictionary
                atom_mapping_el_w_dist_label_val = {'closest24': value['closest24'], 'dist': value['dist']}

            # # atom_mapping_el_w_dist_label_val = {}
            # # atom_mapping_el_w_dist_label[tuple(coorreference)] = []
            # atom_mapping_el_w_dist_label_val = {'closest24': atom_mapping_el_w_dist[tuple(coorreference)][0]['closest24'], 'dist': atom_mapping_el_w_dist[tuple(coorreference)][0]['dist']}

            for idx_li24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
                if (coorreference == coor_li24g_ref_temp).all():
                    atom_mapping_el_w_dist_label_val["label"] = "24g"

                    atom_mapping_el_w_dist_label[tuple(coorreference)] = atom_mapping_el_w_dist_label_val
                    # atom_mapping_el_w_dist_label[tuple(coorreference)].append(atom_mapping_el_w_dist_label_val)

            for i in range(1, litype+1):
                coor_li48htype_ref = locals()[f"coor_li48htype{i}_ref"]
                label = f"48htype{i}"

                for idx_temp, coor_ref_temp in enumerate(coor_li48htype_ref):
                    if (coorreference == coor_ref_temp).all():
                        atom_mapping_el_w_dist_label_val["label"] = label
                        atom_mapping_el_w_dist_label[tuple(coorreference)] = atom_mapping_el_w_dist_label_val
                        # atom_mapping_el_w_dist_label[tuple(coorreference)].append(atom_mapping_el_w_dist_label_val)


            # # for idx_li48htype1_temp, coor_li48htype1_ref_temp in enumerate(coor_li48htype1_ref):
            # #     if (coorreference == coor_li48htype1_ref_temp).all():
            # #         atom_mapping_el_w_dist_label_val["label"] = "48htype1"

            # #         atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_el_w_dist_label_val)

            # # for idx_li48htype2_temp, coor_li48htype2_ref_temp in enumerate(coor_li48htype2_ref):
            # #     if (coorreference == coor_li48htype2_ref_temp).all():
            # #         atom_mapping_el_w_dist_label_val["label"] = "48htype2"

            # #         atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_el_w_dist_label_val)

            # # for idx_li24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
            # #     if (coorreference == coor_li24g_ref_temp).all():
            # #         atom_mapping_el_w_dist_label_val["label"] = "24g"

            # #         atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_el_w_dist_label_val)

            # # if litype == 3:
            # #     for idx_li48htype3_temp, coor_li48htype3_ref_temp in enumerate(coor_li48htype3_ref):
            # #         if (coorreference == coor_li48htype3_ref_temp).all():
            # #             atom_mapping_el_w_dist_label_val["label"] = "48htype3"

            # #             atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_el_w_dist_label_val)

            # # elif litype == 4:
            # #     for idx_li48htype3_temp, coor_li48htype3_ref_temp in enumerate(coor_li48htype3_ref):
            # #         if (coorreference == coor_li48htype3_ref_temp).all():
            # #             atom_mapping_el_w_dist_label_val["label"] = "48htype3"

            # #             atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_el_w_dist_label_val)

            # #     for idx_li48htype4_temp, coor_li48htype4_ref_temp in enumerate(coor_li48htype4_ref):
            # #         if (coorreference == coor_li48htype4_ref_temp).all():
            # #             atom_mapping_el_w_dist_label_val["label"] = "48htype4"

            # #             atom_mapping_el_w_dist[tuple(coorreference)].append(atom_mapping_el_w_dist_label_val)
    
        # # dataframe.at[idx, col_atom_mapping_el_w_dist_label] = atom_mapping_el_w_dist
        dataframe.at[idx, col_atom_mapping_el_w_dist_label] = atom_mapping_el_w_dist_label


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
    

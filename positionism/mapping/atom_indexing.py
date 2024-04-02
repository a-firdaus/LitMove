import numpy as np

from positionism.functional import func_distance


def weirdos_el(dataframe, el, activate_radius):
    # rename from: get_idx_weirdos_el
    """
    Identifies indexes of weirdo for a specific element within structures.

    Parameters
    ==========
    dataframe: pandas.DataFrame
        DataFrame containing necessary data columns.
    el: str
        Element symbol for which anomalies to be identified.
    activate_radius: int
        Number indicating how many level of mapping radius to be used.

    Returns
    =======
    idx0_weirdos_el: list
        List of indexes of weirdos. Numeration starts from 0.
    #weirdos_el: int
        Amount of weirdos.
    idx_coor_weirdos_el: dict
        Whose key: index of weirdo, value: coordinate of weirdo.
    """
    col_coor_structure_init_dict = "coor_structure_init_dict"

    if activate_radius == 2 or activate_radius == 3:
        col_coor_weirdos_el = f"coor_weirdos_48htype2_{el}"
    elif activate_radius == 1:
        col_coor_weirdos_el = f"coor_weirdos_{el}"

    col_idx0_weirdos_el = f"idx0_weirdos_{el}"
    col_nr_of_weirdos_el = f"#weirdos_{el}"
    col_idx_coor_weirdos_el = f"idx_coor_weirdos_{el}"

    dataframe[col_idx0_weirdos_el] = [np.array([]) for _ in range(len(dataframe.index))]
    dataframe[col_nr_of_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor_weirdos_el] = [{} for _ in range(len(dataframe.index))]

    for idx in range(dataframe["geometry"].size):
        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_init_dict][el]
        coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]
        
        idx_weirdos_el = []
        idx_coor_weirdos_el = {}

        for i, given_arr in enumerate(coor_origin24_el_init):
            
            # coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]
            if len(coor_weirdos_el) > 0:
                for arr in coor_weirdos_el:
                    if (arr == given_arr).all():
                        idx_weirdos_el.append(i)

                        idx_coor_weirdos_el[i] = arr

        dataframe.at[idx, col_idx0_weirdos_el] = idx_weirdos_el
        dataframe.at[idx, col_nr_of_weirdos_el] = len(idx_weirdos_el)
        dataframe.at[idx, col_idx_coor_weirdos_el] = idx_coor_weirdos_el


def correct_idx_and_order_mapped_el(dataframe, el, activate_radius):
    # rename from: idx_correcting_mapped_el
    """
    Given list of reference structure's coordinate (that's considered already for mapping), 
    then they're re-ordered again aka being corrected for its indexing.
    Correcting done by calculating the distance of given reference to each atom of the structures,
    for which the closest one is considered.

    Parameters
    ==========
    dataframe: pandas.DataFrame
        DataFrame containing necessary data columns.
    el: str
        Element symbol for which anomalies to be identified.
    activate_radius: int
        Number indicating how many level of mapping radius to be used.

    Returns
    =======
    idx_correcting_el: list
        Corrected index
    coor_reducedreference_sorted_el: list
        Reference coordinate with corrected ordering
    """
    col_coor_structure_init_dict = "coor_structure_init_dict"

    if activate_radius == 2 or activate_radius == 3:
        col_coor_reducedreference_el = f"coor_reducedreference_48htypesmerged_{el}"
    elif activate_radius == 1:
        col_coor_reducedreference_el = f"coor_reducedreference_{el}_closestduplicate"

    col_idx_correcting_el = f"idx_correcting_{el}"
    # col_atom_mapping_el_w_dist_idx24 = f"atom_mapping_{el}_w_dist_idx24"
    # col_coor_reducedreference_closestduplicate_el = f"coor_reducedreference_closestduplicate_{el}"
    col_coor_reducedreference_sorted_el = f"coor_reducedreference_sorted_{el}"

    dataframe[col_idx_correcting_el] = [np.array([]) for _ in range(len(dataframe.index))]
    # dataframe[col_atom_mapping_el_w_dist_idx24] = [{} for _ in range(len(dataframe.index))]
    # dataframe[col_coor_reducedreference_closestduplicate_el] = None
    dataframe[col_coor_reducedreference_sorted_el] = [np.array([]) for _ in range(len(dataframe.index))]

    for idx in range(dataframe["geometry"].size):
        coor_origin24_el_init = dataframe.at[idx, col_coor_structure_init_dict][el] 
        coor_reducedreference_el = dataframe.at[idx, col_coor_reducedreference_el]

        idx_correcting_el = []
        # atom_mapping_el_w_dist_idx24 = {} 
        idx_coor_el = {}

        for idxreference, coorreference in enumerate(coor_reducedreference_el):
            distance_prev = float("inf")
            closest24 = None
            idx_closest24 = None
            # atom_mapping_w_dist_idx24_dict = {}

            # atom_mapping_el_w_dist_idx24[tuple(coorreference)] = []

            for idx24, coor24 in enumerate(coor_origin24_el_init):
                distance = func_distance.mic_eucledian_distance(coorreference, coor24)

                # if distance != 0:
                if distance < distance_prev:
                    distance_prev = distance
                    closest24 = coor24
                    idx_closest24 = idx24
            
            idx_correcting_el.append(idx_closest24)

            # if closest24 is not None:
            #     atom_mapping_w_dist_idx24_dict['closest24'] = tuple(closest24)
            #     atom_mapping_w_dist_idx24_dict['dist'] = distance_prev
            #     atom_mapping_w_dist_idx24_dict['idx_closest24'] = idx_closest24

            #     # if tuple(coorreference) in atom_mapping_el_w_dist_idx24:
            #     #     atom_mapping_el_w_dist_idx24[tuple(coorreference)].append(atom_mapping_w_dist_idx24_dict)
            #     # else:
            #     #     atom_mapping_el_w_dist_idx24[tuple(coorreference)] = atom_mapping_w_dist_idx24_dict
    
            #     if tuple(coorreference) in atom_mapping_el_w_dist_idx24:
            #         new_entry = atom_mapping_el_w_dist_idx24[tuple(coorreference)].copy()
            #         new_entry.append(atom_mapping_w_dist_idx24_dict)
            #         atom_mapping_el_w_dist_idx24[tuple(coorreference)] = new_entry
            #     else:
            #         atom_mapping_el_w_dist_idx24[tuple(coorreference)] = [atom_mapping_w_dist_idx24_dict.copy()]

        # coor_reducedreference_closestduplicate_el = [coor_reducedreference_el[i] for i in idx_correcting_el]
        # # coor_reducedreference_closestduplicate_el_closestduplicate = [coor_reducedreference_el_closestduplicate[i] for i in idx_correcting_el]

        # create a dictionary, for which key: corrected index, value: corresponding reference coordinate
        for i in range(len(idx_correcting_el)):
            idx_coor_el[idx_correcting_el[i]] = coor_reducedreference_el[i]

        sorted_idx_coor_el = {key: val for key, val in sorted(idx_coor_el.items())}
        sorted_coor = list(sorted_idx_coor_el.values())

        dataframe.at[idx, col_idx_correcting_el] = idx_correcting_el
        # dataframe.at[idx, col_atom_mapping_el_w_dist_idx24] = atom_mapping_el_w_dist_idx24
        # dataframe.at[idx, col_coor_reducedreference_closestduplicate_el] = coor_reducedreference_closestduplicate_el
        # # dataframe.at[idx, col_coor_reducedreference_closestduplicate_el_closestduplicate] = coor_reducedreference_closestduplicate_el_closestduplicate
        dataframe.at[idx, col_coor_reducedreference_sorted_el] = sorted_coor


def get_idx_coor_limapped_weirdos_dict(dataframe, coor_structure_init_dict, activate_radius, litype, el):
    # rename from: get_idx_coor_limapped_weirdos_dict_litype
    """
    Generates a dictionary mapping each index to its corresponding coordinates and label, 
    considering different Li types and weirdos, and computes various statistics.

    Args
    ====
    dataframe: pandas.DataFrame 
        DataFrame containing the data
    coor_structure_init_dict: dict
        Dictionary containing initial coordinate structures
    activate_radius: int
        Radius activation flag (1, 2, or 3)
    litype: int
        Lithium type (int from 0 to 8)
    el: str
        Element symbol

    Returns
    =======
    None
    """
    coor_reference_el_init = coor_structure_init_dict[el]

    col_idx_without_weirdos = "idx_without_weirdos"
    col_idx_coor_weirdos_el = f"idx_coor_weirdos_{el}"
    col_idx0_weirdos_Li = "idx0_weirdos_Li"
    col_sum_of_weirdos_Li = f"#weirdos_Li"
    if activate_radius == 2 or activate_radius == 3:
        #col_coor_reducedreference_Li = "coor_reducedreference_Li"
        # col_coor_reducedreference_Li = f"coor_reducedreference_48htypesmerged_{el}" # CHANGED
        col_coor_reducedreference_Li = f"coor_reducedreference_sorted_{el}"
        # col_coor_weirdos_48htypesmerged_Li = "coor_weirdos_48htypesmerged_Li"
        # col_coor_weirdos_el = f"coor_weirdos_48htype2_{el}"
        col_sum_sanitycheck_Li = "sum_sanitycheck_48htypesmerged_Li"
    elif activate_radius == 1:
        # col_coor_reducedreference_Li = f"coor_reducedreference_{el}_closestduplicate" # CHANGED
        col_coor_reducedreference_Li = f"coor_reducedreference_sorted_{el}"
        col_sum_sanitycheck_Li = f"sum_sanitycheck_{el}_closestduplicate"


    col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict" #
    col_sum_label_and_weirdo_flag = "#label_and_#weirdo_flag"
    col_amount_types_and_weirdo = "amount_types_and_weirdo" #
    col_ratio_48htype1_Li = "ratio_48htype1_Li"
    col_ratio_48htype2_Li = "ratio_48htype2_Li"
    col_ratio_24g_Li = "ratio_24g_Li"
    col_ratio_weirdo_Li = "ratio_weirdo_Li"
    col_sum_amount = "sum_amount" #
    col_idx_coor_limapped_weirdos_dict_init = "idx_coor_limapped_weirdos_dict_init" #
    col_ndim_coor_reducedreference_Li = "ndim_coor_reducedreference_Li" #
    col_ndim_coor_weirdos_el = "ndim_coor_weirdos_el" #
    col_len_coor_weirdos_el = "len_coor_weirdos_el" #
    col_len_coor_reducedreference_Li = "len_coor_reducedreference_Li" 
    col_len_idx0_weirdos_Li = "len_idx0_weirdos_Li"
    col_len_idx_without_weirdos = "len_idx_without_weirdos"
    col_ndim_flag_coor = "ndim_flag_coor"

    dataframe[col_idx_coor_limapped_weirdos_dict] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_sum_label_and_weirdo_flag] = "False"
    dataframe[col_amount_types_and_weirdo] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_sum_amount] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor_limapped_weirdos_dict_init] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_reducedreference_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_coor_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_coor_weirdos_el] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_coor_reducedreference_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_idx0_weirdos_Li] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_len_idx_without_weirdos] = [0 for _ in range(len(dataframe.index))]
    dataframe[col_ndim_flag_coor] = "False"

    coor_li24g_ref      = coor_reference_el_init[0:24]
    if litype == 1:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
    elif litype == 2:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
    elif litype == 3:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
    elif litype == 4:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
    elif litype == 5:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
    elif litype == 6:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
    elif litype == 7:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
        coor_li48htype7_ref = coor_reference_el_init[312:360]
    elif litype == 8:
        coor_li48htype1_ref = coor_reference_el_init[24:72]
        coor_li48htype2_ref = coor_reference_el_init[72:120]
        coor_li48htype3_ref = coor_reference_el_init[120:168]
        coor_li48htype4_ref = coor_reference_el_init[168:216]
        coor_li48htype5_ref = coor_reference_el_init[216:264]
        coor_li48htype6_ref = coor_reference_el_init[264:312]
        coor_li48htype7_ref = coor_reference_el_init[312:360]
        coor_li48htype8_ref = coor_reference_el_init[360:408]

    for idx in range(dataframe["geometry"].size):
        coor_limapped_weirdos = []
        idx0_limapped_weirdos = []
        idx_coor_limapped_weirdos_dict_init = {}
        idx_coor_limapped_weirdos_dict = {}

        idx_without_weirdos = dataframe.at[idx, col_idx_without_weirdos]
        coor_reducedreference_Li = np.array(dataframe.at[idx, col_coor_reducedreference_Li])
        idx_coor_weirdos_el = dataframe.at[idx, col_idx_coor_weirdos_el]
        # coor_weirdos_48htype2_el = np.array(dataframe.at[idx, col_coor_weirdos_48htype2_el])
        coor_weirdos_el = np.array(list(idx_coor_weirdos_el.values()))
        idx0_weirdos_Li = dataframe.at[idx, col_idx0_weirdos_Li]
        nr_of_weirdos_Li = dataframe.at[idx, col_sum_of_weirdos_Li]
        sum_sanitycheck_48htypesmerged_Li = dataframe.at[idx, col_sum_sanitycheck_Li]

        ndim_coor_reducedreference_Li = coor_reducedreference_Li.ndim
        ndim_coor_weirdos_el = coor_weirdos_el.ndim
        len_coor_weirdos_el = len(coor_weirdos_el)
        len_coor_reducedreference_Li = len(coor_reducedreference_Li)
        len_idx0_weirdos_Li = len(idx0_weirdos_Li)
        len_idx_without_weirdos = len(idx_without_weirdos)

        # if coor_weirdos_el.ndim == 2:
        if ndim_coor_reducedreference_Li == ndim_coor_weirdos_el & ndim_coor_weirdos_el == 2:
            coor_limapped_weirdos = np.concatenate((coor_reducedreference_Li, coor_weirdos_el), axis=0)
            dataframe.at[idx, col_ndim_flag_coor] = "True"
        # elif coor_weirdos_el.ndim == 1:
        elif ndim_coor_weirdos_el == 1:
            coor_limapped_weirdos = np.array(coor_reducedreference_Li.copy())
        elif ndim_coor_reducedreference_Li == 1:
            coor_limapped_weirdos = np.array(coor_weirdos_el.copy())
        else:
            print(f"coor_weirdos_el or coor_reducedreference_Li has no correct dimension at idx: {idx}")
            # break

        if len(idx0_weirdos_Li) > 0:
            idx0_limapped_weirdos = np.concatenate((idx_without_weirdos, idx0_weirdos_Li), axis=0)
        elif len(idx0_weirdos_Li) == 0:
            idx0_limapped_weirdos = np.array(idx_without_weirdos.copy())
        elif len(idx_without_weirdos) == 0:
            idx0_limapped_weirdos = np.array(idx0_weirdos_Li.copy())
        else:
            print(f"idx0_weirdos_Li or idx_without_weirdos has no correct len at idx: {idx}")
            # break

        idx_coor_limapped_weirdos_dict_init = dict(zip(idx0_limapped_weirdos, coor_limapped_weirdos))

        coor_24g_Li = []; coor_weirdo_Li = []
        coor_48htype1_Li = []; coor_48htype2_Li = []; coor_48htype3_Li = []; coor_48htype4_Li = []; coor_48htype5_Li = []; coor_48htype6_Li = []; coor_48htype7_Li = []; coor_48htype8_Li = []   
        
        for key, value in idx_coor_limapped_weirdos_dict_init.items():
            idx_coor_limapped_weirdos_dict_val = {}

            idx_coor_limapped_weirdos_dict_val['coor'] = tuple(value)

            for idx_24g_temp, coor_li24g_ref_temp in enumerate(coor_li24g_ref):
                if (value == coor_li24g_ref_temp).all():
                    idx_coor_limapped_weirdos_dict_val["label"] = "24g"
                    coor_24g_Li.append(np.array(list(value)))
                    if int(key) in idx_coor_limapped_weirdos_dict:
                        idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                    else:
                        idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val
            for idx_weirdos_temp, coor_weirdos_ref_temp in enumerate(coor_weirdos_el):
                if (value == coor_weirdos_ref_temp).all():
                    idx_coor_limapped_weirdos_dict_val["label"] = "weirdos"
                    coor_weirdo_Li.append(np.array(list(value)))
                    if int(key) in idx_coor_limapped_weirdos_dict:
                        idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                    else:
                        idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val

            for i in range(1, litype+1):
                coor_li48htype_ref = locals()[f"coor_li48htype{i}_ref"]
                label = f"48htype{i}"

                for idx_temp, coor_ref_temp in enumerate(coor_li48htype_ref):
                    if (value == coor_ref_temp).all():
                        idx_coor_limapped_weirdos_dict_val["label"] = label
                        locals()[f"coor_48htype{i}_Li"].append(np.array(list(value)))
                        if int(key) in idx_coor_limapped_weirdos_dict:
                            idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
                        else:
                            idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val
                    

            # if int(key) in idx_coor_limapped_weirdos_dict:
            #     idx_coor_limapped_weirdos_dict[int(key)].append(idx_coor_limapped_weirdos_dict_val)
            # else:
            #     idx_coor_limapped_weirdos_dict[int(key)] = idx_coor_limapped_weirdos_dict_val

        # amount of each type
        amount_24g_Li = len(coor_24g_Li)
        amount_weirdo = len(coor_weirdo_Li)
        if litype == 0:
            sum_amount = amount_24g_Li + amount_weirdo
        elif litype == 1:
            amount_48htype1_Li = len(coor_48htype1_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li
        elif litype == 2:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li
        elif litype == 3:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li
        elif litype == 4:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li
        elif litype == 5:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            amount_48htype5_Li = len(coor_48htype5_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li + amount_48htype5_Li
        elif litype == 6:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            amount_48htype5_Li = len(coor_48htype5_Li)
            amount_48htype6_Li = len(coor_48htype6_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li + amount_48htype5_Li + amount_48htype6_Li
        elif litype == 7:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            amount_48htype5_Li = len(coor_48htype5_Li)
            amount_48htype6_Li = len(coor_48htype6_Li)
            amount_48htype7_Li = len(coor_48htype7_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li + amount_48htype5_Li + amount_48htype6_Li + amount_48htype7_Li
        elif litype == 8:
            amount_48htype1_Li = len(coor_48htype1_Li)
            amount_48htype2_Li = len(coor_48htype2_Li)
            amount_48htype3_Li = len(coor_48htype3_Li)
            amount_48htype4_Li = len(coor_48htype4_Li)
            amount_48htype5_Li = len(coor_48htype5_Li)
            amount_48htype6_Li = len(coor_48htype6_Li)
            amount_48htype7_Li = len(coor_48htype7_Li)
            amount_48htype8_Li = len(coor_48htype8_Li)
            sum_amount = amount_24g_Li + amount_weirdo + amount_48htype1_Li + amount_48htype2_Li + amount_48htype3_Li + amount_48htype4_Li + amount_48htype5_Li + amount_48htype6_Li + amount_48htype7_Li + amount_48htype8_Li


        # sanity check for the amount
        # if amount_weirdo == nr_of_weirdos_Li & sum_amount == sum_sanitycheck_48htypesmerged_Li:
        # if int(amount_weirdo) == int(nr_of_weirdos_Li) & int(sum_amount) == int(sum_sanitycheck_48htypesmerged_Li):
        if int(amount_weirdo) == int(nr_of_weirdos_Li):
            if int(sum_amount) == int(sum_sanitycheck_48htypesmerged_Li):
                dataframe.at[idx, col_sum_label_and_weirdo_flag] = "True"

        if litype == 0:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}"
        elif litype == 1:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}"
        elif litype == 2:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}"
        elif litype == 3:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}"
        elif litype == 4:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}"
        elif litype == 5:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}, 48htype5: {amount_48htype5_Li}"
        elif litype == 6:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}, 48htype5: {amount_48htype5_Li}, 48htype6: {amount_48htype6_Li}"
        elif litype == 7:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}, 48htype5: {amount_48htype5_Li}, 48htype6: {amount_48htype6_Li}, 48htype7: {amount_48htype7_Li}"
        elif litype == 8:
            amount_types_and_weirdo = f"24g: {amount_24g_Li}, weirdo: {amount_weirdo}, 48htype1: {amount_48htype1_Li}, 48htype2: {amount_48htype2_Li}, 48htype3: {amount_48htype3_Li}, 48htype4: {amount_48htype4_Li}, 48htype5: {amount_48htype5_Li}, 48htype6: {amount_48htype6_Li}, 48htype7: {amount_48htype7_Li}, 48htype8: {amount_48htype8_Li}"

        # ratio_48htype1_Li = amount_48htype1_Li / sum_amount
        # ratio_48htype2_Li = amount_48htype2_Li / sum_amount
        # ratio_24g_Li = amount_24g_Li / sum_amount
        # ratio_weirdo_Li = amount_weirdo / sum_amount

        dataframe.at[idx, col_idx_coor_limapped_weirdos_dict] = idx_coor_limapped_weirdos_dict
        dataframe.at[idx, col_amount_types_and_weirdo] = amount_types_and_weirdo
        # dataframe.at[idx, col_ratio_48htype1_Li] = ratio_48htype1_Li
        # dataframe.at[idx, col_ratio_48htype2_Li] = ratio_48htype2_Li
        # dataframe.at[idx, col_ratio_24g_Li] = ratio_24g_Li
        # dataframe.at[idx, col_ratio_weirdo_Li] = ratio_weirdo_Li
        dataframe.at[idx, col_sum_amount] = sum_amount
        # # dataframe.at[idx, col_idx_coor_limapped_weirdos_dict_init] = idx_coor_limapped_weirdos_dict_init
        dataframe.at[idx, col_ndim_coor_reducedreference_Li] = ndim_coor_reducedreference_Li
        dataframe.at[idx, col_ndim_coor_weirdos_el] = ndim_coor_weirdos_el
        dataframe.at[idx, col_len_coor_weirdos_el] = len_coor_weirdos_el
        dataframe.at[idx, col_len_coor_reducedreference_Li] = len_coor_reducedreference_Li
        dataframe.at[idx, col_len_idx0_weirdos_Li] = len_idx0_weirdos_Li
        dataframe.at[idx, col_len_idx_without_weirdos] = len_idx_without_weirdos



from collections import defaultdict


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


def get_occupancy(dataframe, coor_structure_init_dict_expanded, tuple_metainfo, litype, el):
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

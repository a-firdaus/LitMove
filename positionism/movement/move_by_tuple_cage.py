import pandas as pd
import plotly.express as px
from collections import defaultdict

from positionism.functional import func_distance


    # class TupleCage:

# # def get_complete_closest_tuple_cage(dataframe, tuple_metainfo):
# #     col_coor24li_tuple_cage_belongin = "coor24li_tuple_cage_belongin"
# #     col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"

# #     col_idx_coor24li_tuple_cage_belongin = "idx_coor24li_tuple_cage_belongin"
# #     col_idx_coor24li_tuple_cage_belongin_complete_closest = "idx_coor24li_tuple_cage_belongin_complete_closest"
# #     col_top_n_distance_coors = "top_n_distance_coors"

# #     dataframe[col_idx_coor24li_tuple_cage_belongin] = [{} for _ in range(len(dataframe.index))]
# #     dataframe[col_idx_coor24li_tuple_cage_belongin_complete_closest] = [{} for _ in range(len(dataframe.index))]
# #     dataframe[col_top_n_distance_coors] = [{} for _ in range(len(dataframe.index))]
    
# #     for idx in range(dataframe["geometry"].size):
# #         idx_coor24li_tuple_cage_belongin = defaultdict(list)

# #         coor24li_tuple_cage_belongin = dataframe[col_coor24li_tuple_cage_belongin][idx]
# #         idx_coor_limapped_weirdos_dict = dataframe[col_idx_coor_limapped_weirdos_dict][idx]

# #         for key_a, val_a in idx_coor_limapped_weirdos_dict.items():
# #             idx_li = key_a
# #             coor_li_mapped_a = val_a['coor']
# #             coor_li_mapped_a_rounded = tuple(round(coordinate, 5) for coordinate in coor_li_mapped_a)
# #             label_li_a = val_a['label']

# #             idx_coor24li_tuple_cage_belongin[idx_li] = []
# #             for key_b, val_b in coor24li_tuple_cage_belongin.items():
# #                 idx_tuple = key_b
# #                 for entry_b in val_b:
# #                     coor_li_mapped_b = entry_b['coor']
# #                     coor_li_mapped_b_rounded = tuple(round(coordinate, 5) for coordinate in coor_li_mapped_b)
# #                     label_li_b = entry_b['type']
# #                     idx_cage_b = entry_b['idx_cage']

# #                     if (coor_li_mapped_a_rounded == coor_li_mapped_b_rounded) and (label_li_a == label_li_b):
# #                         # idx_coor24li_tuple_belongin_val = {'coor': coor_li_mapped_a, 'type':label_li_a, 'idx_tuple':idx_tuple}
# #                         idx_coor24li_tuple_cage_belongin_val = {'coor': coor_li_mapped_a, 'type':label_li_a, 'idx_tuple':idx_tuple, 'idx_cage':idx_cage_b}
# #                         idx_coor24li_tuple_cage_belongin[idx_li].append(idx_coor24li_tuple_cage_belongin_val)
        
# #         dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin] = idx_coor24li_tuple_cage_belongin
                        
# #         distance_coors_all = defaultdict(list)
# #         n = 3
# #         idx_coor_limapped_weirdos_dict = dataframe[col_idx_coor_limapped_weirdos_dict][idx]
# #         idx_coor24li_tuple_cage_belongin = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin]
# #         # idx_coor24li_tuple_cage_belongin_complete_closest = idx_coor24li_tuple_cage_belongin.copy()
# #         idx_coor24li_tuple_cage_belongin_complete_closest = defaultdict(list)

# #         for key_c, val_c in idx_coor24li_tuple_cage_belongin.items():
# #             idx_li = key_c
# #             idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = []

# #             if val_c == []:
# #                 coor_li_mapped_c = idx_coor_limapped_weirdos_dict[idx_li]['coor']
# #                 label_li_c = idx_coor_limapped_weirdos_dict[idx_li]['label']

# #                 distance_prev = float("inf")
# #                 closest_idx_tuple = None
# #                 closest_idx_cage = None
                
# #                 for key_d, val_d in tuple_metainfo.items():
# #                     for entry_d in val_d: 
# #                         idx_tuple = key_d
# #                         coor_tuple_d = entry_d['coor']
# #                         label_li_d = entry_d['type']
# #                         idx_cage_d = entry_d['idx_cage']

# #                         distance = calc_distance.mic_eucledian_distance(coor_li_mapped_c, coor_tuple_d)

# #                         # distance_coors_all_val = {'coor_li_mapped': coor_li_mapped_c, 'coor_tuple': coor_tuple_d, 'dist': distance, 'label':label_li_d}

# #                         distance_coors_all_val = {'coor_tuple': coor_tuple_d, 'dist': distance, 'label':label_li_d, 'idx_tuple':idx_tuple, 'idx_cage':idx_cage_d}

# #                         distance_coors_all[idx_li].append(distance_coors_all_val)

# #                         if distance < distance_prev:
# #                             distance_prev = distance
# #                             closest_idx_tuple = idx_tuple
# #                             closest_idx_cage = idx_cage_d

# #                 idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = {'coor': coor_li_mapped_c, 'type': label_li_c, 'idx_tuple': closest_idx_tuple, 'idx_cage': closest_idx_cage}

# #             elif val_c != []:
# #                 for entry_c in val_c: 
# #                     coor_li_mapped_c = entry_c['coor']
# #                     label_li_c = entry_c['type']
# #                     idx_tuple_c = entry_c['idx_tuple']
# #                     idx_cage_c = entry_c['idx_cage']

# #                     idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = {'coor': coor_li_mapped_c, 'type': label_li_c, 'idx_tuple': idx_tuple_c, 'idx_cage': idx_cage_c}

# #         sorted_distance_coors_all = {key: sorted(value, key=lambda x: x['dist']) for key, value in distance_coors_all.items()}
# #         top_n_distance_coors = {k: v[0:n] for k, v in sorted_distance_coors_all.items()}
# #         # !!! assumed there's NO DUPLICATE with the SECOND distance

# #         dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest] = idx_coor24li_tuple_cage_belongin_complete_closest
# #         dataframe.at[idx, col_top_n_distance_coors] = top_n_distance_coors


def get_complete_closest_tuple_cage(dataframe, tuple_metainfo, coor_48htype2_metainfo):
    col_idx_coor24li_tuple_cage_belongin_empty = "idx_coor24li_tuple_cage_belongin_empty"
    col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"

    # col_idx_coor24li_tuple_cage_belongin_empty_xxx = "idx_coor24li_tuple_cage_belongin_empty_xxx"
    col_idx_coor24li_tuple_cage_belongin_complete_closest = "idx_coor24li_tuple_cage_belongin_complete_closest"
    col_top_n_distance_coors = "top_n_distance_coors"
    col_idx_coor24li_tuple_cage_belongin_complete_id48htype2 = "idx_coor24li_tuple_cage_belongin_complete_id48htype2"

    # dataframe[col_idx_coor24li_tuple_cage_belongin_empty_xxx] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor24li_tuple_cage_belongin_complete_closest] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_top_n_distance_coors] = [{} for _ in range(len(dataframe.index))]
    dataframe[col_idx_coor24li_tuple_cage_belongin_complete_id48htype2] = [{} for _ in range(len(dataframe.index))]
    
    for idx in range(dataframe["geometry"].size):
        # idx_coor24li_tuple_cage_belongin_empty_xxx = defaultdict(list)

        idx_coor24li_tuple_cage_belongin_empty = dataframe[col_idx_coor24li_tuple_cage_belongin_empty][idx]
        idx_coor_limapped_weirdos_dict = dataframe[col_idx_coor_limapped_weirdos_dict][idx]

        # for key_a, val_a in idx_coor_limapped_weirdos_dict.items():
        #     idx_li = key_a
        #     coor_li_mapped_a = val_a['coor']
        #     coor_li_mapped_a_rounded = tuple(round(coordinate, 5) for coordinate in coor_li_mapped_a)
        #     label_li_a = val_a['label']

        #     idx_coor24li_tuple_cage_belongin_empty_xxx[idx_li] = []
        #     for key_b, val_b in idx_coor24li_tuple_cage_belongin_empty.items():
        #         idx_b_li = key_b
        #         for entry_b in val_b:
        #             coor_li_mapped_b = entry_b['coor']
        #             coor_li_mapped_b_rounded = tuple(round(coordinate, 5) for coordinate in coor_li_mapped_b)
        #             label_li_b = entry_b['type']
        #             idx_tuple = entry_b['idx_tuple']
        #             idx_cage_b = entry_b['idx_cage']

        #             if (coor_li_mapped_a_rounded == coor_li_mapped_b_rounded) and (label_li_a == label_li_b):
        #                 # idx_coor24li_tuple_belongin_val = {'coor': coor_li_mapped_a, 'type':label_li_a, 'idx_tuple':idx_tuple}
        #                 idx_coor24li_tuple_cage_belongin_empty_xxx_val = {'coor': coor_li_mapped_a, 'type':label_li_a, 'idx_tuple':idx_tuple, 'idx_cage':idx_cage_b}
        #                 idx_coor24li_tuple_cage_belongin_empty_xxx[idx_li].append(idx_coor24li_tuple_cage_belongin_empty_xxx_val)
        
        # dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_empty_xxx] = idx_coor24li_tuple_cage_belongin_empty_xxx
                        
        distance_coors_all_closest = defaultdict(list)
        n = 3
        # idx_coor_limapped_weirdos_dict = dataframe[col_idx_coor_limapped_weirdos_dict][idx]
        # idx_coor24li_tuple_cage_belongin_empty_xxx = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_empty_xxx]
        # idx_coor24li_tuple_cage_belongin_complete_closest = idx_coor24li_tuple_cage_belongin_empty_xxx.copy()
        idx_coor24li_tuple_cage_belongin_complete_closest = defaultdict(list)
        idx_coor24li_tuple_cage_belongin_complete_id48htype2 = defaultdict(list)

        for key_c, val_c in idx_coor24li_tuple_cage_belongin_empty.items():
            idx_li = key_c
            idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = []

            if val_c == []:
                coor_li_mapped_c = idx_coor_limapped_weirdos_dict[idx_li]['coor']
                label_li_c = idx_coor_limapped_weirdos_dict[idx_li]['label']

                distance_prev = float("inf")
                closest_idx_tuple = None
                closest_idx_cage = None
                
                for key_d, val_d in tuple_metainfo.items():
                    for entry_d in val_d: 
                        idx_tuple = key_d
                        coor_tuple_d = entry_d['coor']
                        label_li_d = entry_d['type']
                        idx_cage_d = entry_d['idx_cage']

                        distance = func_distance.mic_eucledian_distance(coor_li_mapped_c, coor_tuple_d)

                        # distance_coors_all_closest_val = {'coor_li_mapped': coor_li_mapped_c, 'coor_tuple': coor_tuple_d, 'dist': distance, 'label':label_li_d}

                        distance_coors_all_closest_val = {'coor_tuple': coor_tuple_d, 'dist': distance, 'label':label_li_d, 'idx_tuple':idx_tuple, 'idx_cage':idx_cage_d}

                        distance_coors_all_closest[idx_li].append(distance_coors_all_closest_val)

                        if distance < distance_prev:
                            distance_prev = distance
                            closest_idx_tuple = idx_tuple
                            closest_idx_cage = idx_cage_d

                idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = {'coor': coor_li_mapped_c, 'type': label_li_c, 'idx_tuple': closest_idx_tuple, 'idx_cage': closest_idx_cage}

            elif val_c != []:
                for entry_c in val_c: 
                    coor_li_mapped_c = entry_c['coor']
                    label_li_c = entry_c['type']
                    idx_tuple_c = entry_c['idx_tuple']
                    idx_cage_c = entry_c['idx_cage']

                    idx_coor24li_tuple_cage_belongin_complete_closest[idx_li] = {'coor': coor_li_mapped_c, 'type': label_li_c, 'idx_tuple': idx_tuple_c, 'idx_cage': idx_cage_c}

        idx_coor24li_tuple_cage_belongin_complete_id48htype2 = idx_coor24li_tuple_cage_belongin_complete_closest.copy()

        # # Renaming keys
        # for key in idx_coor24li_tuple_cage_belongin_complete_id48htype2:
        #     idx_coor24li_tuple_cage_belongin_complete_id48htype2[key]['idx_tuple'] = idx_coor24li_tuple_cage_belongin_complete_id48htype2[key].pop('idx_tuple_closest')
        #     idx_coor24li_tuple_cage_belongin_complete_id48htype2[key]['idx_cage'] = idx_coor24li_tuple_cage_belongin_complete_id48htype2[key].pop('idx_cage_closest')

        for key_e, val_e in idx_coor24li_tuple_cage_belongin_complete_closest.items():
            coor_li_mapped_e = val_e['coor']
            coor_li_mapped_e_rounded = tuple(round(coordinate, 5) for coordinate in coor_li_mapped_e)
            label_li_e = val_e['type']
            idx_tuple_e = val_e['idx_tuple']
            idx_cage_e = val_e['idx_cage']

            if label_li_e == '48htype1':
                    
                for id_48htype2, val_48htype2_metainfo_temp in coor_48htype2_metainfo.items():
                    coor_48htype2_metainfo_temp = val_48htype2_metainfo_temp['coor']
                    coor_48htype2_metainfo_temp_rounded = tuple(round(coordinate, 5) for coordinate in coor_48htype2_metainfo_temp)

                    if (coor_li_mapped_e_rounded == coor_48htype2_metainfo_temp_rounded):

                        idx_coor24li_tuple_cage_belongin_complete_id48htype2[key_e] = {'coor': coor_li_mapped_e, 'type': label_li_e, 'idx_tuple': id_48htype2, 'idx_cage': idx_cage_e}
            
            elif label_li_e == 'weirdos':
                idx_tuple_weirdo = f'x'
                # print("Value of idx_tuple_weirdo before assignment:", idx_tuple_weirdo)
                idx_coor24li_tuple_cage_belongin_complete_id48htype2[key_e] = {'coor': coor_li_mapped_e, 'type': label_li_e, 'idx_tuple': idx_tuple_weirdo, 'idx_cage': idx_cage_e}
                # print("Value of idx_tuple_weirdo after assignment:", idx_tuple_weirdo)
                
        sorted_distance_coors_all_closest = {key: sorted(value, key=lambda x: x['dist']) for key, value in distance_coors_all_closest.items()}
        top_n_distance_coors = {k: v[0:n] for k, v in sorted_distance_coors_all_closest.items()}
        # !!! assumed there's NO DUPLICATE with the SECOND distance

        dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest] = idx_coor24li_tuple_cage_belongin_complete_closest
        dataframe.at[idx, col_top_n_distance_coors] = top_n_distance_coors
        dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_id48htype2] = idx_coor24li_tuple_cage_belongin_complete_id48htype2


def get_df_movement(dataframe, to_plot, activate_closest_tuple):
    """
    to_plot = idx_tuple, type, idx_cage
    """
    # col_idx_coor24li_tuple_cage_belongin_complete_closest_weight = "idx_coor24li_tuple_cage_belongin_complete_closest_weight"
    if activate_closest_tuple: # small adjustment here
        col_idx_coor24li_tuple_cage_belongin_complete = "idx_coor24li_tuple_cage_belongin_complete_closest"
    else: # small adjustment here
        col_idx_coor24li_tuple_cage_belongin_complete = "idx_coor24li_tuple_cage_belongin_complete_id48htype2"

    df_to_plot = pd.DataFrame()

    for idx in range(dataframe["geometry"].size):

        # idx_coor24li_tuple_cage_belongin_complete_closest_weight = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest_weight]
        idx_coor24li_tuple_cage_belongin_complete = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete]

        # for j in range(len(idx_coor24li_tuple_cage_belongin_complete_closest_weight)):
        for j in range(len(idx_coor24li_tuple_cage_belongin_complete)):
            df_to_plot.at[idx, f"{j}"] = None  

            # coor_Li_ref_mean = np.mean(coor_Li_ref, axis=0)
            # distance = calc_distance.mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

            # dict_weighted[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref_mean}, coor_Li: {coor_Li[j]}'}
            
            # # for key_b, val_b in idx_coor24li_tuple_cage_belongin_complete_closest_weight.items():
            for key_b, val_b in idx_coor24li_tuple_cage_belongin_complete.items():
                # for entry_b in val_b: 
                # if activate_closest_tuple:
                #     df_to_plot.at[idx, f"{key_b}"] = val_b[f'{to_plot}_closest']
                # else:
                df_to_plot.at[idx, f"{key_b}"] = val_b[f'{to_plot}']

            # diameter_24g48h = max_mapping_radius * 2
            # # if distance < diameter_24g48h and index != idx_ref:
            # if distance > diameter_24g48h and idx != idx_ref:
            #     print(f"path: {idx}, Li: {j}, distance: {distance}")

    return df_to_plot


def get_df_movement_category(dataframe, activate_closest_tuple):
    # col_idx_coor24li_tuple_cage_belongin_complete_closest_weight = "idx_coor24li_tuple_cage_belongin_complete_closest_weight"
    if activate_closest_tuple:
        col_idx_coor24li_tuple_cage_belongin_complete_closest = "idx_coor24li_tuple_cage_belongin_complete_closest"
    else:
        col_idx_coor24li_tuple_cage_belongin_complete_closest = "idx_coor24li_tuple_cage_belongin_complete_id48htype2"

    df_to_plot = pd.DataFrame()

    for idx in range(dataframe["geometry"].size - 1):  # CHANGED HERE

        # idx_coor24li_tuple_cage_belongin_complete_closest_weight = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest_weight]
        # idx_coor24li_tuple_cage_belongin_complete_closest_weight_next = dataframe.at[idx+1, col_idx_coor24li_tuple_cage_belongin_complete_closest_weight]
        idx_coor24li_tuple_cage_belongin_complete_closest = dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest]
        idx_coor24li_tuple_cage_belongin_complete_closest_next = dataframe.at[idx+1, col_idx_coor24li_tuple_cage_belongin_complete_closest]

        # for j in range(len(idx_coor24li_tuple_cage_belongin_complete_closest_weight)):
        for j in range(len(idx_coor24li_tuple_cage_belongin_complete_closest)):
            df_to_plot.at[idx, f"{j}"] = None  

            # type = idx_coor24li_tuple_cage_belongin_complete_closest_weight[j]['type']
            # idx_tuple = idx_coor24li_tuple_cage_belongin_complete_closest_weight[j]['idx_tuple']
            # idx_cage = idx_coor24li_tuple_cage_belongin_complete_closest_weight[j]['idx_cage']

            # type_next = idx_coor24li_tuple_cage_belongin_complete_closest_weight_next[j]['type']
            # idx_tuple_next = idx_coor24li_tuple_cage_belongin_complete_closest_weight_next[j]['idx_tuple']
            # idx_cage_next = idx_coor24li_tuple_cage_belongin_complete_closest_weight_next[j]['idx_cage']

            coor = idx_coor24li_tuple_cage_belongin_complete_closest[j]['coor']
            coor_rounded = tuple(round(coordinate, 5) for coordinate in coor)
            type = idx_coor24li_tuple_cage_belongin_complete_closest[j]['type']
            idx_tuple = idx_coor24li_tuple_cage_belongin_complete_closest[j]['idx_tuple']
            idx_cage = idx_coor24li_tuple_cage_belongin_complete_closest[j]['idx_cage']

            coor_next = idx_coor24li_tuple_cage_belongin_complete_closest_next[j]['coor']
            coor_next_rounded = tuple(round(coordinate, 5) for coordinate in coor_next)
            type_next = idx_coor24li_tuple_cage_belongin_complete_closest_next[j]['type']
            idx_tuple_next = idx_coor24li_tuple_cage_belongin_complete_closest_next[j]['idx_tuple']
            idx_cage_next = idx_coor24li_tuple_cage_belongin_complete_closest_next[j]['idx_cage']

            if idx_cage != idx_cage_next:
                type_movement = 'inTERcage'
            elif idx_cage == idx_cage_next and idx_tuple != idx_tuple_next:
                type_movement = 'intracage'
            elif idx_cage == idx_cage_next and idx_tuple == idx_tuple_next and type != type_next:
                type_movement = 'intratriad'
            elif idx_cage == idx_cage_next and idx_tuple == idx_tuple_next and type == type_next and coor_rounded != coor_next_rounded:
                type_movement = 'intratriad'
            elif idx_cage == idx_cage_next and idx_tuple == idx_tuple_next and type == type_next and coor_rounded == coor_next_rounded:
                type_movement = 'staying'

            df_to_plot.at[idx, f"{j}"] = type_movement

            # coor_Li_ref_mean = np.mean(coor_Li_ref, axis=0)
            # distance = calc_distance.mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

            # dict_weighted[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref_mean}, coor_Li: {coor_Li[j]}'}
            
            # for key_b, val_b in idx_coor24li_tuple_cage_belongin_complete_closest_weight.items():
            #     type = val_b['type']
            #     idx_tuple = val_b['idx_tuple']
            #     idx_cage = val_b['idx_cage']
                # for entry_b in val_b: 
                # df_to_plot.at[idx, f"{key_b}"] = val_b[f'{to_plot}']

            # diameter_24g48h = max_mapping_radius * 2
            # # if distance < diameter_24g48h and index != idx_ref:
            # if distance > diameter_24g48h and idx != idx_ref:
            #     print(f"path: {idx}, Li: {j}, distance: {distance}")
    return df_to_plot


def get_and_plot_df_movement_category_counted(df_movement):
    df = pd.DataFrame()
    df['idx_file'] = None
    df['inTERcage'] = None
    df['intracage'] = None
    df['intratriad'] = None
    df['staying'] = None

    for i in range(len(df_movement)):
        counter_inTERcage = 0
        counter_intracage = 0
        counter_intratriad = 0
        counter_staying = 0

        for j in df_movement.iloc[i]:
            # print(j)
            if j == 'inTERcage':
                counter_inTERcage = counter_inTERcage + 1
            elif j == 'intracage':
                counter_intracage = counter_intracage + 1
            elif j == 'intratriad':
                counter_intratriad = counter_intratriad + 1
            elif j == 'staying':
                counter_staying = counter_staying + 1

        df.at[i, 'idx_file'] = i
        df.at[i, 'inTERcage'] = counter_inTERcage
        df.at[i, 'intracage'] = counter_intracage
        df.at[i, 'intratriad'] = counter_intratriad
        df.at[i, 'staying'] = counter_staying

    wide_df = pd.DataFrame(df)

    # Convert wide format to long format
    # long_df = pd.melt(wide_df, var_name='Category', value_name='Count')
    long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

    long_df['idx_file'] += 0.5

    fig = px.bar(long_df, x="idx_file", y="count", color="category", title="Idx of movement vs Category")
    fig.show()

    return df

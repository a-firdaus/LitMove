import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os, sys
from itertools import islice
from itertools import repeat
from addict import Dict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import math
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from chart_studio import plotly
from adjustText import adjust_text

import plotly.offline as pyoff
import re
import mplcursors
import mpldatacursor

# pymatgen libraries
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.inputs import Poscar


# class Orientation:
#     """
#     This class provides methods to determine and process the orientation of structures 
#     based on reference structures and transformations applied.
#     """

class ReadStructure:
    class Coor:
        def get_coor_structure_init_dict(structure):
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


        def get_coor_structure_init_dict_wholedataframe(dataframe, mapping):
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


    class Metainfo:
        def get_tuple_metainfo(coor_structure_init_dict_expanded, litype, el):
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
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                
                elif litype == 2:
                    for j in coor_li48htype1_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype2_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                elif litype == 3:
                    for j in coor_li48htype1_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                        
                    for j in coor_li48htype2_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype3_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                elif litype == 4:
                    for j in coor_li48htype1_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                        
                    for j in coor_li48htype2_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype3_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype4_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                elif litype == 5:
                    for j in coor_li48htype1_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                        
                    for j in coor_li48htype2_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype3_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype4_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype5_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                elif litype == 6:
                    for j in coor_li48htype1_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                        
                    for j in coor_li48htype2_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype3_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype4_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype5_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype6_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                elif litype == 7:
                    for j in coor_li48htype1_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                        
                    for j in coor_li48htype2_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype3_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype4_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype5_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype6_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype7_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype7'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                        
                elif litype == 8:
                    for j in coor_li48htype1_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype1'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)
                        
                    for j in coor_li48htype2_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype2'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype3_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype3'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype4_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype4'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype5_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype5'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype6_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype6'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype7_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype7'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)

                    for j in coor_li48htype8_ref:
                        distance = Operation.Distance.mic_eucledian_distance(i, j)

                        tuple_metainfo_all_dict = {'coor': j, 'dist': distance, 'type': '48htype8'}

                        tuple_metainfo_all[idx_i].append(tuple_metainfo_all_dict)


            sorted_tuple_metainfo_all = {key: sorted(value, key=lambda x: x['dist']) for key, value in tuple_metainfo_all.items()}
            top_n_tuple_metainfo = {k: v[0:n] for k, v in sorted_tuple_metainfo_all.items()}

            for key, values_list in top_n_tuple_metainfo.items():
                selected_values = [{'coor': entry['coor'], "type": entry["type"]} for entry in values_list]
                tuple_metainfo[key] = selected_values
                                
            return tuple_metainfo   


        def get_coor_48htype1_metainfo(coor_structure_init_dict_expanded, el):
            all_coor_48htype1 = coor_structure_init_dict_expanded[el][24:72]

            coor_48htype1_metainfo = defaultdict(list)

            for id, coor in enumerate(all_coor_48htype1):
                coor_48htype1_metainfo[id] = {'coor': coor}

            return coor_48htype1_metainfo


        def get_idx_cage_coor_24g(coor_24g_array, labels, idx_coor_cage_order, amount_clusters):
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


        def get_tuple_cage_metainfo(tuple_metainfo, idx_cage_coor_24g):
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


    class Parameter:
        def get_dx_dz(file_path, litype):
            """
            Get position (dx, dz) from a CIF file.

            Parameters:
            - file_path (str): The path to the file to be read.
            - litype: how many lithium type to be identified.

            Returns:
            - array of all positions from all types.
            """
            dictio = {}

            if litype == 0:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])

            elif litype == 1:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])
                        if line.startswith("Li2"):
                            parts = line.split()
                            dictio["dx1_48h_type1_init"] = float(parts[4])
                            dictio["dx2_48h_type1_init"] = float(parts[5])
                            dictio["dz_48h_type1_init"] = float(parts[6])

            elif litype == 2:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])
                        if line.startswith("Li2"):
                            parts = line.split()
                            dictio["dx1_48h_type1_init"] = float(parts[4])
                            dictio["dx2_48h_type1_init"] = float(parts[5])
                            dictio["dz_48h_type1_init"] = float(parts[6])
                        if line.startswith("Li3"):
                            parts = line.split()
                            dictio["dx1_48h_type2_init"] = float(parts[4])
                            dictio["dx2_48h_type2_init"] = float(parts[5])
                            dictio["dz_48h_type2_init "]= float(parts[6])

                # return dx1_48h_type1_init, dx2_48h_type1_init, dz_48h_type1_init, dx1_48h_type2_init, dx2_48h_type2_init, dz_48h_type2_init, dx_24g_init, dz1_24g_init, dz2_24g_init

            elif litype == 3:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])
                        if line.startswith("Li2"):
                            parts = line.split()
                            dictio["dx1_48h_type1_init"] = float(parts[4])
                            dictio["dx2_48h_type1_init"] = float(parts[5])
                            dictio["dz_48h_type1_init"] = float(parts[6])
                        if line.startswith("Li3"):
                            parts = line.split()
                            dictio["dx1_48h_type2_init"] = float(parts[4])
                            dictio["dx2_48h_type2_init"] = float(parts[5])
                            dictio["dz_48h_type2_init "]= float(parts[6])
                        if line.startswith("Li4"):
                            parts = line.split()
                            dictio["dx1_48h_type3_init"] = float(parts[4])
                            dictio["dx2_48h_type3_init"] = float(parts[5])
                            dictio["dz_48h_type3_init"] = float(parts[6])

                # return dx1_48h_type1_init, dx2_48h_type1_init, dz_48h_type1_init, dx1_48h_type2_init, dx2_48h_type2_init, dz_48h_type2_init, dx_24g_init, dz1_24g_init, dz2_24g_init, dx1_48h_type3_init, dx2_48h_type3_init, dz_48h_type3_init

            elif litype == 4:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])
                        if line.startswith("Li2"):
                            parts = line.split()
                            dictio["dx1_48h_type1_init"] = float(parts[4])
                            dictio["dx2_48h_type1_init"] = float(parts[5])
                            dictio["dz_48h_type1_init"] = float(parts[6])
                        if line.startswith("Li3"):
                            parts = line.split()
                            dictio["dx1_48h_type2_init"] = float(parts[4])
                            dictio["dx2_48h_type2_init"] = float(parts[5])
                            dictio["dz_48h_type2_init "]= float(parts[6])
                        if line.startswith("Li4"):
                            parts = line.split()
                            dictio["dx1_48h_type3_init"] = float(parts[4])
                            dictio["dx2_48h_type3_init"] = float(parts[5])
                            dictio["dz_48h_type3_init"] = float(parts[6])
                        if line.startswith("Li5"):
                            parts = line.split()
                            dictio["dx1_48h_type4_init"] = float(parts[4])
                            dictio["dx2_48h_type4_init"] = float(parts[5])
                            dictio["dz_48h_type4_init"] = float(parts[6])

            elif litype == 5:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])
                        if line.startswith("Li2"):
                            parts = line.split()
                            dictio["dx1_48h_type1_init"] = float(parts[4])
                            dictio["dx2_48h_type1_init"] = float(parts[5])
                            dictio["dz_48h_type1_init"] = float(parts[6])
                        if line.startswith("Li3"):
                            parts = line.split()
                            dictio["dx1_48h_type2_init"] = float(parts[4])
                            dictio["dx2_48h_type2_init"] = float(parts[5])
                            dictio["dz_48h_type2_init "]= float(parts[6])
                        if line.startswith("Li4"):
                            parts = line.split()
                            dictio["dx1_48h_type3_init"] = float(parts[4])
                            dictio["dx2_48h_type3_init"] = float(parts[5])
                            dictio["dz_48h_type3_init"] = float(parts[6])
                        if line.startswith("Li5"):
                            parts = line.split()
                            dictio["dx1_48h_type4_init"] = float(parts[4])
                            dictio["dx2_48h_type4_init"] = float(parts[5])
                            dictio["dz_48h_type4_init"] = float(parts[6])
                        if line.startswith("Li6"):
                            parts = line.split()
                            dictio["dx1_48h_type5_init"] = float(parts[4])
                            dictio["dx2_48h_type5_init"] = float(parts[5])
                            dictio["dz_48h_type5_init"] = float(parts[6])
            elif litype == 6:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])
                        if line.startswith("Li2"):
                            parts = line.split()
                            dictio["dx1_48h_type1_init"] = float(parts[4])
                            dictio["dx2_48h_type1_init"] = float(parts[5])
                            dictio["dz_48h_type1_init"] = float(parts[6])
                        if line.startswith("Li3"):
                            parts = line.split()
                            dictio["dx1_48h_type2_init"] = float(parts[4])
                            dictio["dx2_48h_type2_init"] = float(parts[5])
                            dictio["dz_48h_type2_init "]= float(parts[6])
                        if line.startswith("Li4"):
                            parts = line.split()
                            dictio["dx1_48h_type3_init"] = float(parts[4])
                            dictio["dx2_48h_type3_init"] = float(parts[5])
                            dictio["dz_48h_type3_init"] = float(parts[6])
                        if line.startswith("Li5"):
                            parts = line.split()
                            dictio["dx1_48h_type4_init"] = float(parts[4])
                            dictio["dx2_48h_type4_init"] = float(parts[5])
                            dictio["dz_48h_type4_init"] = float(parts[6])
                        if line.startswith("Li6"):
                            parts = line.split()
                            dictio["dx1_48h_type5_init"] = float(parts[4])
                            dictio["dx2_48h_type5_init"] = float(parts[5])
                            dictio["dz_48h_type5_init"] = float(parts[6])
                        if line.startswith("Li7"):
                            parts = line.split()
                            dictio["dx1_48h_type6_init"] = float(parts[4])
                            dictio["dx2_48h_type6_init"] = float(parts[5])
                            dictio["dz_48h_type6_init"] = float(parts[6])
            elif litype == 7:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])
                        if line.startswith("Li2"):
                            parts = line.split()
                            dictio["dx1_48h_type1_init"] = float(parts[4])
                            dictio["dx2_48h_type1_init"] = float(parts[5])
                            dictio["dz_48h_type1_init"] = float(parts[6])
                        if line.startswith("Li3"):
                            parts = line.split()
                            dictio["dx1_48h_type2_init"] = float(parts[4])
                            dictio["dx2_48h_type2_init"] = float(parts[5])
                            dictio["dz_48h_type2_init "]= float(parts[6])
                        if line.startswith("Li4"):
                            parts = line.split()
                            dictio["dx1_48h_type3_init"] = float(parts[4])
                            dictio["dx2_48h_type3_init"] = float(parts[5])
                            dictio["dz_48h_type3_init"] = float(parts[6])
                        if line.startswith("Li5"):
                            parts = line.split()
                            dictio["dx1_48h_type4_init"] = float(parts[4])
                            dictio["dx2_48h_type4_init"] = float(parts[5])
                            dictio["dz_48h_type4_init"] = float(parts[6])
                        if line.startswith("Li6"):
                            parts = line.split()
                            dictio["dx1_48h_type5_init"] = float(parts[4])
                            dictio["dx2_48h_type5_init"] = float(parts[5])
                            dictio["dz_48h_type5_init"] = float(parts[6])
                        if line.startswith("Li7"):
                            parts = line.split()
                            dictio["dx1_48h_type6_init"] = float(parts[4])
                            dictio["dx2_48h_type6_init"] = float(parts[5])
                            dictio["dz_48h_type6_init"] = float(parts[6])
                        if line.startswith("Li8"):
                            parts = line.split()
                            dictio["dx1_48h_type7_init"] = float(parts[4])
                            dictio["dx2_48h_type7_init"] = float(parts[5])
                            dictio["dz_48h_type7_init"] = float(parts[6])
            elif litype == 8:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith("Li1"):
                            parts = line.split()
                            dictio["dx_24g_init"] = float(parts[4])
                            dictio["dz1_24g_init"] = float(parts[5])
                            dictio["dz2_24g_init"] = float(parts[6])
                        if line.startswith("Li2"):
                            parts = line.split()
                            dictio["dx1_48h_type1_init"] = float(parts[4])
                            dictio["dx2_48h_type1_init"] = float(parts[5])
                            dictio["dz_48h_type1_init"] = float(parts[6])
                        if line.startswith("Li3"):
                            parts = line.split()
                            dictio["dx1_48h_type2_init"] = float(parts[4])
                            dictio["dx2_48h_type2_init"] = float(parts[5])
                            dictio["dz_48h_type2_init "]= float(parts[6])
                        if line.startswith("Li4"):
                            parts = line.split()
                            dictio["dx1_48h_type3_init"] = float(parts[4])
                            dictio["dx2_48h_type3_init"] = float(parts[5])
                            dictio["dz_48h_type3_init"] = float(parts[6])
                        if line.startswith("Li5"):
                            parts = line.split()
                            dictio["dx1_48h_type4_init"] = float(parts[4])
                            dictio["dx2_48h_type4_init"] = float(parts[5])
                            dictio["dz_48h_type4_init"] = float(parts[6])
                        if line.startswith("Li6"):
                            parts = line.split()
                            dictio["dx1_48h_type5_init"] = float(parts[4])
                            dictio["dx2_48h_type5_init"] = float(parts[5])
                            dictio["dz_48h_type5_init"] = float(parts[6])
                        if line.startswith("Li7"):
                            parts = line.split()
                            dictio["dx1_48h_type6_init"] = float(parts[4])
                            dictio["dx2_48h_type6_init"] = float(parts[5])
                            dictio["dz_48h_type6_init"] = float(parts[6])
                        if line.startswith("Li8"):
                            parts = line.split()
                            dictio["dx1_48h_type7_init"] = float(parts[4])
                            dictio["dx2_48h_type7_init"] = float(parts[5])
                            dictio["dz_48h_type7_init"] = float(parts[6])
                        if line.startswith("Li9"):
                            parts = line.split()
                            dictio["dx1_48h_type8_init"] = float(parts[4])
                            dictio["dx2_48h_type8_init"] = float(parts[5])
                            dictio["dz_48h_type8_init"] = float(parts[6])


                # return dx1_48h_type1_init, dx2_48h_type1_init, dz_48h_type1_init, dx1_48h_type2_init, dx2_48h_type2_init, dz_48h_type2_init, dx_24g_init, dz1_24g_init, dz2_24g_init, dx1_48h_type3_init, dx2_48h_type3_init, dz_48h_type3_init, dx1_48h_type4_init, dx2_48h_type4_init, dz_48h_type4_init 
            
            # return dx1_48h_type1_init, dx2_48h_type1_init, dz_48h_type1_init, dx1_48h_type2_init, dx2_48h_type2_init, dz_48h_type2_init, dx_24g_init, dz1_24g_init, dz2_24g_init, dx1_48h_type3_init, dx2_48h_type3_init, dz_48h_type3_init
            return tuple(dictio[key] for key in dictio.keys())


class Movement:    
    class Distance:
        def get_distance_litoli(dataframe, max_mapping_radius, destination_directory, idx_file_group, idx_ref, mean_ref, var_filename):
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
                        distance = Operation.Distance.mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

                        dict_distance[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref_mean}, coor_Li: {coor_Li[j]}'}
                        df_distance.at[index, f"{j}"] = distance

                        diameter_24g48h = max_mapping_radius * 2
                        # if distance < diameter_24g48h and index != idx_ref:
                        if distance > diameter_24g48h and index != idx_ref:
                            print(f"path: {index}, Li: {j}, distance: {distance}")

                elif mean_ref == False:
                    for j in range(len(coor_Li)):
                        df_distance.at[index, f"{j}"] = None  

                        distance = Operation.Distance.mic_eucledian_distance(coor_Li_ref[j], coor_Li[j])

                        dict_distance[f"{j}"] = {f'dist: {distance}, coor_ref: {coor_Li_ref[j]}, coor_Li: {coor_Li[j]}'}
                        df_distance.at[index, f"{j}"] = distance

                        diameter_24g48h = max_mapping_radius * 2
                        # if distance < diameter_24g48h and index != idx_ref:
                        if distance > diameter_24g48h and index != idx_ref:
                            print(f"path: {index}, Li: {j}, distance: {distance}")

            #         coors_Li_dist_structures_dict = {}
                    
            #         for k in range(len(coor_Li)):

            #             distance_litoli = Operation.Distance.mic_eucledian_distance(coor_Li[j], coor_Li[k])

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

            # #     # #     distance = Operation.Distance.mic_eucledian_distance(coor_Li_ref[j], coor_Li[j])

            # #     # #     df_distance.at[i, f"{j}"] = distance

            # #     # #     diameter_24g48h = max_mapping_radius * 2
            # #     # #     if distance < diameter_24g48h and i != 0:
            # #     # #         print(f"path: {i}, Li: {j}, distance: {distance}")

            return df_distance, dataframe_group


    class Occupancy:
        # # def get_occupancy(dataframe, coor_structure_init_dict_expanded, tuple_metainfo, destination_directory, var_filename, el):
        # #     col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"
            
        # #     col_occupancy = "occupancy"
        # #     col_coor24li_tuple_cage_belongin = "coor24li_tuple_cage_belongin"

        # #     dataframe[col_occupancy] = [{} for _ in range(len(dataframe.index))]
        # #     dataframe[col_coor24li_tuple_cage_belongin] = [{} for _ in range(len(dataframe.index))]

        # #     coor_structure_init_dict_expanded_el = coor_structure_init_dict_expanded[el]
        # #     coor_li48htype1_ref = coor_structure_init_dict_expanded_el[24:72]

        # #     for idx in range(dataframe["geometry"].size):
        # #         coor24li_tuple_cage_belongin = defaultdict(list)

        # #         file_24Li = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}.cif"
        # #         file_path_24Li = os.path.join(destination_directory, file_24Li)

        # #         coor_structure_24Li_dict_el = ReadStructure.Coor.get_coor_structure_init_dict(Structure.from_file(file_path_24Li))[el]
                
        # #         # Convert lists of arrays to sets for efficient comparison
        # #         set_coor_structure = set(map(tuple, coor_structure_24Li_dict_el))
        # #         set_coor_li48htype1_ref = set(map(tuple, coor_li48htype1_ref))

        # #         # Find the difference between the two sets
        # #         result_set = set_coor_structure.difference(set_coor_li48htype1_ref)

        # #         # Convert the result back to a list of arrays
        # #         result_list = list(map(np.array, result_set))
        # #         # for idx_triad, val in tuple_metainfo.items():

        # #         for idx_triad, values_list in tuple_metainfo.items():
        # #             coor24li_tuple_cage_belongin[idx_triad] = []    # WRONG! should be idx atom
                    
        # #             for entry in values_list:
        # #                 for i in result_list:
                    
        # #                     if (i == entry['coor']).all():
        # #                         # if (tuple(i) == tuple(entry['coor'])).all():
        # #                         # coor24li_tuple_belongin_dict = {'coor': i, 'type':entry['type']}
        # #                         coor24li_tuple_cage_belongin_dict = {'coor': i, 'type':entry['type'], 'idx_cage':entry['idx_cage']}
        # #                         coor24li_tuple_cage_belongin[idx_triad].append(coor24li_tuple_cage_belongin_dict)

        # #         # idx_coor_weirdos_Li_dict = dataframe['idx_coor_weirdos_Li'][idx]

        # #         # for idx_weirdo, values_list in idx_coor_weirdos_Li_dict.items():
        # #         #         coorweirdo_tuple_belongin_dict = {'coor': values_list, 'type':'weirdo'}
        # #         #         coor24li_tuple_cage_belongin['weirdo'].append(coorweirdo_tuple_belongin_dict)
                
        # #         # for key, val in coor24li_tuple_cage_belongin.items():
        # #         #     for i

        # #         len_occupancy = []
        # #         for key, val in coor24li_tuple_cage_belongin.items():
        # #             len_occupancy.append(len(val))


        # #         amount_48htype1 = (len(coor_structure_24Li_dict_el)-len(result_list))
        # #         amount_weirdo = dataframe['#weirdos_Li'][idx]
        # #         occupancy_2 = len_occupancy.count(2)
        # #         occupancy_1 = len_occupancy.count(1)
        # #         occupancy_0 = len_occupancy.count(0) - amount_48htype1 - amount_weirdo

        # #         sanity_check_occupancy = occupancy_2 * 2 + occupancy_1

        # #         # if sanity_check_occupancy != 24:
        # #         #     print(f'sum of occupancy not achieved at idx {idx}')
        # #         #     sys.exit()

        # #         # print(f"idx: {idx}")

        # #         # if sanity_check_occupancy != 24:
        # #         #     sys.exit()

        # #         occupancy = {'2': occupancy_2, '1': occupancy_1, '0': occupancy_0, '48htype1': amount_48htype1,'weirdo': amount_weirdo}

        # #         dataframe.at[idx, col_occupancy] = occupancy
        # #         dataframe.at[idx, col_coor24li_tuple_cage_belongin] = coor24li_tuple_cage_belongin


        def get_occupancy(dataframe, coor_structure_init_dict_expanded, tuple_metainfo, el):
            col_idx_coor_limapped_weirdos_dict = "idx_coor_limapped_weirdos_dict"
            col_sum_of_weirdos_Li = f"#weirdos_Li"

            col_occupancy = "occupancy"
            col_idx_coor24li_tuple_cage_belongin_empty = "idx_coor24li_tuple_cage_belongin_empty"

            dataframe[col_occupancy] = [{} for _ in range(len(dataframe.index))]
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
                occupancy_0 = len_occupancy.count(0) - amount_48htype1 - amount_weirdo

                sanity_check_occupancy = occupancy_2 * 2 + occupancy_1 + amount_48htype1 + amount_weirdo + occupancy_0

                # if sanity_check_occupancy != 24:
                #     print(f'sum of occupancy not achieved at idx {idx}')
                #     sys.exit()

                # print(f"idx: {idx}")

                # if sanity_check_occupancy != 24:
                #     sys.exit()

                occupancy = {'2': occupancy_2, '1': occupancy_1, '0': occupancy_0, '48htype1': amount_48htype1,'weirdo': amount_weirdo}

                dataframe.at[idx, col_occupancy] = occupancy
                dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_empty] = idx_coor24li_tuple_cage_belongin_empty


    class TupleCage:
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

        # #                         distance = Operation.Distance.mic_eucledian_distance(coor_li_mapped_c, coor_tuple_d)

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

                                distance = Operation.Distance.mic_eucledian_distance(coor_li_mapped_c, coor_tuple_d)

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
                    # distance = Operation.Distance.mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

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
                    # distance = Operation.Distance.mic_eucledian_distance(coor_Li_ref_mean, coor_Li[j])

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


class Optimizer:
    class GetSumWeirdos:
        # # not yet changed from 3665 - 4444 (???)
        def get_sum_weirdos_Li_var_not_complete(max_mapping_radius, max_mapping_radius_48htype2, activate_radius, file_perfect_poscar_24_wo_cif, file_perfect_poscar_48n24_wo_cif, litype, var_optitype, iter_type, foldermapping_namestyle_all, cif_namestyle_all, modif_all_litype, full_calculation):
            # renamed from get_sum_weirdos_Li_var
            """
                Parameters:
                + max_mapping_radius 
                + max_mapping_radius_48htype2
                + activate_radius
                - file_perfect_poscar_24_wo_cif
                - file_perfect_poscar_48n24_wo_cif
                + litype
                + var_optitype
                + iter_type: "varying_dx_dz", "varying_radius", none
                - foldermapping_namestyle_all
                - cif_namestyle_all
                - modif_all_litype
                - full_calculation

                    - ref_positions_array
                    - dataframe_init
                    - file_perfect_poscar_24
                    - file_ori_ref_48n24

                iter_type: varying_dx_dz, varying_radius, none
                cif_namestyle_all: True, False, None
                full_calculation: True, False
            """
            
            direc = os.getcwd() # get current working directory

            # # user input
            # max_mapping_radius = 0.043
            # max_mapping_radius_48htype2 = 0.076
            # activate_radius = 2
            lattice_constant = 10.2794980000

            folder_name_init_system = "/Init_System"
            file_new_system = "CONTCAR"
            file_name_toten = "toten_final.ods"
            col_excel_geo = "geometry"
            col_excel_path = "path"
            reference_folder = "_reference_cif"
            results_folder = "_results"

            file_perfect_poscar_24 = f"{file_perfect_poscar_24_wo_cif}.cif"
            file_perfect_poscar_48n24 = f"{file_perfect_poscar_48n24_wo_cif}.cif"

            file_path_ori_ref_48n24 = f"/{reference_folder}/{file_perfect_poscar_48n24}"
            path_ori_ref_48n24 = direc+str(file_path_ori_ref_48n24)
            path_reference_folder = direc+"/"+str(reference_folder)

            ref_positions_array_all = np.array(ReadStructure.Parameter.get_dx_dz(path_ori_ref_48n24, litype))

            if litype == 0:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                
                ref_positions_array_singlelitype = ref_positions_array_all[0:3]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g]

            elif litype == 1:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]

                ref_positions_array_singlelitype = ref_positions_array_all[3:6]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1]

            elif litype == 2:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
                dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]

                ref_positions_array_singlelitype = ref_positions_array_all[6:9]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2]

            elif litype == 3:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
                dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
                dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]

                ref_positions_array_singlelitype = ref_positions_array_all[9:12]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3]

            elif litype == 4:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
                dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
                dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
                dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]

                ref_positions_array_singlelitype = ref_positions_array_all[12:15]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4]

            elif litype == 5:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
                dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
                dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
                dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]
                dx1_48h_type5, dx2_48h_type5, dz_48h_type5 = ref_positions_array_all[15:18]

                ref_positions_array_singlelitype = ref_positions_array_all[15:18]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4, dx1_48h_type5, dx2_48h_type5]
            
            elif litype == 6:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
                dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
                dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
                dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]
                dx1_48h_type5, dx2_48h_type5, dz_48h_type5 = ref_positions_array_all[15:18]
                dx1_48h_type6, dx2_48h_type6, dz_48h_type6 = ref_positions_array_all[18:21]

                ref_positions_array_singlelitype = ref_positions_array_all[18:21]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4, dx1_48h_type5, dx2_48h_type5, dx1_48h_type6, dx2_48h_type6]
            
            elif litype == 7:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
                dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
                dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
                dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]
                dx1_48h_type5, dx2_48h_type5, dz_48h_type5 = ref_positions_array_all[15:18]
                dx1_48h_type6, dx2_48h_type6, dz_48h_type6 = ref_positions_array_all[18:21]
                dx1_48h_type7, dx2_48h_type7, dz_48h_type7 = ref_positions_array_all[21:24]

                ref_positions_array_singlelitype = ref_positions_array_all[21:24]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4, dx1_48h_type5, dx2_48h_type5, dx1_48h_type6, dx2_48h_type6, dx1_48h_type7, dx2_48h_type7]

            elif litype == 8:
                dx_24g, dz1_24g, dz2_24g = ref_positions_array_all[0:3]
                dx1_48h_type1, dx2_48h_type1, dz_48h_type1 = ref_positions_array_all[3:6]
                dx1_48h_type2, dx2_48h_type2, dz_48h_type2 = ref_positions_array_all[6:9]
                dx1_48h_type3, dx2_48h_type3, dz_48h_type3 = ref_positions_array_all[9:12]
                dx1_48h_type4, dx2_48h_type4, dz_48h_type4 = ref_positions_array_all[12:15]
                dx1_48h_type5, dx2_48h_type5, dz_48h_type5 = ref_positions_array_all[15:18]
                dx1_48h_type6, dx2_48h_type6, dz_48h_type6 = ref_positions_array_all[18:21]
                dx1_48h_type7, dx2_48h_type7, dz_48h_type7 = ref_positions_array_all[21:24]
                dx1_48h_type8, dx2_48h_type8, dz_48h_type8 = ref_positions_array_all[24:27]

                ref_positions_array_singlelitype = ref_positions_array_all[24:27]
                ref_positions_array_all_compactform = [dx_24g, dz1_24g, dx1_48h_type1, dx2_48h_type1, dx1_48h_type2, dx2_48h_type2, dx1_48h_type3, dx2_48h_type3, dx1_48h_type4, dx2_48h_type4, dx1_48h_type5, dx2_48h_type5, dx1_48h_type6, dx2_48h_type6, dx1_48h_type7, dx2_48h_type7, dx1_48h_type8, dx2_48h_type8]


            # max_mapping_radius_48htype1_48htype2 = (max_mapping_radius + max_mapping_radius_48htype2) / 2
            # file_perfect_poscar_48n24 = "Li6PS5Cl_type2.cif"
            # file_perfect_poscar_24 = "Li6PS5Cl_24_mod_2p27291.cif" # copy this manually to folder_name_perfect_poscar  

            folder_name_iter_type = f"/{results_folder}/_{iter_type}/{file_perfect_poscar_48n24_wo_cif}/"
            path_folder_name_iter_type = direc+str(folder_name_iter_type)
            Operation.File.check_folder_existance(path_folder_name_iter_type, empty_folder=False)


            if foldermapping_namestyle_all == True:
                if activate_radius == 2:
                    if litype == 0:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 1:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 2:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 3:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 4:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 5:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 6:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"            
                    elif litype == 7:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"            
                    elif litype == 8:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                
                elif activate_radius == 1:
                    if litype == 0:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{max_mapping_radius}/"
                    elif litype == 1:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}/"
                    elif litype == 2:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}/"
                    elif litype == 3:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}/"
                    elif litype == 4:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}/"
                    elif litype == 5:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}/"
                    elif litype == 6:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}/"            
                    elif litype == 7:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}/"            
                    elif litype == 8:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}/"
                
            else:
                if activate_radius == 2:
                    if litype == 0:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 1:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 2:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 3:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 4:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 5:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 6:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 7:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                    elif litype == 8:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}_{max_mapping_radius_48htype2}/"
                        
                elif activate_radius == 1:
                    if litype == 0:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx_24g}_{dz1_24g}_{max_mapping_radius}/"
                    elif litype == 1:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}/"
                    elif litype == 2:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}/"
                    elif litype == 3:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}/"
                    elif litype == 4:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}/"
                    elif litype == 5:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}/"
                    elif litype == 6:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}/"
                    elif litype == 7:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}/"
                    elif litype == 8:
                        folder_name_destination_restructure = f"{folder_name_iter_type}{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}/"


            folder_name_perfect_poscar = "/_reference_cif/cif_matrix/Li1/"
            cif_line_nr_start = 26  # index from 0
            poscar_line_nr_start = 8
            poscar_line_nr_end = 60
            contcar_columns_type2 = ['coord_x', 'coord_y', 'coord_z']
            amount_Li = 24
            col_excel_toten = "toten [eV]" 
            amount_P = 4 
            amount_S = 20 
            amount_Cl = 4
            file_restructure = "CONTCAR" 
            cif_columns = ['species', 'idx_species', 'unkownvar_1', 'coord_x', 'coord_y', 'coord_z', 'unkownvar_2'] 

            direc_restructure_destination = direc+str(folder_name_destination_restructure)
            # direc_perfect_poscar = direc+str(folder_name_iter_type) ### direc+str(folder_name_perfect_poscar)
            path_perfect_poscar_24 = os.path.join(path_folder_name_iter_type, file_perfect_poscar_24)
            direc_init_system = direc+str(folder_name_init_system)

            dtype = {col_excel_geo: float, col_excel_path: float}

            data_toten = pd.read_excel(file_name_toten, dtype=dtype, engine="odf")
            data_toten_ori = data_toten
            data_toten = data_toten.sort_values(by=["geometry","path"],ignore_index=True,ascending=False)

            file_loc = CreateDataFrame.base(data_toten, file_new_system)

            # just refreshing folder
            Operation.File.check_folder_existance(direc_restructure_destination, empty_folder=True)

            # copy ref.cif inside _results/../.. 
            Operation.File.copy_rename_single_file(path_folder_name_iter_type, reference_folder, file_perfect_poscar_48n24, prefix=None)

            Operation.File.copy_rename_files(file_loc, direc_restructure_destination, file_restructure, prefix=None, savedir = False)
            PreProcessingCONTCAR.get_positive_lessthan1_poscarorcontcar(file_loc, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)

            file_loc_mask_1, file_loc_important_cols = Orientation.calculate(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation="False")
            
            if modif_all_litype == True:
                ref_positions_array = ref_positions_array_all
            elif modif_all_litype == False:
                ref_positions_array = ref_positions_array_singlelitype
            elif modif_all_litype == None:
                ref_positions_array = ref_positions_array_all

            if cif_namestyle_all == True:
                ref_positions_array_filename = ref_positions_array_all_compactform
            elif cif_namestyle_all == False:
                ref_positions_array_filename = ref_positions_array_singlelitype
            # # DUNNO WHAT TO DO HERE
            elif cif_namestyle_all == None:
                ref_positions_array_filename = ref_positions_array_all_compactform

            # path_perfect_poscar_48n24 = modif_dx_dz_cif(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
            # path_perfect_poscar_48n24 = Optimizer.Position.Modify.modif_dx_dz_get_filepath(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, ref_positions_array_filename, litype, var_optitype, modif_all_litype)
            path_perfect_poscar_48n24 = Optimizer.Position.Modify.modif_dx_dz_get_filepath(path_folder_name_iter_type, path_ori_ref_48n24, ref_positions_array, ref_positions_array_filename, litype, var_optitype, modif_all_litype)

            # just copy file
            # Operation.File.copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_24, prefix=None)
            # !!! had to copy file_perfect_poscar_48n24 into Li1
            # Operation.File.copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_48n24, prefix=None)
            Operation.File.copy_rename_single_file(direc_restructure_destination, path_folder_name_iter_type, file_perfect_poscar_48n24, prefix=None)

            # Operation.File.copy_rename_files(file_loc_important_cols, direc_restructure_destination, file_restructure, prefix=None,  savedir = True)

            # # var_c = "trf_w_linalg_orientated"
            # # Orientation.get_structure_with_linalg_orientated(file_loc_important_cols, direc_restructure_destination, file_restructure, var_c)


            # # var_name_in = "trf_w_linalg_orientated"
            # # var_name_out = "trf_w_linalg_orientated_positive"
            # # n_decimal = 8
            # # get_orientated_positive_lessthan1_cif(file_loc_important_cols, direc_restructure_destination, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal)

            # max_mapping_radius = 0.05282658993283027
            # max_mapping_radius = 0.045
            # max_mapping_radius = 0.055
            # max_mapping_radius = 0.04197083906
            ref_structure_48n24 = Structure.from_file(path_perfect_poscar_48n24)
            cif_structure = Structure(ref_structure_48n24.lattice, ref_structure_48n24.species, ref_structure_48n24.frac_coords)
            cif = CifWriter(cif_structure)
            cif.write_file(f"{direc_restructure_destination}{file_perfect_poscar_48n24_wo_cif}_expanded.cif")

            coor_structure_init_dict = ReadStructure.Coor.get_coor_structure_init_dict(ref_structure_48n24)
            coor_structure_init_dict_expanded = ReadStructure.Coor.get_coor_structure_init_dict(Structure.from_file(f"{direc_restructure_destination}{file_perfect_poscar_48n24_wo_cif}_expanded.cif"))

            # PreProcessingCONTCAR.get_positive_lessthan1_poscarorcontcar(file_loc_important_cols, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)
            ReadStructure.Coor.get_coor_structure_init_dict_wholedataframe(file_loc_important_cols, mapping = "False")

            # if activate_radius == 3:
            #     get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
            #     get_flag_map_weirdos_48htype1_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype1_48htype2)
            #     get_flag_map_weirdos_48htypesmerged_level1_el(file_loc_important_cols, "Li")
            #     get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
            #     get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
            if activate_radius == 2:
                get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
                get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
                get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
            elif activate_radius == 1:
                get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)

            # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "P", max_mapping_radius)
            # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "S", max_mapping_radius)
            # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Cl", max_mapping_radius)

            Mapping.AtomIndexing.get_idx_weirdos_el(file_loc_important_cols, "Li", activate_radius)

            # Mapping.AtomIndexing.idx_correcting_mapped_el(file_loc_important_cols, el="Li")
            Mapping.AtomIndexing.idx_correcting_mapped_el(file_loc_important_cols, "Li", activate_radius)
            # # Mapping.OutputCIF.create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, var_savefilename = "mapLi")
            
            Mapping.Labelling.get_distance_weirdos_label_el(file_loc_important_cols, coor_structure_init_dict, "Li", litype)

            # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","coor_weirdos_48htypesmerged_Li","top3_dist_weirdos_dict_Li","idx0_weirdos_Li","#weirdos_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
            # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_weirdos_Li","sum_mapped_48htype2_Li_new","#weirdos_Li","sum_mapped_48htypesmerged_Li_new","sum_sanitycheck_48htypesmerged_Li_new","idx0_weirdos_Li","top3_sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_Li","duplicate_closest24_w_data_Li","duplicate_closest24_w_data_48htype2_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)

            # if activate_radius == 3:
            #     file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","sum_weirdos_Li","sum_mapped_48htype1_48htype2_Li_closestduplicate","sum_weirdos_48htype1_48htype2_Li","sum_mapped_48htype2_Li_closestduplicate","#weirdos_Li","sum_mapped_48htypesmerged_Li","sum_sanitycheck_48htypesmerged_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
            #     file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","sum_weirdos_Li","sum_mapped_48htype1_48htype2_Li_closestduplicate","sum_weirdos_48htype1_48htype2_Li","sum_mapped_48htype2_Li_closestduplicate","#weirdos_Li","sum_mapped_48htypesmerged_Li","sum_sanitycheck_48htypesmerged_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","toten [eV]"]]

            #     sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())

            #     var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type1}_{dx2_48h_type1}_{formatted_dz_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{formatted_dz_48h_type2}_{dx_24g}_{dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}_{max_mapping_radius_48htype1_48htype2}"
            
            if full_calculation == False:
                pass
            elif full_calculation == True:
                Mapping.OutputCIF.create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, activate_radius, var_savefilename = "mapLi")
                Mapping.OutputCIF.rewrite_cif_w_correct_Li_idx(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, amount_Cl, var_savefilename_init = "mapLi", var_savefilename_new = "mapLi_reindexed")
                Mapping.OutputCIF.format_spacing_cif(file_loc_important_cols, direc_restructure_destination, var_savefilename_init = "mapLi_reindexed", var_savefilename_new = "mapLi_reindexed")
                # # # # Operation.File.delete_files(file_loc_important_cols, direc_restructure_destination, file_name_w_format = "mapLi_reindexed.cif")

                Mapping.OutputCIF.rewrite_cif_w_correct_Li_idx_weirdos_appended(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, amount_Cl, activate_radius,var_savefilename_init = "mapLi", var_savefilename_new = "mapLi_reindexed_weirdos_appended")
                Mapping.OutputCIF.format_spacing_cif(file_loc_important_cols, direc_restructure_destination, var_savefilename_init = "mapLi_reindexed_weirdos_appended", var_savefilename_new = "mapLi_reindexed_weirdos_appended")
                # # # Operation.File.delete_files(file_loc_important_cols, direc_restructure_destination, file_name_w_format = "mapLi_reindexed_weirdos_appended.cif")

                PreProcessingCONTCAR.create_cif_pymatgen(file_loc_important_cols, direc_restructure_destination, file_restructure = "CONTCAR_positive", var_name = "CONTCAR_positive_pymatgen")

                # # # Mapping.OutputCIF.ascending_Li(file_loc_important_cols, direc_restructure_destination, var_filename_init = "mapLi_reindexed_weirdos_appended", var_savefilename_new = "mapLi_reindexed_weirdos_appended_reordered")
                # # # Mapping.OutputCIF.format_spacing_cif(file_loc_important_cols, direc_restructure_destination, var_savefilename_init = "mapLi_reindexed_weirdos_appended_reordered", var_savefilename_new = "mapLi_reindexed_weirdos_appended_reordered")

                Mapping.AtomIndexing.get_idx_coor_limapped_weirdos_dict_litype(file_loc_important_cols, coor_structure_init_dict, activate_radius, litype, el="Li")

                PreProcessingCONTCAR.get_latticeconstant_structure_dict_iterated(file_loc_important_cols, direc_restructure_destination, var_filename = "CONTCAR")
                # Plot.StructureAnalysis.energy_vs_latticeconstant(file_loc_important_cols, var_filename = "CONTCAR")
                Plot.StructureAnalysis.weirdos_directcoor(file_loc_important_cols, activate_radius)

                coor_weirdos_Li = Mapping.OutputWeirdos.get_coor_weirdos_array(file_loc_important_cols, activate_radius)
                Mapping.OutputWeirdos.create_POSCAR_weirdos(coor_weirdos_Li, direc_restructure_destination, lattice_constant, filename = "POSCAR_weirdos")

                Mapping.Labelling.get_label_mapping(file_loc_important_cols, coor_structure_init_dict, "Li", activate_radius, litype)

            if litype == 0:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 1:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 2:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 3:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 4:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 5:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]]   
            elif litype == 6:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]] 
            elif litype == 7:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]] 
            elif litype == 8:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]]   

                # var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type1}_{dx2_48h_type1}_{formatted_dz_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{formatted_dz_48h_type2}_{dx_24g}_{dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
            
            sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())



            if foldermapping_namestyle_all == True:
                if activate_radius == 2:
                    if litype == 0:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 1:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 2:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 3:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 4:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 5:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 6:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 7:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 8:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                
                elif activate_radius == 1:
                    if litype == 0:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{max_mapping_radius}"
                    elif litype == 1:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}"
                    elif litype == 2:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}"
                    elif litype == 3:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}"
                    elif litype == 4:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}"
                    elif litype == 5:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}"
                    elif litype == 6:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}"
                    elif litype == 7:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}"
                    elif litype == 8:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{dx1_48h_type1}_{dx2_48h_type1}_{dx1_48h_type2}_{dx2_48h_type2}_{dx1_48h_type3}_{dx2_48h_type3}_{dx1_48h_type4}_{dx2_48h_type4}_{dx1_48h_type5}_{dx2_48h_type5}_{dx1_48h_type6}_{dx2_48h_type6}_{dx1_48h_type7}_{dx2_48h_type7}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}"
                
            
            else:
                if activate_radius == 2:
                    if litype == 0:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 1:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 2:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 3:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 4:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 5:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 6:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 7:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}_{max_mapping_radius_48htype2}"
                    elif litype == 8:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}_{max_mapping_radius_48htype2}"

                elif activate_radius == 1:
                    if litype == 0:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx_24g}_{dz1_24g}_{max_mapping_radius}"
                    elif litype == 1:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type1}_{dx2_48h_type1}_{max_mapping_radius}"
                    elif litype == 2:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type2}_{dx2_48h_type2}_{max_mapping_radius}"
                    elif litype == 3:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type3}_{dx2_48h_type3}_{max_mapping_radius}"
                    elif litype == 4:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type4}_{dx2_48h_type4}_{max_mapping_radius}"
                    elif litype == 5:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type5}_{dx2_48h_type5}_{max_mapping_radius}"
                    elif litype == 6:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type6}_{dx2_48h_type6}_{max_mapping_radius}"
                    elif litype == 7:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type7}_{dx2_48h_type7}_{max_mapping_radius}"
                    elif litype == 8:
                        var_excel_file = f"{int(sum_weirdos_Li)}_{dx1_48h_type8}_{dx2_48h_type8}_{max_mapping_radius}"


            # path_excel_file = os.path.join(direc_perfect_poscar, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
            path_excel_file = os.path.join(path_folder_name_iter_type, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
            file_loc_important_cols_sorted_toten.to_excel(path_excel_file, index=False)

            if activate_radius == 1:
                file_loc_important_cols.to_pickle(f'{path_folder_name_iter_type}file_loc_important_cols_{max_mapping_radius}_{file_perfect_poscar_48n24_wo_cif}.pkl') 
            elif activate_radius == 2:
                file_loc_important_cols.to_pickle(f'{path_folder_name_iter_type}file_loc_important_cols_{max_mapping_radius}_{max_mapping_radius_48htype2}_{file_perfect_poscar_48n24_wo_cif}.pkl')
            # elif activate_radius == 3:
            #     file_loc_important_cols.to_pickle(f'{path_folder_name_iter_type}file_loc_important_cols_{max_mapping_radius}_{max_mapping_radius_48htype2}_{max_mapping_radius_48htype1_48htype2}_{file_perfect_poscar_48n24_wo_cif}.pkl')

            return sum_weirdos_Li


        def get_sum_weirdos_Li(ref_positions_array, max_mapping_radius, max_mapping_radius_48htype2, dataframe_init, activate_radius, file_perfect_poscar_24, file_ori_ref_48n24, litype, var_optitype, iter_type):
            # renamed from get_sum_weirdos_Li_var_wo_weirdo_litype
            """
            

            Parameters:
            - ref_positions_array
            + max_mapping_radius
            + max_mapping_radius_48htype2
            - dataframe_init
            + activate_radius
            - file_perfect_poscar_24
            - file_ori_ref_48n24
            + litype
            + var_optitype
            + iter_type: "varying_dx_dz", "varying_radius", none

            Returns:
            - 

            Notes:
            - Only tested for activate_radius max 2
            """
            dataframe = dataframe_init

            formatted_positions = [Operation.Float.format_float(pos) for pos in ref_positions_array]
            new_dx1_type, new_dx2_type, new_dz_type = formatted_positions
            
            direc = os.getcwd() # get current working directory

            results_folder = "_results"
            reference_folder = "_reference_cif"

            file_path_ori_ref_48n24 = f"./{reference_folder}/{file_ori_ref_48n24}"
            # # max_mapping_radius_48htype1_48htype2 = (max_mapping_radius + max_mapping_radius_48htype2) / 2

            # # folder_name_init_system = "/Init_System"
            # # file_new_system = "CONTCAR"
            file_name_toten = "toten_final.ods"
            col_excel_geo = "geometry"
            col_excel_path = "path"

            folder_name_iter_type = f"/{results_folder}/_{iter_type}/{file_ori_ref_48n24}/"
            path_folder_name_iter_type = direc+str(folder_name_iter_type)
            Operation.File.check_folder_existance(path_folder_name_iter_type, empty_folder=False)

            # copy ref.cif inside _results/../.. 
            Operation.File.copy_rename_single_file(path_folder_name_iter_type, reference_folder, file_ori_ref_48n24, prefix=None)

            if activate_radius == 2:
                folder_name_destination_restructure = f"{path_folder_name_iter_type}restructure_{new_dx1_type}_{new_dx2_type}_{max_mapping_radius}_{max_mapping_radius_48htype2}_optimizer/"
            elif activate_radius == 1:
                folder_name_destination_restructure = f"{path_folder_name_iter_type}restructure_{new_dx1_type}_{new_dx2_type}_{max_mapping_radius}_optimizer/"

            folder_name_perfect_poscar = folder_name_iter_type
            cif_line_nr_start = 26  # index from 0
            poscar_line_nr_start = 8
            poscar_line_nr_end = 60
            contcar_columns_type2 = ['coord_x', 'coord_y', 'coord_z']
            amount_Li = 24
            col_excel_toten = "toten [eV]" 
            amount_P = 4 
            amount_S = 20 
            file_restructure = "CONTCAR" 
            cif_columns = ['species', 'idx_species', 'unkownvar_1', 'coord_x', 'coord_y', 'coord_z', 'unkownvar_2'] 

            direc_restructure_destination = direc+str(folder_name_destination_restructure)
            direc_perfect_poscar = direc+str(folder_name_perfect_poscar)
            path_perfect_poscar_24 = os.path.join(direc_perfect_poscar, file_perfect_poscar_24)
            direc_init_system = direc+str(folder_name_init_system)

            dtype = {col_excel_geo: float, col_excel_path: float}

            data_toten = pd.read_excel(file_name_toten, dtype=dtype, engine="odf")
            data_toten_ori = data_toten
            data_toten = data_toten.sort_values(by=["geometry","path"],ignore_index=True,ascending=False)

            # just refreshing folder
            Operation.File.check_folder_existance(direc_restructure_destination, empty_folder=True)

            # path_perfect_poscar_48n24 = modif_dx_dz_cif(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type, dx2_48h_type, dz_48h_type, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
            # path_perfect_poscar_48n24 = modif_dx_dz_cif_specificlitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
            path_perfect_poscar_48n24 = Optimizer.Position.Modify.modif_dx_dz_get_filepath(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, ref_positions_array, litype, var_optitype, modif_all_litype = False)

            # just copy file
            # Operation.File.copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_24, prefix=None)
            # !!! had to copy file_ori_ref_48n24 into Li1
            Operation.File.copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_ori_ref_48n24, prefix=None)

            # file_loc_mask_1, dataframe = Orientation.calculate(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation="False")

            Operation.File.copy_rename_files(dataframe, direc_restructure_destination, file_restructure, prefix=None, savedir = True)


            # # var_c = "trf_w_linalg_orientated"
            # # Orientation.get_structure_with_linalg_orientated(dataframe, direc_restructure_destination, file_restructure, var_c)


            # # var_name_in = "trf_w_linalg_orientated"
            # # var_name_out = "trf_w_linalg_orientated_positive"
            # # n_decimal = 8
            # # get_orientated_positive_lessthan1_cif(dataframe, direc_restructure_destination, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal)

            # max_mapping_radius = 0.05282658993283027
            # max_mapping_radius = 0.045
            # max_mapping_radius = 0.055
            # max_mapping_radius = 0.04197083906
            ref_structure_48n24 = Structure.from_file(path_perfect_poscar_48n24)

            coor_structure_init_dict = ReadStructure.Coor.get_coor_structure_init_dict(ref_structure_48n24)
            PreProcessingCONTCAR.get_positive_lessthan1_poscarorcontcar(dataframe, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)
            ReadStructure.Coor.get_coor_structure_init_dict_wholedataframe(dataframe, mapping = "False")

            if activate_radius == 2:
                get_flag_map_weirdos_el(dataframe, coor_structure_init_dict, "Li", max_mapping_radius)
                get_flag_map_weirdos_48htype2_el(dataframe, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
                get_flag_map_48htypesmerged_el(dataframe, "Li", activate_radius)
            elif activate_radius == 1:
                get_flag_map_weirdos_el(dataframe, coor_structure_init_dict, "Li", max_mapping_radius)

            # get_flag_map_weirdos_el(dataframe, coor_structure_init_dict, "P", max_mapping_radius)
            # get_flag_map_weirdos_el(dataframe, coor_structure_init_dict, "S", max_mapping_radius)
            # get_flag_map_weirdos_el(dataframe, coor_structure_init_dict, "Cl", max_mapping_radius)

            Mapping.AtomIndexing.get_idx_weirdos_el(dataframe, "Li", activate_radius)

            # Mapping.AtomIndexing.idx_correcting_mapped_el(dataframe, el="Li")
            Mapping.AtomIndexing.idx_correcting_mapped_el(dataframe, "Li", activate_radius)
            # # Mapping.OutputCIF.create_combine_structure(dataframe, direc_restructure_destination, amount_Li, amount_P, amount_S, var_savefilename = "mapLi")
            
            Mapping.Labelling.get_distance_weirdos_label_el(dataframe, coor_structure_init_dict, "Li", litype)

            # dataframe_sorted_toten = dataframe[["geometry","path","coor_weirdos_48htypesmerged_Li","top3_dist_weirdos_dict_Li","idx0_weirdos_Li","#weirdos_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
            # dataframe_sorted_toten = dataframe[["geometry","path","sum_weirdos_Li","sum_mapped_48htype2_Li_new","#weirdos_Li","sum_mapped_48htypesmerged_Li_new","sum_sanitycheck_48htypesmerged_Li_new","idx0_weirdos_Li","top3_sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_Li","duplicate_closest24_w_data_Li","duplicate_closest24_w_data_48htype2_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)

            if litype == 0:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 1:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 2:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 3:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 4:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 5:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]]
            elif litype == 6:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]] 
            elif litype == 7:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]] 
            elif litype == 8:
                dataframe_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                dataframe_not_sorted_toten = dataframe[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]]   

            sum_weirdos_Li = float(dataframe_sorted_toten["#weirdos_Li"].sum())
                    
            if activate_radius == 2:
                var_excel_file = f"{int(sum_weirdos_Li)}_{new_dx1_type}_{new_dx2_type}_{new_dz_type}_{max_mapping_radius}_{max_mapping_radius_48htype2}"

            elif activate_radius == 1:
                var_excel_file = f"{int(sum_weirdos_Li)}_{new_dx1_type}_{new_dx2_type}_{new_dz_type}_{max_mapping_radius}"

            path_excel_file = os.path.join(direc_perfect_poscar, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
            dataframe_sorted_toten.to_excel(path_excel_file, index=False)

            return sum_weirdos_Li


        def get_sum_weirdos_Li_var_litype(ref_positions_array, max_mapping_radius, max_mapping_radius_48htype2, activate_radius, file_perfect_poscar_24, file_ori_ref_48n24, litype, var_optitype):
            """
            

            Parameters:
            - ref_positions_array (str): 
            - max_mapping_radius
            - max_mapping_radius_48htype2
            - activate_radius
            - file_perfect_poscar_24
            - file_ori_ref_48n24
            - litype
            - var_optitype

            Returns:
            - 
            """
            formatted_positions = [Operation.Float.format_float(pos) for pos in ref_positions_array]
            new_dx1_type, new_dx2_type, new_dz_type = formatted_positions

            direc = os.getcwd() # get current working directory

            file_path_ori_ref_48n24 = f"./perfect_poscar/cif_matrix/ori/{file_ori_ref_48n24}"
            # max_mapping_radius_48htype1_48htype2 = (max_mapping_radius + max_mapping_radius_48htype2) / 2
            # file_ori_ref_48n24 = "Li6PS5Cl_type2.cif"
            # file_perfect_poscar_24 = "Li6PS5Cl_24_mod_2p27291.cif" # copy this manually to folder_name_perfect_poscar  

            folder_name_init_system = "/Init_System"
            file_new_system = "CONTCAR"
            file_name_toten = "toten_final.ods"
            col_excel_geo = "geometry"
            col_excel_path = "path"

            if activate_radius == 2:
                folder_name_destination_restructure = f"/restructure_{new_dx1_type}_{new_dx2_type}_{max_mapping_radius}_{max_mapping_radius_48htype2}_optimizer/"
            elif activate_radius == 1:
                folder_name_destination_restructure = f"/restructure_{new_dx1_type}_{new_dx2_type}_{max_mapping_radius}_optimizer/"

            folder_name_perfect_poscar = "/perfect_poscar/cif_matrix/Li1/"
            cif_line_nr_start = 26  # index from 0
            poscar_line_nr_start = 8
            poscar_line_nr_end = 60
            contcar_columns_type2 = ['coord_x', 'coord_y', 'coord_z']
            amount_Li = 24
            col_excel_toten = "toten [eV]" 
            amount_P = 4 
            amount_S = 20 
            file_restructure = "CONTCAR" 
            cif_columns = ['species', 'idx_species', 'unkownvar_1', 'coord_x', 'coord_y', 'coord_z', 'unkownvar_2'] 

            direc_restructure_destination = direc+str(folder_name_destination_restructure)
            direc_perfect_poscar = direc+str(folder_name_perfect_poscar)
            path_perfect_poscar_24 = os.path.join(direc_perfect_poscar, file_perfect_poscar_24)
            direc_init_system = direc+str(folder_name_init_system)

            dtype = {col_excel_geo: float, col_excel_path: float}

            data_toten = pd.read_excel(file_name_toten, dtype=dtype, engine="odf")
            data_toten_ori = data_toten
            data_toten = data_toten.sort_values(by=["geometry","path"],ignore_index=True,ascending=False)

            geometry = np.array([])
            path = np.array([])
            subdir_col = np.array([])
            for subdir, dirs, files in os.walk(direc,topdown=False):
                # source: https://stackoverflow.com/questions/27805919/how-to-only-read-lines-in-a-text-file-after-a-certain-string
                for file in files:
                    filepath = subdir + os.sep
                    # get directory of CONTCAR
                    if os.path.basename(file) == file_new_system:
                        geometry_nr = Operation.File.splitall(subdir)[-2]
                        path_nr = Operation.File.splitall(subdir)[-1]
                        geometry = pd.DataFrame(np.append(geometry, int(geometry_nr)), columns=["geometry"])
                        geometry_ori = geometry
                        geometry.dropna(axis=1)
                        path = pd.DataFrame(np.append(path, int(path_nr)), columns=["path"])#
                        path.dropna(axis=1)
                        path_sorted = path.sort_values(by="path",ascending=False)
                        subdir_file = os.path.join(subdir,file_new_system)
                        # # create directory of POSCAR of init system
                        subdir_init_system = direc_init_system + os.sep + geometry_nr + os.sep + path_nr
                        subdir_col = pd.DataFrame(np.append(subdir_col, subdir_file), columns=["subdir_new_system"])
                        file_loc = geometry.join(path)
                        file_loc["subdir_new_system"] = subdir_col#
                        path_ori = path

            file_loc_ori_notsorted = file_loc.copy()
            file_loc = file_loc.sort_values(by=["geometry","path"],ignore_index=True,ascending=False) # sort descendingly based on path

            file_loc["g+p"] = (file_loc["geometry"] + file_loc["path"]).fillna(0) # replace NaN with 0
            file_loc["g+p+1"] = file_loc["g+p"].shift(1)
            file_loc["g+p+1"][0] = 0 # replace 1st element with 0
            file_loc["g+p-1"] = file_loc["g+p"].shift(-1)
            file_loc["g+p-1"][(file_loc["g+p-1"]).size - 1] = 0.0 # replace last element with 0
            file_loc["perfect_system"] = file_loc["g+p"][(file_loc["g+p+1"] > file_loc["g+p"]) & (file_loc["g+p-1"] > file_loc["g+p"])]
            file_loc["perfect_system"][file_loc["geometry"].size-1] = 0.0 # hardcode the path 0/0
            file_loc["p_s_mask"] = [0 if np.isnan(item) else 1 for item in file_loc["perfect_system"]]



            if data_toten[col_excel_geo].all() == file_loc["geometry"].all() & data_toten[col_excel_path].all() == file_loc["path"].all():
                file_loc[col_excel_toten] = data_toten[col_excel_toten]
            else:
                print("check the compatibility of column geometry and path between data_toten file and file_loc")

            # just refreshing folder
            Operation.File.check_folder_existance(direc_restructure_destination, empty_folder=True)

            # path_perfect_poscar_48n24 = modif_dx_dz_cif(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type, dx2_48h_type, dz_48h_type, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
            path_perfect_poscar_48n24 = modif_dx_dz_cif_specificlitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, var_optitype, litype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)

            # just copy file
            # Operation.File.copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_24, prefix=None)
            # !!! had to copy file_ori_ref_48n24 into Li1
            Operation.File.copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_ori_ref_48n24, prefix=None)

            file_loc_mask_1, file_loc_important_cols = Orientation.calculate(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation="False")

            Operation.File.copy_rename_files(file_loc_important_cols, direc_restructure_destination, file_restructure, prefix=None, savedir = True)


            # # var_c = "trf_w_linalg_orientated"
            # # Orientation.get_structure_with_linalg_orientated(file_loc_important_cols, direc_restructure_destination, file_restructure, var_c)


            # # var_name_in = "trf_w_linalg_orientated"
            # # var_name_out = "trf_w_linalg_orientated_positive"
            # # n_decimal = 8
            # # get_orientated_positive_lessthan1_cif(file_loc_important_cols, direc_restructure_destination, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal)

            # max_mapping_radius = 0.05282658993283027
            # max_mapping_radius = 0.045
            # max_mapping_radius = 0.055
            # max_mapping_radius = 0.04197083906
            ref_structure_48n24 = Structure.from_file(path_perfect_poscar_48n24)

            coor_structure_init_dict = ReadStructure.Coor.get_coor_structure_init_dict(ref_structure_48n24)
            PreProcessingCONTCAR.get_positive_lessthan1_poscarorcontcar(file_loc_important_cols, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)
            ReadStructure.Coor.get_coor_structure_init_dict_wholedataframe(file_loc_important_cols, mapping = "False")

            if activate_radius == 2:
                get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
                get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
                get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
            elif activate_radius == 1:
                get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)

            # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "P", max_mapping_radius)
            # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "S", max_mapping_radius)
            # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Cl", max_mapping_radius)

            Mapping.AtomIndexing.get_idx_weirdos_el(file_loc_important_cols, "Li", activate_radius)

            # Mapping.AtomIndexing.idx_correcting_mapped_el(file_loc_important_cols, el="Li")
            Mapping.AtomIndexing.idx_correcting_mapped_el(file_loc_important_cols, "Li", activate_radius)
            # # Mapping.OutputCIF.create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, var_savefilename = "mapLi")
            
            Mapping.Labelling.get_distance_weirdos_label_el(file_loc_important_cols, coor_structure_init_dict, "Li", litype)

            # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","coor_weirdos_48htypesmerged_Li","top3_dist_weirdos_dict_Li","idx0_weirdos_Li","#weirdos_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
            # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_weirdos_Li","sum_mapped_48htype2_Li_new","#weirdos_Li","sum_mapped_48htypesmerged_Li_new","sum_sanitycheck_48htypesmerged_Li_new","idx0_weirdos_Li","top3_sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_Li","duplicate_closest24_w_data_Li","duplicate_closest24_w_data_48htype2_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)

            if litype == 0:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
            elif litype == 1:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
            elif litype == 2:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
            elif litype == 3:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
            elif litype == 4:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_48htype4_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_48htype4_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
            elif litype == 5:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","atom_mapping_Li_w_dist_label","toten [eV]"]]
            elif litype == 6:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_24g_Li","toten [eV]"]] 
            elif litype == 7:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_24g_Li","toten [eV]"]] 
            elif litype == 8:
                file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
                file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_48htype6_Li","#closest_48htype7_Li","#closest_48htype8_Li","#closest_24g_Li","toten [eV]"]]   


            sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())
                
            if activate_radius == 2:
                var_excel_file = f"{int(sum_weirdos_Li)}_{new_dx1_type}_{new_dx2_type}_{new_dz_type}_{max_mapping_radius}_{max_mapping_radius_48htype2}"

            elif activate_radius == 1:
                var_excel_file = f"{int(sum_weirdos_Li)}_{new_dx1_type}_{new_dx2_type}_{new_dz_type}_{max_mapping_radius}"

            path_excel_file = os.path.join(direc_perfect_poscar, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
            file_loc_important_cols_sorted_toten.to_excel(path_excel_file, index=False)

            return sum_weirdos_Li


    class Position:
        class Modify:
            def change_dx_dz_alllitype(file_path, file_path_new, ref_positions_array, litype):
                # old_name = change_dx_dz
                # ref_positions_array = ALL values in this array

                formatted_positions = [Operation.Float.format_float(pos) for pos in ref_positions_array]
                print(f"formatted_positions: {formatted_positions}")

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                if litype == 0:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)

                elif litype == 1:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
                    new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li2"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)

                elif litype == 2:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
                    new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
                    new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li2"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li3"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)

                elif litype == 3:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
                    new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
                    new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
                    new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li2"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li3"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li4"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)

                elif litype == 4:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
                    new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
                    new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
                    new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
                    new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li2"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li3"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li4"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li5"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)

                elif litype == 5:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
                    new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
                    new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
                    new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
                    new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
                    new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li2"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li3"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li4"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li5"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li6"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)
                elif litype == 6:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
                    new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
                    new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
                    new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
                    new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
                    new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
                    new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li2"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li3"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li4"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li5"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li6"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li7"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)
                elif litype == 7:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
                    new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
                    new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
                    new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
                    new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
                    new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
                    new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]
                    new_dx1_48h_type7, new_dx2_48h_type7, new_dz_48h_type7 = formatted_positions[21:24]

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li2"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li3"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li4"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li5"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li6"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li7"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li8"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type7}",f"{new_dx2_48h_type7}",f"{new_dz_48h_type7}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)
                elif litype == 8:
                    new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
                    new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
                    new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
                    new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
                    new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
                    new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
                    new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]
                    new_dx1_48h_type7, new_dx2_48h_type7, new_dz_48h_type7 = formatted_positions[21:24]
                    new_dx1_48h_type8, new_dx2_48h_type8, new_dz_48h_type8 = formatted_positions[24:27]

                    with open(file_path_new, 'w') as f:
                        for line in lines:
                            if line.startswith("Li1"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                                parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li2"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li3"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li4"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li5"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li6"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li7"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li8"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type7}",f"{new_dx2_48h_type7}",f"{new_dz_48h_type7}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            if line.startswith("Li9"):
                                parts = line.split()
                                # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                                parts[4:6+1] = [f"{new_dx1_48h_type8}",f"{new_dx2_48h_type8}",f"{new_dz_48h_type8}"]
                                parts[2] = f" {parts[2]}"
                                parts[-1] = f"{parts[-1]}\n"
                                line = " ".join(parts)
                            f.write(line)


            def change_dx_dz_specificlitype(file_path, file_path_new, ref_positions_array, litype):

                formatted_positions = [Operation.Float.format_float(pos) for pos in ref_positions_array]

                new_dx1_type, new_dx2_type, new_dz_type = formatted_positions

                with open(file_path, 'r') as f:
                    lines = f.readlines()

                with open(file_path_new, 'w') as f:
                    for line in lines:
                        if line.startswith(f"Li{litype+1}"):
                            parts = line.split()
                            # parts[4:6+1] = [f"{new_dx1_type:.5f}",f"{new_dx2_type:.5f}",f"{new_dz_type:.5f}"]
                            parts[4:6+1] = [f"{new_dx1_type}",f"{new_dx2_type}",f"{new_dz_type}"]
                            parts[2] = f" {parts[2]}"
                            parts[-1] = f"{parts[-1]}\n"
                            line = " ".join(parts)
                        f.write(line)


            def modif_dx_dz_get_filepath(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, ref_positions_array_filename, litype, var_optitype, modif_all_litype):
                file_path_new = Optimizer.Position.Output.create_file_name(direc_perfect_poscar, ref_positions_array_filename, var_optitype)
                if modif_all_litype == True:
                    Optimizer.Position.Modify.change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)
                elif modif_all_litype == False:
                    Optimizer.Position.Modify.change_dx_dz_specificlitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)
                elif modif_all_litype == None:
                    Optimizer.Position.Modify.change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

                return file_path_new


            # # SEEMS NO USAGE
            # def modif_dx_dz_cif_alllitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, litype, var_optitype):
            #     file_path_new = Optimizer.Position.Output.create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype)
            #     Optimizer.Position.Modify.change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

            #     return file_path_new


            # # SEEMS NO USAGE
            # def modif_dx_dz_cif_specificlitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, litype, var_optitype):
            #     file_path_new = Optimizer.Position.Output.create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype)
            #     Optimizer.Position.Modify.change_dx_dz_specificlitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

            #     return file_path_new


        class Output:
            def create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype):
                formatted_positions = [Operation.Float.format_float(pos) for pos in ref_positions_array]
                formatted_positions_str = list(map(str, formatted_positions))
                return os.path.join(direc_perfect_poscar, f"Li6PS5Cl_{'_'.join(formatted_positions_str)}_{var_optitype}.cif")


# def get_sum_weirdos_Li_var_wo_weirdo(dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, max_mapping_radius, max_mapping_radius_48htype2, dataframe, activate_radius, file_perfect_poscar_24, file_ori_ref_48n24, litype, var_optitype):

#     file_loc_important_cols = dataframe

#     formatted_dx1_48h_type1 = Operation.Float.format_float(dx1_48h_type1)
#     formatted_dx2_48h_type1 = Operation.Float.format_float(dx2_48h_type1)
#     formatted_dz_48h_type1 = Operation.Float.format_float(dz_48h_type1)
#     formatted_dx1_48h_type2 = Operation.Float.format_float(dx1_48h_type2)
#     formatted_dx2_48h_type2 = Operation.Float.format_float(dx2_48h_type2)
#     formatted_dz_48h_type2 = Operation.Float.format_float(dz_48h_type2)
#     formatted_dx_24g = Operation.Float.format_float(dx_24g)
#     formatted_dz1_24g = Operation.Float.format_float(dz1_24g)
#     formatted_dz2_24g = Operation.Float.format_float(dz2_24g)
#     direc = os.getcwd() # get current working directory

#     # # user input
#     # max_mapping_radius = 0.043
#     # max_mapping_radius_48htype2 = 0.076
#     # activate_radius = 2

#     file_path_ori_ref_48n24 = f"./perfect_poscar/cif_matrix/ori/{file_ori_ref_48n24}"
#     # max_mapping_radius_48htype1_48htype2 = (max_mapping_radius + max_mapping_radius_48htype2) / 2
#     # file_ori_ref_48n24 = "Li6PS5Cl_type2.cif"
#     # file_perfect_poscar_24 = "Li6PS5Cl_24_mod_2p27291.cif" # copy this manually to folder_name_perfect_poscar  

#     folder_name_init_system = "/Init_System"
#     file_new_system = "CONTCAR"
#     file_name_toten = "toten_final.ods"
#     col_excel_geo = "geometry"
#     col_excel_path = "path"

#     if activate_radius == 2:
#         folder_name_destination_restructure = f"/restructure_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}_optimizer/"
#     elif activate_radius == 1:
#         folder_name_destination_restructure = f"/restructure_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{max_mapping_radius}_optimizer/"

#     folder_name_perfect_poscar = "/perfect_poscar/cif_matrix/Li1/"
#     cif_line_nr_start = 26  # index from 0
#     poscar_line_nr_start = 8
#     poscar_line_nr_end = 60
#     contcar_columns_type2 = ['coord_x', 'coord_y', 'coord_z']
#     amount_Li = 24
#     col_excel_toten = "toten [eV]" 
#     amount_P = 4 
#     amount_S = 20 
#     file_restructure = "CONTCAR" 
#     cif_columns = ['species', 'idx_species', 'unkownvar_1', 'coord_x', 'coord_y', 'coord_z', 'unkownvar_2'] 

#     direc_restructure_destination = direc+str(folder_name_destination_restructure)
#     direc_perfect_poscar = direc+str(folder_name_perfect_poscar)
#     path_perfect_poscar_24 = os.path.join(direc_perfect_poscar, file_perfect_poscar_24)
#     direc_init_system = direc+str(folder_name_init_system)

#     dtype = {col_excel_geo: float, col_excel_path: float}

#     data_toten = pd.read_excel(file_name_toten, dtype=dtype, engine="odf")
#     data_toten_ori = data_toten
#     data_toten = data_toten.sort_values(by=["geometry","path"],ignore_index=True,ascending=False)

#     # just refreshing folder
#     Operation.File.check_folder_existance(direc_restructure_destination)

#     # path_perfect_poscar_48n24 = modif_dx_dz_cif(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)
#     path_perfect_poscar_48n24 = modif_dx_dz_cif_allvariables(direc_perfect_poscar, file_path_ori_ref_48n24, dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, var_optitype) # os.path.join(direc_perfect_poscar, file_perfect_poscar_48n24)

#     # just copy file
#     # Operation.File.copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_perfect_poscar_24, prefix=None)
#     # !!! had to copy file_ori_ref_48n24 into Li1
#     Operation.File.copy_rename_single_file(direc_restructure_destination, direc_perfect_poscar, file_ori_ref_48n24, prefix=None)

#     # file_loc_mask_1, file_loc_important_cols = Orientation.calculate(file_loc, direc_restructure_destination, file_restructure, path_perfect_poscar_24, col_excel_toten, orientation="False")

#     Operation.File.copy_rename_files(file_loc_important_cols, direc_restructure_destination, file_restructure, prefix=None, savedir = True)


#     # # var_c = "trf_w_linalg_orientated"
#     # # Orientation.get_structure_with_linalg_orientated(file_loc_important_cols, direc_restructure_destination, file_restructure, var_c)


#     # # var_name_in = "trf_w_linalg_orientated"
#     # # var_name_out = "trf_w_linalg_orientated_positive"
#     # # n_decimal = 8
#     # # get_orientated_positive_lessthan1_cif(file_loc_important_cols, direc_restructure_destination, cif_line_nr_start, cif_columns, var_name_in, var_name_out, n_decimal)

#     # max_mapping_radius = 0.05282658993283027
#     # max_mapping_radius = 0.045
#     # max_mapping_radius = 0.055
#     # max_mapping_radius = 0.04197083906
#     ref_structure_48n24 = Structure.from_file(path_perfect_poscar_48n24)

#     coor_structure_init_dict = ReadStructure.Coor.get_coor_structure_init_dict(ref_structure_48n24)
#     PreProcessingCONTCAR.get_positive_lessthan1_poscarorcontcar(file_loc_important_cols, direc_restructure_destination, poscar_line_nr_start, poscar_line_nr_end, contcar_columns_type2, file_type = "CONTCAR", var_name_in = None, var_name_out = "positive", n_decimal=16)
#     ReadStructure.Coor.get_coor_structure_init_dict_wholedataframe(file_loc_important_cols, mapping = "False")

#     # if activate_radius == 3:
#     #     get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
#     #     get_flag_map_weirdos_48htype1_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype1_48htype2)
#     #     get_flag_map_weirdos_48htypesmerged_level1_el(file_loc_important_cols, "Li")
#     #     get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
#     #     get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
#     if activate_radius == 2:
#         get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)
#         get_flag_map_weirdos_48htype2_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius_48htype2, activate_radius)
#         get_flag_map_48htypesmerged_el(file_loc_important_cols, "Li", activate_radius)
#     elif activate_radius == 1:
#         get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Li", max_mapping_radius)

#     # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "P", max_mapping_radius)
#     # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "S", max_mapping_radius)
#     # get_flag_map_weirdos_el(file_loc_important_cols, coor_structure_init_dict, "Cl", max_mapping_radius)

#     Mapping.AtomIndexing.get_idx_weirdos_el(file_loc_important_cols, "Li", activate_radius)

#     # Mapping.AtomIndexing.idx_correcting_mapped_el(file_loc_important_cols, el="Li")
#     Mapping.AtomIndexing.idx_correcting_mapped_el(file_loc_important_cols, "Li", activate_radius)
#     # # Mapping.OutputCIF.create_combine_structure(file_loc_important_cols, direc_restructure_destination, amount_Li, amount_P, amount_S, var_savefilename = "mapLi")
    
#     Mapping.Labelling.get_distance_weirdos_label_el(file_loc_important_cols, coor_structure_init_dict, "Li", litype)

#     # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","coor_weirdos_48htypesmerged_Li","top3_dist_weirdos_dict_Li","idx0_weirdos_Li","#weirdos_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#     # file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_weirdos_Li","sum_mapped_48htype2_Li_new","#weirdos_Li","sum_mapped_48htypesmerged_Li_new","sum_sanitycheck_48htypesmerged_Li_new","idx0_weirdos_Li","top3_sorted_coorweirdo_dist_label_coorreference_idxweirdo_idxreference_Li","duplicate_closest24_w_data_Li","duplicate_closest24_w_data_48htype2_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)

#     # if activate_radius == 3:
#     #     file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","sum_weirdos_Li","sum_mapped_48htype1_48htype2_Li_closestduplicate","sum_weirdos_48htype1_48htype2_Li","sum_mapped_48htype2_Li_closestduplicate","#weirdos_Li","sum_mapped_48htypesmerged_Li","sum_sanitycheck_48htypesmerged_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#     #     file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","sum_weirdos_Li","sum_mapped_48htype1_48htype2_Li_closestduplicate","sum_weirdos_48htype1_48htype2_Li","sum_mapped_48htype2_Li_closestduplicate","#weirdos_Li","sum_mapped_48htypesmerged_Li","sum_sanitycheck_48htypesmerged_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","toten [eV]"]]

#     #     sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())

#     #     var_excel_file = f"{int(sum_weirdos_Li)}_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dz_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dz_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}_{max_mapping_radius_48htype1_48htype2}"
    
#     if activate_radius == 2:
#         if litype == 0:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 1:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 2:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 3:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 4:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 5:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]]
        
#         sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())

#         var_excel_file = f"{int(sum_weirdos_Li)}_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dz_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dz_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}_{max_mapping_radius_48htype2}"

#     elif activate_radius == 1:
#         if litype == 0:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 1:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 2:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 3:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 4:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_24g_Li","toten [eV]"]]
#         elif litype == 5:
#             file_loc_important_cols_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]].sort_values("toten [eV]", ascending=True)
#             file_loc_important_cols_not_sorted_toten = file_loc_important_cols[["geometry","path","sum_mapped_Li_closestduplicate","#weirdos_Li","idx0_weirdos_Li","top3_sorted_idxweirdo_dist_Li","top3_sorted_idxweirdo_label_Li","#closest_48htype1_Li","#closest_48htype2_Li","#closest_48htype3_Li","#closest_48htype4_Li","#closest_48htype5_Li","#closest_24g_Li","toten [eV]"]]

#         sum_weirdos_Li = float(file_loc_important_cols_sorted_toten["#weirdos_Li"].sum())

#         var_excel_file = f"{int(sum_weirdos_Li)}_{formatted_dx1_48h_type1}_{formatted_dx2_48h_type1}_{formatted_dz_48h_type1}_{formatted_dx1_48h_type2}_{formatted_dx2_48h_type2}_{formatted_dz_48h_type2}_{formatted_dx_24g}_{formatted_dz1_24g}_{formatted_dz2_24g}_{max_mapping_radius}"

#     path_excel_file = os.path.join(direc_perfect_poscar, f'04_outputs_{var_excel_file}_{var_optitype}.xlsx')
#     file_loc_important_cols_sorted_toten.to_excel(path_excel_file, index=False)

#     return sum_weirdos_Li


# def varying_radius_vs_sumweirdosLi(dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, max_mapping_radius, max_mapping_radius_48htype2, delta_radius, n_sample, var_optitype):
    
#     col_radius_type1 = "radius_type1"
#     col_radius_type2 = "radius_type2"
#     col_sumweirdosLi = "sumweirdosLi"

#     radius_sumweirdosLi_df = pd.DataFrame()
#     radius_sumweirdosLi_df[col_radius_type1] = None
#     radius_sumweirdosLi_df[col_radius_type2] = None
#     radius_sumweirdosLi_df[col_sumweirdosLi] = None

#     idx_sumweirdosLi_df = 0
#     forward_max_mapping_radius = max_mapping_radius
#     forward_max_mapping_radius_48htype2 = max_mapping_radius_48htype2
#     backward_max_mapping_radius = max_mapping_radius
#     backward_max_mapping_radius_48htype2 = max_mapping_radius_48htype2

#     for i in range(int(n_sample/2)):
#         forward_max_mapping_radius += i * delta_radius
#         forward_max_mapping_radius_48htype2 += i * delta_radius
    
#         sum_weirdos_Li = get_sum_weirdos_Li_w_radius(dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, forward_max_mapping_radius, forward_max_mapping_radius_48htype2, var_optitype)

#         radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_radius_type1] = forward_max_mapping_radius
#         radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_radius_type2] = forward_max_mapping_radius_48htype2
#         radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_sumweirdosLi] = sum_weirdos_Li
#         print(idx_sumweirdosLi_df)

#     # for j in range(int(n_sample/2)):
#     #     backward_max_mapping_radius -= j * delta_radius
#     #     backward_max_mapping_radius_48htype2 -= j * delta_radius
    
#     #     sum_weirdos_Li = get_sum_weirdos_Li_w_radius(dx1_48h_type1, dx2_48h_type1, dz_48h_type1, dx1_48h_type2, dx2_48h_type2, dz_48h_type2, dx_24g, dz1_24g, dz2_24g, backward_max_mapping_radius, backward_max_mapping_radius_48htype2, var_optitype)

#     #     radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_radius_type1] = backward_max_mapping_radius
#     #     radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_radius_type2] = backward_max_mapping_radius_48htype2
#     #     radius_sumweirdosLi_df.at[idx_sumweirdosLi_df, col_sumweirdosLi] = sum_weirdos_Li
#     #     print(idx_sumweirdosLi_df)

#     return radius_sumweirdosLi_df


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


# def get_triads_movement(destination_directory, geo, var_filename, filename_ref_72):
#     # df_coor = pd.DataFrame()
#     df_triad = pd.DataFrame()
#     df_ratio = pd.DataFrame()
#     df_dist = pd.DataFrame()

#     df_dist["dist"] = None
#     # coor_Li_ref = []

#     # col_xyz_coor = "xyz_coor"

#     # df_coor[col_xyz_coor] = None

#     if geo == 0:
#         path_geo = path_geo_0
#     elif geo == 1:
#         path_geo = path_geo_1
#     elif geo == 2:
#         path_geo = path_geo_2
#     elif geo == 3:
#         path_geo = path_geo_3
#     elif geo == 4:
#         path_geo = path_geo_4
#     elif geo == 5:
#         path_geo = path_geo_5
#     elif geo == 6:
#         path_geo = path_geo_6
#     elif geo == 7:
#         path_geo = path_geo_7
#     elif geo == 8:
#         path_geo = path_geo_8

#     file_ref_24 = f"{geo}_0_{var_filename}.cif"
#     file_path_ref_24 = os.path.join(destination_directory, file_ref_24)

#     file_ref_72 = f"{filename_ref_72}.cif"
#     file_path_ref_72 = os.path.join(destination_directory, file_ref_72)


#     idx_coor_Li_dict_ref_24 = get_idx_coor_Li_dict(file_path_ref_24)    # key is the pointer to 24
#     idx_coor_Li_dict_ref_72 = get_idx_coor_Li_dict(file_path_ref_72)    # key is the pointer to 24

#     idx_coor_Li_dict_ref_triad = get_idx_coor_Li_dict_ref_triad(idx_coor_Li_dict_ref_24, idx_coor_Li_dict_ref_72)
#     # idxs_Li_ref_24 = list(idx_coor_Li_dict_ref_24.keys())
#     # idxs_Li_ref_72 = list(idx_coor_Li_dict_ref_72.keys())

#     for i in path_geo:
#         # coor_Li = []
#         file = f"{geo}_{i}_{var_filename}.cif"
#         file_path = os.path.join(destination_directory, file)

#         idx_coor_Li_dict = get_idx_coor_Li_dict(file_path)
#         # idxs_Li = list(idx_coor_Li_dict.keys())

#         # # idx_coor_Li_triad_belonging_initial = defaultdict(list)
#         # # idx_coor_Li_triad_belonging_initial_centroid = defaultdict(list)

#         ### does the numeration of Li is important?
#         ### 1) check which triad it does belong to initially

#         idx_coor_Li_idx_centroid_triad_ref = get_idx_coor_Li_idx_centroid_triad(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_ref_24)
#         idx_coor_Li_idx_centroid_triad = get_idx_coor_Li_idx_centroid_triad(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict)
#         idxs_Li_dict = [i for i in range(24) if i in idx_coor_Li_idx_centroid_triad.keys()]

#         ## get ratio of 24:48
#         counter_48 = 0
#         for Li_idx, val in idx_coor_Li_idx_centroid_triad.items():
#             # print(Operation.Distance.mic_eucledian_distance(val['coor'], val['centroid_triad']))
#             if val['structure'] == 48:
#                 counter_48 = counter_48 + 1
#         # print(f"path {i} has ratio of 48 of: {counter_48/len(idx_coor_Li_idx_centroid_triad)}")
#         df_ratio.at[i, "ratio of 48"] = counter_48/len(idx_coor_Li_idx_centroid_triad)

#         ## get li-to-li-distance 
#         dist_ascending, sorted_coors_Li_dist_structures = get_dist_ascending(idx_coor_Li_idx_centroid_triad)
#         # print(dist_ascending)
#         df_dist.at[i, "dist"] = dist_ascending[1:6]

#         for j in idxs_Li_dict:
#             # df_triad.at[i, f"{j}"] = None  

#             triad = idx_coor_Li_idx_centroid_triad[j]["idx_triad"]

#             df_triad.at[i, f"{j}"] = triad

#             if triad == df_triad.at[0, f"{j}"] and i != 0:
#                 print(f"path: {i}, Li: {j}, triad: {triad}")

#     return df_triad, df_ratio, df_dist, sorted_coors_Li_dist_structures



# def get_triads_fullness(destination_directory, geo, var_filename, filename_ref_72):
#     # df_idx_triad_counts = pd.DataFrame #(np.zeros((24, 1)))
#     # df_idx_triad_counts["idx_triad_counts"] = None

#     idx_coor_Li_idx_centroid_triad_weirdos_appended_dict = defaultdict(list)

#     if geo == 0:
#         path_geo = path_geo_0
#     elif geo == 1:
#         path_geo = path_geo_1
#     elif geo == 2:
#         path_geo = path_geo_2
#     elif geo == 3:
#         path_geo = path_geo_3
#     elif geo == 4:
#         path_geo = path_geo_4
#     elif geo == 5:
#         path_geo = path_geo_5
#     elif geo == 6:
#         path_geo = path_geo_6
#     elif geo == 7:
#         path_geo = path_geo_7
#     elif geo == 8:
#         path_geo = path_geo_8

#     df_idx_triad_counts = pd.DataFrame(np.zeros((24, len(path_geo))))

#     file_ref_24 = f"{geo}_0_{var_filename}.cif"
#     file_path_ref_24 = os.path.join(destination_directory, file_ref_24)

#     file_ref_72 = f"{filename_ref_72}.cif"
#     file_path_ref_72 = os.path.join(destination_directory, file_ref_72)

#     idx_coor_Li_dict_ref_24 = get_idx_coor_Li_dict(file_path_ref_24)    # key is the pointer to 24
#     idx_coor_Li_dict_ref_72 = get_idx_coor_Li_dict(file_path_ref_72)    # key is the pointer to 24

#     idx_coor_Li_dict_ref_triad = get_idx_coor_Li_dict_ref_triad(idx_coor_Li_dict_ref_24, idx_coor_Li_dict_ref_72)
#     # idxs_Li_ref_24 = list(idx_coor_Li_dict_ref_24.keys())
#     # idxs_Li_ref_72 = list(idx_coor_Li_dict_ref_72.keys())

#     for i in path_geo:
#         # coor_Li = []
#         file = f"{geo}_{i}_{var_filename}.cif"
#         file_path = os.path.join(destination_directory, file)

#         idx_coor_Li_dict = get_idx_coor_Li_dict(file_path)

#         file_weirdos_appended = f"{geo}_{i}_{var_filename}_weirdos_appended.cif"
#         file_path_weirdos_appended = os.path.join(destination_directory, file_weirdos_appended)

#         idx_coor_Li_dict_weirdos_appended = get_idx_coor_Li_dict(file_path_weirdos_appended)
#         # idx_coor_Li_dict_ref_triad_weirdos_appended = get_idx_coor_Li_dict_ref_triad(idx_coor_Li_dict_weirdos_appended, idx_coor_Li_dict_ref_72)

#         idxs_Li = list(idx_coor_Li_dict.keys())
#         idxs_Li_not = sorted(i for i in range(24) if i not in idxs_Li)
#         # idxs_Li = list(idx_coor_Li_dict.keys())

#         # # idx_coor_Li_triad_belonging_initial = defaultdict(list)
#         # # idx_coor_Li_triad_belonging_initial_centroid = defaultdict(list)

#         ### does the numeration of Li is important?
#         ### 1) check which triad it does belong to initially

#         idx_coor_Li_idx_centroid_triad_ref = get_idx_coor_Li_idx_centroid_triad(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_ref_24)
#         idx_coor_Li_idx_centroid_triad = get_idx_coor_Li_idx_centroid_triad_w_closest_dist(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict)
#         idx_coor_Li_idx_centroid_triad_weirdos_appended = get_idx_coor_Li_idx_centroid_triad_w_closest_dist_weirdos_appended(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_weirdos_appended, idxs_Li_not)
#         # idx_coor_Li_idx_centroid_triad_weirdos_appended = get_idx_coor_Li_idx_centroid_triad_w_closest_dist(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_weirdos_appended)
#         idxs_Li_dict = [i for i in range(24) if i in idx_coor_Li_idx_centroid_triad.keys()]
        
#         idx_triad_array = sorted([val['idx_triad'] for val in idx_coor_Li_idx_centroid_triad.values()])
#         idx_triad_array_not = sorted(i for i in range(24) if i not in idx_triad_array)
#         # idxs_Li_triad_dict = [i for i in range(24) if i in idx_coor_Li_idx_centroid_triad()]

#         # idx_triad_series = pd.Series(idx_triad_array)
#         # df_idx_triad_counts[i] = idx_triad_series.value_counts()

#         idx_triad_counts = defaultdict(int)
#         # Count the occurrences of each idx_triad
#         # idx_triad_counts = pd.DataFrame(np.zeros((24, 1)))
#         for key, val in idx_coor_Li_idx_centroid_triad.items():
#             idx_triad = val['idx_triad']
#             idx_triad_counts[idx_triad] += 1
#         for j in idx_triad_array_not:
#             idx_triad_counts[j] = 0

#         # df_idx_triad_counts.at[i, "idx_triad_counts"] = dict(idx_triad_counts)
#         # df_idx_triad_counts = pd.DataFrame(np.zeros((24, 1)))
#         df_idx_triad_counts[i] = dict(idx_triad_counts)
#         # df_idx_triad_counts[i].fillna(0)

#         idx_coor_Li_idx_centroid_triad_weirdos_appended_dict[i] = dict(idx_coor_Li_idx_centroid_triad_weirdos_appended)

#     return df_idx_triad_counts, idx_coor_Li_idx_centroid_triad_weirdos_appended_dict


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


# def weighing_movement(dataframe, litype):
#     col_idx_coor24li_tuple_cage_belongin_complete_closest = "idx_coor24li_tuple_cage_belongin_complete_closest"
#     col_idx_coor24li_tuple_cage_belongin_complete_closest_weight = "idx_coor24li_tuple_cage_belongin_complete_closest_weight"

#     dataframe[col_idx_coor24li_tuple_cage_belongin_complete_closest_weight] = [{} for _ in range(len(dataframe.index))]

#     multiplicator = litype + 2

#     # TO DO: to be refined with different litype
#     if litype == 4:
#         weight_24g = 0
#         weight_48htype4 = 1
#         weight_48htype2 = 2
#         weight_48htype3 = 3
#         weight_48htype1 = 4
#         weight_weirdos = 5 

#     for idx in range(dataframe["geometry"].size):
#         # dict_weighted = defaultdict(list)

#         idx_coor24li_tuple_cage_belongin_complete_closest = dataframe[col_idx_coor24li_tuple_cage_belongin_complete_closest][idx]

#         idx_coor24li_tuple_cage_belongin_complete_closest_weight = defaultdict(list)

#         for key_a, val_a in idx_coor24li_tuple_cage_belongin_complete_closest.items():
#             idx_li = key_a
#             coor_li_mapped = val_a['coor']
#             type = val_a['type']
#             idx_tuple = val_a['idx_tuple']
#             idx_cage = val_a['idx_cage']

#             if type == "24g":
#                 weighted_type = weight_24g
#             elif type == "48htype4":
#                 weighted_type = weight_48htype4
#             elif type == "48htype2":
#                 weighted_type = weight_48htype2
#             elif type == "48htype3":
#                 weighted_type = weight_48htype3
#             elif type == "48htype1":
#                 weighted_type = weight_48htype1
#             elif type == "weirdos":
#                 weighted_type = weight_weirdos
#             else:
#                 print("wrong type")
            
#             weight = idx_tuple * multiplicator + weighted_type
        
#             idx_coor24li_tuple_cage_belongin_complete_closest_weight[idx_li] = {'coor': coor_li_mapped, 'type': type, 'idx_tuple': idx_tuple, 'weight':weight, 'idx_cage': idx_cage}

#         dataframe.at[idx, col_idx_coor24li_tuple_cage_belongin_complete_closest_weight] = idx_coor24li_tuple_cage_belongin_complete_closest_weight


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


# def get_idxs_val(df, val):
#     idxs = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1]) if df.iloc[row, col] == val]
#     return idxs


# def get_key_Li_idx(dict, path, idx_triad):
#     idxs_li = [key for key, value in dict[path].items() if value.get('idx_triad') == idx_triad]
#     return idxs_li

# def get_idx_coor_Li_dict_ref_triad(idx_coor_Li_dict_ref_24, idx_coor_Li_dict_ref_72):
#     idxs_Li_ref_24 = list(idx_coor_Li_dict_ref_24.keys())
#     idxs_Li_ref_72 = list(idx_coor_Li_dict_ref_72.keys())

#     idx_coor_Li_dict_ref_triad = defaultdict(list)

#     for key_72, coor_72 in idx_coor_Li_dict_ref_72.items():
#         for key_24, coor_24 in idx_coor_Li_dict_ref_24.items():
#             distance = Operation.Distance.mic_eucledian_distance(coor_72, coor_24)
#             if distance == 0:
#                 idx_coor_Li_dict_ref_triad[key_24].append(coor_72)

#     for key_72, coor_72 in idx_coor_Li_dict_ref_72.items():
#         for key_24, coor_24 in idx_coor_Li_dict_ref_24.items():
#             distance = Operation.Distance.mic_eucledian_distance(coor_72, coor_24)
#             if distance <= 0.086399 and distance != 0:                      # to edit this number
#                 idx_coor_Li_dict_ref_triad[key_24].append(coor_72)
#     return idx_coor_Li_dict_ref_triad


# def get_idx_coor_Li_idx_centroid_triad_w_closest_dist_weirdos_appended(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict_weirdos_appended, idxs_Li_not):
#     idx_coor_Li_idx_centroid_triad_weirdos_appended = defaultdict(list)
#     distance_array = []
#     for key, coor in idx_coor_Li_dict_weirdos_appended.items():
#         # if key in idxs_Li_not:
#         #     idx_coor_Li_idx_centroid_triad_dict = {}    
#         #     for key_triad, coor_triad in idx_coor_Li_dict_ref_triad.items():
#         #         for coor_triad_component in coor_triad:
#         #             distance = Operation.Distance.mic_eucledian_distance(coor_triad_component, coor)
#         #             distance_array.append(distance)  
#         #     distance_array_sorted = sorted(set(distance_array))
#         #     distance_array_sorted_top3 = distance_array_sorted[0:4]
#         #     idx_coor_Li_idx_centroid_triad_dict['dist_top3'] = distance_array_sorted_top3  
#         #     if key in idx_coor_Li_idx_centroid_triad_weirdos_appended:
#         #         idx_coor_Li_idx_centroid_triad_weirdos_appended[key].append(idx_coor_Li_idx_centroid_triad_dict)
#         #     else:
#         #         idx_coor_Li_idx_centroid_triad_weirdos_appended[key] = idx_coor_Li_idx_centroid_triad_dict         
#         # else:
#         idx_coor_Li_idx_centroid_triad_dict = {}    
#         for key_triad, coor_triad in idx_coor_Li_dict_ref_triad.items():
#             for coor_triad_component in coor_triad:
#                 distance = Operation.Distance.mic_eucledian_distance(coor_triad_component, coor)
#                 distance_array.append(distance)
#                 if distance == 0:
#                     idx_coor_Li_idx_centroid_triad_dict['coor'] = coor
#                     idx_coor_Li_idx_centroid_triad_dict['idx_triad'] = key_triad
#                     idx_coor_Li_idx_centroid_triad_dict['centroid_triad'] = coor_triad[0]
#                     # check if it's at 24g or 48h
#                     if coor == coor_triad[0]:
#                         idx_coor_Li_idx_centroid_triad_dict['structure'] = 24
#                     else:
#                         idx_coor_Li_idx_centroid_triad_dict['structure'] = 48           
#         distance_array_sorted = sorted(set(distance_array))
#         distance_array_sorted_top3 = distance_array_sorted[0:4]
#         # idx_coor_Li_idx_centroid_triad_dict['dist_top3'] = distance_array_sorted_top3
#         if key in idx_coor_Li_idx_centroid_triad_weirdos_appended:
#             idx_coor_Li_idx_centroid_triad_weirdos_appended[key].append(idx_coor_Li_idx_centroid_triad_dict)
#         else:
#             idx_coor_Li_idx_centroid_triad_weirdos_appended[key] = idx_coor_Li_idx_centroid_triad_dict
#     return idx_coor_Li_idx_centroid_triad_weirdos_appended


# def get_idx_coor_Li_idx_centroid_triad_w_closest_dist(idx_coor_Li_dict_ref_triad, idx_coor_Li_dict):
#     idx_coor_Li_idx_centroid_triad = defaultdict(list)
#     distance_array = []
#     for key, coor in idx_coor_Li_dict.items():
#         idx_coor_Li_idx_centroid_triad_dict = {}    
#         for key_triad, coor_triad in idx_coor_Li_dict_ref_triad.items():
#             for coor_triad_component in coor_triad:
#                 distance = Operation.Distance.mic_eucledian_distance(coor_triad_component, coor)
#                 distance_array.append(distance)
#                 if distance == 0:
#                     idx_coor_Li_idx_centroid_triad_dict['coor'] = coor
#                     idx_coor_Li_idx_centroid_triad_dict['idx_triad'] = key_triad
#                     idx_coor_Li_idx_centroid_triad_dict['centroid_triad'] = coor_triad[0]
#                     # check if it's at 24g or 48h
#                     if coor == coor_triad[0]:
#                         idx_coor_Li_idx_centroid_triad_dict['structure'] = 24
#                     else:
#                         idx_coor_Li_idx_centroid_triad_dict['structure'] = 48                    
#         distance_array_sorted = sorted(distance_array)
#         distance_array_sorted_top3 = distance_array_sorted[1:4]
#         idx_coor_Li_idx_centroid_triad_dict['dist_top3'] = distance_array_sorted_top3
#         # idx_coor_Li_idx_centroid_triad_dict['dist'] = distance_array_sorted
#         if key in idx_coor_Li_idx_centroid_triad:
#             idx_coor_Li_idx_centroid_triad[key].append(idx_coor_Li_idx_centroid_triad_dict)
#         else:
#             idx_coor_Li_idx_centroid_triad[key] = idx_coor_Li_idx_centroid_triad_dict
#     return idx_coor_Li_idx_centroid_triad


# def get_dist_ascending(idx_coor_Li_idx_centroid_triad):
#     coors_Li_dist_structures = defaultdict(list)

#     for Li_idx_temp1, val_temp1 in idx_coor_Li_idx_centroid_triad.items():
#         coors_Li_dist_structures_dict = {}
#         for Li_idx_temp2, val_temp2 in idx_coor_Li_idx_centroid_triad.items():
#             distance = Operation.Distance.mic_eucledian_distance(val_temp1['coor'], val_temp2['coor'])
#             coors_Li_dist_structures_dict['coors'] = (val_temp1['coor'], val_temp2['coor'])
#             coors_Li_dist_structures_dict['dist'] = distance
#             coors_Li_dist_structures_dict['structures'] = (val_temp1['structure'], val_temp2['structure'])

#             key = (Li_idx_temp1, Li_idx_temp2)
#             if key in coors_Li_dist_structures:
#                 coors_Li_dist_structures[key].append(coors_Li_dist_structures_dict)
#             else:
#                 coors_Li_dist_structures[key] = coors_Li_dist_structures_dict

#     sorted_coors_Li_dist_structures = dict(sorted(coors_Li_dist_structures.items(), key=lambda item: item[1]['dist']))
#     dist_ascending = list({val['dist'] for idx, val in sorted_coors_Li_dist_structures.items()})

#     return dist_ascending, sorted_coors_Li_dist_structures


# # def get_idx_coor_Li_dict(file_path):
# #     with open(file_path, 'r') as f:
# #         content = f.read()

# #     # Initialize a dictionary to store the data
# #     Li_idx_coor_dict = {}

# #     # Use regular expressions to extract Li indices and coordinates
# #     li_pattern = re.compile(r'Li\s+Li(\d+)\s+1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
# #     matches = li_pattern.findall(content)

# #     # Iterate through the matches and populate the dictionary
# #     for match in matches:
# #         index = int(match[0])
# #         x = float(match[1])
# #         y = float(match[2])
# #         z = float(match[3])
# #         Li_idx_coor_dict[index] = (x, y, z)

# #     return Li_idx_coor_dict


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################


class Plot:
    class StructureAnalysis:
        def energy_vs_latticeconstant(dataframe, var_filename):
            col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"
            col_toten = "toten [eV]"

            for idx in range(dataframe["geometry"].size):
                latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict]
                toten = dataframe.at[idx, col_toten]
                
                a = latticeconstant_structure_dict["a"]

                plt.scatter(a, toten)
            
            plt.title("Lattice constant vs Total energy")
            plt.xlabel("Lattice constant [Å]")
            plt.ylabel("Total energy [eV]")
            plt.show()


        def weirdos_directcoor(dataframe, activate_radius):
            if activate_radius == 2 or activate_radius == 3:
                col_coor_weirdos_el = f"coor_weirdos_48htype2_Li"
            elif activate_radius == 1:
                col_coor_weirdos_el = f"coor_weirdos_Li"

            # Create a figure and a 3D axis
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for idx in range(dataframe["geometry"].size):
                coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]

                # Plot each set of coordinates in the loop
                for coordinates in coor_weirdos_el:
                    ax.scatter(*coordinates, marker='o')

            # Set labels
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            ax.set_title('Weirdos direct coordinate')

            # Show the plot
            plt.show()


    class Mapping:
        class Labelling:
            def plot_amount_type(dataframe, litype, el, style, category_labels = None):
                """
                    style: scatter, bar
                """
                col_amount_type_el = f"amount_type_{el}"

                df = pd.DataFrame()
                df['idx_file'] = None

                if litype == 0:
                    df['24g'] = None; df['weirdo'] = None
                elif litype == 1:
                    df['48htype1'] = None; df['24g'] = None; df['weirdo'] = None
                elif litype == 2:
                    df['48htype1'] = None; df['48htype2'] = None; df['24g'] = None; df['weirdo'] = None
                elif litype == 3:
                    df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['24g'] = None; df['weirdo'] = None
                elif litype == 4:
                    df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['24g'] = None; df['weirdo'] = None
                elif litype == 5:
                    df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['24g'] = None; df['weirdo'] = None
                elif litype == 6:
                    df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['24g'] = None; df['weirdo'] = None
                elif litype == 7:
                    df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['48htype7'] = None; df['24g'] = None; df['weirdo'] = None
                elif litype == 8:
                    df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['48htype7'] = None; df['48htype8'] = None; df['24g'] = None; df['weirdo'] = None

                for idx in range(dataframe["geometry"].size):

                    amount_type = dataframe.at[idx, col_amount_type_el]
                    df.at[idx, 'idx_file'] = idx

                    if litype == 0:
                        df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
                    elif litype == 1:
                        df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
                    elif litype == 2:
                        df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
                    elif litype == 3:
                        df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
                    elif litype == 4:
                        df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
                    elif litype == 5:
                        df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
                    elif litype == 6:
                        df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
                    elif litype == 7:
                        df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '48htype7'] = amount_type['48htype7']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
                    elif litype == 8:
                        df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '48htype7'] = amount_type['48htype7']; df.at[idx, '48htype8'] = amount_type['48htype8']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']

                wide_df = pd.DataFrame(df)

                long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

                if category_labels:
                    # # long_df['category'] = long_df['category'].replace(category_labels)
                    long_df['category'] = Operation.String.replace_values_in_series(long_df['category'], category_labels)

                if style == "bar":
                    fig = px.bar(long_df, x="idx_file", y="count", color="category", title="Idx file vs Li type")
                elif style == "scatter":
                    fig = px.scatter(long_df, x="idx_file", y="count", color="category", title="Idx file vs Li type")
                fig.show()

                return df


            def plot_mapped_label_vs_dist_and_histogram(dataframe, litype, category_data, el):
                """
                    category_data: mapping, weirdo
                    TO DO: correct the map for histogram
                """
                dist_weirdos_el_appendend = []
                label_weirdos_el_appended = []
                idx_appended = []

                if category_data == "mapping":
                    col_dist_label_el = f"atom_mapping_{el}_w_dist_label"
                elif category_data == "weirdo":
                    col_dist_label_el = f'top1_sorted_idxweirdo_dist_label_{el}'
                    col_top1_sorted_idxweirdo_coor_el = f"top1_sorted_idxweirdo_coor_{el}"
                    coor_weirdos_el_appended = []    

                for idx in range(dataframe["geometry"].size):
                    atom_mapping_el_w_dist_label = dataframe.at[idx, col_dist_label_el]
                    if category_data == "weirdo":
                        coor_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_coor_el].values()

                    for i in atom_mapping_el_w_dist_label.values():
                        if category_data == "mapping":
                            dist = i['dist']
                            label = i['label']
                        elif category_data == "weirdo":
                            dist = i[0]['dist']
                            label = i[0]['label']

                        dist_weirdos_el_appendend.append(dist)
                        label_weirdos_el_appended.append(label)
                        idx_appended.append(idx)

                    if category_data == "weirdo":
                        for single_coor in coor_weirdos_el:
                            coor_weirdos_el_appended.append(single_coor[0])

                if category_data == "mapping":
                    df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'idx_nr': idx_appended})
                elif category_data == "weirdo":
                    df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'idx_nr': idx_appended, 'coor': coor_weirdos_el_appended})

                fig = px.scatter(df, 
                                x = 'label', 
                                y = 'dist',
                                title = 'Mapped atom type vs its distance'
                                )

                fig.show(config={'scrollZoom': True})

                # if litype == 0:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                # elif litype == 1:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                        
                #     df_48htype1 = df.loc[df['label'] == '48htype1']
                #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype1')
                # elif litype == 2:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                        
                #     df_48htype1 = df.loc[df['label'] == '48htype1']
                #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype1')

                #     df_48htype2 = df.loc[df['label'] == '48htype2']
                #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype2')
                # elif litype == 3:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                        
                #     df_48htype1 = df.loc[df['label'] == '48htype1']
                #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype1')

                #     df_48htype2 = df.loc[df['label'] == '48htype2']
                #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype2')

                #     df_48htype3 = df.loc[df['label'] == '48htype3']
                #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype3')
                # elif litype == 4:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                        
                #     df_48htype1 = df.loc[df['label'] == '48htype1']
                #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype1')

                #     df_48htype2 = df.loc[df['label'] == '48htype2']
                #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype2')

                #     df_48htype3 = df.loc[df['label'] == '48htype3']
                #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype3')

                #     df_48htype4 = df.loc[df['label'] == '48htype4']
                #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype4')
                # elif litype == 5:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                        
                #     df_48htype1 = df.loc[df['label'] == '48htype1']
                #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype1')

                #     df_48htype2 = df.loc[df['label'] == '48htype2']
                #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype2')

                #     df_48htype3 = df.loc[df['label'] == '48htype3']
                #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype3')

                #     df_48htype4 = df.loc[df['label'] == '48htype4']
                #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype4')

                #     df_48htype5 = df.loc[df['label'] == '48htype5']
                #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype5')
                # elif litype == 6:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                        
                #     df_48htype1 = df.loc[df['label'] == '48htype1']
                #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype1')

                #     df_48htype2 = df.loc[df['label'] == '48htype2']
                #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype2')

                #     df_48htype3 = df.loc[df['label'] == '48htype3']
                #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype3')

                #     df_48htype4 = df.loc[df['label'] == '48htype4']
                #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype4')

                #     df_48htype5 = df.loc[df['label'] == '48htype5']
                #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype5')

                #     df_48htype6 = df.loc[df['label'] == '48htype6']
                #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype6')
                # elif litype == 7:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                        
                #     df_48htype1 = df.loc[df['label'] == '48htype1']
                #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype1')

                #     df_48htype2 = df.loc[df['label'] == '48htype2']
                #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype2')

                #     df_48htype3 = df.loc[df['label'] == '48htype3']
                #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype3')

                #     df_48htype4 = df.loc[df['label'] == '48htype4']
                #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype4')

                #     df_48htype5 = df.loc[df['label'] == '48htype5']
                #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype5')

                #     df_48htype6 = df.loc[df['label'] == '48htype6']
                #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype6')

                #     df_48htype7 = df.loc[df['label'] == '48htype7']
                #     plt.hist(df_48htype7['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype7')
                # elif litype == 8:
                #     df_24g = df.loc[df['label'] == '24g']
                #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 24g')
                        
                #     df_48htype1 = df.loc[df['label'] == '48htype1']
                #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype1')

                #     df_48htype2 = df.loc[df['label'] == '48htype2']
                #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype2')

                #     df_48htype3 = df.loc[df['label'] == '48htype3']
                #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype3')

                #     df_48htype4 = df.loc[df['label'] == '48htype4']
                #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype4')

                #     df_48htype5 = df.loc[df['label'] == '48htype5']
                #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype5')

                #     df_48htype6 = df.loc[df['label'] == '48htype6']
                #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype6')

                #     df_48htype7 = df.loc[df['label'] == '48htype7']
                #     plt.hist(df_48htype7['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype7')

                #     df_48htype8 = df.loc[df['label'] == '48htype8']
                #     plt.hist(df_48htype8['dist'], color='lightgreen', ec='black', bins=15)
                #     plt.title('Distribution of Distances for 48htype8')

                return df


    class Movement:
        class Distance:
            def plot_distance(df_distance, max_mapping_radius, activate_shifting_x, activate_diameter_line, Li_idxs):

                diameter_24g48h = max_mapping_radius * 2

                # x = df_distance.index
                if activate_shifting_x == True:
                    x = [xi + 0.5 for xi in range(len(df_distance))]
                else:
                    x = range(len(df_distance))

                # # fig = plt.figure()
                # fig = plt.figure(figsize=(800/96, 600/96))  # 800x600 pixels, assuming 96 DPI
                # ax = plt.subplot(111)

                fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size in inches

                lines = []

                colors = ['b', 'g', 'r', 'c', 'm', 'y'] #, 'k']

                # for i in df_distance.index:
                for i in range(len(df_distance.columns)):

                    line_color = colors[i % len(colors)]  # Cycle through colors list

                    if Li_idxs == "all" or i in Li_idxs:
                        # # i = i
                        # # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
                        line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", color=line_color, linewidth=2)  # Set line width to 2 pixels
                        lines.append(line)
                        # label = f"{i}" if Li_idxs == "all" else None
                        # line, = ax.plot(x, df_distance[f"{i}"], label=label)
                        # lines.append(line)

                    # if type(Li_idxs) == list:
                    #     for j in Li_idxs:
                    #         if i == j:
                    #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
                    #             lines.append(line)

                # # ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}')
                if activate_diameter_line == True:
                    ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}', linewidth=1)  # Set line width to 1 pixel

                # plt.title(f"Geometry {geo} with d={diameter_24g48h}")
                    
                # Explicitly set x-ticks
                # ax.set_xticks(x)
                if activate_shifting_x == True:
                    ax.set_xticks([0,1,2,3,4,5,6,7,8])

                # Optionally, if you want to label each tick with the original index before adjustment:
                # # # if activate_shifting_x == True:
                # # #     ax.set_xticklabels([str(int(xi - 0.5)) for xi in x])

                # Shrink current axis's height by 10% on the bottom
                    # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])

                handles, labels = ax.get_legend_handles_labels()

                ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                        fancybox=True, shadow=True, ncol=5)

                # Enable cursor information
                mplcursors.cursor(hover=True)

                # Enable zooming with cursor
                mpldatacursor.datacursor(display='multiple', draggable=True)

                plt.show()


        class Occupancy:
            def plot_occupancy(dataframe, category_labels = None):
                col_occupancy = "occupancy"

                df = pd.DataFrame()
                df['idx_file'] = None
                df['2'] = None
                df['1'] = None
                df['0'] = None
                df['48htype1'] = None
                df['weirdo'] = None

                for idx in range(dataframe["geometry"].size):

                    occupancy = dataframe.at[idx, col_occupancy]

                    # for key, val in occupancy.items():
                    df.at[idx, 'idx_file'] = idx
                    df.at[idx, '2'] = occupancy['2']
                    df.at[idx, '1'] = occupancy['1']
                    df.at[idx, '0'] = occupancy['0']
                    df.at[idx, '48htype1'] = occupancy['48htype1']
                    df.at[idx, 'weirdo'] = occupancy['weirdo']

                wide_df = pd.DataFrame(df)

                # Convert wide format to long format
                # long_df = pd.melt(wide_df, var_name='Category', value_name='Count')
                long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

                if category_labels:
                    # # long_df['category'] = long_df['category'].replace(category_labels)
                    long_df['category'] = Operation.String.replace_values_in_series(long_df['category'], category_labels)

                fig = px.bar(long_df, x="idx_file", y="count", color="category", title="Idx of file vs Occupancy")
                fig.show()

                return df


        class TupleCage:
            # # # # def plot_cage_tuple_label(df_distance, df_type, df_idx_tuple, max_mapping_radius, litype, category_labels, activate_diameter_line, activate_relabel_s_i, Li_idxs):

            # # # #     # df_distance = df_distance.iloc[:,:amount_Li]
            # # # #     # df_type = df_type.iloc[:,:amount_Li]
            # # # #     # df_idx_tuple = df_idx_tuple.iloc[:,:amount_Li]

            # # # #     diameter_24g48h = max_mapping_radius * 2

            # # # #     x = range(len(df_distance))

            # # # #     # # fig = plt.figure()
            # # # #     # fig = plt.figure(figsize=(800/96, 600/96))  # 800x600 pixels, assuming 96 DPI
            # # # #     # ax = plt.subplot(111)

            # # # #     fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size in inches

            # # # #     lines = []
            # # # #     texts = []

            # # # #     # type_marker_mapping = {
            # # # #     #     '48htype1': 'o',
            # # # #     #     '48htype2': 's',
            # # # #     #     '48htype3': '^',
            # # # #     #     '48htype4': 'D',
            # # # #     #     'weirdos': 'X',
            # # # #     #     '24g': 'v'    
            # # # #     # }

            # # # #     # # type_marker_mapping = {
            # # # #     # #     '48htype1': ('o', 'r'),  # Example: Circle marker with red color
            # # # #     # #     '48htype2': ('s', 'g'),  # Square marker with green color
            # # # #     # #     '48htype3': ('^', 'b'),  # Triangle marker with blue color
            # # # #     # #     '48htype4': ('D', 'c'),  # Diamond marker with cyan color
            # # # #     # #     'weirdos': ('X', 'm'),   # X marker with magenta color
            # # # #     # #     '24g': ('v', 'y')        # Triangle_down marker with yellow color
            # # # #     # # }

            # # # #     type_marker_mapping = {
            # # # #         '48htype1': ('o'),  # Example: Circle marker with red color
            # # # #         '48htype2': ('s'),  # Square marker with green color
            # # # #         '48htype3': ('^'),  # Triangle marker with blue color
            # # # #         '48htype4': ('D'),  # Diamond marker with cyan color
            # # # #         'weirdos': ('X'),   # X marker with magenta color
            # # # #         '24g': ('v')        # Triangle_down marker with yellow color
            # # # #     }

            # # # #     colors = ['b', 'g', 'r', 'c', 'm', 'y'] #, 'k']  # Example color list
            # # # #     # colors = list(mcolors.CSS4_COLORS.values())
            # # # #     # colors = [color + (0.7,) for color in mcolors.CSS4_COLORS.values()]
            # # # #     # colors = mcolors
            # # # #     # names = list(colors)

            # # # #     # Define offsets for text position
            # # # #     x_offset = 0.02  # Adjust these values as needed
            # # # #     y_offset = -0.05  # Adjust these values as needed

            # # # #     # Track which labels have been added
            # # # #     added_labels = set()

            # # # #     # for i in range(24):
            # # # #     for i in range(len(df_distance.columns)):
            # # # #         if Li_idxs == "all" or i in Li_idxs:
            # # # #             column_data = df_distance[f"{i}"]
            # # # #             column_val = df_type[f"{i}"]
            # # # #             column_idx_tuple = df_idx_tuple[f"{i}"]
            # # # #             # type_val = df_type[0, i]
            # # # #             # print(type_val)

            # # # #             line_color = colors[i % len(colors)]  # Cycle through colors list
            # # # #             # # # # # # # line_color = colors[i % len(colors)] if i < len(colors) else 'black'  # Use a default color if the index exceeds available colors

            # # # #             # # # for j in x:
            # # # #             for j, (y_val, type_val, idx_tuple_val) in enumerate(zip(column_data, column_val, column_idx_tuple)):
            # # # #                 # type = column_val[j]
            # # # #                 # idx_tuple = column_idx_tuple[j]

            # # # #                 # marker_style = type_marker_mapping.get(column_val, 'o')  # Get the marker style for the type
            # # # #                 # # marker_style = type_marker_mapping.get(type, 'o')  # Get the marker style for the type
            # # # #                 # # # # # # # marker_style, marker_color = type_marker_mapping.get(type_val, ('o','k'))  # Get the marker style for the type
            # # # #                 marker_style = type_marker_mapping.get(type_val, ('o'))  # Get the marker style for the type
            # # # #                 # # # # # # ax.scatter(j, df_distance[f"{i}"][j], label=f"Type: {column_val}", marker=marker_style, s=100)
            # # # #                 # # # # # label = f"{type_val}" if type_val not in added_labels else None
            # # # #                 # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100)
            # # # #                 # # # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
            # # # #                 # # # # # ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
            # # # #                 # # # # # added_labels.add(type_val)
            # # # #                 mapped_label = category_labels.get(type_val, type_val)  # Use the original type_val if it's not found in category_labels
            # # # #                 # Use mapped_label for the label. Only add it if it's not already added.
            # # # #                 label = mapped_label if mapped_label not in added_labels else None
            # # # #                 ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color=line_color, alpha=0.5)
            # # # #                 if label:  # If a label was added, record it as added
            # # # #                     added_labels.add(mapped_label)

            # # # #                 # # # # ax.text(j, df_distance.iloc[j, i], str(int(idx_tuple_val)), color=line_color, fontsize=20)
            # # # #                 # # # # # ax.text(j, y_val, str(int(idx_tuple_val)), color=line_color, fontsize=20)
            # # # #                 # Apply offsets to text position
            # # # #                 text_x = j + x_offset * ax.get_xlim()[1]  # Adjust text x-position
            # # # #                 text_y = y_val + y_offset * ax.get_ylim()[1]  # Adjust text y-position

            # # # #                 # Check if type_val is 'weirdos' and change text color to black, else use the line color
            # # # #                 text_color = 'black' if type_val in ['weirdos', '48htype1'] else line_color

            # # # #                 print(idx_tuple_val)
            # # # #                 if idx_tuple_val == 'x':
            # # # #                     text = ax.text(text_x, text_y, idx_tuple_val, color=text_color, fontsize=18)
            # # # #                 else:
            # # # #                     # if activate_relabel_s_i:
            # # # #                     #     if litype == 4:
            # # # #                     #         if idx_tuple_val in ['48htype2', '48htype3', '48htype4']:
            # # # #                     #             text = ax.text(text_x, text_y, str(int(idx_tuple_val))+"s", color=text_color, fontsize=18)
            # # # #                     #         elif idx_tuple_val in ['48htype1']:
            # # # #                     #             text = ax.text(text_x, text_y, str(int(idx_tuple_val))+"i", color=text_color, fontsize=18)
            # # # #                     # else:
            # # # #                     text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=text_color, fontsize=18)
            # # # #                 texts.append(text)

            # # # #                 # # # # # # # # # # # # # # text = ax.text(j+x_offset, y_val+y_offset, str(int(idx_tuple_val)), color=line_color, fontsize=15)
            # # # #                 # # # # # # # # if idx_tuple_val == f' ':
            # # # #                 # # # # # # # #     text = ax.text(text_x, text_y, idx_tuple_val, color=line_color, fontsize=18)
            # # # #                 # # # # # # # # else:
            # # # #                 # # # # # # # #     text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=line_color, fontsize=18)
            # # # #                 # # # # # # # # texts.append(text)

            # # # #             # # i = i
            # # # #             # # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
            # # # #             # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", linewidth=2, marker=marker_style, markersize=10)  # Set line width to 2 pixels

            # # # #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", color=line_color, linewidth=2)  # Set line width to 2 pixels
            # # # #             # ax.text(i, value, str(int(idx_value)), color=line_color, fontsize=8)

            # # # #             lines.append(line)
            # # # #             # label = f"{i}" if Li_idxs == "all" else None
            # # # #             # line, = ax.plot(x, df_distance[f"{i}"], label=label)
            # # # #             # lines.append(line)

            # # # #         # if type(Li_idxs) == list:
            # # # #         #     for j in Li_idxs:
            # # # #         #         if i == j:
            # # # #         #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
            # # # #         #             lines.append(line)

            # # # #     adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
                
            # # # #     # # ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}')
            # # # #     if activate_diameter_line == True:
            # # # #         ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}', linewidth=1)  # Set line width to 1 pixel

            # # # #     # Set the y-axis to only show ticks at 0, 1, 2, 3
            # # # #     plt.yticks([0, 1, 2, 3])

            # # # #     # plt.title(f"Geometry {geo} with d={diameter_24g48h}")

            # # # #     # Shrink current axis's height by 10% on the bottom
            # # # #         # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
            # # # #     box = ax.get_position()
            # # # #     ax.set_position([box.x0, box.y0 + box.height * 0.1,
            # # # #                     box.width, box.height * 0.9])

            # # # #     handles, labels = ax.get_legend_handles_labels()

            # # # #     # Set marker color in legend box to black
            # # # #     # # legend_handles = [(h[0], h[1], {'color': 'black'}) for h in handles]
            # # # #     legend_handles = [(h, {'color': 'black'}) for h in handles]

            # # # #     ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
            # # # #             fancybox=True, shadow=True, ncol=5)
                
            # # # #     # # # ax.legend(handles=legend_handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
            # # # #     # # #         fancybox=True, shadow=True, ncol=5, handlelength=2, handler_map={tuple: HandlerTuple(ndivide=None)})
            # # # #     # ax.legend(handles=legend_handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
            # # # #     #         fancybox=True, shadow=True, ncol=5)

            # # # #     # Enable cursor information
            # # # #     mplcursors.cursor(hover=True)

            # # # #     # Enable zooming with cursor
            # # # #     mpldatacursor.datacursor(display='multiple', draggable=True)

            # # # #     plt.show()


            def plot_cage_tuple_label(df_distance, df_type, df_idx_tuple, max_mapping_radius, litype, category_labels, activate_diameter_line, activate_relabel_s_i, Li_idxs):

                # df_distance = df_distance.iloc[:,:amount_Li]
                # df_type = df_type.iloc[:,:amount_Li]
                # df_idx_tuple = df_idx_tuple.iloc[:,:amount_Li]

                diameter_24g48h = max_mapping_radius * 2

                x = range(len(df_distance))

                # # fig = plt.figure()
                # fig = plt.figure(figsize=(800/96, 600/96))  # 800x600 pixels, assuming 96 DPI
                # ax = plt.subplot(111)

                fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size in inches

                lines = []
                texts = []

                # type_marker_mapping = {
                #     '48htype1': 'o',
                #     '48htype2': 's',
                #     '48htype3': '^',
                #     '48htype4': 'D',
                #     'weirdos': 'X',
                #     '24g': 'v'    
                # }

                # # type_marker_mapping = {
                # #     '48htype1': ('o', 'r'),  # Example: Circle marker with red color
                # #     '48htype2': ('s', 'g'),  # Square marker with green color
                # #     '48htype3': ('^', 'b'),  # Triangle marker with blue color
                # #     '48htype4': ('D', 'c'),  # Diamond marker with cyan color
                # #     'weirdos': ('X', 'm'),   # X marker with magenta color
                # #     '24g': ('v', 'y')        # Triangle_down marker with yellow color
                # # }

                type_marker_mapping = {
                    '48htype1': ('o'),  # Example: Circle marker with red color
                    '48htype2': ('s'),  # Square marker with green color
                    '48htype3': ('^'),  # Triangle marker with blue color
                    '48htype4': ('D'),  # Diamond marker with cyan color
                    'weirdos': ('X'),   # X marker with magenta color
                    '24g': ('v')        # Triangle_down marker with yellow color
                }

                colors = ['b', 'g', 'r', 'c', 'm', 'y'] #, 'k']  # Example color list
                # colors = list(mcolors.CSS4_COLORS.values())
                # colors = [color + (0.7,) for color in mcolors.CSS4_COLORS.values()]
                # colors = mcolors
                # names = list(colors)

                # Define offsets for text position
                x_offset = 0.02  # Adjust these values as needed
                y_offset = -0.05  # Adjust these values as needed

                # Track which labels have been added
                added_labels = set()

                # for i in range(24):
                for i in range(len(df_distance.columns)):
                    if Li_idxs == "all" or i in Li_idxs:
                        column_data = df_distance[f"{i}"]
                        column_val = df_type[f"{i}"]
                        column_idx_tuple = df_idx_tuple[f"{i}"]
                        # type_val = df_type[0, i]
                        # print(type_val)

                        line_color = colors[i % len(colors)]  # Cycle through colors list
                        # # # # # # # line_color = colors[i % len(colors)] if i < len(colors) else 'black'  # Use a default color if the index exceeds available colors

                        # # # for j in x:
                        for j, (y_val, type_val, idx_tuple_val) in enumerate(zip(column_data, column_val, column_idx_tuple)):
                            # type = column_val[j]
                            # idx_tuple = column_idx_tuple[j]

                            # marker_style = type_marker_mapping.get(column_val, 'o')  # Get the marker style for the type
                            # # marker_style = type_marker_mapping.get(type, 'o')  # Get the marker style for the type
                            # # # # # # # marker_style, marker_color = type_marker_mapping.get(type_val, ('o','k'))  # Get the marker style for the type
                            marker_style = type_marker_mapping.get(type_val, ('o'))  # Get the marker style for the type
                            # # # # # # ax.scatter(j, df_distance[f"{i}"][j], label=f"Type: {column_val}", marker=marker_style, s=100)
                            # # # # # label = f"{type_val}" if type_val not in added_labels else None
                            # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100)
                            # # # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
                            # # # # # ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
                            # # # # # added_labels.add(type_val)
                            mapped_label = category_labels.get(type_val, type_val)  # Use the original type_val if it's not found in category_labels
                            # Use mapped_label for the label. Only add it if it's not already added.
                            label = mapped_label if mapped_label not in added_labels else None
                            ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color=line_color, alpha=0.5)
                            if label:  # If a label was added, record it as added
                                added_labels.add(mapped_label)

                            # # # # ax.text(j, df_distance.iloc[j, i], str(int(idx_tuple_val)), color=line_color, fontsize=20)
                            # # # # # ax.text(j, y_val, str(int(idx_tuple_val)), color=line_color, fontsize=20)
                            # Apply offsets to text position
                            text_x = j + x_offset * ax.get_xlim()[1]  # Adjust text x-position
                            text_y = y_val + y_offset * ax.get_ylim()[1]  # Adjust text y-position

                            # Check if type_val is 'weirdos' and change text color to black, else use the line color
                            text_color = 'black' if type_val in ['weirdos', '48htype1'] else line_color

                            if activate_relabel_s_i:        
                                # print(idx_tuple_val)
                                if type_val in ['48htype2', '48htype3', '48htype4', '24g']:
                                    text = ax.text(text_x, text_y, str(int(idx_tuple_val))+"s", color=text_color, fontsize=18)
                                elif type_val in ['48htype1']:
                                    text = ax.text(text_x, text_y, str(int(idx_tuple_val))+"i", color=text_color, fontsize=18)
                                elif type_val == 'weirdos':
                                    # idx_tuple_val = 'x'
                                    print(idx_tuple_val)
                                    text = ax.text(text_x, text_y, idx_tuple_val, color=text_color, fontsize=18)
                                    print(text)
                            else:
                                if idx_tuple_val == 'x':
                                    text = ax.text(text_x, text_y, idx_tuple_val, color=text_color, fontsize=18)
                                else:
                                    text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=text_color, fontsize=18)
                            texts.append(text)

                            # # # # # # # # # # # # # # text = ax.text(j+x_offset, y_val+y_offset, str(int(idx_tuple_val)), color=line_color, fontsize=15)
                            # # # # # # # # if idx_tuple_val == f' ':
                            # # # # # # # #     text = ax.text(text_x, text_y, idx_tuple_val, color=line_color, fontsize=18)
                            # # # # # # # # else:
                            # # # # # # # #     text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=line_color, fontsize=18)
                            # # # # # # # # texts.append(text)

                        # # i = i
                        # # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
                        # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", linewidth=2, marker=marker_style, markersize=10)  # Set line width to 2 pixels

                        line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", color=line_color, linewidth=2)  # Set line width to 2 pixels
                        # ax.text(i, value, str(int(idx_value)), color=line_color, fontsize=8)

                        lines.append(line)
                        # label = f"{i}" if Li_idxs == "all" else None
                        # line, = ax.plot(x, df_distance[f"{i}"], label=label)
                        # lines.append(line)

                    # if type(Li_idxs) == list:
                    #     for j in Li_idxs:
                    #         if i == j:
                    #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
                    #             lines.append(line)

                adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
                
                # # ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}')
                if activate_diameter_line == True:
                    ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}', linewidth=1)  # Set line width to 1 pixel

                # ax.set_ylim(-0.5, 3.5)

                # Set the y-axis to only show ticks at 0, 1, 2, 3
                plt.yticks([0, 1, 2, 3])

                # plt.title(f"Geometry {geo} with d={diameter_24g48h}")

                # Shrink current axis's height by 10% on the bottom
                    # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                box.width, box.height * 0.9])

                handles, labels = ax.get_legend_handles_labels()

                # Set marker color in legend box to black
                # # legend_handles = [(h[0], h[1], {'color': 'black'}) for h in handles]
                legend_handles = [(h, {'color': 'black'}) for h in handles]

                ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                        fancybox=True, shadow=True, ncol=5)
                
                # # # ax.legend(handles=legend_handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                # # #         fancybox=True, shadow=True, ncol=5, handlelength=2, handler_map={tuple: HandlerTuple(ndivide=None)})
                # ax.legend(handles=legend_handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                #         fancybox=True, shadow=True, ncol=5)

                # Enable cursor information
                mplcursors.cursor(hover=True)

                # Enable zooming with cursor
                mpldatacursor.datacursor(display='multiple', draggable=True)

                plt.show()


    class NotYetClassified:
        def plot_varying_radius_vs_sumweirdosLi(dataframe):
            col_radius_type1 = "radius_type1"
            col_radius_type2 = "radius_type2"
            col_sumweirdosLi = "sumweirdosLi"

            # # dataframe_to_float = dataframe.copy()
            # dataframe_to_float[col_radius_type1] = dataframe_to_float[col_radius_type1].apply(lambda x: float(x[0]))
            # dataframe_to_float[col_radius_type2] = dataframe_to_float[col_radius_type2].apply(lambda x: float(x[0]))
            # dataframe_to_float[col_sumweirdosLi] = dataframe_to_float[col_sumweirdosLi].apply(lambda x: float(x[0]))

            # # %matplotlib inline
            matrix_map = dataframe.pivot_table(index=col_radius_type2, columns=col_radius_type1,values=col_sumweirdosLi)  
            sns.heatmap(matrix_map)


        def plot_distweirdos(dataframe):
            col_top1_sorted_idxweirdo_dist_el = "top1_sorted_idxweirdo_dist_Li"
            col_top1_sorted_idxweirdo_label_el = "top1_sorted_idxweirdo_label_Li"
            col_top1_sorted_idxweirdo_coor_el = "top1_sorted_idxweirdo_coor_Li"
            col_top1_sorted_idxweirdo_file_el = "top1_sorted_idxweirdo_file_Li"

            val = 0.

            dist_weirdos_el_appendend = []
            y_appended = []
            label_weirdos_el_appended = []
            coor_weirdos_el_appended = []
            file_weirdos_el_appended = []

            for idx in range(dataframe["geometry"].size):
                dist_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_dist_el].values()
                label_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_label_el].values()
                coor_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_coor_el].values()
                file_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_file_el].values()

                for single_dist in dist_weirdos_el:
                    dist_weirdos_el_appendend.append(single_dist[0])
                    y_appended.append(np.zeros_like(single_dist[0]) + val)

                for single_label in label_weirdos_el:
                    label_weirdos_el_appended.append(single_label[0])

                for single_coor in coor_weirdos_el:
                    coor_weirdos_el_appended.append(single_coor[0])
                
                for single_file in file_weirdos_el:
                    file_weirdos_el_appended.append(single_file[0])

            df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'y': y_appended, 'coor': coor_weirdos_el_appended, 'file': file_weirdos_el_appended})

            # fig = px.scatter(df, x = 'dist', y = 'y', color='label', color_discrete_map={'48htype1': 'red', '48htype2': 'blue', '24g': 'green'})
            fig = px.scatter(df, 
                            x = 'dist', 
                            y = 'y', 
                            color ='label', 
                            hover_data = ['dist', 'label', 'coor', 'file'])

            fig.show(config={'scrollZoom': True})


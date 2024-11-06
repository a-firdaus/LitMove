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


# class Optimizer:
#     # class GetSumWeirdos:

#     class Position:
#         # class Modify:


#         class Output:


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


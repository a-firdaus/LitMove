import pandas as pd
import numpy as np
import os

from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter


# # not yet changed from 3665 - 4444 (???)
def get_sum_weirdos_Li_var_not_complete(max_mapping_radius, max_mapping_radius_48htype2, activate_radius, 
                                        file_perfect_poscar_24_wo_cif, file_perfect_poscar_48n24_wo_cif, 
                                        litype, var_optitype, iter_type, foldermapping_namestyle_all, 
                                        cif_namestyle_all, modif_all_litype, full_calculation):
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


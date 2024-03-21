import numpy as np
import os

from pymatgen.core.structure import Structure

from functional import calc_distance


class Cartesian:
    """
    Note: 
    - to be checked!
    """
    # def get_fractional_to_cartesian_matrix(a, b, c, alpha, beta, gamma, angle_in_degrees=True):
    def get_fractional_to_cartesian_matrix(dataframe, var_filename, angle_in_degrees=True): 
        # source: https://gist.github.com/Bismarrck/a68da01f19b39320f78a
        col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"

        col_fractional_to_cartesian_matrix = f"fractional_to_cartesian_matrix_{var_filename}"

        dataframe[col_fractional_to_cartesian_matrix] = None

        for idx in range(dataframe["geometry"].size):
            latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict] 

            a = latticeconstant_structure_dict["a"]
            b = latticeconstant_structure_dict["b"]
            c = latticeconstant_structure_dict["c"]

            alpha = latticeconstant_structure_dict["alpha"]
            beta = latticeconstant_structure_dict["beta"]
            gamma = latticeconstant_structure_dict["gamma"]
            
            if angle_in_degrees:
                alpha = np.deg2rad(alpha)
                beta = np.deg2rad(beta)
                gamma = np.deg2rad(gamma)

            cosa = np.cos(alpha)
            sina = np.sin(alpha)
            cosb = np.cos(beta)
            sinb = np.sin(beta)
            cosg = np.cos(gamma)
            sing = np.sin(gamma)
            volume = 1.0 - cosa**2.0 - cosb**2.0 - cosg**2.0 + 2.0 * cosa * cosb * cosg
            volume = np.sqrt(volume)
            r = np.zeros((3, 3))
            r[0, 0] = a
            r[0, 1] = b * cosg
            r[0, 2] = c * cosb
            r[1, 1] = b * sing
            r[1, 2] = c * (cosa - cosb * cosg) / sing
            r[2, 2] = c * volume / sing

            # print(f"type_a_alpha_r: {type(a), type(alpha), type(r)}")
            # print(f"a_alpha_r: {a, alpha, r}")
            dataframe.at[idx, col_fractional_to_cartesian_matrix] = r


    def get_fractional_to_cartesian_coor(dataframe, destination_directory, var_filename):
        col_fractional_to_cartesian_matrix = f"fractional_to_cartesian_matrix_{var_filename}"

        col_coor_structure_dict_cartesian = f"coor_structure_dict_cartesian_{var_filename}"

        dataframe[col_coor_structure_dict_cartesian] = None

        for idx in range(dataframe["geometry"].size):
            coor_origin_Li_init_cartesian = []; coor_origin_P_init_cartesian = []; coor_origin_S_init_cartesian = []; coor_origin_Cl_init_cartesian = []
            coor_structure_dict_cartesian = {}

            if var_filename == "CONTCAR":
                source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}"
            else:
                source_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_filename}.cif"
            source_filename_path = os.path.join(destination_directory, source_filename)

            new_structure = Structure.from_file(source_filename_path)

            r = dataframe.at[idx, col_fractional_to_cartesian_matrix]

            for idx24, coor24 in enumerate(new_structure):
                if coor24.species_string == "Li":
                    coor_origin_Li_init_cartesian.append(np.dot(coor24.frac_coords, r.T)) 
                if coor24.species_string == "P":
                    coor_origin_P_init_cartesian.append(np.dot(coor24.frac_coords, r.T))
                if coor24.species_string == "S":
                    coor_origin_S_init_cartesian.append(np.dot(coor24.frac_coords, r.T))  
                if coor24.species_string == "Cl":
                    coor_origin_Cl_init_cartesian.append(np.dot(coor24.frac_coords, r.T)) 
            
            coor_structure_dict_cartesian["Li"] = coor_origin_Li_init_cartesian
            coor_structure_dict_cartesian["P"] = coor_origin_P_init_cartesian
            coor_structure_dict_cartesian["S"] = coor_origin_S_init_cartesian
            coor_structure_dict_cartesian["Cl"] = coor_origin_Cl_init_cartesian
        
            dataframe.at[idx, col_coor_structure_dict_cartesian] = coor_structure_dict_cartesian


    def get_closest_neighbors_el_cartesian_coor(dataframe, max_neighbors_radius, el, var_filename):
        col_distance_el = f"distance_cartesian_{var_filename}_{el}"
        col_closest_neighbors_w_dist_el = f"closest_neighbors_w_dist_{var_filename}_{el}"

        col_coor_structure_dict_cartesian = f"coor_structure_dict_cartesian_{var_filename}"
        col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"

        dataframe[col_distance_el] = None
        dataframe[col_closest_neighbors_w_dist_el] = None
        
        for idx in range(dataframe["geometry"].size):

            distance_el = {} 
            closest_neighbors_w_dist_el = {}

            coor_cartesion_el = dataframe.at[idx, col_coor_structure_dict_cartesian][el]

            latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict] 

            a = latticeconstant_structure_dict["a"]
            b = latticeconstant_structure_dict["b"]
            c = latticeconstant_structure_dict["c"]

            for idx24_temp1, coor24_temp1 in enumerate(coor_cartesion_el):        
                closest_neighbors_w_dist_dict = {}
                distance_array = []

                for idx24_temp2, coor24_temp2 in enumerate(coor_cartesion_el):
                    distance = calc_distance.mic_eucledian_distance_cartesian(coor24_temp1, coor24_temp2, a, b, c)
                    
                    if distance < max_neighbors_radius:
                        distance_array.append(distance)

                        closest_neighbors_w_dist_dict['neighbor'] = tuple(coor24_temp2)
                        closest_neighbors_w_dist_dict['dist'] = distance

                        # Get the list of neighbors for the current coordinate, or create one if it doesn't exist
                        neighbors_list = closest_neighbors_w_dist_el.setdefault(tuple(coor24_temp1), [])
                        neighbors_list.append(closest_neighbors_w_dist_dict)

                        # if tuple(coor24_temp1) in closest_neighbors_w_dist_el:
                        #     closest_neighbors_w_dist_el[tuple(coor24_temp1)].append(closest_neighbors_w_dist_dict)
                        # else:
                        #     closest_neighbors_w_dist_el[tuple(coor24_temp1)] = closest_neighbors_w_dist_dict
                
                distance_array_sorted = sorted(set(distance_array))
                if tuple(coor24_temp1) in distance_el:
                    distance_el[tuple(coor24_temp1)].append(distance_array_sorted)
                else:
                    distance_el[tuple(coor24_temp1)] = distance_array_sorted        

            dataframe.at[idx, col_distance_el] = distance_el
            dataframe.at[idx, col_closest_neighbors_w_dist_el] = closest_neighbors_w_dist_el

class Float:
    def format_float(number):
        """
        format float
        """
        # # basically nothing is formatted here
        # if number < 0:
        #     # return f'{(number*-1):.5f}0'
        #     return f'{number:.5f}'
        # else:
        #     return f'{number:.5f}'
        return number

import numpy as np
import os
import shutil
import math
from matplotlib import pyplot as plt
from collections import defaultdict

from pymatgen.core.structure import Structure


class String:
    def replace(i):
        """
        Replace a string with NaN if it cannot be converted to a float.

        Source: https://stackoverflow.com/questions/57048617/how-do-i-replace-all-string-values-with-nan-dynamically
        
        Args
        ====
        i: str
            The input string.

        Returns
        =======
        i : float or np.nan
            The converted float or NaN.

        """
        try:
            float(i)
            return float(i)
        except:
            return np.nan


    def modify_line(line, old_part, new_label):
        """
        Replace a specific part of a line with a new label.
        
        Args:
            line (str): The original line of text.
            old_part (str): The part of the line to be replaced.
            new_label (str): The new label to insert in place of the old part.
        
        Returns:
            str: The modified line with the new label.
        """
        return line.replace(old_part, new_label)
    

    def replace_values_in_series(series, replacements):
        """
        Replace values in a Pandas Series according to a replacements dictionary.
        
        Args:
            series (pd.Series): The Pandas Series to modify.
            replacements (dict): A dictionary where keys are old values and values are new values.
        
        Returns:
            pd.Series: The modified Series with replaced values.
        """
        return series.replace(replacements)


class File:
    def splitall(path):
        """
        Splitting path into its individual components of each sub-/folder.

        Args:
            path (str): The input path to be split.

        Returns:
            list: A list containing the components of the path.

        Source: 
            https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
        """
        allparts = []
        while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                allparts.insert(0, parts[0])
                break
            elif parts[1] == path: # sentinel for relative paths
                allparts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                allparts.insert(0, parts[1])
        return allparts


    def copy_rename_single_file(destination_directory, source_directory, filename, prefix):
        """
        Copy a file from the source directory to the destination directory with an optional filename prefix.

        Args:
            destination_directory (str): The directory where the file should be copied.
            source_directory (str): The directory from which the file should be copied.
            filename (str): The name of the file to be copied.
            prefix (str): An optional prefix to be added to the new filename.

        Returns:
            None
        """
        # Generate the new filename
        if prefix == None:
            new_filename = f"{filename}"
        else:
            new_filename = f"{filename}_{prefix}"
        
        # Get the source file path and destination file path
        destination_path = os.path.join(destination_directory, new_filename)
        
        # Copy the file to the destination directory with the new name
        source_path = os.path.join(source_directory, filename)
        shutil.copy2(source_path, destination_path)
        # print(f"File copied and renamed: {filename} -> {new_filename}")


    def copy_rename_files(dataframe, destination_directory, filename, prefix, savedir):
        """
        Copy and rename multiple files based on the contents of a DataFrame.

        Args:
            dataframe (pd.DataFrame): DataFrame containing file information.
            destination_directory (str): The directory where files should be copied.
            filename (str): The base name of the files.
            prefix (str): An optional prefix to be added to the new filenames.
            savedir (bool): If True, save the new file paths to the DataFrame.

        Returns:
            None
        """
        if savedir == True:
            col_subdir_copiedrenamed_files = f"subdir_{filename}"

            dataframe[col_subdir_copiedrenamed_files] = None

        elif savedir == False:
            pass

        for index in range(dataframe["geometry"].size):
            # Generate the new filename
            if prefix == None:
                new_filename = f"{int(dataframe['geometry'][index])}_{int(dataframe['path'][index])}_{filename}"
            else:
                new_filename = f"{int(dataframe['geometry'][index])}_{int(dataframe['path'][index])}_{filename}_{prefix}"


            # Get the source file path and destination file path
            destination_path = os.path.join(destination_directory, new_filename)
            
            # Copy the file to the destination directory with the new name
            shutil.copy2(dataframe['subdir_new_system'][index], destination_path)
            # print(f"File copied and renamed: {filename} -> {new_filename}")
        
            if savedir == True:
                dataframe.at[int(index), col_subdir_copiedrenamed_files] = destination_path
            elif savedir == False:
                pass


    # def copy_rename_single_file_and_delete_elements(destination_directory, source_directory, filename, prefix, line_ranges, line_numbers_edit, new_contents):
    #     # Generate the new filename
    #     new_filename = f"{filename}_{prefix}"
        
    #     # Get the source file path and destination file path
    #     destination_path = os.path.join(destination_directory, new_filename)
        
    #     # Copy the file to the destination directory with the new name
    #     source_path = os.path.join(source_directory, filename)
    #     shutil.copy2(source_path, destination_path)
    #     print(f"File copied and renamed: {filename} -> {new_filename}")

    #     File.delete_elements(destination_path, line_ranges, line_numbers_edit, new_contents)


    # def copy_rename_files_and_delete_elements(file_loc, destination_directory, filename, index, prefix, line_ranges, line_numbers_edit, new_contents):
    #     # Generate the new filename
    #     new_filename = f"{int(file_loc['geometry'][index])}_{int(file_loc['path'][index])}_{filename}_{prefix}"
        
    #     # Get the source file path and destination file path
    #     destination_path = os.path.join(destination_directory, new_filename)
        
    #     # Copy the file to the destination directory with the new name
    #     shutil.copy2(file_loc['subdir_new_system'][index], destination_path)
    #     print(f"File copied and renamed: {filename} -> {new_filename}")

    #     File.delete_elements(destination_path, line_ranges, line_numbers_edit, new_contents)


    def delete_lines(file_path, line_ranges):
        """
        Delete specified lines from a file.

        Args:
            file_path (str): The path to the file to be modified.
            line_ranges (list): A list of tuples representing the ranges of lines to be deleted.

        Returns:
            None
        """
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        lines_to_delete = set()
        for line_range in line_ranges:
            start, end = line_range
            if 1 <= start <= len(lines) and start <= end <= len(lines):
                lines_to_delete.update(range(start - 1, end))

        modified_lines = [line for i, line in enumerate(lines) if i not in lines_to_delete]

        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

        # print(f"Lines deleted successfully in file: {file_path}")


    def delete_elements(file_path, line_ranges, line_numbers_edit, new_contents):
        """
        Delete specified lines and edit others in a file.

        Args:
            file_path (str): The path to the file to be modified.
            line_ranges (list): A list of tuples representing the ranges of lines to be deleted.
            line_numbers_edit (list): A list of line numbers to be edited.
            new_contents (list): A list of new contents for the edited lines.

        Returns:
            None
        """
        File.delete_lines(file_path, line_ranges)
        File.edit_lines(file_path, line_numbers_edit, new_contents)


    def edit_lines(file_path, line_numbers, new_contents):
        """
        Edit specified lines in a file.

        Args:
            file_path (str): The path to the file to be modified.
            line_numbers (list): A list of line numbers to be edited.
            new_contents (list): A list of new contents for the edited lines.

        Returns:
            None
        """
        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify the line content
        for line_number, new_content in zip(line_numbers, new_contents):
            if 1 <= line_number <= len(lines):
                lines[line_number - 1] = new_content + '\n'

            # Write the modified content back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)
                # # print(f"Line edited successfully.")
        # # else:
        # # #     print(f"Invalid line number: {line_number}")


    def check_folder_existance(folder_name, empty_folder):
        """
        Check if a folder exists, create it if not, and optionally empty it.

        Args:
            folder_name (str): The name of the folder to check or create.
            empty_folder (bool): If True, empty the folder if it already exists.

        Returns:
            None
        """
        # Check if the folder exists
        if not os.path.exists(folder_name):
            # Create the folder if it doesn't exist
            os.makedirs(folder_name)
            # print(f"Folder '{folder_name}' created.")
        else:
            if empty_folder == True:
                File.empty_folder(folder_name)
                # print(f"Folder '{folder_name}' already exists. Emptying it.")
            elif empty_folder == False:
                pass


    def empty_folder(folder_name):
        """
        Empty the contents of a folder.

        Args:
            folder_name (str): The name of the folder to be emptied.

        Returns:
            None
        """
        files_inside_folder = os.listdir(folder_name)
        for i in files_inside_folder:
            path_to_files = folder_name + i 
            os.remove(path_to_files)


    def delete_files(dataframe, folder_name, file_name_w_format):
        """
        Delete files based in the specified folder.

        Args:
            dataframe (pd.DataFrame): A DataFrame containing information about files ('geometry' and 'path').
            folder_name (str): The directory where the files are located.
            file_name_w_format (str): The base name of the files with format information.

        Returns:
            None

        Raises:
            OSError: If an error occurs during file deletion, an exception is caught,
                    and an error message is printed.
        """
        for idx in range(dataframe["geometry"].size):
            filename_to_delete = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{file_name_w_format}"
            filename_to_delete_path = os.path.join(folder_name, filename_to_delete)
            
            try:
                # Attempt to delete the file
                os.remove(filename_to_delete_path)
                # print(f"{filename_to_delete_path} has been deleted.")
            except OSError as e:
                # Handle any errors that occur during file deletion
                print(f"Error deleting {filename_to_delete_path}: {e}")


class Distance:
    # def eucledian_distance(coor1, coor2):
    #     distance = math.sqrt(sum((x1 - x2)**2 for x1, x2 in zip(coor1, coor2)))
    #     return distance


    def apply_pbc(distance_1D):
        """
        Apply Periodic Boundary Conditions to a given 1D distance.

        Parameters:
        - distance_1D (float): The value to apply the periodic boundary conditions to.

        Returns:
        - distance_1D (float): The adjusted value after applying the periodic boundary conditions, ensuring it remains within the normalized range [0, 0.5].
        """
        while abs(distance_1D) > 0.5:
            return 1 - abs(distance_1D)
        return distance_1D


    def mic_eucledian_distance(coor1, coor2):
        """
        This function computes the minimum image convention (MIC) Euclidean distance 
        between two points (coor1 and coor2),considering periodic boundary conditions. 
        It ensures that the distance measured is the shortest possible path 
        between these points in a periodic system.

        Parameters:
        - coor1 (tuple): The (x, y, z) coordinates of the first point.
        - coor2 (tuple): The (x, y, z) coordinates of the second point.

        Returns:
        - distance (float): The minimum image convention Euclidean distance between the two points.
        """
        x_coor1, y_coor1, z_coor1 = coor1
        x_coor2, y_coor2, z_coor2 = coor2
        
        delta_x = x_coor1 - x_coor2
        delta_y = y_coor1 - y_coor2
        delta_z = z_coor1 - z_coor2

        distance = math.sqrt(sum([(Distance.apply_pbc(delta_x))**2, (Distance.apply_pbc(delta_y))**2, (Distance.apply_pbc(delta_z))**2]))
        return distance


    def apply_pbc_cartesian(distance_1D, length_1D):
        """
        Apply Periodic Boundary Conditions to a given 1D distance in a Cartesian system.

        Parameters:
        - distance_1D (float): The original distance along one axis (X, Y, or Z).
        - length (float): The length of the domain along the same axis.

        Returns:
        - distance_1D (float): The adjusted distance considering PBC, ensuring it's the shortest possible
        within the given domain length.

        Notes:
        - Angle is ignored in the calculation
        """
        if abs(distance_1D) > 0.5 * length_1D:
            return length_1D - abs(distance_1D)
        return distance_1D


    def mic_eucledian_distance_cartesian(coor1, coor2, a, b, c):
        """
        Calculates the minimum image convention (MIC) Euclidean distance between two points in 3D space
        with periodic boundary conditions in a Cartesian coordinate system, 
        considering the box dimensions a, b, and c along the x, y, and z axes, respectively.

        Parameters:
        - coor1 (tuple): Coordinates (x, y, z) of the first point.
        - coor2 (tuple): Coordinates (x, y, z) of the second point.
        - a (float): Length of the simulation box along the x-axis.
        - b (float): Length of the simulation box along the y-axis.
        - c (float): Length of the simulation box along the z-axis.

        Returns:
        - distance (float): The MIC Euclidean distance between the two points.

        Notes:
        - I'm actually confused with this function
        """
        x_coor1, y_coor1, z_coor1 = coor1
        x_coor2, y_coor2, z_coor2 = coor2
        
        delta_x = x_coor1 - x_coor2
        delta_y = y_coor1 - y_coor2
        delta_z = z_coor1 - z_coor2

        distance = math.sqrt(sum([(Distance.apply_pbc_cartesian(delta_x, a))**2, (Distance.apply_pbc_cartesian(delta_y, b))**2, (Distance.apply_pbc_cartesian(delta_z, c))**2]))
        return distance


class Dict:
    def merge_dictionaries(dict1, dict2):
        """
        Merges two dictionaries into a single dictionary, combining the values of 
        any duplicate keys into lists.

        Parameters:
        - dict1 (dict): The first input dictionary.
        - dict2 (dict): The second input dictionary.

        Returns:
        - defaultdict(list): A dictionary where each key holds a list of values 
        from both input dictionaries. If a key is present in both dict1 and dict2, 
        both values will be in the list. Otherwise, the list will contain the 
        single value from whichever dictionary the key originates from.
        """
        merged_dict = defaultdict(list)

        for d in (dict1, dict2): # Extendable for more dictionaries
            for key, value in d.items():
                merged_dict[key].append(value)
        
        return merged_dict


    def get_duplicate_values(dictionary):
        """
        Identifies and returns a list of duplicate values in the given dictionary. If a value appears
        more than once across all values in the dictionary, it will be included in the return list.

        Parameters:
        - dictionary (dict): The dictionary whose values are to be checked for duplicates.

        Returns:
        - list: A list of values that appear more than once in the dictionary.

        Note: 
        This implementation is designed to work effectively with immutable value types (e.g.,
        integers, strings, tuples). For mutable types like lists, it treats the value as seen only once
        because lists cannot be used as dictionary keys or added to sets directly without conversion.
        """
        seen_values = []
        duplicate_values = []

        for value in dictionary.values():
            if value in seen_values:
                duplicate_values.append(value)
            else:
                seen_values.append(value)

        return duplicate_values
    

    class Mapping:
        def get_duplicate_closest24_w_data(dict):
            """
            Identifies and returns a list of duplicate values (closest24 with its data: coor)
            """
            duplicate_closest24 = {}
            for coorreference, values in dict.items():
                for entry in values:
                    closest24 = entry["closest24"]
                    dist = entry["dist"]

                if closest24 in duplicate_closest24:
                    duplicate_closest24[closest24].append({"coorreference": coorreference, "dist": dist})
                else:
                    duplicate_closest24[closest24] = [{"coorreference": coorreference, "dist": dist}]

            duplicate_closest24_w_data = {}
            for closest24, coorreferences_dists in duplicate_closest24.items():
                if len(coorreferences_dists) > 1:
                    duplicate_closest24_w_data[f"Duplicate closest24: {closest24}"] = [{"coorreferences and dists": coorreferences_dists}]

            return duplicate_closest24_w_data


        def get_atom_mapping_el_w_dist_closestduplicate(dict):
            """
            Identifies and returns closest mapped atom with its distance
            """
            filtered_data = {}
            for coorreference, values in dict.items():
                for entry in values:
                    closest24 = entry["closest24"]
                    dist = entry["dist"]
                    
                if closest24 in filtered_data:
                    if dist < filtered_data[closest24]["dist"]:
                        filtered_data[closest24] = {"coorreference": coorreference, "dist": dist}
                else:
                    filtered_data[closest24] = {"coorreference": coorreference, "dist": dist}

            atom_mapping_el_w_dist_closestduplicate = {entry["coorreference"]: {"closest24": key, "dist": entry["dist"]} for key, entry in filtered_data.items()}
            return atom_mapping_el_w_dist_closestduplicate


class KMeans:
    def kmeans_cluster_atoms(coor_atoms, amount_clusters):
        # source: https://stackoverflow.com/questions/64987810/3d-plotting-of-a-dataset-that-uses-k-means
        kmeans = KMeans(n_clusters=amount_clusters)                # Number of clusters
        kmeans = kmeans.fit(coor_atoms)                          # Fitting the input data
        labels = kmeans.predict(coor_atoms)                      # Getting the cluster labels
        centroids = kmeans.cluster_centers_             # Centroid values
        # print("Centroids are:", centroids)              # From sci-kit learn

        fig = plt.figure(figsize=(10,10))
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')

        # x = np.array(labels==0)
        # y = np.array(labels==1)
        # z = np.array(labels==2)


        # ax.scatter(coor_atoms[x][:, 0], coor_atoms[x][:, 1], coor_atoms[x][:, 2], color='red')
        # ax.scatter(coor_atoms[y][:, 0], coor_atoms[y][:, 1], coor_atoms[y][:, 2], color='blue')
        # ax.scatter(coor_atoms[z][:, 0], coor_atoms[z][:, 1], coor_atoms[z][:, 2], color='green')
        # ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
        #             marker='x', s=169, linewidths=10,
        #             color='black', zorder=50)
        # # ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c="black",s=150,label="Centers",alpha=1) # for dot marker

        # Define a colormap for different clusters
        colors = plt.cm.rainbow(np.linspace(0, 1, amount_clusters))

        for cluster_label in range(amount_clusters):
            cluster_mask = np.array(labels == cluster_label)
            ax.scatter(
                coor_atoms[cluster_mask][:, 0],
                coor_atoms[cluster_mask][:, 1],
                coor_atoms[cluster_mask][:, 2],
                color=colors[cluster_label],
                label=f'Cluster {cluster_label}'
            )

        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            marker='x',
            s=169,
            linewidths=10,
            color='black',
            zorder=50,
            label='Centroids'
        )

        ax.legend()

        return centroids, labels


    def create_POSCAR_atoms_centroids_appended(coor_atoms, coor_centroids, destination_directory, lattice_constant, filename):

        # Define lattice constants (you might need to adjust these based on your actual system)
        lattice_constants_matrix = np.array([
            [lattice_constant, 0.0000000000000000, 0.0000000000000000],
            [0.0000000000000000, lattice_constant, 0.0000000000000000],
            [0.0000000000000000, 0.0000000000000000, lattice_constant]
        ])

        # Define the header and comment lines for the POSCAR file
        header = "Generated POSCAR"
        comment = "1.0"

        # Define filename
        filename_path = os.path.join(destination_directory, filename)

        # Write the POSCAR file
        with open(filename_path, "w") as f:
            # Write the header and comment
            f.write(header + "\n")
            f.write(comment + "\n")

            # Write the lattice constants
            for row in lattice_constants_matrix:
                f.write(" ".join(map(str, row)) + "\n")

            # K as mock element for centroids
            f.write("Li K\n")

            # Write the number of atoms for each element
            # f.write(" ".join(map(str, np.ones(len(coordinates), dtype=int))) + "\n")
            f.write(str(len(coor_atoms)) + " " + str(len(coor_centroids)) + "\n")  # Number of Li atoms

            # Write the selective dynamics tag (in this case, 'Direct')
            f.write("Direct\n")

            # Write the atomic coordinates
            for coor_atom in coor_atoms:
                # f.write(" ".join(map(str, coord)) + "\n")
                formatted_coor_atom = [format(x, ".16f") for x in coor_atom]
                f.write(" ".join(formatted_coor_atom) + "\n")
            for coor_centroid in coor_centroids:
                formatted_coor_centroid = [format(x, ".16f") for x in coor_centroid]
                f.write(" ".join(formatted_coor_centroid) + "\n")

        # print("POSCAR file created successfully.")


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
                    distance = Distance.mic_eucledian_distance_cartesian(coor24_temp1, coor24_temp2, a, b, c)
                    
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

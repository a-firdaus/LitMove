import os
import numpy as np

from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.inputs import Poscar


def with_library(dataframe, destination_directory, filename, structure_reference, var_name, prefix):
    # rename from: get_structure_with_library
    """
    Generates transformed structures from input data and saves them as CIF files,
    comparing each structure to a reference structure using StructureMatcher from pymatgen library.
    The comparison result for each structure is stored in the 'verify_w_lib' column
    of the DataFrame.

    Parameters
    ==========
    dataframe: pd.DataFrame
        DataFrame containing necessary data columns.
    destination_directory: str
        Path to the directory where CIF files will be saved.
    filename: str
        Base filename for the CIF files.
    structure_reference: pymatgen Structure
        Reference structure to compare with.
    var_name: str
        Variable name for saving the transformerd CIF files.
    prefix: str or None
        Prefix to append to the filenames (optional).

    Returns
    =======
    None
        The function saves CIF files but does not return any value.
    """
    for idx in range(dataframe["geometry"].size):
        # Generate filename for the transformed structure
        if prefix is None:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}"
        else:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}_{prefix}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)
        
        # Load structure from file
        structure = Structure.from_file(filename_to_transform_path)
        
        # Use StructureMatcher to compare structures
        # StructureMatcher can accept different tolerances for judging equivalence
        matcher = StructureMatcher(primitive_cell=False)
        # first, we can verify these lattices are equivalent. should return True
        matcher_verify = matcher.fit(structure_reference, structure)
        if matcher_verify == False:
                print(f"Matcher doesn't match at geo/ path: {int(dataframe['geometry'][idx])}/{int(dataframe['path'][idx])}.")
        # Store the comparison result in the DataFrame
        dataframe.at[idx, 'verify_w_lib'] = matcher_verify
        
        # Generate CIF filename
        if prefix is None:
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        else:
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}_{prefix}.cif"
        destination_path = os.path.join(destination_directory, cif_filename)
        
        # Write transformed structure to CIF file
        transformed_structure = matcher.get_s2_like_s1(structure_reference, structure, include_ignored_species=True)
        cif = CifWriter(transformed_structure)
        cif.write_file(destination_path)


def with_linalg(dataframe, destination_directory, filename, structure_reference, var_name, prefix):
    # rename from: get_structure_w_linalg
    """
    Applies linear algebra transformations to structures and saves them as CIF files,
    comparing each structure to a reference structure using StructureMatcher from pymatgen library.
    The comparison result for each structure is stored in the 'verify_w_linalg' column
    of the DataFrame.

    Parameters
    ==========
    dataframe: pd.DataFrame
        DataFrame containing necessary data columns.
    destination_directory: str
        Path to the directory where CIF files will be saved.
    filename: str
        Base filename for the CIF files.
    structure_reference: pymatgen Structure
        Reference structure to compare with.
    var_name: str
        Variable name for the CIF files.
    prefix: str or None
        Prefix to append to the filenames (optional).

    Returns
    =======
    None
        The function saves CIF files but does not return any value.

    Note
    ====
    - The functionality related to transformation matrices has been commented out and may need further debugging.
    - If prefix is given for other element, i.e. P, this needs further debugging
    """
    for idx in range(dataframe["geometry"].size):
        # Generate filename for the transformed structure
        if prefix == None: 
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}"
        else:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}_{prefix}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)
        structure = Structure.from_file(filename_to_transform_path)

        # Use StructureMatcher to compare structures
        # StructureMatcher can accept different tolerances for judging equivalence
        matcher = StructureMatcher(primitive_cell=False) # don't work if it's True
        matcher_verify = matcher.fit(structure_reference, structure) # returns True
        dataframe.at[idx, 'verify_w_linalg'] = matcher_verify
        if not matcher_verify:
            print(f"Matcher doesn't match at geo/ path: {int(dataframe['geometry'][idx])}/{int(dataframe['path'][idx])}.")

        # Transform structure using linear algebra
            # output of transformation:
                # 3x3 matrix of supercell transformation;
                # 1x3 vector of fractional translation;
                # 1x4 mapping to transform struct2 to be similar to struct1
        transformation = matcher.get_transformation(structure_reference, structure)
        if transformation is None:
            return None
        
        scaling, translation, mapping = transformation

        # Store transformation information in DataFrame
        if prefix == None:
            dataframe.at[idx, 'scaling'] = scaling
            dataframe.at[idx, 'translation'] = translation
            dataframe.at[idx, 'mapping'] = mapping
        else:
            dataframe.at[idx, 'transformation_P'] = transformation

        # Apply transformation to coordinates
        scaled_coords = np.dot(structure.frac_coords, scaling.T) # scaling
        # # scaled_coords = np.round(scaled_coords, decimals=16)
        translated_coords = scaled_coords + translation          # translation
        transformed_coords = translated_coords[mapping]               # mapping
        # # # long story short
        # # transformed_coords = np.dot(structure.frac_coords[mapping], scaling.T) + translation

        # Create a new structure with the transformed coordinates
        transformed_structure = Structure(structure.lattice, structure.species, transformed_coords)

        # Write transformed structure to CIF file
        cif = CifWriter(transformed_structure)
        if prefix == None: 
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        else:
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}_{prefix}.cif"
        destination_path = os.path.join(destination_directory, cif_filename)
        cif.write_file(destination_path)


# for sanity check
def get_structure_with_linalg_combinded_with_library(dataframe, destination_directory, filename, structure_reference, var_name, prefix):
    """
    Note
    ====
    - will be checked later
    """
    for idx in range(dataframe["geometry"].size):
        if prefix == None: 
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}"
        else:
            filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}_{prefix}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)
        structure = Structure.from_file(filename_to_transform_path)

        # StructureMatcher can accept different tolerances for judging equivalence
        matcher = StructureMatcher(primitive_cell=False)
        # first, we can verify these lattices are equivalent
        matcher_verify = matcher.fit(structure_reference, structure)  # returns True
        dataframe['verify_w_linalg'][idx] = matcher_verify
        if matcher_verify == False:
            print(f"Matcher doesn't match.")

        # Transform desired structure into structure_reference
            # output of transformation:
                # 3x3 matrix of supercell transformation;
                # 1x3 vector of fractional translation;
                # 1x4 mapping to transform struct2 to be similar to struct1
        transformation = matcher.get_transformation(structure_reference, structure)
        if transformation is None:
            return None
        
        scaling, translation, mapping = transformation

        if prefix == None:
            dataframe.at[idx, 'transformation'] = transformation
        else:
            dataframe.at[idx, f'transformation_{prefix}'] = transformation

        sites = list(structure)
        # Append the ignored sites at the end.
        # # sites.extend([site for site in struct2 if site not in s2])
        temp = Structure.from_sites(sites)

        # Apply scaling
        temp.make_supercell(scaling)
        # apply translation
        temp.translate_sites(list(range(len(temp))), translation)

        # Apply some modification from library
        for i, j in enumerate(mapping[: len(structure_reference)]):
            if j is not None:
                vec = np.round(structure_reference[i].frac_coords - temp[j].frac_coords)
                temp.translate_sites(j, vec, to_unit_cell=False)
        sites = [temp.sites[i] for i in mapping if i is not None]
        # if include_ignored_species:
        #     start = int(round(len(temp) / len(struct2) * len(s2)))
        #     sites.extend(temp.sites[start:])
        transformed_structure = Structure.from_sites(sites)
        # transformed_structure = Structure(structure.lattice, structure.species, mapped_coords) # similar to above but different input
        
        cif = CifWriter(transformed_structure)
        if prefix == None: 
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        else:
            cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}_{prefix}.cif"
        destination_path = os.path.join(destination_directory, cif_filename)
        cif.write_file(destination_path)


def get_structure_with_linalg_orientated(dataframe, destination_directory, filename, var_name):
    """
    Note
    ====
    - will be checked later
    - WHAT IS THIS DOING?
    """
    ## POSCAR file is also created
    dataframe['subdir_orientated'] = None
    for idx in range(dataframe["geometry"].size):
        filename_to_transform = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{filename}"
        filename_to_transform_path = os.path.join(destination_directory, filename_to_transform)
        structure = Structure.from_file(filename_to_transform_path)

        # Transform desired structure into structure_reference
        scaling = dataframe["scaling"][idx]
        translation = dataframe["translation"][idx]
        mapping = dataframe["mapping"][idx]

        # Apply scaling
        scaled_coords = np.dot(structure.frac_coords, scaling.T)
        # apply translation
        translated_coords = scaled_coords + translation
        # apply mapping (no mapping here)
        mapped_coords = translated_coords[mapping]
        # # # long story short
        # # transformed_coords = np.dot(structure.frac_coords[mapping], scaling.T) + translation

        # Create a new structure with the transformed coordinates
        transformed_structure = Structure(structure.lattice, structure.species, mapped_coords)
        # transformed_structure = Structure.from_sites(mapped_coords) # similar to above but different input
        
        cif = CifWriter(transformed_structure)
        cif_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_{var_name}.cif"
        destination_path = os.path.join(destination_directory, cif_filename)
        # dataframe
        cif.write_file(destination_path)

        poscar = Poscar(transformed_structure)
        poscar_filename = f"{int(dataframe['geometry'][idx])}_{int(dataframe['path'][idx])}_POSCAR_{var_name}"
        destination_path_poscar = os.path.join(destination_directory, poscar_filename)
        poscar.write_file(destination_path_poscar)
        
        dataframe['subdir_orientated'][idx] = destination_path

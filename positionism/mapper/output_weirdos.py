import numpy as np
import os


def as_array(dataframe, activate_radius):
    # rename from: get_coor_weirdos_array
    """
    Extracts and compiles coordinates of "weirdo" elements into a single numpy array based on the specified activation radius.

    Parameters
    ==========
    dataframe: pd.DataFrame
        A pandas DataFrame containing the data from which coordinates of weirdos are to be extracted. 
    activate_radius: int
        An integer (1, 2, or 3) specifying the activation radius.

    Returns
    =======
    coor_weirdos_el_appended: numpy.ndarray
        A numpy array containing all the coordinates of "weirdo" elements extracted from the specified column(s) in the DataFrame. Each element in this array is a coordinate set (typically x, y, z) of a "weirdo" element.
    """
    if activate_radius == 2 or activate_radius == 3:
        col_coor_weirdos_el = f"coor_weirdos_48htype2_Li"
    elif activate_radius == 1:
        col_coor_weirdos_el = f"coor_weirdos_Li"

    coor_weirdos_el_appended = []

    for idx in range(dataframe["geometry"].size):
        coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]

        for single_coor in coor_weirdos_el:
            coor_weirdos_el_appended.append(single_coor)

    coor_weirdos_el_appended = np.array(coor_weirdos_el_appended)

    return coor_weirdos_el_appended


def create_POSCAR(coor_weirdos, destination_directory, lattice_constant, filename):
    # rename from: create_POSCAR_weirdos
    """
    Generates a POSCAR file for visualization or computation in software that utilizes the VASP POSCAR format,
    using specified coordinates for "weirdo" atoms.

    This function writes a POSCAR file with a simple cubic lattice structure where the provided "weirdo" atom coordinates 
    are used to position lithium (Li) atoms. The lattice constant is uniform across all dimensions. The file is saved 
    with the given filename in the specified directory.

    Parameters
    ==========
    coor_weirdos: list or numpy.ndarray
        A list or numpy array of coordinates (x, y, z) for the "weirdo" atoms. These coordinates should be in fractional coordinates relative to the lattice vectors.
    destination_directory: str
        The directory path where the POSCAR file will be saved.
    lattice_constant: float
        The lattice constant to use for the cubic lattice.
    filename: str
        The filename for the saved POSCAR file. 

    Returns
    =======
    None
        This function writes a file to a POSCAR file and does not return any value.

    Note
    ====
    - The function is tailored for lithium atoms but can be modified for other elements by adjusting the element symbol 
      written to the file.
    - The POSCAR file generated is in the 'Direct' format, meaning the atomic coordinates are given in fractional coordinates 
      relative to the lattice vectors.
    """
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

        # Write the element symbols (in this case, using 'Li' for lithium)
        f.write("Li\n")

        # Write the number of atoms for each element
        # f.write(" ".join(map(str, np.ones(len(coordinates), dtype=int))) + "\n")
        f.write(str(len(coor_weirdos)) + "\n")  # Number of Li atoms

        # Write the selective dynamics tag (in this case, 'Direct')
        f.write("Direct\n")

        # Write the atomic coordinates
        for coor in coor_weirdos:
            # f.write(" ".join(map(str, coord)) + "\n")
            formatted_coords = [format(x, ".16f") for x in coor]
            f.write(" ".join(formatted_coords) + "\n")

    # print("POSCAR file created successfully.")

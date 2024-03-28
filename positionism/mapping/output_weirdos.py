import numpy as np
import os


def get_coor_weirdos_array(dataframe, activate_radius):
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


def create_POSCAR_weirdos(coor_weirdos, destination_directory, lattice_constant, filename):

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

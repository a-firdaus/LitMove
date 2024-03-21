import math


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

    distance = math.sqrt(sum([(apply_pbc(delta_x))**2, (apply_pbc(delta_y))**2, (apply_pbc(delta_z))**2]))
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

    distance = math.sqrt(sum([(apply_pbc_cartesian(delta_x, a))**2, (apply_pbc_cartesian(delta_y, b))**2, (apply_pbc_cartesian(delta_z, c))**2]))
    return distance

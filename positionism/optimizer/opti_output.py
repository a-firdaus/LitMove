import os

def create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype):
    # formatted_positions = [Operation.Float.format_float(pos) for pos in ref_positions_array]
    formatted_positions = [pos for pos in ref_positions_array]
    formatted_positions_str = list(map(str, formatted_positions))
    return os.path.join(direc_perfect_poscar, f"Li6PS5Cl_{'_'.join(formatted_positions_str)}_{var_optitype}.cif")

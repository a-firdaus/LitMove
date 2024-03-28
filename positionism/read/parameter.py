
# class Parameter:
def dx_dz(file_path, litype):
    # rename from: get_dx_dz
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


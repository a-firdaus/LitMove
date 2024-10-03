from positionism.optimizer import opti_output


def change_dx_dz_alllitype(file_path, file_path_new, ref_positions_array, litype):
    # old_name = change_dx_dz
    # ref_positions_array = ALL values in this array

    # formatted_positions = [Operation.Float.format_float(pos) for pos in ref_positions_array]
    formatted_positions = [pos for pos in ref_positions_array]
    print(f"formatted_positions: {formatted_positions}")

    with open(file_path, 'r') as f:
        lines = f.readlines()

    if litype == 0:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 1:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 2:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 3:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 4:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)

    elif litype == 5:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
        new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li6"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)
    elif litype == 6:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
        new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
        new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li6"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li7"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)
    elif litype == 7:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
        new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
        new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]
        new_dx1_48h_type7, new_dx2_48h_type7, new_dz_48h_type7 = formatted_positions[21:24]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li6"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li7"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li8"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type7}",f"{new_dx2_48h_type7}",f"{new_dz_48h_type7}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)
    elif litype == 8:
        new_dx_24g, new_dz1_24g, new_dz2_24g = formatted_positions[0:3]
        new_dx1_48h_type1, new_dx2_48h_type1, new_dz_48h_type1 = formatted_positions[3:6]
        new_dx1_48h_type2, new_dx2_48h_type2, new_dz_48h_type2 = formatted_positions[6:9]
        new_dx1_48h_type3, new_dx2_48h_type3, new_dz_48h_type3 = formatted_positions[9:12]
        new_dx1_48h_type4, new_dx2_48h_type4, new_dz_48h_type4 = formatted_positions[12:15]
        new_dx1_48h_type5, new_dx2_48h_type5, new_dz_48h_type5 = formatted_positions[15:18]
        new_dx1_48h_type6, new_dx2_48h_type6, new_dz_48h_type6 = formatted_positions[18:21]
        new_dx1_48h_type7, new_dx2_48h_type7, new_dz_48h_type7 = formatted_positions[21:24]
        new_dx1_48h_type8, new_dx2_48h_type8, new_dz_48h_type8 = formatted_positions[24:27]

        with open(file_path_new, 'w') as f:
            for line in lines:
                if line.startswith("Li1"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type1:.5f}",f"{new_dx2_48h_type1:.5f}",f"{new_dz_48h_type1:.5f}"]
                    parts[4:6+1] = [f"{new_dx_24g}",f"{new_dz1_24g}",f"{new_dz2_24g}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li2"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx_24g:.5f}",f"{new_dz1_24g:.5f}",f"{new_dz2_24g:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type1}",f"{new_dx2_48h_type1}",f"{new_dz_48h_type1}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li3"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type2}",f"{new_dx2_48h_type2}",f"{new_dz_48h_type2}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li4"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type3}",f"{new_dx2_48h_type3}",f"{new_dz_48h_type3}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li5"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type4}",f"{new_dx2_48h_type4}",f"{new_dz_48h_type4}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li6"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type5}",f"{new_dx2_48h_type5}",f"{new_dz_48h_type5}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li7"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type6}",f"{new_dx2_48h_type6}",f"{new_dz_48h_type6}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li8"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type7}",f"{new_dx2_48h_type7}",f"{new_dz_48h_type7}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                if line.startswith("Li9"):
                    parts = line.split()
                    # parts[4:6+1] = [f"{new_dx1_48h_type2:.5f}",f"{new_dx2_48h_type2:.5f}",f"{new_dz_48h_type2:.5f}"]
                    parts[4:6+1] = [f"{new_dx1_48h_type8}",f"{new_dx2_48h_type8}",f"{new_dz_48h_type8}"]
                    parts[2] = f" {parts[2]}"
                    parts[-1] = f"{parts[-1]}\n"
                    line = " ".join(parts)
                f.write(line)


def change_dx_dz_specificlitype(file_path, file_path_new, ref_positions_array, litype):

    # formatted_positions = [Operation.Float.format_float(pos) for pos in ref_positions_array]
    formatted_positions = [pos for pos in ref_positions_array]

    new_dx1_type, new_dx2_type, new_dz_type = formatted_positions

    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path_new, 'w') as f:
        for line in lines:
            if line.startswith(f"Li{litype+1}"):
                parts = line.split()
                # parts[4:6+1] = [f"{new_dx1_type:.5f}",f"{new_dx2_type:.5f}",f"{new_dz_type:.5f}"]
                parts[4:6+1] = [f"{new_dx1_type}",f"{new_dx2_type}",f"{new_dz_type}"]
                parts[2] = f" {parts[2]}"
                parts[-1] = f"{parts[-1]}\n"
                line = " ".join(parts)
            f.write(line)


# # SEEMS NO USAGE
def modif_dx_dz_cif_specificlitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, litype, var_optitype):
    file_path_new = opti_output.create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype)
    change_dx_dz_specificlitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

    return file_path_new


def modif_dx_dz_get_filepath(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, ref_positions_array_filename, litype, var_optitype, modif_all_litype):
    file_path_new = opti_output.create_file_name(direc_perfect_poscar, ref_positions_array_filename, var_optitype)
    if modif_all_litype == True:
        change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)
    elif modif_all_litype == False:
        change_dx_dz_specificlitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)
    elif modif_all_litype == None:
        change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

    return file_path_new


# # SEEMS NO USAGE
# def modif_dx_dz_cif_alllitype(direc_perfect_poscar, file_path_ori_ref_48n24, ref_positions_array, litype, var_optitype):
#     file_path_new = output.create_file_name(direc_perfect_poscar, ref_positions_array, var_optitype)
#     change_dx_dz_alllitype(file_path_ori_ref_48n24, file_path_new, ref_positions_array, litype)

#     return file_path_new

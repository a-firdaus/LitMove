#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
import os
import math

def apply_pbc(value):
    if abs(value) > 0.5:
        return 1 - abs(value)
    return value

def mic_eucledian_distance(coor1, coor2):
    x_coor1, y_coor1, z_coor1 = coor1
    x_coor2, y_coor2, z_coor2 = coor2
    
    delta_x = x_coor1 - x_coor2
    delta_y = y_coor1 - y_coor2
    delta_z = z_coor1 - z_coor2

    distance = math.sqrt(sum([(apply_pbc(delta_x))**2, (apply_pbc(delta_y))**2, (apply_pbc(delta_z))**2]))
    return distance

poscar_line_nr_start = 8        # index from 0
poscar_line_nr_end = 60
amount_Li = 25
n_decimal = 16

if len(sys.argv) != 3:
    print("Usage: python reindexPOSCAR.py file_path_init file_path")
    sys.exit(1)

file_path_init = sys.argv[1]
file_path = sys.argv[2]

with open(file_path_init, 'r') as file_init:
    lines_init = file_init.readlines()
data_init = lines_init[poscar_line_nr_start:poscar_line_nr_start + amount_Li]

df_init = pd.DataFrame([string.strip().split() for string in data_init])
df_float_init = df_init.astype(float)

check_negative_init = df_float_init.lt(0).any().any()

if check_negative_init:
    sys.exit()

coordinate_init = [row.to_numpy() for index, row in df_float_init.iterrows()]

with open(file_path, 'r') as file:
    lines = file.readlines()
data = lines[poscar_line_nr_start:poscar_line_nr_start + amount_Li]
data_non_Li = lines[poscar_line_nr_start + amount_Li:poscar_line_nr_end]

df = pd.DataFrame([string.strip().split() for string in data])
df_float = df.astype(float)
df_non_Li = pd.DataFrame([string.strip().split() for string in data_non_Li])
df_float_non_Li = df_non_Li.astype(float)

check_negative = df_float.lt(0).any().any()

if check_negative:
    sys.exit()

coordinate = [row.to_numpy() for index, row in df_float.iterrows()]

coor_final_reindex = []

for coor_init in coordinate_init:
    distance_prev = float("inf")
    closest24 = None

    for coor in coordinate:
        distance = mic_eucledian_distance(coor_init, coor)

        if distance < distance_prev:
            distance_prev = distance
            closest_coor = coor

    coor_final_reindex.append(closest_coor)

are_equal = all(np.array_equal(coor_reindex, coor) for coor_reindex, coor in zip(coor_final_reindex, coordinate))

if are_equal:
    print("The arrays are the same.")
else:
    print("The arrays are different.")
    sys.exit()

df_coordinate = pd.DataFrame(coordinate)
df_concat = pd.concat([df_coordinate, df_float_non_Li], ignore_index=True)

for i in range(df_concat.shape[0]):
    for j in range(df_concat.shape[1]):
        temp = df_concat[j][i]
        df_concat[j][i] = '{:.{width}f}'.format(float(temp), width=n_decimal)

row_list = df_concat.to_string(index=False, header=False).split('\n')
row_list_space = ['  '.join(string.split()) for string in row_list]  # 2 spaces of distance
row_list_w_beginning = ['  ' + row for row in row_list_space]        # 2 spaces in the beginning
absolute_correct_list = '\n'.join(row_list_w_beginning).splitlines()

line_append_list = []
for idx_c, line in enumerate(absolute_correct_list):
    line_new_line = str(line) + '\n'
    line_append_list.append(line_new_line)

file_list = lines[:poscar_line_nr_start] + line_append_list

poscar_filename = file_path + "_reindexed"
destination_path = poscar_filename

with open(destination_path, "w") as poscar_positive_file:
    for item in file_list:
        poscar_positive_file.writelines(item)

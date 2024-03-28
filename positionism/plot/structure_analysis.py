from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import numpy as np


def energy_vs_latticeconstant(dataframe, var_filename):
    col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"
    col_toten = "toten [eV]"

    for idx in range(dataframe["geometry"].size):
        latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict]
        toten = dataframe.at[idx, col_toten]
        
        a = latticeconstant_structure_dict["a"]

        plt.scatter(a, toten)
    
    plt.title("Lattice constant vs Total energy")
    plt.xlabel("Lattice constant [Å]")
    plt.ylabel("Total energy [eV]")
    plt.show()


def weirdos_directcoor(dataframe, activate_radius):
    if activate_radius == 2 or activate_radius == 3:
        col_coor_weirdos_el = f"coor_weirdos_48htype2_Li"
    elif activate_radius == 1:
        col_coor_weirdos_el = f"coor_weirdos_Li"

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx in range(dataframe["geometry"].size):
        coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]

        # Plot each set of coordinates in the loop
        for coordinates in coor_weirdos_el:
            ax.scatter(*coordinates, marker='o')

    # Set labels
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Weirdos direct coordinate')

    # Show the plot
    plt.show()


def plot_varying_radius_vs_sumweirdosLi(dataframe):
    col_radius_type1 = "radius_type1"
    col_radius_type2 = "radius_type2"
    col_sumweirdosLi = "sumweirdosLi"

    # # dataframe_to_float = dataframe.copy()
    # dataframe_to_float[col_radius_type1] = dataframe_to_float[col_radius_type1].apply(lambda x: float(x[0]))
    # dataframe_to_float[col_radius_type2] = dataframe_to_float[col_radius_type2].apply(lambda x: float(x[0]))
    # dataframe_to_float[col_sumweirdosLi] = dataframe_to_float[col_sumweirdosLi].apply(lambda x: float(x[0]))

    # # %matplotlib inline
    matrix_map = dataframe.pivot_table(index=col_radius_type2, columns=col_radius_type1,values=col_sumweirdosLi)  
    sns.heatmap(matrix_map)


def plot_distweirdos(dataframe):
    col_top1_sorted_idxweirdo_dist_el = "top1_sorted_idxweirdo_dist_Li"
    col_top1_sorted_idxweirdo_label_el = "top1_sorted_idxweirdo_label_Li"
    col_top1_sorted_idxweirdo_coor_el = "top1_sorted_idxweirdo_coor_Li"
    col_top1_sorted_idxweirdo_file_el = "top1_sorted_idxweirdo_file_Li"

    val = 0.

    dist_weirdos_el_appendend = []
    y_appended = []
    label_weirdos_el_appended = []
    coor_weirdos_el_appended = []
    file_weirdos_el_appended = []

    for idx in range(dataframe["geometry"].size):
        dist_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_dist_el].values()
        label_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_label_el].values()
        coor_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_coor_el].values()
        file_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_file_el].values()

        for single_dist in dist_weirdos_el:
            dist_weirdos_el_appendend.append(single_dist[0])
            y_appended.append(np.zeros_like(single_dist[0]) + val)

        for single_label in label_weirdos_el:
            label_weirdos_el_appended.append(single_label[0])

        for single_coor in coor_weirdos_el:
            coor_weirdos_el_appended.append(single_coor[0])
        
        for single_file in file_weirdos_el:
            file_weirdos_el_appended.append(single_file[0])

    df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'y': y_appended, 'coor': coor_weirdos_el_appended, 'file': file_weirdos_el_appended})

    # fig = px.scatter(df, x = 'dist', y = 'y', color='label', color_discrete_map={'48htype1': 'red', '48htype2': 'blue', '24g': 'green'})
    fig = px.scatter(df, 
                    x = 'dist', 
                    y = 'y', 
                    color ='label', 
                    hover_data = ['dist', 'label', 'coor', 'file'])

    fig.show(config={'scrollZoom': True})


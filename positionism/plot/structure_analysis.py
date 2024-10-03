from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import numpy as np

# Enable LaTeX and set the font to Computer Modern
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

plt.rcParams['axes.titlesize'] = 12  # Set the font size for the plot title
plt.rcParams['axes.labelsize'] = 12  # Set the font size for the x and y labels
plt.rcParams['xtick.labelsize'] = 12  # Set the font size for the x tick labels
plt.rcParams['ytick.labelsize'] = 12  # Set the font size for the y tick labels
plt.rcParams['legend.fontsize'] = 12  # Set the font size for legend

def energy_vs_latticeconstant(dataframe, direc_restructure_destination, litype, var_filename, interpolate):
    col_latticeconstant_structure_dict = f"latticeconstant_structure_dict_{var_filename}"
    col_toten = "toten [eV]"

    if interpolate == True:

        lattice_constants = []
        total_energies = []

        for idx in range(dataframe["geometry"].size):
            latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict]
            toten = dataframe.at[idx, col_toten]
            
            a = latticeconstant_structure_dict["a"]

            lattice_constants.append(a)
            total_energies.append(toten)

        lattice_constants = np.array(lattice_constants)
        total_energies = np.array(total_energies)        

        # # Linear interpolation
        # interp_func = interp1d(lattice_constants, total_energies, kind='linear')

        # Perform linear regression to find the slope (m) and intercept (c)
        m, c = np.polyfit(lattice_constants, total_energies, 1)

        # Set the figure size to 5x3
        plt.figure(figsize=(5, 3))

        # Plot scatter plot
        plt.scatter(lattice_constants, total_energies, label='Data points')

        # # Plot linear interpolation line
        # x_values = np.linspace(min(lattice_constants), max(lattice_constants), 100)
        # plt.plot(x_values, interp_func(x_values), color='red', label='Linear interpolation')

        # Set x and y ticks at every 0.5 interval
        # plt.xticks(np.arange(round(min(lattice_constants),2), round(max(lattice_constants),2) + 0.5, 0.5))
        plt.yticks(np.arange(round(min(total_energies),2), round(max(total_energies),2) + 0.5, 1.0))

        # Plot linear regression line
        plt.plot(lattice_constants, m*lattice_constants + c, color='red', label=f'Linear fit: y = {m:.2f}x + {c:.2f}')
        
        # plt.title(r"Lattice constant vs Total energy")
        plt.xlabel(r"Lattice constant [\AA]")
        plt.ylabel(r"Energy [eV]")

        # Save the plot to a PDF file
        plt.tight_layout()
        plt.savefig(f"{direc_restructure_destination}/energy_vs_latticeconstant_litype{litype}.pdf", format='pdf')

        plt.show()

    else:
        for idx in range(dataframe["geometry"].size):
            latticeconstant_structure_dict = dataframe.at[idx, col_latticeconstant_structure_dict]
            toten = dataframe.at[idx, col_toten]
            
            a = latticeconstant_structure_dict["a"]

            plt.scatter(a, toten)
        
        plt.title("Lattice constant vs Total energy")
        plt.xlabel("Lattice constant [Å]")
        plt.ylabel("Total energy [eV]")
        plt.show()


def weirdos_directcoor(dataframe, direc_restructure_destination, activate_radius, litype):
    if activate_radius == 2 or activate_radius == 3:
        col_coor_weirdos_el = f"coor_weirdos_48htype2_Li"
    elif activate_radius == 1:
        col_coor_weirdos_el = f"coor_weirdos_Li"

    # Create a figure and a 3D axis
    fig, ax = plt.subplots(figsize=(6.7, 5.9))
    # fig, ax = plt.subplots(figsize=(3.3, 3))
    ax = fig.add_subplot(111, projection='3d')

    for idx in range(dataframe["geometry"].size):
        coor_weirdos_el = dataframe.at[idx, col_coor_weirdos_el]

        # Plot each set of coordinates in the loop
        for coordinates in coor_weirdos_el:
            ax.scatter(*coordinates, marker='o')

    # Set labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # ax.set_title('Weirdos direct coordinate')

    plt.tight_layout()
    plt.savefig(f"{direc_restructure_destination}/weirdos_coor_litype{litype}.pdf", format='pdf')

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


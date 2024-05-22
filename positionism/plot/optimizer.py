import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

plt.rcParams['axes.titlesize'] = 12  # Set the font size for the plot title
plt.rcParams['axes.labelsize'] = 12  # Set the font size for the x and y labels
plt.rcParams['xtick.labelsize'] = 12  # Set the font size for the x tick labels
plt.rcParams['ytick.labelsize'] = 12  # Set the font size for the y tick labels
plt.rcParams['legend.fontsize'] = 12  # Set the font size for legend

def heatmap_x_y(dataframe, folder_results_iter_type):
    decimals = 5
    dataframe['x'] = dataframe['x'].apply(lambda x: round(x, decimals))
    dataframe['y'] = dataframe['y'].apply(lambda x: round(x, decimals))
    dataframe['z']  = dataframe['z'].apply(lambda x: round(x, decimals))

    # Create pivot table
    matrix_map = dataframe.pivot_table(index='y', columns='x', values='sumweirdosLi')

    # Create the heatmap
    plt.figure(figsize=(3.3, 3))
    ax = sns.heatmap(matrix_map, cmap='viridis')

    # Adjust layout to fit labels
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig(f".{folder_results_iter_type}heatmap.pdf", format='pdf')

    # Show the plot
    plt.show()


def radius_plot(dataframe, folder_results_iter_type):
    # Create the heatmap
    plt.figure(figsize=(5, 3))
    plt.plot(dataframe["radius_type1"], dataframe["sumweirdosLi"], marker='o', linestyle='-')

    plt.xlabel(r'$\text{Radius}$')
    plt.ylabel(r'$\text{Unassigned Li-ions}$')

    # Adjust layout to fit labels
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig(f".{folder_results_iter_type}radius_unassigned.pdf", format='pdf')

    # Show the plot
    plt.show()

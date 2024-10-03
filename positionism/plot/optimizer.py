import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

plt.rcParams['axes.titlesize'] = 12  # Set the font size for the plot title
plt.rcParams['axes.labelsize'] = 12  # Set the font size for the x and y labels
plt.rcParams['xtick.labelsize'] = 12  # Set the font size for the x tick labels
plt.rcParams['ytick.labelsize'] = 12  # Set the font size for the y tick labels
plt.rcParams['legend.fontsize'] = 12  # Set the font size for legend

def heatmap_x_y(dataframe, folder_results_iter_type, litype):
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
    plt.savefig(f".{folder_results_iter_type}heatmap_litype{litype}.pdf", format='pdf')

    # Show the plot
    plt.show()


def radius_plot(dataframe, folder_results_iter_type, litype):
    dataframe = dataframe[0:36]

    # Create the heatmap
    plt.figure(figsize=(5, 3))
    plt.plot(dataframe["radius_type1"], dataframe["sumweirdosLi"], marker='s', linestyle='-')

    plt.xlabel(r'$r_{\text{mapping}}$')
    # plt.ylabel(r'$n_{\text{unassigned Li-ions}}$')
    plt.ylabel(r'Unassigned Li-ions')

    # Adjust layout to fit labels
    plt.tight_layout()

    # Save the plot to a PDF file
    plt.savefig(f".{folder_results_iter_type}radius_unassigned_litype{litype}.pdf", format='pdf')

    # Show the plot
    plt.show()


def plot_bar_with_latex_font(dataframe, folder_results_iter_type, litype):
    # Create the bar plot
    fig = px.bar(dataframe, x="radius_type1", y="amount_empty")

    # Update layout to use a font that resembles LaTeX's Computer Modern and adjust margins
    fig.update_layout(
        font=dict(
            family="Serif",
            size=12
        ),
        # title={
        #     'text': r'$\text{Amount of folder w/o weirdo}$',
        #     'x': 0.5,
        #     'xanchor': 'center',
        #     'yanchor': 'top'
        # },
        xaxis_title=r'Radius',
        yaxis_title=r'Empty folder',
        margin=dict(l=20, r=20, t=50, b=50)  # Adjust margins for a tighter layout
    )

    # Save the plot to a PDF file using Kaleido with a tight layout
    pio.write_image(fig, f".{folder_results_iter_type}bar_plot_empty_df_litype{litype}.pdf", format='pdf', scale=1, width=800, height=600)

    # Show the plot
    fig.show()
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt

from positionism.functional import func_string

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

plt.rcParams['axes.titlesize'] = 12  # Set the font size for the plot title
plt.rcParams['axes.labelsize'] = 12  # Set the font size for the x and y labels
plt.rcParams['xtick.labelsize'] = 12  # Set the font size for the x tick labels
plt.rcParams['ytick.labelsize'] = 12  # Set the font size for the y tick labels
plt.rcParams['legend.fontsize'] = 12  # Set the font size for legend

def get_df_amount_type(dataframe, litype, el):
    # rename from: plot_amount_type
    """
        style: scatter, bar
    """
    col_amount_type_el = f"amount_type_{el}"

    df = pd.DataFrame()
    df['idx_file'] = None

    if litype == 0:
        df['24g'] = None; df['weirdo'] = None
    elif litype == 1:
        df['48htype2'] = None; df['24g'] = None; df['weirdo'] = None    # "48htype2" instead of 1 because 48htype1 is holy grail for interstitial
    elif litype == 2:
        df['48htype1'] = None; df['48htype2'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 3:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 4:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 5:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 6:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 7:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['48htype7'] = None; df['24g'] = None; df['weirdo'] = None
    elif litype == 8:
        df['48htype1'] = None; df['48htype2'] = None; df['48htype3'] = None; df['48htype4'] = None; df['48htype5'] = None; df['48htype6'] = None; df['48htype7'] = None; df['48htype8'] = None; df['24g'] = None; df['weirdo'] = None

    for idx in range(dataframe["geometry"].size):

        amount_type = dataframe.at[idx, col_amount_type_el]
        df.at[idx, 'idx_file'] = idx

        if litype == 0:
            df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 1:
            df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']      # "48htype2" instead of 1 because 48htype1 is holy grail for interstitial
        elif litype == 2:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 3:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 4:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 5:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 6:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 7:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '48htype7'] = amount_type['48htype7']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
        elif litype == 8:
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '48htype2'] = amount_type['48htype2']; df.at[idx, '48htype3'] = amount_type['48htype3']; df.at[idx, '48htype4'] = amount_type['48htype4']; df.at[idx, '48htype5'] = amount_type['48htype5']; df.at[idx, '48htype6'] = amount_type['48htype6']; df.at[idx, '48htype7'] = amount_type['48htype7']; df.at[idx, '48htype8'] = amount_type['48htype8']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']

    return df


def plot_amount_type(df, sorted, direc_restructure_destination, litype, style):
    category_labels = {
        '48htype2': '48h type 1',
        '48htype1': '48h type 2',
        '48htype3': '48h type 3',
        '48htype4': '48h type 4',
        '48htype5': '48h type 5',
        '48htype6': '48h type 6',
        '48htype7': '48h type 7',
        '48htype8': '48h type 8',
        '24g': '24g',
        'weirdo': 'Unassigned'
        # ... add more as needed
    }

    # Define the colors for each category
    category_colors = {
        '48htype2': '#1f77b4',  # Blue
        '48htype1': '#d62728',  # Red  
        '48htype3': '#2ca02c',  # Green #
        '48htype4': '#bcbd22',  # Yellow-green
        '48htype5': '#9467bd',  # Purple
        '48htype6': '#8c564b',  # Brown
        '48htype7': '#e377c2',  # Pink
        '48htype8': '#7f7f7f',  # Gray
        '24g':      '#ff7f0e',  # Orange    #
        'weirdo':   '#17becf'     # Cyan
    }    

    # shift 'idx_file' by 1
    df['idx_file'] = df['idx_file'] + 1
    
    # Define categories
    if litype == 0:
        categories = ["24g", "weirdo"]
    elif litype == 1:
        categories = ["24g", "48htype2", "weirdo"]
    elif litype == 2:
        categories = ["24g", "48htype1", "48htype2", "weirdo"]
    elif litype == 3:
        categories = ["24g", "48htype1", "48htype2", "48htype3", "weirdo"]
    elif litype == 4:
        categories = ["24g", "48htype1", "48htype2", "48htype3", "48htype4", "weirdo"]
    elif litype == 5:
        categories = ["24g", "48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "weirdo"]
    elif litype == 6:
        categories = ["24g", "48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "weirdo"]
    elif litype == 7:
        categories = ["24g", "48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7", "weirdo"]
    elif litype == 8:
        categories = ["24g", "48htype1", "48htype2", "48htype3", "48htype4", "48htype5", "48htype6", "48htype7", "48htype8", "weirdo"]

    wide_df = pd.DataFrame(df)

    long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

    if category_labels:
        # # long_df['category'] = long_df['category'].replace(category_labels)
        long_df['category'] = func_string.replace_values_in_series(long_df['category'], category_labels)

    if style == "bar":
        fig1 = px.bar(long_df, x="idx_file", y="count", color="category")
    elif style == "scatter":
        fig1 = px.scatter(long_df, x="idx_file", y="count", color="category")
    fig1.show()

    # Update layout for LaTeX-like font settings
    fig1.update_layout(
        font=dict(
            family="serif",
            size=12
        ),
        xaxis_title=r'$\text{File index}$',
        yaxis_title=r'$\text{Amount of Li-types}$',
        margin=dict(l=20, r=20, t=50, b=50)  # Adjust margins for a tighter layout
    )

    def create_bar_plot(ax, figsize, font_size):
        fig, ax = plt.subplots(figsize=figsize)
        bottom_positions = [0] * len(df)

        for category in categories:
            ax.bar(df['idx_file'], df[category], bottom=bottom_positions, 
                color=category_colors[category], label=category_labels.get(category, category))
            bottom_positions = [i + j for i, j in zip(bottom_positions, df[category])]

        ax.set_xlabel(r'$\text{File index}$', fontsize=font_size)
        ax.set_ylabel(r'$\text{Amount of Li-types}$', fontsize=font_size)

        return fig, ax

    def adjust_legend_and_figsize(fig, ax, categories, font_size):
        legend = ax.legend(title=r'$\text{Category}$', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(categories))
        plt.setp(legend.get_title(), fontsize=font_size)      
        
        # Check if the legend fits
        fig.canvas.draw()
        legend_width = legend.get_window_extent().width / fig.dpi
        fig_width = fig.get_figwidth()
        
        if legend_width > fig_width:
            # Split legend into two rows
            legend = ax.legend(title=r'$\text{Category}$', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=int(len(categories) / 2))
            plt.setp(legend.get_title(), fontsize=font_size)
            
            # Adjust figure size
            fig_height = fig.get_figheight()
            fig.set_size_inches(fig_width, fig_height + 0.5)  # Increase height to accommodate legend

        plt.tight_layout()
        return fig, ax

    # Create and save multiple plots with different sizes
    fig2, ax2 = create_bar_plot(ax=None, figsize=(8, 3), font_size=12)
    # ax2.legend(title=r'$\text{Category}$', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(categories))
    plt.subplots_adjust(right=0.8)  # Adjust the right side to make space for the legend  
    plt.tight_layout()
    if sorted == "True":
        plt.savefig(f"{direc_restructure_destination}/licategory_plot_sorted_litype{litype}.pdf", format='pdf')
    else:
        plt.savefig(f"{direc_restructure_destination}/licategory_plot_litype{litype}.pdf", format='pdf')

    fig3, ax3 = create_bar_plot(ax=None, figsize=(9.37, 3.9), font_size=12)
    # legend = ax3.legend(title=r'$\text{Category}$', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(categories))
    # # Set the font size for the legend title
    fig3, ax3 = adjust_legend_and_figsize(fig3, ax3, categories, font_size=12)
    # plt.setp(legend.get_title(), fontsize=12)      
    # plt.tight_layout()
    if sorted == "True":
        plt.savefig(f"{direc_restructure_destination}/licategory_plot_sorted_legend_litype{litype}.pdf", format='pdf')
    else:
        plt.savefig(f"{direc_restructure_destination}/licategory_plot_legend_litype{litype}.pdf", format='pdf')

    fig4, ax4 = create_bar_plot(ax=None, figsize=(3.3, 2.1), font_size=10)
    # No legend for this plot
    plt.tight_layout()
    if sorted == "True":
        plt.savefig(f"{direc_restructure_destination}/licategory_plot_sorted_small_litype{litype}.pdf", format='pdf')
    else:
        plt.savefig(f"{direc_restructure_destination}/licategory_plot_small_litype{litype}.pdf", format='pdf')

    # Show the plotly figure
    fig1.show()

    # Show the last matplotlib figure (for illustration purposes)
    plt.show()

def plot_mapped_label_vs_dist_and_histogram(dataframe, litype, category_data, el):
    # rename from: plot_mapped_label_vs_dist_and_histogram
    """
        category_data: mapping, weirdo
        TO DO: correct the map for histogram
    """
    dist_weirdos_el_appendend = []
    label_weirdos_el_appended = []
    idx_appended = []

    if category_data == "mapping":
        col_dist_label_el = f"atom_mapping_{el}_w_dist_label"
    elif category_data == "weirdo":
        col_dist_label_el = f'top1_sorted_idxweirdo_dist_label_{el}'
        col_top1_sorted_idxweirdo_coor_el = f"top1_sorted_idxweirdo_coor_{el}"
        coor_weirdos_el_appended = []    

    for idx in range(dataframe["geometry"].size):
        atom_mapping_el_w_dist_label = dataframe.at[idx, col_dist_label_el]
        if category_data == "weirdo":
            coor_weirdos_el = dataframe.at[idx, col_top1_sorted_idxweirdo_coor_el].values()

        for i in atom_mapping_el_w_dist_label.values():
            if category_data == "mapping":
                dist = i['dist']
                label = i['label']
            elif category_data == "weirdo":
                dist = i[0]['dist']
                label = i[0]['label']

            dist_weirdos_el_appendend.append(dist)
            label_weirdos_el_appended.append(label)
            idx_appended.append(idx)

        if category_data == "weirdo":
            for single_coor in coor_weirdos_el:
                coor_weirdos_el_appended.append(single_coor[0])

    if category_data == "mapping":
        df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'idx_nr': idx_appended})
    elif category_data == "weirdo":
        df = pd.DataFrame({'dist': dist_weirdos_el_appendend, 'label': label_weirdos_el_appended, 'idx_nr': idx_appended, 'coor': coor_weirdos_el_appended})

    fig = px.scatter(df, 
                    x = 'label', 
                    y = 'dist',
                    title = 'Mapped atom type vs its distance'
                    )

    fig.show(config={'scrollZoom': True})

    # if litype == 0:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
    # elif litype == 1:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')
    # elif litype == 2:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')
    # elif litype == 3:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')
    # elif litype == 4:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')
    # elif litype == 5:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')

    #     df_48htype5 = df.loc[df['label'] == '48htype5']
    #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype5')
    # elif litype == 6:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')

    #     df_48htype5 = df.loc[df['label'] == '48htype5']
    #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype5')

    #     df_48htype6 = df.loc[df['label'] == '48htype6']
    #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype6')
    # elif litype == 7:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')

    #     df_48htype5 = df.loc[df['label'] == '48htype5']
    #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype5')

    #     df_48htype6 = df.loc[df['label'] == '48htype6']
    #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype6')

    #     df_48htype7 = df.loc[df['label'] == '48htype7']
    #     plt.hist(df_48htype7['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype7')
    # elif litype == 8:
    #     df_24g = df.loc[df['label'] == '24g']
    #     plt.hist(df_24g['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 24g')
            
    #     df_48htype1 = df.loc[df['label'] == '48htype1']
    #     plt.hist(df_48htype1['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype1')

    #     df_48htype2 = df.loc[df['label'] == '48htype2']
    #     plt.hist(df_48htype2['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype2')

    #     df_48htype3 = df.loc[df['label'] == '48htype3']
    #     plt.hist(df_48htype3['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype3')

    #     df_48htype4 = df.loc[df['label'] == '48htype4']
    #     plt.hist(df_48htype4['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype4')

    #     df_48htype5 = df.loc[df['label'] == '48htype5']
    #     plt.hist(df_48htype5['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype5')

    #     df_48htype6 = df.loc[df['label'] == '48htype6']
    #     plt.hist(df_48htype6['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype6')

    #     df_48htype7 = df.loc[df['label'] == '48htype7']
    #     plt.hist(df_48htype7['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype7')

    #     df_48htype8 = df.loc[df['label'] == '48htype8']
    #     plt.hist(df_48htype8['dist'], color='lightgreen', ec='black', bins=15)
    #     plt.title('Distribution of Distances for 48htype8')

    return df

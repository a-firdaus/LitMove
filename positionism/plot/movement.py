from matplotlib import pyplot as plt
import pandas as pd
import mplcursors
import mpldatacursor
import plotly.express as px
from adjustText import adjust_text
import plotly.io as pio
from varname import nameof

from positionism.functional import func_string

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

plt.rcParams['axes.titlesize'] = 12  # Set the font size for the plot title
plt.rcParams['axes.labelsize'] = 12  # Set the font size for the x and y labels
plt.rcParams['xtick.labelsize'] = 12  # Set the font size for the x tick labels
plt.rcParams['ytick.labelsize'] = 12  # Set the font size for the y tick labels
plt.rcParams['legend.fontsize'] = 12  # Set the font size for legend

pio.templates.default = "plotly_white"

# class Distance:
def plot_distance(df_distance, max_mapping_radius, activate_shifting_x, activate_diameter_line, Li_idxs):

    diameter_24g48h = max_mapping_radius * 2

    # x = df_distance.index
    if activate_shifting_x == True:
        x = [xi + 0.5 for xi in range(len(df_distance))]
    else:
        x = range(len(df_distance))

    # # fig = plt.figure()
    # fig = plt.figure(figsize=(800/96, 600/96))  # 800x600 pixels, assuming 96 DPI
    # ax = plt.subplot(111)

    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size in inches

    lines = []

    colors = ['b', 'g', 'r', 'c', 'm', 'y'] #, 'k']

    # for i in df_distance.index:
    for i in range(len(df_distance.columns)):

        line_color = colors[i % len(colors)]  # Cycle through colors list

        if Li_idxs == "all" or i in Li_idxs:
            # # i = i
            # # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
            line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", color=line_color, linewidth=2)  # Set line width to 2 pixels
            lines.append(line)
            # label = f"{i}" if Li_idxs == "all" else None
            # line, = ax.plot(x, df_distance[f"{i}"], label=label)
            # lines.append(line)

        # if type(Li_idxs) == list:
        #     for j in Li_idxs:
        #         if i == j:
        #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
        #             lines.append(line)

    # # ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}')
    if activate_diameter_line == True:
        ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}', linewidth=1)  # Set line width to 1 pixel

    # plt.title(f"Geometry {geo} with d={diameter_24g48h}")
        
    # Explicitly set x-ticks
    # ax.set_xticks(x)
    if activate_shifting_x == True:
        ax.set_xticks([0,1,2,3,4,5,6,7,8])

    # Optionally, if you want to label each tick with the original index before adjustment:
    # # # if activate_shifting_x == True:
    # # #     ax.set_xticklabels([str(int(xi - 0.5)) for xi in x])

    # Shrink current axis's height by 10% on the bottom
        # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    handles, labels = ax.get_legend_handles_labels()

    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)

    # Enable cursor information
    mplcursors.cursor(hover=True)

    # Enable zooming with cursor
    mpldatacursor.datacursor(display='multiple', draggable=True)

    plt.show()


# class Occupancy:
def get_df_occupancy(dataframe, strict_count):
    # rename from: plot_occupancy
    """
    strict_count: True, False
    """
    if strict_count == True:
        col_occupancy = "occupancy_strict"
    else:
        col_occupancy = "occupancy_notstrict"

    df = pd.DataFrame()
    df['idx_file'] = None
    df['2'] = None
    df['1'] = None
    df['0'] = None
    df['48htype1'] = None
    df['weirdo'] = None

    for idx in range(dataframe["geometry"].size):

        occupancy = dataframe.at[idx, col_occupancy]

        # for key, val in occupancy.items():
        df.at[idx, 'idx_file'] = idx
        df.at[idx, '2'] = occupancy['2']
        df.at[idx, '1'] = occupancy['1']
        df.at[idx, '0'] = occupancy['0']
        df.at[idx, '48htype1'] = occupancy['48htype1']
        df.at[idx, 'weirdo'] = occupancy['weirdo']

    return df

def plot_occupancy(dataframe, sorted, direc_restructure_destination, litype, strict_count):
    if strict_count:
        col_occupancy = "occupancy_strict"
    else:
        col_occupancy = "occupancy_notstrict"

    # Define categories
    categories = ['2', '1', '0', '48htype1', 'weirdo']

    # Initialize DataFrame with the necessary columns
    df = pd.DataFrame(columns=['idx_file'] + categories)

    category_labels = {
        '2': 'Doubly occupied',
        '1': 'Singly occupied',
        '0': 'Empty',
        '48htype1': '48h type 2',
        'weirdo': 'Unassigned'
        # ... add more as needed
    }

     # Define the colors for each category
    category_colors = {
        '2': '#1f77b4',  # Blue
        '1': '#ff7f0e',  # Orange
        '0': '#313131',  # Dar grey
        '48htype1': '#d62728',  # Red  
        'weirdo':   '#17becf'     # Cyan
    }       

    # Define variable name for file saving based on "sorted" or not
    nameof_dataframe = nameof(dataframe)
    sorted = "True" if "sorted" in "df_mapping_metainfo_sorted" else "False"

    for idx in range(dataframe["geometry"].size):
        occupancy = dataframe.at[idx, col_occupancy]
        df.at[idx, 'idx_file'] = idx + 1  # Shift file index by 1
        df.at[idx, '2'] = occupancy['2']
        df.at[idx, '1'] = occupancy['1']
        df.at[idx, '0'] = occupancy['0']
        df.at[idx, '48htype1'] = occupancy['48htype1']
        df.at[idx, 'weirdo'] = occupancy['weirdo']

    wide_df = pd.DataFrame(df)

    # Convert wide format to long format
    long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

    if category_labels:
        long_df['category'] = func_string.replace_values_in_series(long_df['category'], category_labels)

    fig1 = px.bar(long_df, x="idx_file", y="count", color="category")

    # Update layout for LaTeX-like font settings
    fig1.update_layout(
        font=dict(
            family="serif",
            size=12
        ),
        xaxis_title=r'$\text{File index}$',
        yaxis_title=r'$\text{Amount of Li occupancy}$',
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

    # Create and save multiple plots with different sizes
    fig2, ax2 = create_bar_plot(ax=None, figsize=(8, 3), font_size=12)
    # ax2.legend(title=r'$\text{Category}$', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(categories))
    plt.subplots_adjust(right=0.8)  # Adjust the right side to make space for the legend  
    plt.tight_layout()
    if sorted == "True":
        if strict_count:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_sorted_strict_litype{litype}.pdf", format='pdf')
        else:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_sorted_litype{litype}.pdf", format='pdf')
    else:
        if strict_count:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_strict_litype{litype}.pdf", format='pdf')
        else:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_litype{litype}.pdf", format='pdf')

    fig3, ax3 = create_bar_plot(ax=None, figsize=(9.37, 3.9), font_size=12)
    legend = ax3.legend(title=r'$\text{Category}$', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(categories))
    # Set the font size for the legend title
    plt.setp(legend.get_title(), fontsize=12)      
    plt.tight_layout()
    if sorted == "True":
        if strict_count:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_sorted_legend_strict_litype{litype}.pdf", format='pdf')
        else:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_sorted_legend_litype{litype}.pdf", format='pdf')
    else:
        if strict_count:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_legend_strict_litype{litype}.pdf", format='pdf')
        else:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_legend_litype{litype}.pdf", format='pdf')


    fig4, ax4 = create_bar_plot(ax=None, figsize=(3.3, 2.1), font_size=10)
    # No legend for this plot
    plt.tight_layout()
    if sorted == "True":
        if strict_count:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_sorted_strict_small_litype{litype}.pdf", format='pdf')
        else:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_sorted_small_litype{litype}.pdf", format='pdf')
    else:
        if strict_count:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_strict_small_litype{litype}.pdf", format='pdf')
        else:
            plt.savefig(f"{direc_restructure_destination}/occupancy_plot_small_litype{litype}.pdf", format='pdf')


    # fig5, ax5 = create_bar_plot(ax=None, figsize=(6.55, 3.33))
    # ax5.legend(title=r'$\text{Category}$', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(categories))
    # plt.tight_layout()
    # if strict_count:
    #     plt.savefig(f"{direc_restructure_destination}/occupancy_plot_legend_strict_small_litype{litype}.pdf", format='pdf')
    # else:
    #     plt.savefig(f"{direc_restructure_destination}/occupancy_plot_legend_small_litype{litype}.pdf", format='pdf')

    # Show the plotly figure
    fig1.show()

    # Show the last matplotlib figure (for illustration purposes)
    plt.show()


# class TupleCage:
# # # # def plot_cage_tuple_label(df_distance, df_type, df_idx_tuple, max_mapping_radius, litype, category_labels, activate_diameter_line, activate_relabel_s_i, Li_idxs):

# # # #     # df_distance = df_distance.iloc[:,:amount_Li]
# # # #     # df_type = df_type.iloc[:,:amount_Li]
# # # #     # df_idx_tuple = df_idx_tuple.iloc[:,:amount_Li]

# # # #     diameter_24g48h = max_mapping_radius * 2

# # # #     x = range(len(df_distance))

# # # #     # # fig = plt.figure()
# # # #     # fig = plt.figure(figsize=(800/96, 600/96))  # 800x600 pixels, assuming 96 DPI
# # # #     # ax = plt.subplot(111)

# # # #     fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size in inches

# # # #     lines = []
# # # #     texts = []

# # # #     # type_marker_mapping = {
# # # #     #     '48htype1': 'o',
# # # #     #     '48htype2': 's',
# # # #     #     '48htype3': '^',
# # # #     #     '48htype4': 'D',
# # # #     #     'weirdos': 'X',
# # # #     #     '24g': 'v'    
# # # #     # }

# # # #     # # type_marker_mapping = {
# # # #     # #     '48htype1': ('o', 'r'),  # Example: Circle marker with red color
# # # #     # #     '48htype2': ('s', 'g'),  # Square marker with green color
# # # #     # #     '48htype3': ('^', 'b'),  # Triangle marker with blue color
# # # #     # #     '48htype4': ('D', 'c'),  # Diamond marker with cyan color
# # # #     # #     'weirdos': ('X', 'm'),   # X marker with magenta color
# # # #     # #     '24g': ('v', 'y')        # Triangle_down marker with yellow color
# # # #     # # }

# # # #     type_marker_mapping = {
# # # #         '48htype1': ('o'),  # Example: Circle marker with red color
# # # #         '48htype2': ('s'),  # Square marker with green color
# # # #         '48htype3': ('^'),  # Triangle marker with blue color
# # # #         '48htype4': ('D'),  # Diamond marker with cyan color
# # # #         'weirdos': ('X'),   # X marker with magenta color
# # # #         '24g': ('v')        # Triangle_down marker with yellow color
# # # #     }

# # # #     colors = ['b', 'g', 'r', 'c', 'm', 'y'] #, 'k']  # Example color list
# # # #     # colors = list(mcolors.CSS4_COLORS.values())
# # # #     # colors = [color + (0.7,) for color in mcolors.CSS4_COLORS.values()]
# # # #     # colors = mcolors
# # # #     # names = list(colors)

# # # #     # Define offsets for text position
# # # #     x_offset = 0.02  # Adjust these values as needed
# # # #     y_offset = -0.05  # Adjust these values as needed

# # # #     # Track which labels have been added
# # # #     added_labels = set()

# # # #     # for i in range(24):
# # # #     for i in range(len(df_distance.columns)):
# # # #         if Li_idxs == "all" or i in Li_idxs:
# # # #             column_data = df_distance[f"{i}"]
# # # #             column_val = df_type[f"{i}"]
# # # #             column_idx_tuple = df_idx_tuple[f"{i}"]
# # # #             # type_val = df_type[0, i]
# # # #             # print(type_val)

# # # #             line_color = colors[i % len(colors)]  # Cycle through colors list
# # # #             # # # # # # # line_color = colors[i % len(colors)] if i < len(colors) else 'black'  # Use a default color if the index exceeds available colors

# # # #             # # # for j in x:
# # # #             for j, (y_val, type_val, idx_tuple_val) in enumerate(zip(column_data, column_val, column_idx_tuple)):
# # # #                 # type = column_val[j]
# # # #                 # idx_tuple = column_idx_tuple[j]

# # # #                 # marker_style = type_marker_mapping.get(column_val, 'o')  # Get the marker style for the type
# # # #                 # # marker_style = type_marker_mapping.get(type, 'o')  # Get the marker style for the type
# # # #                 # # # # # # # marker_style, marker_color = type_marker_mapping.get(type_val, ('o','k'))  # Get the marker style for the type
# # # #                 marker_style = type_marker_mapping.get(type_val, ('o'))  # Get the marker style for the type
# # # #                 # # # # # # ax.scatter(j, df_distance[f"{i}"][j], label=f"Type: {column_val}", marker=marker_style, s=100)
# # # #                 # # # # # label = f"{type_val}" if type_val not in added_labels else None
# # # #                 # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100)
# # # #                 # # # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
# # # #                 # # # # # ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
# # # #                 # # # # # added_labels.add(type_val)
# # # #                 mapped_label = category_labels.get(type_val, type_val)  # Use the original type_val if it's not found in category_labels
# # # #                 # Use mapped_label for the label. Only add it if it's not already added.
# # # #                 label = mapped_label if mapped_label not in added_labels else None
# # # #                 ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color=line_color, alpha=0.5)
# # # #                 if label:  # If a label was added, record it as added
# # # #                     added_labels.add(mapped_label)

# # # #                 # # # # ax.text(j, df_distance.iloc[j, i], str(int(idx_tuple_val)), color=line_color, fontsize=20)
# # # #                 # # # # # ax.text(j, y_val, str(int(idx_tuple_val)), color=line_color, fontsize=20)
# # # #                 # Apply offsets to text position
# # # #                 text_x = j + x_offset * ax.get_xlim()[1]  # Adjust text x-position
# # # #                 text_y = y_val + y_offset * ax.get_ylim()[1]  # Adjust text y-position

# # # #                 # Check if type_val is 'weirdos' and change text color to black, else use the line color
# # # #                 text_color = 'black' if type_val in ['weirdos', '48htype1'] else line_color

# # # #                 print(idx_tuple_val)
# # # #                 if idx_tuple_val == 'x':
# # # #                     text = ax.text(text_x, text_y, idx_tuple_val, color=text_color, fontsize=18)
# # # #                 else:
# # # #                     # if activate_relabel_s_i:
# # # #                     #     if litype == 4:
# # # #                     #         if idx_tuple_val in ['48htype2', '48htype3', '48htype4']:
# # # #                     #             text = ax.text(text_x, text_y, str(int(idx_tuple_val))+"s", color=text_color, fontsize=18)
# # # #                     #         elif idx_tuple_val in ['48htype1']:
# # # #                     #             text = ax.text(text_x, text_y, str(int(idx_tuple_val))+"i", color=text_color, fontsize=18)
# # # #                     # else:
# # # #                     text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=text_color, fontsize=18)
# # # #                 texts.append(text)

# # # #                 # # # # # # # # # # # # # # text = ax.text(j+x_offset, y_val+y_offset, str(int(idx_tuple_val)), color=line_color, fontsize=15)
# # # #                 # # # # # # # # if idx_tuple_val == f' ':
# # # #                 # # # # # # # #     text = ax.text(text_x, text_y, idx_tuple_val, color=line_color, fontsize=18)
# # # #                 # # # # # # # # else:
# # # #                 # # # # # # # #     text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=line_color, fontsize=18)
# # # #                 # # # # # # # # texts.append(text)

# # # #             # # i = i
# # # #             # # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
# # # #             # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", linewidth=2, marker=marker_style, markersize=10)  # Set line width to 2 pixels

# # # #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", color=line_color, linewidth=2)  # Set line width to 2 pixels
# # # #             # ax.text(i, value, str(int(idx_value)), color=line_color, fontsize=8)

# # # #             lines.append(line)
# # # #             # label = f"{i}" if Li_idxs == "all" else None
# # # #             # line, = ax.plot(x, df_distance[f"{i}"], label=label)
# # # #             # lines.append(line)

# # # #         # if type(Li_idxs) == list:
# # # #         #     for j in Li_idxs:
# # # #         #         if i == j:
# # # #         #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
# # # #         #             lines.append(line)

# # # #     adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    
# # # #     # # ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}')
# # # #     if activate_diameter_line == True:
# # # #         ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}', linewidth=1)  # Set line width to 1 pixel

# # # #     # Set the y-axis to only show ticks at 0, 1, 2, 3
# # # #     plt.yticks([0, 1, 2, 3])

# # # #     # plt.title(f"Geometry {geo} with d={diameter_24g48h}")

# # # #     # Shrink current axis's height by 10% on the bottom
# # # #         # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
# # # #     box = ax.get_position()
# # # #     ax.set_position([box.x0, box.y0 + box.height * 0.1,
# # # #                     box.width, box.height * 0.9])

# # # #     handles, labels = ax.get_legend_handles_labels()

# # # #     # Set marker color in legend box to black
# # # #     # # legend_handles = [(h[0], h[1], {'color': 'black'}) for h in handles]
# # # #     legend_handles = [(h, {'color': 'black'}) for h in handles]

# # # #     ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
# # # #             fancybox=True, shadow=True, ncol=5)
    
# # # #     # # # ax.legend(handles=legend_handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
# # # #     # # #         fancybox=True, shadow=True, ncol=5, handlelength=2, handler_map={tuple: HandlerTuple(ndivide=None)})
# # # #     # ax.legend(handles=legend_handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
# # # #     #         fancybox=True, shadow=True, ncol=5)

# # # #     # Enable cursor information
# # # #     mplcursors.cursor(hover=True)

# # # #     # Enable zooming with cursor
# # # #     mpldatacursor.datacursor(display='multiple', draggable=True)

# # # #     plt.show()


def plot_cage_tuple_label(df_distance, df_type, df_idx_tuple, max_mapping_radius, activate_diameter_line, activate_relabel_s_i, Li_idxs):
    category_labels = {
        '48htype2': '48h type 1',
        '48htype1': '48h type 2',
        '48htype3': '48h type 3',
        '48htype4': '48h type 4',
        '24g': '24g',
        'weirdo': 'Unassigned'
        # ... add more as needed
    }    

    # df_distance = df_distance.iloc[:,:amount_Li]
    # df_type = df_type.iloc[:,:amount_Li]
    # df_idx_tuple = df_idx_tuple.iloc[:,:amount_Li]

    diameter_24g48h = max_mapping_radius * 2

    x = range(len(df_distance))

    # # fig = plt.figure()
    # fig = plt.figure(figsize=(800/96, 600/96))  # 800x600 pixels, assuming 96 DPI
    # ax = plt.subplot(111)

    fig, ax = plt.subplots(figsize=(10, 6))  # Set the figure size in inches

    lines = []
    texts = []

    # type_marker_mapping = {
    #     '48htype1': 'o',
    #     '48htype2': 's',
    #     '48htype3': '^',
    #     '48htype4': 'D',
    #     'weirdos': 'X',
    #     '24g': 'v'    
    # }

    # # type_marker_mapping = {
    # #     '48htype1': ('o', 'r'),  # Example: Circle marker with red color
    # #     '48htype2': ('s', 'g'),  # Square marker with green color
    # #     '48htype3': ('^', 'b'),  # Triangle marker with blue color
    # #     '48htype4': ('D', 'c'),  # Diamond marker with cyan color
    # #     'weirdos': ('X', 'm'),   # X marker with magenta color
    # #     '24g': ('v', 'y')        # Triangle_down marker with yellow color
    # # }

    type_marker_mapping = {
        '48htype1': ('o'),  # Example: Circle marker with red color
        '48htype2': ('s'),  # Square marker with green color
        '48htype3': ('^'),  # Triangle marker with blue color
        '48htype4': ('D'),  # Diamond marker with cyan color
        'weirdos': ('X'),   # X marker with magenta color
        '24g': ('v')        # Triangle_down marker with yellow color
    }

    colors = ['b', 'g', 'r', 'c', 'm', 'y'] #, 'k']  # Example color list
    # colors = list(mcolors.CSS4_COLORS.values())
    # colors = [color + (0.7,) for color in mcolors.CSS4_COLORS.values()]
    # colors = mcolors
    # names = list(colors)

    # Define offsets for text position
    x_offset = 0.02  # Adjust these values as needed
    y_offset = -0.05  # Adjust these values as needed

    # Track which labels have been added
    added_labels = set()

    # for i in range(24):
    for i in range(len(df_distance.columns)):
        if Li_idxs == "all" or i in Li_idxs:
            column_data = df_distance[f"{i}"]
            column_val = df_type[f"{i}"]
            column_idx_tuple = df_idx_tuple[f"{i}"]
            # type_val = df_type[0, i]
            # print(type_val)

            line_color = colors[i % len(colors)]  # Cycle through colors list
            # # # # # # # line_color = colors[i % len(colors)] if i < len(colors) else 'black'  # Use a default color if the index exceeds available colors

            # # # for j in x:
            for j, (y_val, type_val, idx_tuple_val) in enumerate(zip(column_data, column_val, column_idx_tuple)):
                # type = column_val[j]
                # idx_tuple = column_idx_tuple[j]

                # marker_style = type_marker_mapping.get(column_val, 'o')  # Get the marker style for the type
                # # marker_style = type_marker_mapping.get(type, 'o')  # Get the marker style for the type
                # # # # # # # marker_style, marker_color = type_marker_mapping.get(type_val, ('o','k'))  # Get the marker style for the type
                marker_style = type_marker_mapping.get(type_val, ('o'))  # Get the marker style for the type
                # # # # # # ax.scatter(j, df_distance[f"{i}"][j], label=f"Type: {column_val}", marker=marker_style, s=100)
                # # # # # label = f"{type_val}" if type_val not in added_labels else None
                # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100)
                # # # # # # # # # ax.scatter(j, df_distance.iloc[j, i], label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
                # # # # # ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color = marker_color, alpha = 0.5)
                # # # # # added_labels.add(type_val)
                mapped_label = category_labels.get(type_val, type_val)  # Use the original type_val if it's not found in category_labels
                # Use mapped_label for the label. Only add it if it's not already added.
                label = mapped_label if mapped_label not in added_labels else None
                ax.scatter(j, y_val, label=label, marker=marker_style, s=100, color=line_color, alpha=0.5)
                if label:  # If a label was added, record it as added
                    added_labels.add(mapped_label)

                # # # # ax.text(j, df_distance.iloc[j, i], str(int(idx_tuple_val)), color=line_color, fontsize=20)
                # # # # # ax.text(j, y_val, str(int(idx_tuple_val)), color=line_color, fontsize=20)
                # Apply offsets to text position
                text_x = j + x_offset * ax.get_xlim()[1]  # Adjust text x-position
                text_y = y_val + y_offset * ax.get_ylim()[1]  # Adjust text y-position

                # Check if type_val is 'weirdos' and change text color to black, else use the line color
                text_color = 'black' if type_val in ['weirdos', '48htype1'] else line_color

                if activate_relabel_s_i:        
                    # print(idx_tuple_val)
                    if type_val in ['48htype2', '48htype3', '48htype4', '24g']:
                        text = ax.text(text_x, text_y, str(int(idx_tuple_val))+"s", color=text_color, fontsize=18)
                    elif type_val in ['48htype1']:
                        text = ax.text(text_x, text_y, str(int(idx_tuple_val))+"i", color=text_color, fontsize=18)
                    elif type_val == 'weirdos':
                        # idx_tuple_val = 'x'
                        # # print(idx_tuple_val)
                        text = ax.text(text_x, text_y, idx_tuple_val, color=text_color, fontsize=18)
                        # # print(text)
                else:
                    if idx_tuple_val == 'x':
                        text = ax.text(text_x, text_y, idx_tuple_val, color=text_color, fontsize=18)
                    else:
                        text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=text_color, fontsize=18)
                texts.append(text)

                # # # # # # # # # # # # # # text = ax.text(j+x_offset, y_val+y_offset, str(int(idx_tuple_val)), color=line_color, fontsize=15)
                # # # # # # # # if idx_tuple_val == f' ':
                # # # # # # # #     text = ax.text(text_x, text_y, idx_tuple_val, color=line_color, fontsize=18)
                # # # # # # # # else:
                # # # # # # # #     text = ax.text(text_x, text_y, str(int(idx_tuple_val)), color=line_color, fontsize=18)
                # # # # # # # # texts.append(text)

            # # i = i
            # # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
            # line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", linewidth=2, marker=marker_style, markersize=10)  # Set line width to 2 pixels

            line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}", color=line_color, linewidth=2)  # Set line width to 2 pixels
            # ax.text(i, value, str(int(idx_value)), color=line_color, fontsize=8)

            lines.append(line)
            # label = f"{i}" if Li_idxs == "all" else None
            # line, = ax.plot(x, df_distance[f"{i}"], label=label)
            # lines.append(line)

        # if type(Li_idxs) == list:
        #     for j in Li_idxs:
        #         if i == j:
        #             line, = ax.plot(x, df_distance[f"{i}"], label=f"{i}")
        #             lines.append(line)

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    # # ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}')
    if activate_diameter_line == True:
        ax.axhline(y=diameter_24g48h, color='b', linestyle=':', label=f'd_mapping = {diameter_24g48h:.3f}', linewidth=1)  # Set line width to 1 pixel

    # ax.set_ylim(-0.5, 3.5)

    # Set the y-axis to only show ticks at 0, 1, 2, 3
    plt.yticks([0, 1, 2, 3])

    # plt.title(f"Geometry {geo} with d={diameter_24g48h}")

    # Shrink current axis's height by 10% on the bottom
        # source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])

    handles, labels = ax.get_legend_handles_labels()

    # Set marker color in legend box to black
    # # legend_handles = [(h[0], h[1], {'color': 'black'}) for h in handles]
    legend_handles = [(h, {'color': 'black'}) for h in handles]

    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=5)
    
    # # # ax.legend(handles=legend_handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
    # # #         fancybox=True, shadow=True, ncol=5, handlelength=2, handler_map={tuple: HandlerTuple(ndivide=None)})
    # ax.legend(handles=legend_handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #         fancybox=True, shadow=True, ncol=5)

    # Enable cursor information
    mplcursors.cursor(hover=True)

    # Enable zooming with cursor
    mpldatacursor.datacursor(display='multiple', draggable=True)

    plt.show()

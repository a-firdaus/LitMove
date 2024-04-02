import pandas as pd
import plotly.express as px

from functional import func_string


def plot_amount_type(dataframe, litype, el, style, category_labels = None):
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
        df['48htype1'] = None; df['24g'] = None; df['weirdo'] = None
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
            df.at[idx, '48htype1'] = amount_type['48htype1']; df.at[idx, '24g'] = amount_type['24g']; df.at[idx, 'weirdo'] = amount_type['weirdo']
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

    wide_df = pd.DataFrame(df)

    long_df = pd.melt(wide_df, id_vars=['idx_file'], var_name='category', value_name='count')

    if category_labels:
        # # long_df['category'] = long_df['category'].replace(category_labels)
        long_df['category'] = func_string.replace_values_in_series(long_df['category'], category_labels)

    if style == "bar":
        fig = px.bar(long_df, x="idx_file", y="count", color="category", title="Idx file vs Li type")
    elif style == "scatter":
        fig = px.scatter(long_df, x="idx_file", y="count", color="category", title="Idx file vs Li type")
    fig.show()

    return df


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

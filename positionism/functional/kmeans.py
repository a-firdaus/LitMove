import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def kmeans_cluster_atoms(coor_atoms, amount_clusters):
    # source: https://stackoverflow.com/questions/64987810/3d-plotting-of-a-dataset-that-uses-k-means
    kmeans = KMeans(n_clusters=amount_clusters)                # Number of clusters
    kmeans = kmeans.fit(coor_atoms)                          # Fitting the input data
    labels = kmeans.predict(coor_atoms)                      # Getting the cluster labels
    centroids = kmeans.cluster_centers_             # Centroid values
    # print("Centroids are:", centroids)              # From sci-kit learn

    fig = plt.figure(figsize=(10,10))
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    # x = np.array(labels==0)
    # y = np.array(labels==1)
    # z = np.array(labels==2)


    # ax.scatter(coor_atoms[x][:, 0], coor_atoms[x][:, 1], coor_atoms[x][:, 2], color='red')
    # ax.scatter(coor_atoms[y][:, 0], coor_atoms[y][:, 1], coor_atoms[y][:, 2], color='blue')
    # ax.scatter(coor_atoms[z][:, 0], coor_atoms[z][:, 1], coor_atoms[z][:, 2], color='green')
    # ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
    #             marker='x', s=169, linewidths=10,
    #             color='black', zorder=50)
    # # ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c="black",s=150,label="Centers",alpha=1) # for dot marker

    # Define a colormap for different clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, amount_clusters))

    for cluster_label in range(amount_clusters):
        cluster_mask = np.array(labels == cluster_label)
        ax.scatter(
            coor_atoms[cluster_mask][:, 0],
            coor_atoms[cluster_mask][:, 1],
            coor_atoms[cluster_mask][:, 2],
            color=colors[cluster_label],
            label=f'Cluster {cluster_label}'
        )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        marker='x',
        s=169,
        linewidths=10,
        color='black',
        zorder=50,
        label='Centroids'
    )

    ax.legend()

    return centroids, labels


def create_POSCAR_atoms_centroids_appended(coor_atoms, coor_centroids, destination_directory, lattice_constant, filename):

    # Define lattice constants (you might need to adjust these based on your actual system)
    lattice_constants_matrix = np.array([
        [lattice_constant, 0.0000000000000000, 0.0000000000000000],
        [0.0000000000000000, lattice_constant, 0.0000000000000000],
        [0.0000000000000000, 0.0000000000000000, lattice_constant]
    ])

    # Define the header and comment lines for the POSCAR file
    header = "Generated POSCAR"
    comment = "1.0"

    # Define filename
    filename_path = os.path.join(destination_directory, filename)

    # Write the POSCAR file
    with open(filename_path, "w") as f:
        # Write the header and comment
        f.write(header + "\n")
        f.write(comment + "\n")

        # Write the lattice constants
        for row in lattice_constants_matrix:
            f.write(" ".join(map(str, row)) + "\n")

        # K as mock element for centroids
        f.write("Li K\n")

        # Write the number of atoms for each element
        # f.write(" ".join(map(str, np.ones(len(coordinates), dtype=int))) + "\n")
        f.write(str(len(coor_atoms)) + " " + str(len(coor_centroids)) + "\n")  # Number of Li atoms

        # Write the selective dynamics tag (in this case, 'Direct')
        f.write("Direct\n")

        # Write the atomic coordinates
        for coor_atom in coor_atoms:
            # f.write(" ".join(map(str, coord)) + "\n")
            formatted_coor_atom = [format(x, ".16f") for x in coor_atom]
            f.write(" ".join(formatted_coor_atom) + "\n")
        for coor_centroid in coor_centroids:
            formatted_coor_centroid = [format(x, ".16f") for x in coor_centroid]
            f.write(" ".join(formatted_coor_centroid) + "\n")

    # print("POSCAR file created successfully.")


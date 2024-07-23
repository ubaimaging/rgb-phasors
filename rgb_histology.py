import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from tools import phasor, cluster_phasor_plot, rgb2bgr, median_filter

from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_to_polar

from phasorpy.cursors import (
    mask_from_circular_cursor,
    pseudo_color,
)

# ---------------------------------------------------------
#   Analisis de clustering y comparacion de segmentaciones
#   Segmento las RGB con diferentes métodos de clustering, 
#   al menos 3, y comparo con las máscaras que tengo del 
#   fiji, y con la segmentación manual. Los clusters 
#   segmentan 4 tipos grupos: nucleos, eritrocitos, tejido 
#   y espacio alveolar. 

#   Luego en base a eso elijo un algoritmo de clustering 
#   para aplicar a la segmentacion de la base de datos.
# ---------------------------------------------------------

# PARTES: 
    # 1 - Segmentar con clustering 
    # 2 - Cargar las mascaras de fiji
    # 3 - Cargar la segmentación manual 
    # 4 - Comparar todos los métodos, usar alguna correlacion de imagenes 
    # y alguna medida de diferencia pixel a pixel


# Parte 1
part1 = True
if part1:
    im = plt.imread('/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/46-21B6ND02.tif')
    im = im[0:500, 0:500]
    bgr = rgb2bgr(im)
    dc, g, s = phasor(bgr)
    g = median_filter(g, 1)
    s = median_filter(s, 1)

    plt.figure(1)
    plt.imshow(im)

    clusters = True
    # Try different types of clusters
    if clusters:
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

        coord_g = g.flatten()
        coord_s = s.flatten()
        x = np.asarray([coord_g, coord_s]).transpose()
        num_clusters = 4

        par = True
        if par:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
            kmeans.fit(x)
            kmeans_labels = kmeans.predict(x)

            gmm = GaussianMixture(n_components=num_clusters, random_state=0)
            gmm.fit(x)
            gmm_labels = gmm.predict(x)

            otros = False
            if otros:
                spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
                spectral_labels = spectral.fit_predict(x)

                dbscan = DBSCAN(eps=3, min_samples=4)
                dbscan.fit(x)
                dbscan_labels = dbscan.labels_

                cluster_phasor_plot(x, spectral_labels, nclusters=num_clusters)
                cluster_phasor_plot(x, dbscan_labels, nclusters=num_clusters)

            cluster_phasor_plot(x, gmm_labels, nclusters=num_clusters)
            cluster_phasor_plot(x, kmeans_labels, nclusters=num_clusters)


            gmm_mask = gmm_labels.reshape(g.shape[0], g.shape[1])
            kmeans_mask = kmeans_labels.reshape(g.shape[0], g.shape[1])

            # TODO arreglar para que la mascara salga con los colores del cluster
            # bgr = [CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[0], CATEGORICAL[3]] 
            # segmented_image = pseudo_color(dc, mask, colors=bgr)

            plt.figure()
            plt.imshow(gmm_mask)

            plt.figure()
            plt.imshow(kmeans_mask)
            plt.show()

# Parte 2
parte2 = False
if parte2:
    # elegir las 20 imagenes a segmentar y ahí guardarlas en una carpeta
    aux = 1

# ------------------------------------------
#       Extra segmentacion con cursores
# ------------------------------------------
cursores = False
if cursores:
    cursors_real = [0.18, -0.08, -0.2]
    cursors_imag = [-0.07, 0.12, -0.32]
    radius=[0.15, 0.15, 0.35]
    circular_mask = mask_from_circular_cursor(
        g, s, cursors_real, cursors_imag, radius=radius
    )

    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(g.flatten(), s.flatten(), cmap="RdYlGn_r")
    # Plot cursors Blue, Green, Red
    plot.cursor(
        cursors_real[0],
        cursors_imag[0],
        radius=radius[0],
        color=CATEGORICAL[1],
        linestyle='-',
    )

    plot.cursor(
        cursors_real[1],
        cursors_imag[1],
        radius=radius[1],
        color=CATEGORICAL[2],
        linestyle='-',
    )

    plot.cursor(
        cursors_real[2],
        cursors_imag[2],
        radius=radius[2],
        color=CATEGORICAL[0],
        linestyle='-',
    )
    bgr = [CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[0]] 
    segmented_image = pseudo_color(dc, circular_mask, colors=bgr)

    fig, ax = plt.subplots()
    ax.set_title('Segmented image with circular cursors')
    ax.imshow(segmented_image)
    plt.show()
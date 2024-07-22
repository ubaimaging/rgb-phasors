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

# ------------
#    Part 1
# ------------
part1 = True
if part1:
    im = plt.imread('/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/46-21B6ND02.tif')
    bgr = rgb2bgr(im)
    dc, g, s = phasor(bgr)
    g = median_filter(g, 1)
    s = median_filter(s, 1)

    plt.figure(1)
    plt.imshow(im)

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

    clusters = True
    # Try different types of clusters
    if clusters:
        from sklearn.mixture import GaussianMixture
        from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

        coord_g = g.flatten()
        coord_s = s.flatten()
        x = np.asarray([coord_g, coord_s]).transpose()
        num_clusters = 4


        gmm = GaussianMixture(n_components=num_clusters, random_state=0)
        gmm.fit(x)
        gmm_labels = gmm.predict(x)

        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
        kmeans.fit(x)
        kmeans_labels = kmeans.predict(x)

        spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
        spectral.fit(x)
        spectral_labels = spectral.predict(x)

        dbscan = DBSCAN(eps=3, min_samples=num_clusters)
        dbscan.fit(x)
        dbscan_labels = dbscan.predict(x)
    

        cluster_phasor_plot(x, gmm_labels, nclusters=num_clusters)
        cluster_phasor_plot(x, kmeans_labels, nclusters=num_clusters)
        cluster_phasor_plot(x, spectral_labels, nclusters=num_clusters)
        cluster_phasor_plot(x, dbscan_labels, nclusters=num_clusters)

        # mask = labels.reshape(g.shape[0], g.shape[1])
        # bgr = [CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[0], CATEGORICAL[3]] 
        # segmented_image = pseudo_color(dc, mask, colors=bgr)

        # plt.figure()
        # plt.imshow(mask)

        plt.show()
import numpy as np
import matplotlib.pyplot as plt
from phasorpy.plot import PhasorPlot
import tools 

from sklearn.mixture import GaussianMixture

from sklearn.cluster import (
    KMeans, AgglomerativeClustering, SpectralClustering
)

from tools import  construct_label_array_optimized, map_values_to_rgb


def plot_pseudocolor_im(x, xt):
    labels_new = construct_label_array_optimized(xt, x, labels+1)
    imcolor = map_values_to_rgb(labels_new.reshape([256, 256]))
    plt.figure(figsize=(7, 7))
    plt.imshow(imcolor)
    plt.title("Pseudocolor image")

# Use the simulated color wheel and phasor plot to test differets clustering method
# Kmeans 
# GMM 
# Spectral
# Agglomerative 

# Generates RGB color wheel
color_wheel_image = tools.generate_color_wheel_image(256)

# Plot color wheel
plt.figure(figsize=(6, 6))
plt.imshow(color_wheel_image, extent=(-1, 1, -1, 1))
plt.axis('off')
plt.title("RGB Image")

color_wheel_image_nan = tools.replace_with_nan(color_wheel_image)
aux = tools.rgb2bgr(color_wheel_image_nan)

# Compute phasor
dc, g, s = np.asarray(tools.phasor(aux))

# Plot phasor
plot = PhasorPlot(allquadrants=True, title='Phasor plot')
plot.hist2d(g.flatten(), s.flatten(), cmap="RdYlGn_r")

# cluster coordinates in x
coord_g = g.flatten()[~np.isnan(g.flatten())]
coord_s = s.flatten()[~np.isnan(s.flatten())]
x = np.asarray([coord_g, coord_s]).transpose()
xt = np.asarray([g.flatten(), s.flatten()]).transpose()
num_clusters = 3

# Test kmeans clustering
test_km = True
if test_km:
    kmeans = KMeans(n_clusters=3, random_state=100, n_init="auto").fit(x)
    labels = kmeans.fit_predict(x)
    cm = kmeans.cluster_centers_

    tools.cluster_phasor_plot(x, labels, nclusters=3, title="Kmeans")
    labels_new = construct_label_array_optimized(xt, x, labels+1)
    imcolor = map_values_to_rgb(labels_new.reshape([256, 256]))
    
    plt.figure(figsize=(7, 7))
    plt.imshow(imcolor)
    plt.title("Pseudocolor Kmeans")
    plt.show()
    
# Test different GMM clustering methods
test_gmm = False
if test_gmm:
    cov_types = ['full', 'tied', 'diag', 'spherical']
    for i in range(len(cov_types)):
        gmm = GaussianMixture(n_components=3, random_state=100, covariance_type=cov_types[i], 
                                init_params='kmeans')
        gmm.fit(x)
        labels = gmm.predict(x)
        means = gmm.means_
        covariances = gmm.covariances_

        tools.cluster_phasor_plot(x, labels, nclusters=3, title=cov_types[i])

        labels_new = tools.construct_label_array_optimized(xt, x, labels+1)
        imcolor = tools.map_values_to_rgb(labels_new.reshape([256, 256]))

        plt.figure(figsize=(7, 7))
        plt.imshow(imcolor)
        plt.title("Pseudocolor image: " + cov_types[i])
    plt.show()

# test Agglomerative clustering
test_ag = False
if test_ag:
    agglo = AgglomerativeClustering(n_clusters=3)
    labels = agglo.fit_predict(x)
    tools.cluster_phasor_plot(x, labels, nclusters=3, title="Agglomerative Clustering")
    plot_pseudocolor_im(x, xt)
    plt.show()

# Test Spectral Clustering
sp = False
if sp: 
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
    labels = spectral.fit_predict(x)
    tools.cluster_phasor_plot(x, labels, nclusters=3, title="Spectral Clustering")
    plot_pseudocolor_im(x, xt)
    plt.show()

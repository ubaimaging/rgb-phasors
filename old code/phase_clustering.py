import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from tools import cluster_phasor_plot, rgb2bgr, phasor, median_filter, invert_mask
import tifffile
from phasorpy.plot import PhasorPlot

rgb_example = False
if rgb_example:
    im = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig1/im_rgb.png")

    im1 = im[:, :, 0:1]
    im1 = im1.reshape(im1.shape[0], im1.shape[1])
    im2 = im[:, :, 1:2]
    im2 = im2.reshape(im2.shape[0], im2.shape[1])
    im3 = im[:, :, 2:3]
    im3 = im3.reshape(im3.shape[0], im3.shape[1])

    sum = im1 + im2 + im3
    mask = np.where(sum == 3, np.nan * np.ones(sum.shape), np.ones(sum.shape))

    aux = np.asarray([im3 * mask, im2 * mask, im1 * mask])

    _, real, imag = np.asarray(phasor(aux))

imagen_example = True
if imagen_example:
    im = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif")

    im = rgb2bgr(im)
    _, real, imag = phasor(im)
    real = median_filter(real, 3).astype(np.float16)
    imag = median_filter(imag, 3).astype(np.float16)


# Tus datos calculados previamente
coordinates = np.asarray([real.flatten(), imag.flatten()]).transpose()
phase = np.angle(real + 1j * imag)
phase = np.degrees(phase) + 180
# phase = np.degrees(phase)
modulation = np.abs(real + 1j * imag)

aux = phase * modulation
phase2 = phase.flatten()[~np.isnan(aux).flatten()]
modulation2 = modulation.flatten()[~np.isnan(aux).flatten()]

plot = PhasorPlot(allquadrants=True, title='Phasor plot')
plot.hist2d(real, imag, cmap="RdYlGn_r")

plt.figure(2)
plt.scatter(phase2, modulation2)


x = np.asarray([phase2, modulation2]).transpose()
xt = np.asarray([phase.flatten(), modulation.flatten()]).transpose()
num_clusters = 2

km = True
if km:
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=100, n_init="auto").fit(x)
    labels = kmeans.fit_predict(x)
    cm = kmeans.cluster_centers_

    cluster_phasor_plot(x, labels, nclusters=num_clusters, title="Kmeans", cluster_type=2)
    
    from tools import  construct_label_array_optimized, map_values_to_rgb

    labels_new = construct_label_array_optimized(xt, x, labels+1)
    # imcolor = map_values_to_rgb(labels_new.reshape([465, 465]) * mask)
    imcolor = map_values_to_rgb(labels_new.reshape([2048, 2448]))
    
    plt.figure(figsize=(7, 7))
    plt.imshow(imcolor)
    plt.title("Pseudocolor Kmeans")

    coord_g = real.flatten()[~np.isnan(real.flatten())]
    coord_s = imag.flatten()[~np.isnan(imag.flatten())]
    x = np.asarray([coord_g, coord_s]).transpose()
    cluster_phasor_plot(x, labels, nclusters=2, cluster_type=1)

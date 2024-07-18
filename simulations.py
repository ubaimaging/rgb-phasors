# Create the simulations to do phasors with the RGB circle

import numpy as np
import matplotlib.pyplot as plt
import phasorpy

from phasorpy.cursors import (
    mask_from_circular_cursor,
    pseudo_color,
)
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal

from tools import phasor, cluster_phasor_plot

im = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/simulations/rgbw.png")

im1 = im[:, :, 0:1]
im1 = im1.reshape(im1.shape[0], im1.shape[1])
im2 = im[:, :, 1:2]
im2 = im2.reshape(im2.shape[0], im2.shape[1])
im3 = im[:, :, 2:3]
im3 = im3.reshape(im3.shape[0], im3.shape[1])

sum = im1 + im2 + im3
mask = np.where(sum == 3, np.nan * np.ones(sum.shape), np.ones(sum.shape))

aux = np.asarray([im3 * mask, im2 * mask, im1 * mask])

dc, g, s = np.asarray(phasor(aux))

plt.figure(1)
plt.imshow(im)

cursors = True
if cursors:
    cursors_real = [0.5, -0.245, -0.245]
    cursors_imag = [0, 0.43, -0.43]
    circular_mask = mask_from_circular_cursor(
        g, s, cursors_real, cursors_imag, radius=0.5
    )

    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(g.flatten(), s.flatten(), cmap="RdYlGn_r")
    # Plot cursors Blue, Green, Red
    plot.cursor(
        cursors_real[0],
        cursors_imag[0],
        radius=0.5,
        color=CATEGORICAL[1],
        linestyle='-',
    )

    plot.cursor(
        cursors_real[1],
        cursors_imag[1],
        radius=0.5,
        color=CATEGORICAL[2],
        linestyle='-',
    )

    plot.cursor(
        cursors_real[2],
        cursors_imag[2],
        radius=0.5,
        color=CATEGORICAL[0],
        linestyle='-',
    )
    bgr = [CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[0]] 
    segmented_image = pseudo_color(dc, circular_mask, colors=bgr)

    fig, ax = plt.subplots()
    ax.set_title('Segmented image with circular cursors')
    ax.imshow(segmented_image)

    # plt.show()

clusters = True
if clusters:
    from sklearn.cluster import KMeans

    coord_g = g.flatten()[~np.isnan(g.flatten())]
    coord_s = s.flatten()[~np.isnan(s.flatten())]
    x = np.asarray([coord_g, coord_s]).transpose()

    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(x)
    pred_y = kmeans.fit_predict(x)
    cm = kmeans.cluster_centers_
    cluster_phasor_plot(x, pred_y, nclusters=3)

    im = np.zeros(dc.shape)
    # TODO armar la imagen de pseudoclor con los valores del cluster
    # imcolor = mask_with_predict_clusters(x, pred_y + 1, g, s, im)

spectral = True
if spectral: 
    # define pure components blue, green, red
    bgr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    comb = [1, 0.5, 0.3]
    avg, g, s = phasor_from_signal(bgr)
    _, gc, sc = phasor_from_signal(comb)

    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.plot(g[0], s[0], color="b", markersize=10)
    plot.plot(g[1], s[1], color="g", markersize=10)
    plot.plot(g[2], s[2], color="r", markersize=10)
    plot.plot(gc, sc, color="k", markersize=10)
    plt.show()

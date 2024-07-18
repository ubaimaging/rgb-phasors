from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
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
from tools import phasor

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

coord_g = g.flatten()[~np.isnan(g.flatten())]
coord_s = s.flatten()[~np.isnan(s.flatten())]
data = np.asarray([coord_g, coord_s]).transpose()

# Ejemplo de K-Means
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
labels_kmeans = kmeans.labels_

# Ejemplo de DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(data)
labels_dbscan = dbscan.labels_

# Ejemplo de Gaussian Mixture Models
gmm = GaussianMixture(n_components=3)
gmm.fit(data)
labels_gmm = gmm.predict(data)

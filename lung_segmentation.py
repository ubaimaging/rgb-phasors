import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tools
import os

# this code segmentate each image separtatly
# we want to segmentate al all them according to the phasor of al them.

# Calcular los phadors dc, g y s the todas las imagenes y 
# segmnetar ese gran phasor para encontrar dos familias de datos

"""
1 - Read TIFF files
2 - Reorganize channels
3 - Compute dc, g and s
4 - Filter g and s
5 - Segmentation with kmeans
6 - Calculate area
"""

path = '/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND-INST/'
name = sorted(os.listdir(path))
# n = [46, 47, 52, 54, 70, 91, 92, 93, 94, 95]
n = ["046", "063", "065", "066", "067", 103, 104, 105, 106, 125, 126]

table = np.zeros([2, len(n)])
cal_areas = False

if cal_areas == True:
    i = 0
    k = 0
    while k < len(n):
        cont = 0
        at = 0
        while i < len(name) and name[i][0:3] == str(n[k]):
            rgb = plt.imread(path + name[i])
            bgr = tools.rgb2bgr(rgb)
            dc, g, s = tools.phasor(bgr)
            g = tools.median_filter(g, 3)
            s = tools.median_filter(s, 3)

            # Clustering segmentation
            X = np.asarray([g.flatten(), s.flatten()]).transpose()
            nclusters = 2
            kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(X)
            pred_y = kmeans.fit_predict(X)
            imp = pred_y.reshape(g.shape)

            # Areas
            hist = np.histogram(imp, bins=2)[0]
            area_tissue = hist[0] / np.sum(hist)
            at = at + area_tissue
            cont = cont + 1
            i = i + 1

        table[0][k] = int(n[k])
        table[1][k] = at / cont
        k = k + 1
    print(table)
    np.savetxt("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/data_b6nd_inst.csv", table, delimiter=",")
    
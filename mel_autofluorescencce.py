import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tools
import os

# Open the RGB images 
imm = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/nev-mel/18852_10x_r1.tif")
imn = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/nev-mel/16252_10x_r1.tif")

rgbm = tools.rgb2bgr(imm)
rgbn = tools.rgb2bgr(imn)

# calculate the phasor
dcm, gm, sm = tools.phasor(rgbm)
dcn, gn, sn = tools.phasor(rgbn)
# threshold the background on the nevus image
gn = np.where(dcn > 35, gn, np.NaN)
sn = np.where(dcn > 35, sn, np.NaN)
# median filter
gm = tools.median_filter(gm, 3)
sm = tools.median_filter(sm, 3)
gn = tools.median_filter(gn, 3)
sn = tools.median_filter(sn, 3)

# Segmentar con clustering el phasor y tener las dos pseudocolor
gaux = np.concatenate([gm.flatten(), gn.flatten()])
saux = np.concatenate([sm.flatten(), sn.flatten()])
aux = gaux * saux
g = gaux[~np.isnan(aux)]
s = saux[~np.isnan(aux)]

nclusters = 3
X = np.asarray([g, s]).transpose()
kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(X)
pred_y = kmeans.fit_predict(X)
p0 = np.where(pred_y == 0)
p1 = np.where(pred_y == 1)
p2 = np.where(pred_y == 2)

# uso un ciclo para rescontruir la imagen de pseudocolor con las coordenadas y los clusters
maskm = np.zeros(imm.shape)
maskn = np.zeros(imn.shape)

for i in range(gm.shape[0]):
    for j in range(gm.shape[1]):
        for k in range(len(X)):
            if gm[i][j] == X[k][0] and sm[i][j] == X[k][1]:
                if pred_y[k] == 0:
                    maskm[i, j, :] = 0, 0, 1
                if pred_y[k] == 1:
                    maskm[i, j, :] = 0, 1, 0
                if pred_y[k] == 2:
                    maskm[i, j, :] = 1, 0, 0

for i in range(gm.shape[0]):
    for j in range(gm.shape[1]):
        for k in range(len(X)):
            if gn[i][j] == X[k][0] and sn[i][j] == X[k][1]:
                if pred_y[k] == 0:
                    maskn[i, j, :] = 0, 0, 1
                if pred_y[k] == 1:
                    maskn[i, j, :] = 0, 1, 0
                if pred_y[k] == 2:
                    maskn[i, j, :] = 1, 0, 0


plt.figure()
plt.imshow(maskm)

plt.figure()
plt.imshow(maskn)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(X[p0[0], 0], X[p0[0], 1], c='b')
ax.scatter(X[p1[0], 0], X[p1[0], 1], c='g')
ax.scatter(X[p2[0], 0], X[p2[0], 1], c='r')
tools.phasor_circle(ax)
plt.show()

# Muestro las imagenes segmentadas

plotty = False
if plotty:
    plt.figure(1)
    plt.imshow(imm)
    plt.figure(2)
    plt.imshow(imn)
    plt.show()

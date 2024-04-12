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

# tools.interactive(dcm, gm, sm, 0.1, 8)

# threshold the background on the nevus image
# gn = np.where(dcn > 35, gn, np.NaN)
# median filter
gm = tools.median_filter(gm, 3)
sm = tools.median_filter(sm, 3)
gn = tools.median_filter(gn, 3)
sn = tools.median_filter(sn, 3)

# Segmentar con clustering el phasor y tener las dos pseudocolor
g = np.concatenate([gm.flatten(), gn.flatten()])
s = np.concatenate([sm.flatten(), sn.flatten()])
# aux = gaux * saux
# g = gaux[~np.isnan(aux)]
# s = saux[~np.isnan(aux)]

nclusters = 5
X = np.asarray([g.flatten(), s.flatten()]).transpose()
kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(X)
pred_y = kmeans.fit_predict(X)
p0 = np.where(pred_y == 0)
p1 = np.where(pred_y == 1)
p2 = np.where(pred_y == 2)
p3 = np.where(pred_y == 3)
p4 = np.where(pred_y == 4)

plottyf = False
if plottyf:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[p0[0], 0], X[p0[0], 1], c='b')
    ax.scatter(X[p1[0], 0], X[p1[0], 1], c='g')
    ax.scatter(X[p2[0], 0], X[p2[0], 1], c='r')
    tools.phasor_circle(ax)

mask = pred_y.reshape([2, imm.shape[0], imm.shape[1]])
plt.figure()
plt.imshow(mask[0], cmap="Set1")
plt.figure()
plt.imshow(mask[1], cmap="Set1")
plt.show()

# Muestro las imagenes segmentadas

plotty = False
if plotty:
    plt.figure(1)
    plt.imshow(imm)
    plt.figure(2)
    plt.imshow(imn)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tools
import os
from matplotlib import colors



# README 
# with this script we created the files: 
# - cluster_coord and cluster_coord2.npy contain the coordinates g and s 
#   with normal and instilation groups respectively. 
# - lung_mask (normal) and lung_mask2.npy (instilated) have the 100 image mask calculated for 
#   each image of each mouse.
# - areas and areas.npy have the percetange of tissue area for each image

# Part 1: read all the files and obtain the cluster
# Part 2: segmentate all images and creat mask 
# Part 3: calculates the areas with the mask and stats analysis

# ------------
#    Part 1
# ------------
part1 = False
if part1:
    path = '/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND2/'
    name = sorted(os.listdir(path))
    auxg = []
    auxs = []
    for i in range(len(name)):
        rgb = plt.imread(path + name[i])
        bgr = tools.rgb2bgr(rgb)
        _, g, s = tools.phasor(bgr)
        g = tools.median_filter(g, 1)
        s = tools.median_filter(s, 1)
        auxg.append(g.flatten())
        auxs.append(s.flatten())
    auxg = np.concatenate(auxg)
    auxs = np.concatenate(auxs)
    data = np.asarray([auxg, auxs])
    np.save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/cluster_coord_B6ND2.npy", data)

    path = '/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst2/'
    name = sorted(os.listdir(path))
    auxg = []
    auxs = []
    for i in range(len(name)):
        rgb = plt.imread(path + name[i])
        bgr = tools.rgb2bgr(rgb)
        _, g, s = tools.phasor(bgr)
        g = tools.median_filter(g, 1)
        s = tools.median_filter(s, 1)
        auxg.append(g.flatten())
        auxs.append(s.flatten())
    auxg = np.concatenate(auxg)
    auxs = np.concatenate(auxs)
    data = np.asarray([auxg, auxs])
    np.save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/cluster_coord_B6ND_inst2.npy", data)

# ------------
#    Part 2
# ------------

# Load both files and concatenate them to apply the clustering 
part2 = False
if part2:
    data1 = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/cluster_coord_B6ND2.npy")
    data2 = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/cluster_coord_B6ND_inst2.npy")
    g = np.concatenate([data1[0], data2[0]])
    s = np.concatenate([data1[1], data2[1]])
    aux = np.zeros([2, len(g)])
    aux[0] = g
    aux[1] = s
    np.save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/cluster_coord2.npy", aux)

part2_2 = False
if part2_2:
    data = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/cluster_coord2.npy")
    m, n = 2048, 2448
    nclusters = 2
    X = np.asarray([data[0], data[1]]).transpose()
    cluster = False
    if cluster:
        kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(X)
        pred_y = kmeans.fit_predict(X)
        im = pred_y.reshape([100, m, n])
        np.save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/lung_mask2.npy", im)
        # data = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/lung_mask.npy")

plotty = False
if plotty:
    data = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/cluster_coord.npy")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist2d(data[0], data[1], bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    plt.show()


# Calculo de area y estudio estadístico
part3 = False
if part3:
    data = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/lung_mask2.npy")
    area = int(data.shape[1] * data.shape[2])
    aux = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        aux[i] = np.sum(data[i]) / area
    np.save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/areas2.npy", aux)

parte3_1 = False
if parte3_1:
    import seaborn as sns

    data1 = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/datos/areas.npy")
    data2 = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/datos/areas2.npy")
    data = np.zeros(200)
    data = np.concatenate([data1[0:50], data2[0:50], data1[50:100], data2[50:100]])

    d = data.reshape([2, 100])
    d = 1 - d
    dmean = np.mean(d, axis=1)
    dstd = np.std(d, axis=1)

    from tools import plot_separated_boxplots_and_violin
    plot_separated_boxplots_and_violin(d)


plotty2 = False
if plotty2:
    # data = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/datos/lung_mask.npy")
    data = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/datos/lung_mask2.npy")

    # imagen para el paper numero 15 nombre 52-21B6ND05.tif que es data[ind] con ind=14 
    # y para instilado es ind = 56
    ind = 56
    # im = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif")
    im = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst2/063-21B6ND_INS06.tif") 

    bgr = tools.rgb2bgr(im)
    _, g, s = tools.phasor(bgr)
    g = tools.median_filter(g, 1)
    s = tools.median_filter(s, 1)

    nclusters = 2
    X = np.asarray([g.flatten(), s.flatten()]).transpose()
    kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(X)
    pred_y = kmeans.fit_predict(X)
    p0 = np.where(pred_y == 0)
    p1 = np.where(pred_y == 1)
    # p2 = np.where(pred_y == 2)

    plt.figure(1)
    plt.imshow(im)
    # plt.savefig("/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2/rgb.png", dpi=300)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist2d(g.flatten(), s.flatten(), bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    tools.phasor_circle(ax)
    # plt.savefig("/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2/phasor.png", dpi=300)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X[p0[0], 0], X[p0[0], 1], c='r')
    ax.scatter(X[p1[0], 0], X[p1[0], 1], c='b')
    # ax.scatter(X[p2[0], 0], X[p2[0], 1], c='r')
    tools.phasor_circle(ax)
    # plt.savefig("/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2/phasor_c.png", dpi=300)

    imcolor = np.zeros([data.shape[1], data.shape[2], 3])
    
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if data[ind][i][j] == 1:
                imcolor[i, j, :] = 0, 0, 1
            else:
                imcolor[i, j, :] = 1, 0, 0
    
    imcolor = imcolor.reshape([data.shape[1], data.shape[2], 3])
                
    plt.figure(4)
    plt.imshow(imcolor)
    # plt.savefig("/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2/mask.png", dpi=300)
    plt.show()
    

# buscar mascara 

control = True
data = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/datos/lung_mask2.npy")

if control:
    data = np.load("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/datos/lung_mask2.npy")
    ind = 0
    imcolor = np.zeros([data.shape[1], data.shape[2], 3])

    for ind in range(100):
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                if data[ind][i][j] == 1:
                    imcolor[i, j, :] = 0, 0, 1
                else:
                    imcolor[i, j, :] = 1, 0, 0

        imcolor = imcolor.reshape([data.shape[1], data.shape[2], 3])
                    
        plt.figure(4)
        plt.imshow(imcolor)
        # plt.savefig("/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2/mask.png", dpi=300)
        plt.show()
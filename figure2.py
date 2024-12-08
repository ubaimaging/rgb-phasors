# Read two image expalmes of lung ND and ND Inst 
# Read the mask done in fiji for this samples
# Use the phasor to segmentate plot:
# phasor, phasor cluster and phasor mask
# Statistical analysis with data already calculated 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tools import plot_all_tifs_in_folder
import tifffile as tiff
import tools

# Part 1
# Plot image and mask from fiji
# Plot phasor, phasor cluster and phasor segmentation
part1 = False
if part1:
    # Ejemplo de uso
    folder_path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2/mask" 
    plot_all_tifs_in_folder(folder_path)
    plt.show()

part2 = True
if part2: 
        impath1 = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2/mask/52-21B6ND05.tif"
        impath2 = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2/mask/063-21B6ND_INS06.tif"
        img1 = plt.imread(impath1)
        img2 = plt.imread(impath2)

        img1 = tools.rgb2bgr(img1)
        img2 = tools.rgb2bgr(img2)

        _, g1, s1 = tools.phasor(img1)
        g1 = tools.median_filter(g1, 3)
        s1 = tools.median_filter(s1, 3)


        _, g2, s2 = tools.phasor(img2)
        g2 = tools.median_filter(g2, 3)
        s2 = tools.median_filter(s2, 3)

        # Clustering segmentation
        nclusters = 2
        
        X1 = np.asarray([g1.flatten(), s1.flatten()]).transpose()
        kmeans = KMeans(n_clusters=nclusters, random_state=42, n_init="auto").fit(X1)
        pred_y1 = kmeans.fit_predict(X1)
        imp1 = pred_y1.reshape(g1.shape)

        X2 = np.asarray([g2.flatten(), s2.flatten()]).transpose()
        kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(X2)
        pred_y2 = kmeans.fit_predict(X2)
        imp2 = pred_y2.reshape(g2.shape)

        plt.figure(1)
        plt.imshow(imp1,cmap = 'cividis')
        plt.title("Segmented ND")

        plt.figure(2)
        plt.imshow(imp2, cmap = 'cividis')
        plt.title("Segmented ND inst")

        plt.show()

# Part 3
# Perfomer statistical analysis
part3 = False
if part3:
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

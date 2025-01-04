import numpy as np
import matplotlib.pyplot as plt
import time

from tools import rgb2bgr, phasor, median_filter
import tools 
import tifffile
from phasorpy.plot import PhasorPlot

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering


def mask_from_cluster_phasor_kmeans(real, x, nclusters=3):
    kmeans = KMeans(n_clusters= nclusters, random_state=42, n_init="auto").fit(x)
    labels = kmeans.predict(x)
    mask = labels.reshape(real.shape)
    return mask

def mask_from_cluster_phasor_gmm(real, x, nclusters=3, cov_type="full"):
    gmm = GaussianMixture(n_components=nclusters, random_state=42, 
                          covariance_type=cov_type, init_params='kmeans')
    gmm.fit(x)
    labels = gmm.predict(x)
    mask = labels.reshape(real.shape)
    return mask

def mask_from_cluster_phasor_spectral(real, x, nclusters=3):
    spectral = SpectralClustering(n_clusters=nclusters, affinity='nearest_neighbors',
                                  random_state=42)
    labels = spectral.fit_predict(x)
    mask = labels.reshape(real.shape)
    return mask


# Plot the examples images 
plotty = False
if plotty:
    im1 = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif")
    im2 = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/063-21B6ND_INS04.tif")
    
    plt.figure()
    plt.imshow(im1)

    plt.figure()
    plt.imshow(im2)

    im1 = rgb2bgr(im1)
    _, real1, imag1 = phasor(im1)
    real1 = median_filter(real1, 2)
    imag1 = median_filter(imag1, 2)

    im2 = rgb2bgr(im2)
    _, real2, imag2 = phasor(im2)
    real2 = median_filter(real2, 2)
    imag2 = median_filter(imag2, 2)

    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(real1.flatten(), imag1.flatten(), cmap="RdYlGn_r")

    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(real2.flatten(), imag2.flatten(), cmap="RdYlGn_r")

    plt.show()

# Part 1:
#   Test the excecution time for some clustering algoritm with the RGB HE image 
#       Kmeans
#       GMM (full)
#       GMM (tied)
#       Spectral

part1 = False
if part1:
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif"
    # path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/063-21B6ND_INS04.tif"
    im = tifffile.imread(path)
    im = rgb2bgr(im)
    _, real, imag = phasor(im)
    real = median_filter(real, 2)
    imag = median_filter(imag, 2)
    x = np.asarray([real.flatten(), imag.flatten()]).transpose()

    # Measure the execution time 
    time_km = False
    if time_km:
        start = time.time()
        mask_kmeans = mask_from_cluster_phasor_kmeans(real, x, nclusters=4)
        print(f"K-Means tiempo: {time.time() - start:.2f} segundos")

    time_gmm_tied = False
    if time_gmm_tied:
        start = time.time()
        mask_gmm_tied = mask_from_cluster_phasor_gmm(real, x, nclusters=4, cov_type="tied")
        print(f"GMM (tied) tiempo: {time.time() - start:.2f} segundos")

    time_gmm_full = False
    if time_gmm_full:
        start = time.time()
        mask_gmm_tied = mask_from_cluster_phasor_gmm(real, x, nclusters=4, cov_type="full")
        print(f"GMM (full) tiempo: {time.time() - start:.2f} segundos")

    time_sp = False
    if time_sp:
        start = time.time()
        mask_gmm_tied = mask_from_cluster_phasor_spectral(real, x, nclusters=4)
        print(f"Spectral tiempo: {time.time() - start:.2f} segundos")

# Part 2
# Segmentates with kmeans two images and analize the effects it produces
part2 = False
if part2:
    # Ejemplo de Normal Diet
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif"
    im = tifffile.imread(path)
    im = rgb2bgr(im)
    _, real, imag = phasor(im)
    real = median_filter(real, 2)
    imag = median_filter(imag, 2)
    x = np.asarray([real.flatten(), imag.flatten()]).transpose()

    nclusters = 4
    kmeans = KMeans(n_clusters= nclusters, random_state=42, n_init="auto").fit(x)
    labels = kmeans.predict(x)
    mask1 = labels.reshape(real.shape)

    plotty = False # To plot the clusterized phasor and the colored mask

    if plotty:
        tools.cluster_phasor_plot_4_clusters(x, labels)
        pseudocolor_image = tools.map_mask_to_colors(mask1, ind=[0, 1, 3, 2])
        plt.figure()
        plt.imshow(pseudocolor_image)

    # Ejemplo de Instilado
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/063-21B6ND_INS04.tif"
    im = tifffile.imread(path)
    im = rgb2bgr(im)
    _, real, imag = phasor(im)
    real = median_filter(real, 2)
    imag = median_filter(imag, 2)
    x = np.asarray([real.flatten(), imag.flatten()]).transpose()

    nclusters = 4
    kmeans = KMeans(n_clusters= nclusters, random_state=42, n_init="auto").fit(x)
    labels = kmeans.predict(x)
    mask2 = labels.reshape(real.shape)

    if plotty:
        tools.cluster_phasor_plot_4_clusters(x, labels, colors=["r", "lime", "b", "k"])
        pseudocolor_image = tools.map_mask_to_colors(mask2, ind=[3, 0, 2, 1])
        plt.figure()
        plt.imshow(pseudocolor_image)
        plt.show

    # Change the color mask to binary mask
    mask1 = np.where(mask1 == 0, np.zeros(mask1.shape), np.ones(mask1.shape))
    mask2 = np.where(mask2 == 3, np.zeros(mask2.shape), np.ones(mask2.shape))

    if plotty:
        plt.figure()
        plt.imshow(mask1)
        plt.figure()
        plt.imshow(mask2)
        plt.show()

# Part 2 Other kind of segmentation 
part2 = False
if part2:

    # Read the images
    im1 = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif")
    im2 = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/063-21B6ND_INS04.tif")
    
    im2 = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/multiotsu/nd/multiotsu_46-21B6ND01.tif")
    
    plt.imshow(im2)
    plt.show()

    # Kmeans
    km = False
    if km:
        from sklearn.cluster import KMeans
        im1 = im1 / 255
        pixels1 = im1.reshape(-1, 3)  # Convertir a Nx3
        kmeans1 = KMeans(n_clusters=4, random_state=42)
        labels1 = kmeans1.fit_predict(pixels1)
        segmented_image1 = labels1.reshape(im1.shape[:2])

        im2 = im2 / 255
        pixels2 = im2.reshape(-1, 3)  # Convertir a Nx3
        kmeans2 = KMeans(n_clusters=4, random_state=42)
        labels2 = kmeans2.fit_predict(pixels2)
        segmented_image2 = labels2.reshape(im1.shape[:2])

        # Plot
        plotty = False
        if plotty:
            plt.figure()
            plt.imshow(segmented_image1)
            plt.title("Segmentación con K-Means")
            plt.axis('off')

            plt.figure()
            plt.imshow(segmented_image2)
            plt.title("Segmentación con K-Means")
            plt.axis('off')
            plt.show()
    
    # Otsu
    otsu = False
    if otsu:
        from skimage.filters import threshold_multiotsu
        from skimage.color import rgb2gray

        gray_image1 = rgb2gray(im1)
        gray_image1 = (gray_image1 * 255).astype(np.uint8)
        thresholds1 = threshold_multiotsu(gray_image1, classes=4)
        mask1 = np.digitize(gray_image1, bins=thresholds1)

        gray_image2 = rgb2gray(im2)
        gray_image2 = (gray_image2 * 255).astype(np.uint8)
        thresholds2 = threshold_multiotsu(gray_image2, classes=4)
        mask2 = np.digitize(gray_image2, bins=thresholds2)

        # Plot
        plotty = False
        if plotty:
            plt.figure()
            plt.imshow(mask1)
            plt.axis('off')

            plt.figure()
            plt.imshow(mask2)
            plt.axis('off')
            plt.show()

# Create the simulations to do phasors with the RGB circle
import numpy as np
import matplotlib.pyplot as plt
import tools
from tools import phasor, cluster_phasor_plot, rgb2bgr, map_to_rgb

from phasorpy.cursors import mask_from_circular_cursor
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal


# Generates RGB color wheel
color_wheel_image = tools.generate_color_wheel_image(256)

# Visualizar la rueda de colores
plt.figure(figsize=(7, 7))
plt.imshow(color_wheel_image, extent=(-1, 1, -1, 1))
plt.axis('off')
plt.title("RGB Image")


color_wheel_image_nan = tools.replace_with_nan(color_wheel_image)
aux = tools.rgb2bgr(color_wheel_image_nan)
dc, g, s = np.asarray(tools.phasor(aux))

plot = PhasorPlot(allquadrants=True, title='Phasor plot')
plot.hist2d(g.flatten(), s.flatten(), cmap="RdYlGn_r")

cursors = True
if cursors:
    cursors_real = [0.5, -0.245, -0.245]
    cursors_imag = [0, 0.43, -0.43]

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

    cursors_mask = mask_from_circular_cursor(
        g, s, cursors_real, cursors_imag, radius=0.5)
    
    auxmask = np.transpose(cursors_mask, (1, 2, 0)).astype(int)

    auxx = map_to_rgb(auxmask)

    plt.figure(figsize=(7, 7))
    plt.imshow(auxx)
    plt.title("Pseudocolor image with cursors")
    # plt.show()

clusters = True
if clusters:
    coord_g = g.flatten()[~np.isnan(g.flatten())]
    coord_s = s.flatten()[~np.isnan(s.flatten())]
    x = np.asarray([coord_g, coord_s]).transpose()
    xt = np.asarray([g.flatten(), s.flatten()]).transpose()
    num_clusters = 3

    km = True 
    if km:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=100, n_init="auto").fit(x)
        labels = kmeans.fit_predict(x)
        cm = kmeans.cluster_centers_

        cluster_phasor_plot(x, labels, nclusters=num_clusters, title="Kmeans")
        
        from tools import  construct_label_array_optimized, map_values_to_rgb

        labels_new = construct_label_array_optimized(xt, x, labels+1)
        imcolor = map_values_to_rgb(labels_new.reshape([256, 256]))
        
        plt.figure(figsize=(7, 7))
        plt.imshow(imcolor)
        plt.title("Pseudocolor Kmeans")

    gmm = True
    if gmm:
        from sklearn.mixture import GaussianMixture
        cov_types = ['full', 'tied', 'diag', 'spherical']
        i = 1
        gmm = GaussianMixture(n_components=3, random_state=100, covariance_type=cov_types[i], 
                              init_params='kmeans')
        gmm.fit(x)
        labels = gmm.predict(x)
        means = gmm.means_
        covariances = gmm.covariances_

        cluster_phasor_plot(x, labels, nclusters=3, title=cov_types[i])
        
        from tools import  construct_label_array_optimized, map_values_to_rgb

        labels_new = construct_label_array_optimized(xt, x, labels+1)
        imcolor = map_values_to_rgb(labels_new.reshape([256, 256]))
        
        plt.figure(figsize=(7, 7))
        plt.imshow(imcolor)
        plt.title("Pseudocolor image: " + cov_types[i])
        # plt.show()

spectral = True
if spectral: 
    # define pure components blue, green, red
    bgr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    comb = [1, 0.5, 0.3]
    avg, gs, ss = phasor_from_signal(bgr)
    _, gc, sc = phasor_from_signal(comb)

    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.plot(gs[0], ss[0], color="b", markersize=10)
    plot.plot(gs[1], ss[1], color="lime", markersize=10)
    plot.plot(gs[2], ss[2], color="r", markersize=10)
    plot.plot(gc, sc, color="k", markersize=10)

    # Implementar el spectral unmixing
    sp_unmixing = True
    if sp_unmixing:
        bgr = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        avg, gs, ss = phasor_from_signal(bgr)
        ncomp = 3
        vecB = np.stack((g, s, np.ones(g.shape)), axis=-1)  # Dimensions: (465, 465, 3)
        # Matrix A with dimensions (3, 3)
        matA = np.array([gs, ss, [1, 1, 1]])
        # Flatten the first two dimensions of vecB to apply lstsq at once
        vecB_flat = vecB.reshape(-1, 3)  # Dimensions: (465*465, 3)
        # Apply lstsq to each row of vecB_flat with respect to matA
        frac_flat, _, _, _ = np.linalg.lstsq(matA, vecB_flat.T, rcond=None)
        # Reshape the result back to its original form
        frac = frac_flat.T.reshape(256, 256, 3)


        # plotear las tres imagenes por separado
        frac_rgb = rgb2bgr(frac)
        # frac_rgb = tools.convert_rgb_to_bgr(frac_rgb)
        plt.figure(figsize=(6, 6))
        plt.imshow(frac_rgb[0], cmap="Blues")
        plt.title("Blue channel")

        plt.figure(figsize=(6, 6))
        plt.imshow(frac_rgb[1], cmap="Greens")
        plt.title("Green channel")

        plt.figure(figsize=(6, 6))
        plt.imshow(frac_rgb[2], cmap="Reds")
        plt.title("Red channel")

        # Recontruir la de pseudocolor          
        plt.figure(figsize=(7, 7))
        plt.imshow(frac)

    plt.show()
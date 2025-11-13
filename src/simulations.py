# Create the simulations to do phasors with the RGB circle
import numpy as np
import matplotlib.pyplot as plt
import tools
from tools import phasor, cluster_phasor_plot, rgb2bgr, map_to_rgb

from phasorpy.cursors import mask_from_circular_cursor
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal


# import color space image
color_space = plt.imread(
    "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig1/color_space.png")

fig1 = plt.figure(figsize=(5, 5))
plt.imshow(color_space)
plt.axis("off")


# Generates RGB color wheel
color_wheel_image = tools.generate_color_wheel_image(256)
fig2 = plt.figure(figsize=(5, 5))
plt.imshow(color_wheel_image, extent=(-1, 1, -1, 1))
plt.axis('off')


color_wheel_image_nan = tools.replace_with_nan(color_wheel_image)
aux = tools.rgb2bgr(color_wheel_image_nan)
dc, g, s = np.asarray(tools.phasor(aux))

# Create the pure components and plot 
bgr = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
avg, gs, ss = phasor_from_signal(bgr)

plot = PhasorPlot(allquadrants=True, title='')
plot.hist2d(g.flatten(), s.flatten(), cmap="RdYlGn_r")
plot.plot(gs[0], ss[0], color="b", markersize=10)
plot.plot(gs[1], ss[1], color="lime", markersize=10)
plot.plot(gs[2], ss[2], color="r", markersize=10)
fig3 = plot.fig
plot.fig.set_size_inches(5, 5) 
plot.ax.set_aspect('equal')  

cursors = True
if cursors:
    cursors_real = [0.5, -0.245, -0.245]
    cursors_imag = [0, 0.43, -0.43]

    plot = PhasorPlot(allquadrants=True, title='')
    plot.hist2d(g.flatten(), s.flatten(), cmap="RdYlGn_r")
    fig4 = plot.fig
    fig4 = plot.fig
    plot.fig.set_size_inches(5, 5) 
    plot.ax.set_aspect('equal')  

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

    fig7 = plt.figure(figsize=(5, 5))
    plt.imshow(auxx)
    plt.axis("off")


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

        fig5, plot = cluster_phasor_plot(x, labels, nclusters=num_clusters, title="")
        plot.fig.set_size_inches(5, 5) 
        plot.ax.set_aspect('equal')
        
        from tools import  construct_label_array_optimized, map_values_to_rgb

        labels_new = construct_label_array_optimized(xt, x, labels+1)
        imcolor = map_values_to_rgb(labels_new.reshape([256, 256]))
        
        fig8 = plt.figure(figsize=(5, 5))
        plt.imshow(imcolor)
        plt.axis("off")

    gmm = False
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

        # Create phasor plot
        plot = PhasorPlot(allquadrants=True, title='')
        ax = plot.ax

        # Plot points
        ax.plot(gs[0], ss[0], 'o', color="r", markersize=12)
        ax.plot(gs[1], ss[1], 'o', color="lime", markersize=12)
        ax.plot(gs[2], ss[2], 'o', color="b", markersize=12)
        ax.plot(gc, sc, 'o', color="k", markersize=12)

        # Draw lines connecting the points to the combination
        for g, s in zip(gs, ss):
            ax.plot([g, gc], [s, sc], linestyle='--', color='gray')

        # Annotate the points
        ax.text(gs[0] - 0.05, ss[0] - 0.06, "C3", fontsize=14, 
                color='r', ha='center', va='top', fontweight='bold') # blue
        ax.text(gs[1], ss[1] + 0.05, "C2", fontsize=14, 
                color='lime', ha='center', va='bottom', fontweight='bold') # green
        ax.text(gs[2] - 0.2, ss[2] + 0.15, "C1", fontsize=14, 
                color='b', ha='left', va='top', fontweight='bold') # red
        ax.text(gc + 0.05, sc + 0.1, "C4", fontsize=14, 
                color='k', ha='left', va='top', fontweight='bold') # CL

        plot.ax.set_xlim(-1.05, 1.05)
        plot.ax.set_ylim(-1.05, 1.05)

        # Output the final figure
        fig6 = plot.fig
        plot.fig.set_size_inches(5, 5)
        plot.ax.set_aspect('equal')

        # Recontruir la de pseudocolor          
        fig9 = plt.figure(figsize=(5, 5))
        plt.imshow(frac)
        plt.axis("off")

    # plt.show()


################################################################

##################  Paper Figure ###############################

################################################################

figure_paper = True
if figure_paper:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import string

    # Lista de tus figuras (ya generadas)
    figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]

    # Títulos correspondientes
    titles = [
        "RGB Color Space", "Simulated Color Wheel", "Phasor Plot",
        "Phasor Plot: Cursor", "Phasor Plot: Clustering", "Phasor Plot: Unmixing",
        "Pseudocolor: Cursors", "Pseudocolor: Clustering", "Pseudocolor: Unmixing"
    ]

    # Crear figura compuesta
    fig_final, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.flatten()

    for idx, (ax, fig, title) in enumerate(zip(axs, figs, titles)):
        # Renderizar figura previa con canvas
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Extraer imagen desde el buffer RGBA
        width, height = fig.get_size_inches() * fig.get_dpi()
        width, height = int(width), int(height)
        image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)

        # Mostrar la imagen en el subplot
        ax.imshow(image[:, :, :3])
        ax.set_title(title, fontsize=16)
        ax.axis('off')

        # Agregar letra (A, B, ..., I) en la esquina superior izquierda
        label = f"{string.ascii_uppercase[idx]}"
        ax.text(
            0.01, 0.98, label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
        )

    plt.tight_layout()
    fig_final.savefig(
        "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig1/fig1_grid.png",
        dpi=300
    )
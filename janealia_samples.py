import numpy as np
import matplotlib.pyplot as plt
import tools

from phasorpy.cursors import (
    mask_from_circular_cursor,
    pseudo_color,
)
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal, phasor_filter, phasor_threshold
import tifffile

 
part1 = False 
if part1:
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/j_samples/epi/"

    # image = plt.imread(path + "janelia_sample_3channes_roi1.tif")

    image = plt.imread(path + "janelia_sample_555_roi1.tif")
    bgr = tools.rgb2bgr(image) # reorder channels to blue, green, red

    # Cut image for SimFCS
    # aux = bgr[:, 500:1524, 500:1524]
    # tifffile.imwrite(path + "janelia_sample_488_cut.tif", aux)

    avg, real, imag = phasor_from_signal(bgr, axis=0)

    plotty = False
    if plotty:
        # plot histogram of intensity
        plt.figure()
        plt.hist(avg.flatten(), bins=256)
        plt.yscale("log")

        mask = avg > 10
        binary_mask = mask.astype(np.uint8)
        mask_nan = np.where(binary_mask == 0, np.nan, binary_mask)

        real = real * mask_nan
        imag = imag * mask_nan

        cursors_real = [0.12, 0.36, -0.1]
        cursors_imag = [0, 0.33, 0.63]
        radius = [0.1, 0.15, 0.35]

        circular_mask = mask_from_circular_cursor(
            real, imag, cursors_real, cursors_imag, radius=radius
        )

        plot = PhasorPlot(allquadrants=True, title='Circular cursors')
        plot.hist2d(real, imag, cmap='RdYlGn_r')
        for i in range(len(cursors_real)):
            plot.cursor(
                cursors_real[i],
                cursors_imag[i],
                radius=radius[i],
                color=CATEGORICAL[i],
                linestyle='-',
            )


        pseudo_color_image = pseudo_color(avg, circular_mask)

        fig, ax = plt.subplots()
        ax.set_title('Pseudo-color image from circular cursors')
        ax.imshow(pseudo_color_image)
        plt.show()


    interactivo = False
    if interactivo:
        import tools
        tools.interactive(avg, real, imag, 0.15, 8)


# Spectral Unmixing Part
part2 = True
if part2:
    # Obtain the center of each components 
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/"

    blue = plt.imread(path + "janelia_sample_dapi_roi1.tif")
    green = plt.imread(path + "janelia_sample_488_roi2.tif")
    red = plt.imread(path + "janelia_sample_555_roi1.tif")

    comp_plot = True
    if comp_plot:
        plt.figure(1)
        plt.imshow(blue)
        plt.figure(2)
        plt.imshow(green)
        plt.figure(3)
        plt.imshow(red)
        # plt.show()

    comp_plot_phasor = True
    if comp_plot_phasor:
        avgb, realb, imagb = phasor_from_signal(tools.rgb2bgr(blue)[0:, 500:700, 700:1100], axis=0)
        plot = PhasorPlot(allquadrants=True, title='Phasor plot blue')
        plot.hist2d(realb.flatten(), imagb.flatten(), cmap="RdYlGn_r")

        avgg, realg, imagg = phasor_from_signal(tools.rgb2bgr(green)[0:, 400:1400, 1000:1600], axis=0)
        plot = PhasorPlot(allquadrants=True, title='Phasor plot green')
        plot.hist2d(realg.flatten(), imagg.flatten(), cmap="RdYlGn_r")

        avgr, realr, imagr = phasor_from_signal(tools.rgb2bgr(red)[0:, 600:1000, 1200:1400], axis=0)
        plot = PhasorPlot(allquadrants=True, title='Phasor plot red')
        plot.hist2d(realr.flatten(), imagr.flatten(), cmap="RdYlGn_r")
        # plt.show()

    sp_unmixing = True
    if sp_unmixing:
        # image = plt.imread(path + "janelia_sample_3channes_roi5.tif")
        image = plt.imread(path + "deconvolved_janelia_sample_3channes_roi5.tif")
        # image = plt.imread(path + "janelia_sample_3channes_roi2.tif")


        plt.figure()
        plt.imshow(image)

        img = tools.rgb2bgr(image)
        avg, real, imag = phasor_from_signal(img, axis=0)

        plot = PhasorPlot(allquadrants=True, title='Phasor')
        plot.hist2d(real.flatten(), imag.flatten(), cmap="RdYlGn_r")
        

        bgr = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        # _, gs, ss = phasor_from_signal(bgr, axis=0)

        gs = np.array([-0.5, -0.35, 0.45])
        ss = np.array([-0.85, 0.78, 0.32])

        # filter
        real, imag, = phasor_filter(real, imag)
        # threshold
        avg, real, imag = phasor_threshold(avg, real, imag, mean_min=5)

        ncomp = 3
        vecB = np.stack((real, imag, np.ones(real.shape)), axis=-1)  # Dimensions: (465, 465, 3)
        # Matrix A with dimensions (3, 3)
        matA = np.array([gs, ss, [1, 1, 1]])
        # Flatten the first two dimensions of vecB to apply lstsq at once
        vecB_flat = vecB.reshape(-1, 3)
        # Apply lstsq to each row of vecB_flat with respect to matA
        frac_flat, _, _, _ = np.linalg.lstsq(matA, vecB_flat.T, rcond=None)
        # Reshape the result back to its original form
        frac = frac_flat.T.reshape(real.shape[0], real.shape[1], 3)

        # plotear las tres imagenes por separado
        frac_rgb = tools.rgb2bgr(frac)
        plt.figure()
        plt.imshow(frac_rgb[0], cmap="Blues")
        plt.title("Blue channel")

        plt.figure()
        plt.imshow(frac_rgb[1], cmap="Greens")
        plt.title("Green channel")

        plt.figure()
        plt.imshow(frac_rgb[2], cmap="Reds")
        plt.title("Red channel")

        # Recontruir la de pseudocolor
        plt.figure()
        plt.imshow(frac)
        plt.show()

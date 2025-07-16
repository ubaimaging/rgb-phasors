import numpy as np
import matplotlib.pyplot as plt
import tools
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal, phasor_threshold
from phasorpy.components import phasor_component_fit


# Obtain the center of each components 
path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/"
blue = plt.imread(path + "janelia_sample_dapi_roi1.tif")
green = plt.imread(path + "janelia_sample_488_roi2.tif")
red = plt.imread(path + "janelia_sample_555_roi1.tif")

path_im = "/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelia 2025/rgb/"
image = plt.imread(path_im + "roi14.tif")[350:2000, 50:1600]

# 🔧 Ajuste de estilo global para figuras científicas
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'savefig.dpi': 600,
    'figure.dpi': 150
})


# This part of the code is used to plot the components of the image
# and their phasor plots
# It is useful to visualize the components of the image and their phasor plots
# to analyze the spectral unmixing of the image
comp_plot = True
if comp_plot:

    # Plotting the components
    plt.figure(1)
    plt.imshow(blue)
    plt.figure(2)
    plt.imshow(green)
    plt.figure(3)
    plt.imshow(red)

    # Phasor Plot for each component 
    avgb, realb, imagb = phasor_from_signal(
        tools.rgb2bgr(blue)[0:, 500:700, 700:1100], axis=0)
    avgg, realg, imagg = phasor_from_signal(
        tools.rgb2bgr(green)[0:, 400:1400, 1000:1600], axis=0)
    avgr, realr, imagr = phasor_from_signal(
        tools.rgb2bgr(red)[0:, 600:1000, 1200:1400], axis=0)

    realb_cm, imagb_cm  = tools.phasor_center_of_mass(realb, imagb, avgb)
    realg_cm, imagg_cm  = tools.phasor_center_of_mass(realg, imagg, avgg)
    realr_cm, imagr_cm  = tools.phasor_center_of_mass(realr, imagr, avgr)

    plot = PhasorPlot(allquadrants=True, title='Components Phasor plot')
    plot.hist2d(realb.flatten(), imagb.flatten(), cmap="RdYlGn_r")
    plt.plot(realb_cm, imagb_cm, color='blue', markersize=12, marker='o', 
                linestyle='None', label='Blue channel')
    
    plot.hist2d(realg.flatten(), imagg.flatten(), cmap="RdYlGn_r")
    plt.plot(realg_cm, imagg_cm, color='green', markersize=12, marker='p', 
                linestyle='None', label='Green channel')
    
    plot.hist2d(realr.flatten(), imagr.flatten(), cmap="RdYlGn_r")
    plt.plot(realr_cm, imagr_cm, color='red', markersize=12, marker='*', 
                linestyle='None', label='Red channel')
    plt.legend()

# Spectral unmixing Analysis
# This part of the code is used to analyze the spectral unmixing of the image
sp_unmixing = True
if sp_unmixing:

    plt.figure()
    plt.imshow(image)
    plt.title("Original image")

    img = tools.rgb2bgr(image)

    img[0] = img[0] + 0
    img[1] = img[1] + 0
    img[1] = img[1] + 0

    avg, real, imag = phasor_from_signal(img, axis=0)

    plot = PhasorPlot(allquadrants=True, title='Phasor plot of the image')
    plot.hist2d(real.flatten(), imag.flatten(), cmap="RdYlGn_r")
    plt.plot(realb_cm, imagb_cm, color='blue', markersize=12, marker='o', 
                linestyle='None', label='Blue channel')
    plt.plot(realg_cm, imagg_cm, color='green', markersize=12, marker='p', 
                linestyle='None', label='Green channel')
    plt.plot(realr_cm, imagr_cm, color='red', markersize=12, marker='*', 
                linestyle='None', label='Red channel')
    plt.legend()

    gs = np.array([realb_cm, realg_cm, realr_cm])
    ss = np.array([imagb_cm, imagg_cm, imagr_cm])

    # threshold
    avg, real, imag = phasor_threshold(avg, real, imag, mean_min=8)

    # fit the phasor components
    frac = phasor_component_fit(avg, real, imag, gs, ss)

    # Umbralizar las fracciones con los rangos del histograma de cada componente
    rang = np.array([[0.5, 0.7], [0.5, 1.0], [0.2, 0.5]])
    # rang = np.array([[-2, 2], [-2, 2], [-2, 2]], dtype=float)
    fracb = tools.umbralizar_por_rango(frac[0], rang[0][0], rang[0][1])
    fracg = tools.umbralizar_por_rango(frac[1], rang[1][0], rang[1][1])
    fracr = tools.umbralizar_por_rango(frac[2], rang[2][0], rang[2][1])

    # Plot Histograms
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    # Subplot 1: Histogramas completos
    axs[0].hist(frac[0].flatten(),
                bins=256, color='blue', alpha=0.5, label='Blue channel', log=True)
    axs[0].hist(frac[1].flatten(),
                bins=256, color='green', alpha=0.5, label='Green channel', log=True)
    axs[0].hist(frac[2].flatten(),
                bins=256, color='red', alpha=0.5, label='Red channel', log=True)
    axs[0].set_title("Histogram of fractions")
    axs[0].set_xlabel("Fraction value")
    axs[0].set_ylabel("Frequency")
    axs[0].legend()
    # Subplot 2: Histogramas sin ceros
    axs[1].hist(fracb.flatten()[fracb.flatten() != 0],
                bins=256, color='blue', alpha=0.5, label='Blue channel', log=True)
    axs[1].hist(fracg.flatten()[fracg.flatten() != 0],
                bins=256, color='green', alpha=0.5, label='Green channel', log=True)
    axs[1].hist(fracr.flatten()[fracr.flatten() != 0],
                bins=256, color='red', alpha=0.5, label='Red channel', log=True)
    axs[1].set_title("Histogram of non-zero fractions")
    axs[1].set_xlabel("Fraction value")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlim(0, 1.2)
    axs[1].legend()
    plt.tight_layout()

    # plotear las tres imagenes por separado
    plt.figure()
    plt.imshow(fracb, cmap="Blues", interpolation='nearest')
    plt.title("Blue channel")

    plt.figure()
    plt.imshow(fracg, cmap="Greens", interpolation='nearest')
    plt.title("Green channel")

    plt.figure()
    plt.imshow(fracr, cmap="Reds", interpolation='nearest')
    plt.title("Red channel")


    # Photon counting Analysis (PCA)
    # This part of the code is used to analyze the photon counting of the image
    # and plot the histograms of the photon counting for each channel
    pca = True
    if pca:
        # Histogram for each component
        plt.figure()
        plt.hist(img[0].flatten()[fracb.flatten() != 0],
                    bins=256, color='blue', alpha=0.5, label='Blue channel', log=True)
        plt.hist(img[1].flatten()[fracg.flatten() != 0], 
                    bins=256, color='green', alpha=0.5, label='Green channel', log=True)
        plt.hist(img[2].flatten()[fracr.flatten() != 0],
                    bins=256, color='red', alpha=0.5, label='Red channel', log=True)
        plt.legend()

        # Photon counting for each componen
        sumb = np.sum(img[0].flatten()[
            (fracb.flatten() >= rang[0][0]) & (fracb.flatten() <= rang[0][1])])
        sumg = np.sum(img[1].flatten()[
            (fracg.flatten() >= rang[1][0]) & (fracg.flatten() <= rang[1][1])])
        sumr = np.sum(img[2].flatten()[
            (fracr.flatten() >= rang[2][0]) & (fracr.flatten() <= rang[2][1])])

        print("Photon counting:", sumb, sumg, sumr)

        # Photon counting image
        blue = sumb * fracb
        green = sumg * fracg
        red = sumr * fracr

        # plot de canales separados con escalas de conteo de fotones
        plt.figure()
        plt.imshow(blue, cmap='plasma', interpolation='nearest')
        plt.title("Blue channel")
        plt.colorbar()

        plt.figure()
        plt.imshow(green, cmap='plasma', interpolation='nearest')
        plt.title("Green channel")
        plt.colorbar()

        plt.figure()
        plt.imshow(red, cmap='plasma', interpolation='nearest')
        plt.title("Red channel")
        plt.colorbar()

        #######################################
        #######################################
        # Plot con fondo cero 
        plt.figure()
        plt.imshow(np.nan_to_num(blue, nan=0.0), cmap='plasma', interpolation='nearest')
        plt.title("Blue channel")
        plt.colorbar()

        plt.figure()
        plt.imshow(np.nan_to_num(green, nan=0.0), cmap='plasma', interpolation='nearest')
        plt.title("Green channel")
        plt.colorbar()

        plt.figure()
        plt.imshow(np.nan_to_num(red, nan=0.0), cmap='plasma', interpolation='nearest')
        plt.title("Red channel")
        plt.colorbar()

        def normalize(img):
            img = img - np.nanmin(img)
            img = img / np.nanmax(img)
            return img

        # Aplicás la normalización
        R = normalize(red)
        G = normalize(green)
        B = normalize(blue)

        # Stack en eje de canales
        rgb_image = np.stack((R, G, B), axis=-1)

        # Mostrar
        plt.figure()
        plt.title("RGB Image")
        plt.imshow(rgb_image, interpolation='nearest')
        plt.axis('off')

        plt.show()


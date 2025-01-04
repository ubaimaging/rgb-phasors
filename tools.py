import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.filters import median
from matplotlib.widgets import Cursor
import math
import seaborn as sns
import pandas as pd
import tifffile as tiff
import os
from matplotlib.colors import hsv_to_rgb
import pandas as pd
from natsort import natsorted


def phasor(image_stack, harmonic=1):
    """
        This function computes the average intensity image, the G and S coordinates of the phasor.
    As well as the modulation and phase.

    :param image_stack: is a file with spectral mxm images to calculate the fast fourier transform from
    numpy library.
    :param harmonic: int. The number of the harmonic where the phasor is calculated.
    :return: avg: is the average intensity image
    :return: g: is mxm image with the real part of the fft.
    :return: s: is mxm imaginary with the real part of the fft.
    :return: md: numpy.ndarray  It is the modulus obtain with Euclidean Distance.
    :return: ph: is the phase between g and s in degrees.
    """

    data = np.fft.fft(image_stack, axis=0, norm='ortho')

    dc = data[0].real
    g = data[harmonic].real
    g /= dc
    s = data[harmonic].imag
    s /= -dc
    return dc, g, s

def unnormalized_phasor(image_stack, harmonic=1):

    data = np.fft.fft(image_stack, axis=0, norm='ortho')

    dc = data[0].real
    g = data[harmonic].real
    s = -data[harmonic].imag
    return dc, g, s


def phasor_circle(ax):
    """
        Built the figure inner and outer circle and the 45 degrees lines in the plot
    :param ax: axis where to plot the phasor circle.
    :return: the axis with the added circle.
    """

    x1 = np.linspace(start=-1, stop=1, num=500)
    yp1 = lambda x1: np.sqrt(1 - x1 ** 2)
    yn1 = lambda x1: -np.sqrt(1 - x1 ** 2)

    x2 = np.linspace(start=-0.5, stop=0.5, num=500)
    yp2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
    yn2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)

    x3 = np.linspace(start=-1, stop=1, num=30)
    x4 = np.linspace(start=-0.7, stop=0.7, num=30)

    ax.plot(x1, list(map(yp1, x1)), color='darkgoldenrod')
    ax.plot(x1, list(map(yn1, x1)), color='darkgoldenrod')
    ax.plot(x2, list(map(yp2, x2)), color='darkgoldenrod')
    ax.plot(x2, list(map(yn2, x2)), color='darkgoldenrod')
    ax.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    ax.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    ax.scatter(x4, x4, marker='_', color='darkgoldenrod')
    ax.scatter(x4, -x4, marker='_', color='darkgoldenrod')

    return ax


def circle_lines(ax, phase):
    """
        Built the figure inner and outer circle and the 45 degrees lines in the plot
    :param phase: array containing the 5 phases in degrees to plot the lines
    :param ax: axis where to plot the phasor circle.
    :return: the axis with the added circle.
    """
    x1 = np.linspace(start=-1, stop=1, num=500)
    yp1 = lambda x1: np.sqrt(1 - x1 ** 2)
    yn1 = lambda x1: -np.sqrt(1 - x1 ** 2)
    x2 = np.linspace(start=-0.5, stop=0.5, num=500)
    yp2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
    yn2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
    x3 = np.linspace(start=-1, stop=1, num=30)
    #  circle
    ax.plot(x1, list(map(yp1, x1)), color='darkgoldenrod')
    ax.plot(x1, list(map(yn1, x1)), color='darkgoldenrod')
    ax.plot(x2, list(map(yp2, x2)), color='darkgoldenrod')
    ax.plot(x2, list(map(yn2, x2)), color='darkgoldenrod')
    #  x = 0 and y = 0
    ax.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
    ax.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
    theta = np.pi / 180
    #  lines
    x11 = np.linspace(start=0, stop=0.34, num=100)
    x12 = np.linspace(start=0, stop=0.21, num=100)
    x13 = np.linspace(start=-0.173, stop=0, num=100)
    x14 = np.linspace(start=-0.5, stop=0, num=100)
    x15 = np.linspace(start=-0.70, stop=0, num=100)
    ax.plot(x11, np.tan(phase[0] * theta) * x11, color='mediumvioletred')
    ax.plot(x12, np.tan(phase[1] * theta) * x12, color='mediumslateblue')
    ax.plot(x13, np.tan(phase[2] * theta) * x13, color='cyan')
    ax.plot(x14, np.tan(phase[3] * theta) * x14, color='lime')
    ax.plot(x15, np.tan(phase[4] * theta) * x15, color='red')
    return ax


def phasor_figure(x, y, phases=None, circle_plot=False, phases_lines=False):
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.suptitle('Phasor')
    ax.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    if circle_plot:
        phasor_circle(ax)
    if phases_lines:
        circle_lines(ax, phases)
    return ax


def rgb2bgr(im):
    """
    Parameters
    ----------
    im : _type_ image data RGB
    Returns
    -------
    _type_ reorganized image BGR
    """
    im1 = im[:, :, 0:1]
    imr = im1.reshape(im1.shape[0], im1.shape[1])
    im2 = im[:, :, 1:2]
    img = im2.reshape(im2.shape[0], im2.shape[1])
    im3 = im[:, :, 2:3]
    imb = im3.reshape(im3.shape[0], im3.shape[1])
    bgr = np.asarray([imb, img, imr])
    return bgr


def convert_rgb_to_bgr(image):
    """
    Cambia la posición de las coordenadas de color en una imagen RGB
    para convertirla a formato BGR.
    
    Args:
        image: Una matriz numpy de forma (H, W, 3) que representa una imagen RGB.
    
    Returns:
        Una matriz numpy con los canales de color intercambiados a formato BGR.
    """
    # Intercambiar los canales rojo y azul
    bgr_image = image[..., [2, 1, 0]]
    return bgr_image


def median_filter(im, n):
    imf = np.copy(im)
    for i in range(n):
        imf = median(imf)
    return imf


def phasor_clustering(dc, x, nclusters = 2):
    # Clustering segmentation
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(x)
    pred_y = kmeans.fit_predict(x)
    cm = kmeans.cluster_centers_
    imp = pred_y.reshape(dc.shape)
    return pred_y, imp, cm


def cluster_phasor_plot(X, pred_y, nclusters=3, title= " ", cluster_type=1):
    from phasorpy.plot import PhasorPlot
    from matplotlib import pyplot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax = pyplot.subplot(1, 1, 1)
    if cluster_type == 1:
        plot = PhasorPlot(ax=ax, allquadrants=True, title='Phasor plot: ' + title)

    if nclusters == 2:
        p0 = np.where(pred_y == 0)
        p1 = np.where(pred_y == 1)
        ax.scatter(X[p0[0], 0], X[p0[0], 1], c='b')
        ax.scatter(X[p1[0], 0], X[p1[0], 1], c='lime')
    
    if nclusters == 3:
        p0 = np.where(pred_y == 0)
        p1 = np.where(pred_y == 1)
        p2 = np.where(pred_y == 2)
        ax.scatter(X[p0[0], 0], X[p0[0], 1], c='r')
        ax.scatter(X[p1[0], 0], X[p1[0], 1], c='b')
        ax.scatter(X[p2[0], 0], X[p2[0], 1], c='lime')

    if nclusters == 4:
        p0 = np.where(pred_y == 0)
        p1 = np.where(pred_y == 1)
        p2 = np.where(pred_y == 2)
        p3 = np.where(pred_y == 3)
        ax.scatter(X[p0[0], 0], X[p0[0], 1], c='r')
        ax.scatter(X[p1[0], 0], X[p1[0], 1], c='blue')
        ax.scatter(X[p2[0], 0], X[p2[0], 1], c='lime')
        ax.scatter(X[p3[0], 0], X[p3[0], 1], c='k')
    
    if nclusters == 6:
        p0 = np.where(pred_y == 0)
        p1 = np.where(pred_y == 1)
        p2 = np.where(pred_y == 2)
        p3 = np.where(pred_y == 3)
        p4 = np.where(pred_y == 4)
        p5 = np.where(pred_y == 5)
        ax.scatter(X[p0[0], 0], X[p0[0], 1], c='cyan')
        ax.scatter(X[p1[0], 0], X[p1[0], 1], c='lime')
        ax.scatter(X[p2[0], 0], X[p2[0], 1], c='yellow')
        ax.scatter(X[p3[0], 0], X[p3[0], 1], c='magenta')
        ax.scatter(X[p4[0], 0], X[p4[0], 1], c='b')
        ax.scatter(X[p5[0], 0], X[p5[0], 1], c='r')
        # phasor_circle(ax)

# -------------------------
#   INTERACTIVE FUNCTIONS 
# -------------------------

def interactive(dc, g, s, Ro, nbit, filter=3):
    """
        This function plot the avg image, its histogram, the phasors and the rbg pseudocolor image.
    To get the phasor the user must pick an intensity cut umbral in the histogram in order to plot the phasor.
    To get the rgb pseudocolor image you must pick three circle in the phasor plot.
    :param nbit: bits of the image
    :param dc: average intensity image. ndarray
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :param Ro: radius of the circle to select pixels in the phasor
    :return: fig: figure contains the avg, histogram, phasor and pseudocolor image.
    """

    median_filter(g, filter)
    median_filter(s, filter)

    nbit = 2**nbit
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    ax[0, 0].imshow(dc, cmap='gray')
    ax[0, 0].axis('off')
    ax[0, 0].set_title('Average intensity image')
    ax[0, 1].hist(dc.flatten(), bins=nbit, range=(0, nbit))
    ax[0, 1].set_yscale("log")
    cursor = Cursor(ax[0, 1], horizOn=False, vertOn=True, color='darkgoldenrod')
    ax[0, 1].set_title('Average intensity image histogram')

    ic = plt.ginput(1, timeout=0)
    ic = int(ic[0][0])

    x, y = histogram_thresholding(dc, g, s, ic)  # x y contain g and s coordinate to pass to hist2d function

    phasor_circle(ax[1, 0])
    ax[1, 0].hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    ax[1, 0].set_title('Phasor')

    center = plt.ginput(3, timeout=0)  # store the center of each circle

    ccolor = ['blue', 'green', 'red']
    for i in range(3):
        circle = plt.Circle((center[i][0], center[i][1]), Ro, color=ccolor[i], fill=False)
        ax[1, 0].add_patch(circle)

    rgba = rgb_coloring(dc, g, s, ic, center, Ro)
    ax[1, 1].imshow(rgba)
    ax[1, 1].set_title('Pseudocolor image')
    ax[1, 1].axis('off')
    plt.show()
    return fig


def histogram_thresholding(dc, g, s, ic):
    """
        Use this function to filter the background deleting, those pixels where the intensity value is under ic.
    :param dc: ndarray. Intensity image.
    :param g:  ndarray. G image.
    :param s:  ndarray. S image.
    :param ic: intensity cut umbral.
    :return: x, y. Arrays contain the G and S phasor coordinates.
    """

    """store the coordinate to plot in the phasor"""
    aux = np.concatenate(np.where(dc > ic, dc, np.zeros(dc.shape)))
    g2 = np.concatenate(g)
    s2 = np.concatenate(s)
    x = np.delete(g2, np.where(aux == 0))
    y = np.delete(s2, np.where(aux == 0))
    return x, y



def rgb_coloring(dc, g, s, ic, center, Ro):
    """
        Create a matrix to see if a pixels is into the circle, using circle equation
    so the negative values of Mi means that the pixel belong to the circle and multiply
    aux1 to set zero where the avg image is under ic value
    :param dc: ndarray. Intensity image.
    :param g:  ndarray. G image.
    :param s:  ndarray. S image.
    :param ic: intensity cut umbral.
    :param Ro: circle radius.
    :param center: ndarray containing the center coordinate of each circle.
    :return: rgba pseudocolored image.
    """
    aux1 = np.where(dc > ic, dc, np.zeros(dc.shape))
    M1 = ((g - center[0][0]) ** 2 + (s - center[0][1]) ** 2 - Ro ** 2) * aux1
    M2 = ((g - center[1][0]) ** 2 + (s - center[1][1]) ** 2 - Ro ** 2) * aux1
    M3 = ((g - center[2][0]) ** 2 + (s - center[2][1]) ** 2 - Ro ** 2) * aux1

    # img_new = np.copy(dc)
    img_new = np.zeros(dc.shape)  # todo si uso esto escribe sobre una img de fondo negro

    indices1 = np.where(M1 < 0)
    indices2 = np.where(M2 < 0)
    indices3 = np.where(M3 < 0)

    cmap = plt.cm.gray
    norm = plt.Normalize(img_new.min(), img_new.max())
    rgba = cmap(norm(img_new))
    # Set the colors
    rgba[indices1[0], indices1[1], :3] = 0, 0, 1  # blue
    rgba[indices2[0], indices2[1], :3] = 0, 1, 0  # green
    rgba[indices3[0], indices3[1], :3] = 1, 0, 0  # red

    return rgba


def interactive2(dc, g, s, Ro, nbit, filter=3):
    """
        This function plot the avg image, its histogram, the phasors and the rbg pseudocolor image.
    To get the phasor the user must pick an intensity cut umbral in the histogram in order to plot the phasor.
    To get the rgb pseudocolor image you must pick three circle in the phasor plot.
    :param nbit: bits of the image
    :param dc: average intensity image. ndarray
    :param g: image. ndarray. Contains the real coordinate G of the phasor
    :param s: image. ndarray. Contains the imaginary coordinate S of the phasor
    :param Ro: radius of the circle to select pixels in the phasor
    :return: fig: figure contains the avg, histogram, phasor and pseudocolor image.
    """

    median_filter(g, filter)
    median_filter(s, filter)

    nbit = 2**nbit

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(dc.flatten(), bins=nbit, range=(0, nbit))
    ax.set_yscale("log")
    cursor = Cursor(ax, horizOn=False, vertOn=True, color='darkgoldenrod')

    ic = plt.ginput(1, timeout=0)
    ic = int(ic[0][0])
    x, y = histogram_thresholding(dc, g, s, ic)  # x y contain g and s coordinate to pass to hist2d function

    fig, ax2 = plt.subplots(figsize=(6, 6))
    phasor_circle(ax2)
    ax2.hist2d(x, y, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    ax2.set_title('Phasor')

    center = plt.ginput(3, timeout=0)  # store the center of each circle

    ccolor = ['blue', 'green', 'red']
    for i in range(3):
        circle = plt.Circle((center[i][0], center[i][1]), Ro, color=ccolor[i], fill=False)
        ax2.add_patch(circle)

    plt.figure(3)
    plt.imshow(dc, cmap="gray")

    plt.figure(4)
    rgba = rgb_coloring(dc, g, s, ic, center, Ro)
    plt.imshow(rgba)
    plt.title('Pseudocolor image')
    plt.axis('off')

    plt.show()
    return fig


def mask_with_predict_clusters(X, Y, g, s, im):
    N = X.shape[0]
    M, K = g.shape
    
    for n in range(N):
        for i in range(M):
            for j in range(K):
                if X[n][0] == g[i][j] and X[n][1] == s[i][j]:
                    im[i][j] = Y[n]
    
    return im


def unmixing_from_phasor(
        multi_harmonic_real,
        multi_harmonic_imag,
        matrixA):
    
    import numpy 
    
    """ Return fractions in each pixel from multiple components.

        Parametres
    ----------
    multi_harmonic_real : array_like
        Real components of the phasor coordinate for many harmonics.
    multi_harmonic_imag : array_like
        Imaginary components of the phasor coordinate for many harmonics.
    matrixA : array_like 
        Coefficiency matrix for each components. 

    Returns
    -------
    fractions : ndarray
        Fractions of each components. """

    multi_harmonic_real = numpy.asarray(multi_harmonic_real)
    multi_harmonic_imag = numpy.asarray(multi_harmonic_imag)
    matrixA = numpy.asarray(matrixA)

    if multi_harmonic_real.shape != multi_harmonic_imag.shape:
        raise ValueError("multi_harmonic_real and multi_harmonic_imag"
                         "have different shape")
    if matrixA.size == 0:
        raise ValueError("matrixA is empty")
    
    ncomp = matrixA.shape[0] - 1
    nh = math.floor(ncomp / 2)
    if len(multi_harmonic_real.shape) == 2:
        vecB = [multi_harmonic_real[j] for j in range(nh)] \
            + [multi_harmonic_imag[j] for j in range(nh)] + [1]
        return numpy.linalg.lstsq(matrixA, vecB, rcond=None)[0]
    else:
        fractions = numpy.zeros([multi_harmonic_real.shape[0], 
                                 multi_harmonic_imag.shape[1], ncomp])
        for r in range(multi_harmonic_real.shape[0]):
            for c in range(multi_harmonic_real.shape[1]):
                vecB = [multi_harmonic_real[r, c, j] for j in range(nh)] \
                    + [multi_harmonic_imag[r, c, j] for j in range(nh)] + [1]
                fractions[r, c] = numpy.linalg.lstsq(matrixA, vecB,
                                                     rcond=None)[0]
        return fractions
    


def map_to_rgb(array):
    """
    Mapea combinaciones de ceros y unos a un espacio de color personalizado.

    :param array: numpy array de dimensiones (n, n, 3) con valores 0 y 1
    :return: numpy array de dimensiones (n, n, 3) con colores personalizados en formato RGB
    """
    # Validar que el array tiene tres canales
    if array.shape[-1] != 3:
        raise ValueError("El array debe tener 3 canales en la última dimensión.")
    
    # Definir el mapeo de combinaciones a colores según la nueva base
    color_map = {
        (0, 0, 0): [0, 0, 0],      # Negro
        (1, 0, 0): [0, 0, 255],    # Azul
        (0, 1, 0): [0, 255, 0],    # Verde
        (0, 0, 1): [255, 0, 0],    # Rojo
        (1, 1, 0): [0, 255, 255],  # Cian (azul + verde)
        (1, 0, 1): [255, 0, 255],  # Magenta (azul + rojo)
        (0, 1, 1): [255, 255, 0],  # Amarillo (verde + rojo)
        (1, 1, 1): [255, 255, 255] # Blanco
    }

    # Crear una imagen RGB del mismo tamaño
    rgb_image = np.zeros(array.shape, dtype=np.uint8)

    # Asignar colores según las combinaciones
    for combination, color in color_map.items():
        mask = (array[:, :, 0] == combination[0]) & \
               (array[:, :, 1] == combination[1]) & \
               (array[:, :, 2] == combination[2])
        rgb_image[mask] = color

    return rgb_image


def construct_label_array_optimized(xt, xn, labels):
    """
    Construye un array optimizado para grandes volúmenes de datos,
    asignando los valores de labels a posiciones de xt coincidentes con xn.
    
    Args:
        xt (np.ndarray): Array de dimensión (M, 2).
        xn (np.ndarray): Array de dimensión (N, 2).
        labels (np.ndarray): Array de dimensión (N,).
    
    Returns:
        np.ndarray: Array de dimensión (M,) con valores de labels o ceros.
    """
    # Asegurarse de que los arrays sean contiguos en memoria
    xt = np.ascontiguousarray(xt)
    xn = np.ascontiguousarray(xn)
    
    # Crear un array inicializado con ceros del mismo tamaño que xt
    result = np.zeros(len(xt), dtype=labels.dtype)

    # Convertir xn y xt a tuplas para usarlas como claves hashables
    xn_tuples = [tuple(row) for row in xn]
    xt_tuples = [tuple(row) for row in xt]

    # Construir un índice que asocia xn con sus labels
    index_map = {coord: label for coord, label in zip(xn_tuples, labels)}

    # Asignar valores usando una comprensión de lista
    result = np.array([index_map.get(coord, 0) for coord in xt_tuples], dtype=labels.dtype)
    return result


def map_values_to_rgb(array):
    """
    Convierte un array con valores 1, 2, 3 a colores RGB.
    
    Args:
        array (np.ndarray): Array con valores 1, 2 y 3.
    
    Returns:
        np.ndarray: Array RGB de dimensiones (H, W, 3).
    """
    # Crear un array RGB inicializado con ceros
    rgb_array = np.zeros((*array.shape, 3), dtype=np.uint8)
    
    # Mapear valores a colores
    rgb_array[array == 1] = [255, 0, 0]    # Azul para 1
    rgb_array[array == 2] = [0, 0, 255]    # Rojo para 2
    rgb_array[array == 3] = [0, 255, 0]    # Verde para 3
    
    return rgb_array


def plot_separated_boxplots_and_violin(d):
    """
    Genera una figura con boxplots y swarmplots, y una figura separada con violin plots,
    mostrando medias y desviaciones estándar con líneas en el violin plot y valores en el xlabel.
    
    Parámetros:
    - d: np.array, un array de tamaño (2, 100).
    """
    # Verificar que los datos tengan la forma esperada
    if d.shape != (2, 100):
        raise ValueError("El array de entrada debe tener forma (2, 100)")

    # Reorganizar a (2, 10, 10)
    d_reshaped = d.reshape(2, 10, 10)

    # Calcular medias y desviaciones estándar para los subgrupos
    means = np.mean(d_reshaped, axis=2)  # (2, 10)
    stds = np.std(d_reshaped, axis=2)    # (2, 10)

    # Colores y etiquetas
    violin_colors = ['#DFF2FF', '#E6FFE6']  # Azul claro y verde claro
    labels = ['ND', 'ND Inst']

    # Figura 1: Boxplots
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Boxplots y swarmplot con los datos reorganizados
    box_data = []
    box_labels = []

    for group_idx in range(2):
        for subgroup_idx in range(10):
            box_data.append(d_reshaped[group_idx, subgroup_idx])
            box_labels.extend([f"{labels[group_idx]}-{subgroup_idx + 1}"] * 10)

    # Combinar datos en un DataFrame para Seaborn
    df = pd.DataFrame({
        'Values': np.concatenate(box_data),
        'Group': box_labels
    })

    # Boxplot con Seaborn
    sns.boxplot(
        data=df,
        x='Group',
        y='Values',
        ax=ax1,
        palette=sns.color_palette([violin_colors[0]] * 10 + [violin_colors[1]] * 10),
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "red",  # Puntos de media en rojo
            "markeredgecolor": "red"
        },
        flierprops={"marker": "*", "color": "black", "alpha": 0.8}  # Outliers como puntos negros
    )

    # Swarmplot sobre el Boxplot
    sns.swarmplot(
        data=df,
        x='Group',
        y='Values',
        ax=ax1,
        color='black',  # Puntos negros
        alpha=0.8,
        size=3
    )

    # Configuración del boxplot y swarmplot
    ax1.set_title("Boxplot with Swarmplot", fontsize=16, weight='bold')
    ax1.set_xlabel("Subgroups", fontsize=14, weight='bold')
    ax1.set_ylabel("Values", fontsize=14, weight='bold')
    ax1.tick_params(axis='x', labelsize=12)  # Configuración del tamaño de las etiquetas
    ax1.tick_params(axis='y', labelsize=12)  # Configuración del tamaño de las etiquetas
    ax1.set_xticklabels(ax1.get_xticklabels(), fontweight='bold', rotation=90)  # Etiquetas X verticales
    ax1.set_yticklabels(ax1.get_yticklabels(), fontweight='bold')  # Pesos en Y

    # Agregar leyenda al gráfico de boxplots
    ax1.legend(
        handles=[
            plt.Line2D([0], [0], color=violin_colors[0], lw=4, label='ND Boxplot'),
            plt.Line2D([0], [0], color=violin_colors[1], lw=4, label='ND Inst Boxplot'),
            plt.Line2D([0], [0], marker='*', color='black', lw=0, label='Outliers'),
            plt.Line2D([0], [0], marker='o', color='red', lw=0, label='Mean')
        ],
        title="Legend",
        loc="upper left",
        fontsize=10,
        title_fontsize=12
    )

    # Ajustar y mostrar la figura 1
    plt.tight_layout()
    # plt.show()

    # Figura 2: Violin Plots
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Violin plot con los datos originales (2, 100)
    x_labels = []  # Para construir el xlabel

    for group_idx in range(2):
        parts = ax2.violinplot(
            d[group_idx],
            positions=[group_idx],
            showmeans=False,
            showextrema=False,
            showmedians=False
        )
        # Personalizar estilo de los violines
        for pc in parts['bodies']:
            pc.set_facecolor(violin_colors[group_idx])  # Color de fondo claro
            pc.set_edgecolor('black')  # Borde negro
            pc.set_alpha(0.7)

        # Calcular media y std del grupo
        group_mean = np.mean(d[group_idx])
        group_std = np.std(d[group_idx])

        # Dibujar líneas cortas para la media y las desviaciones estándar
        line_width = 0.2  # Ancho relativo de las líneas
        ax2.plot(
            [group_idx - line_width, group_idx + line_width],
            [group_mean, group_mean],
            color='red', linestyle='--', linewidth=3, label='Mean' if group_idx == 0 else ""
        )
        ax2.plot(
            [group_idx - line_width, group_idx + line_width],
            [group_mean + 2 * group_std, group_mean + 2 * group_std],
            color='black', linestyle='--', linewidth=3, label='+2 Std' if group_idx == 0 else ""
        )
        ax2.plot(
            [group_idx - line_width, group_idx + line_width],
            [group_mean - 2 * group_std, group_mean - 2 * group_std],
            color='black', linestyle='--', linewidth=3, label='-2 Std' if group_idx == 0 else ""
        )

        # Construir la etiqueta para el xlabel
        x_labels.append(f"{labels[group_idx]}:\nμ={group_mean:.2f}, σ={group_std:.2f}")

    # Configuración del violin plot
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(x_labels, fontsize=12, weight='bold')
    ax2.set_title("Violin Plot", fontsize=16, weight='bold')
    ax2.set_xlabel("Groups", fontsize=14, weight='bold')
    ax2.set_ylabel("Values", fontsize=14, weight='bold')

    # Leyenda de las líneas
    ax2.legend(loc="upper left", fontsize=10, title="Legend", title_fontsize=12)

    # Ajustar y mostrar la figura 2
    plt.tight_layout()
    plt.show()



def read_and_plot_tif_tifffile(image_path):
    """
    Lee y muestra una imagen TIFF usando tifffile.

    Parámetros:
    - image_path (str): Ruta de la imagen TIFF.
    """
    try:
        # Leer la imagen TIFF
        img = tiff.imread(image_path)

        # Mostrar información básica
        print(f"Tamaño de la imagen: {img.shape}")
        print(f"Tipo de datos: {img.dtype}")

        # Determinar colormap si es necesario
        cmap = 'cividis' if img.ndim == 2 else None

        # Mostrar la imagen
        plt.figure(figsize=(7, 7))
        plt.imshow(img, cmap=cmap)
        plt.axis('off')  # Desactiva los ejes
        plt.title(f"Imagen: {os.path.basename(image_path)}", fontsize=14, weight='bold')
        plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"Error al leer la imagen TIFF: {e}")


def plot_all_tifs_in_folder(folder_path):
    """
    Busca y muestra todas las imágenes TIFF en una carpeta usando tifffile.

    Parámetros:
    - folder_path (str): Ruta de la carpeta que contiene imágenes TIFF.
    """
    try:
        # Obtener lista de archivos en la carpeta
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

        if not files:
            print("No se encontraron imágenes TIFF en la carpeta.")
            return

        # Leer y mostrar cada archivo TIFF
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            print(f"Procesando: {file_path}")
            read_and_plot_tif_tifffile(file_path)

    except Exception as e:
        print(f"Error al procesar la carpeta: {e}")


def invert_mask(im):
    return im * - 1 + np.max(im) 


def transform_array(array):
    """
    Transforma un array de dimensión (n, m, 4) a (n, m, 3), 
    eliminando la última coordenada y 
    cambiando el orden de la primera y segunda coordenada.

    Args:
        array (numpy.ndarray): Array de entrada con dimensión (n, m, 4).

    Returns:
        numpy.ndarray: Array transformado con dimensión (n, m, 3).
    """
    if array.shape[-1] != 4:
        raise ValueError("El array debe tener dimensión (n, m, 4) en su última coordenada.")

    # Eliminar la última coordenada
    transformed = array[:, :, :3]

    # Intercambiar la primera entrada con la segunda
    transformed[:, :, [0, 1]] = transformed[:, :, [1, 0]]

    return transformed


def replace_with_nan(array):
    """
    Reemplaza todas las entradas (1, 1, 1) en un array tridimensional con NaN.

    Args:
        array (numpy.ndarray): Array de entrada de dimensión (n, m, k), donde k >= 3.

    Returns:
        numpy.ndarray: Array con las entradas (1, 1, 1) reemplazadas por NaN.
    """
    if array.shape[-1] < 3:
        raise ValueError("El array debe tener al menos 3 dimensiones en la última coordenada.")

    # Crear una máscara para identificar las entradas (1, 1, 1)
    mask = (array[..., 0] == 1) & (array[..., 1] == 1) & (array[..., 2] == 1)

    # Reemplazar las entradas (1, 1, 1) por NaN
    array = array.astype(float)  # Convertir a tipo float para admitir NaN
    array[mask] = np.nan

    return array


def generate_color_wheel_image(resolution=256):
    """
    Genera una imagen RGB de una rueda de colores basada en el modelo HSV.
    
    Args:
        resolution: Resolución de la imagen (pixels por lado, cuadrada).
    
    Returns:
        Una matriz RGB que representa la rueda de colores.
    """
    # Crear coordenadas x, y centradas
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xv, yv = np.meshgrid(x, y)
    
    # Convertir coordenadas a ángulo (hue) y radio (valor)
    angle = np.arctan2(yv, xv)  # Ángulo en radianes
    angle = (angle + 2 * np.pi) % (2 * np.pi)  # Normalizar entre 0 y 2pi
    hue = angle / (2 * np.pi)  # Convertir a rango [0, 1]
    radius = np.sqrt(xv**2 + yv**2)  # Distancia radial desde el centro
    
    # Crear la imagen HSV
    saturation = np.clip(radius, 0, 1)  # Saturación depende del radio
    value = np.ones_like(radius)  # Valor constante (pleno brillo)
    hsv_image = np.dstack((hue, saturation, value))
    
    # Convertir la imagen HSV a RGB
    rgb_image = hsv_to_rgb(hsv_image)
    
    # Aplicar un círculo para recortar los valores fuera del rango [0, 1] en radio
    mask = radius <= 1
    rgb_image[~mask] = 1  # Fuera del círculo, establecer en blanco
    
    return rgb_image


def map_mask_to_colors(mask, ind=[0, 1, 2, 3]):
    """
    Mapea una máscara con valores (0, 1, 2, 3) a colores específicos:
    - 0 -> Negro (0, 0, 0)
    - 1 -> Rojo (255, 0, 0)
    - 2 -> Verde (0, 255, 0)
    - 3 -> Azul (0, 0, 255)

    Args:
        mask: Una matriz 2D numpy con valores 0, 1, 2, 3.

    Returns:
        Una imagen RGB con los colores mapeados.
    """
    # Crear una imagen RGB inicial
    height, width = mask.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Mapear los valores de la máscara a colores
    if mask.max() == 3:
        color_map = {
            ind[0]: [0, 0, 0],       # Negro
            ind[1]: [255, 0, 0],     # Rojo
            ind[2]: [0, 255, 0],   # Verde
            ind[3]: [0, 0, 255]      # Azul
        }
    else:
        color_map = {
            0: [255, 0, 0],     # Rojo
            1: [0, 255, 0],     # Verde
            2: [0, 0, 255]      # Azul
        }

    for value, color in color_map.items():
        color_image[mask == value] = color
    return color_image


def cluster_phasor_plot_4_clusters(X, pred_y, cluster_type=1, colors=["k", "r", "lime", "b"]):
    from phasorpy.plot import PhasorPlot
    from matplotlib import pyplot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax = pyplot.subplot(1, 1, 1)
    if cluster_type == 1:
        plot = PhasorPlot(ax=ax, allquadrants=True, title="Phasor plot")

    p0 = np.where(pred_y == 0)
    p1 = np.where(pred_y == 1)
    p2 = np.where(pred_y == 2)
    p3 = np.where(pred_y == 3)
    ax.scatter(X[p0[0], 0], X[p0[0], 1], c=colors[0])
    ax.scatter(X[p1[0], 0], X[p1[0], 1], c=colors[1])
    ax.scatter(X[p2[0], 0], X[p2[0], 1], c=colors[2])
    ax.scatter(X[p3[0], 0], X[p3[0], 1], c=colors[3])


#################################################
########## Segmentation functions ###############
#################################################


from skimage.filters import threshold_multiotsu
from skimage.color import rgb2gray

def apply_multiotsu_segmentation(image, classes=4):
    """
    Applies Multi-Otsu segmentation to an image.
    Args:
        image (numpy.ndarray): Input image, can be in color or grayscale.
        classes (int): Number of classes for Multi-Otsu thresholding. Default is 4.

    Returns:
        numpy.ndarray: Segmented mask based on Multi-Otsu.
    """
    gray_image = rgb2gray(image) if image.ndim == 3 else image
    gray_image = (gray_image * 255).astype(np.uint8)
    thresholds = threshold_multiotsu(gray_image, classes=classes)
    mask = np.digitize(gray_image, bins=thresholds)
    return mask


from sklearn.cluster import KMeans

def apply_kmeans_segmentation(image, n_clusters=4, random_state=42):
    """
    Applies K-Means clustering segmentation to an image.

    Args:
        image (numpy.ndarray): Input image, must be in RGB or grayscale.
        n_clusters (int): Number of clusters for K-Means. Default is 4.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        numpy.ndarray: Segmented image with cluster labels.
    """
    normalized_image = image / 255.0
    pixels = normalized_image.reshape(-1, normalized_image.shape[-1]) if normalized_image.ndim == 3 else normalized_image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(pixels)
    segmented_image = labels.reshape(image.shape[:2])
    return segmented_image


def segmentation_from_phasor_cluster(real, imag, n_clusters=4, random_state=42):
    """
    Segments data from phasor clusters using K-Means.

    Args:
        real (numpy.ndarray): Real component of the phasor data.
        imag (numpy.ndarray): Imaginary component of the phasor data.
        n_clusters (int): Number of clusters for K-Means. Default is 4.
        random_state (int): Random state for reproducibility. Default is 42.

    Returns:
        numpy.ndarray: Segmentation mask with cluster labels.
    """
    # Combine real and imaginary components into a 2D array (Nx2)
    x = np.asarray([real.flatten(), imag.flatten()]).transpose()
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, 
                    random_state=random_state, n_init="auto").fit(x)
    # Predict cluster labels for each data point
    labels = kmeans.predict(x)
    # Reshape the labels to match the original shape of the input
    mask = labels.reshape(real.shape)
    return mask


def binarize_images(input_folder, output_folder, background_value):
    """
    Reads all .tif images from a folder, binarizes them, and saves the processed images 
    in a specified output folder.
    
    Parameters:
    - input_folder (str): Path to the folder containing .tif images.
    - output_folder (str): Path to the folder where binarized images will be saved.
    - background_value (int or float): The value considered as background.
    
    Output:
    - None. The processed images are saved in the output folder.
    """
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print("The input folder does not exist.")
        return
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder created: {output_folder}")
    
    # Get all .tif files in the input folder
    tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    if not tif_files:
        print("No .tif images found in the input folder.")
        return
    
    for file in tif_files:
        input_file_path = os.path.join(input_folder, file)
        
        # Read the image
        image = tiff.imread(input_file_path)
        
        # Binarize the image: 0 where the pixel equals background_value, 255 otherwise
        binary_image = np.where(image == background_value, 0, 255).astype(np.uint8)
        
        # Save the binarized image in the output folder
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_binary.tif")
        tiff.imwrite(output_file_path, binary_image)
        print(f"Binarized image saved: {output_file_path}")


def calculate_mask_areas_with_stats(main_folder):
    """
    Process subfolders, calculate mask areas, and add mean and std for each individual (every 10 masks)
    and each subfolder.

    Parameters:
    main_folder (str): Path to the main folder containing subfolders with masks.

    Returns:
    pandas.DataFrame: DataFrame with percentage area and calculated stats.
    """
    results_dict = {}

    # Iterate through each subfolder in the main folder
    for subfolder in natsorted(os.listdir(main_folder)):
        subfolder_path = os.path.join(main_folder, subfolder)

        if os.path.isdir(subfolder_path):  # Check if it is a folder
            # Get a sorted list of TIFF files in the subfolder
            files = natsorted(
                f for f in os.listdir(subfolder_path)
                if f.endswith(".tif") or f.endswith(".tiff")
            )

            mask_areas = []

            # Process each file in the subfolder
            for file in files:
                file_path = os.path.join(subfolder_path, file)
                mask = tiff.imread(file_path)

                # Validate mask contains only 0 and 255
                if np.any((mask != 0) & (mask != 255)):
                    print(f"Warning: The mask {file} in {subfolder} contains unexpected values.")
                    continue

                # Calculate percentage of tissue (255)
                total_pixels = mask.size
                tissue = np.sum(mask == 255)
                tissue_percentage = (tissue / total_pixels) * 100

                mask_areas.append(tissue_percentage)

            # Store the tissue percentages for the current subfolder
            results_dict[subfolder] = mask_areas

    # Create a DataFrame from the results dictionary
    df = pd.DataFrame.from_dict(results_dict, orient='index').transpose()

    # Calculate stats for each subfolder
    stats = {}
    for col in df.columns:
        values = df[col].dropna().to_numpy()

        # Calculate stats for each individual (every 10 masks)
        individual_means = []
        individual_stds = []
        for i in range(0, len(values), 10):
            group = values[i:i + 10]
            if len(group) > 0:
                individual_means.append(np.mean(group))
                individual_stds.append(np.std(group))

        # Create new columns of the same length as the DataFrame
        individual_mean_column = [np.nan] * len(df)
        individual_std_column = [np.nan] * len(df)

        # Assign the calculated means and stds to the appropriate positions
        for idx, mean in enumerate(individual_means):
            individual_mean_column[idx * 10] = mean
        for idx, std in enumerate(individual_stds):
            individual_std_column[idx * 10] = std

        df[f"{col}_individual_mean"] = individual_mean_column
        df[f"{col}_individual_std"] = individual_std_column

        # Calculate overall stats for the subfolder
        stats[col] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }

    # Add overall stats to the DataFrame
    for col, stat in stats.items():
        df[f"{col}_overall_mean"] = [stat["mean"]] + [None] * (df.shape[0] - 1)
        df[f"{col}_overall_std"] = [stat["std"]] + [None] * (df.shape[0] - 1)

    return df


def summarize_methods(csv_path):
    """
    Lee un archivo CSV con estadísticas de métodos y familias,
    calcula los valores promedio generales y desviaciones estándar globales
    para cada método.

    Parámetros:
    - csv_path (str): Ruta al archivo CSV generado previamente.

    Retorno:
    - DataFrame con valores promedio y desviación estándar global por método.
    """
    # Leer el archivo CSV
    data = pd.read_csv(csv_path)

    # Identificar las columnas de medias y desviaciones estándar
    mean_columns = [col for col in data.columns if "mean" in col and "overall" not in col]
    std_columns = [col for col in data.columns if "std" in col and "overall" not in col]

    # Inicializar diccionario para los resultados
    summary = {}

    for mean_col, std_col in zip(mean_columns, std_columns):
        method = mean_col.split("_")[0]  # Extraer el nombre del método
        # Combinar medias y desviaciones estándar para el método
        means = data[mean_col].dropna().to_numpy()
        stds = data[std_col].dropna().to_numpy()

        # Calcular promedios generales y std global
        summary[method] = {
            "Overall Mean": np.mean(means),
            "Overall Std": np.mean(stds)
        }

    # Convertir a DataFrame
    summary_df = pd.DataFrame(summary).T.reset_index()
    summary_df.rename(columns={"index": "Method"}, inplace=True)
    return summary_df
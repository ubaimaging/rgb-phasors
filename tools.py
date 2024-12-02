import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import colorsys
from skimage.filters import median
from matplotlib.widgets import Cursor
import math



def phasor(image_stack, harmonic=1, axis=0):
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
    fig, ax = plt.subplots(figsize=(8, 8))
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


def cluster_phasor_plot(X, pred_y, nclusters=3, title= None):
    from phasorpy.plot import PhasorPlot
    from matplotlib import pyplot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = pyplot.subplot(1, 1, 1)
    plot = PhasorPlot(ax=ax, allquadrants=True, title='Phasor plot: ' + title)

    if nclusters == 3:
        p0 = np.where(pred_y == 0)
        p1 = np.where(pred_y == 1)
        p2 = np.where(pred_y == 2)
        ax.scatter(X[p0[0], 0], X[p0[0], 1], c='g')
        ax.scatter(X[p1[0], 0], X[p1[0], 1], c='b')
        ax.scatter(X[p2[0], 0], X[p2[0], 1], c='r')

    if nclusters == 4:
        p0 = np.where(pred_y == 0)
        p1 = np.where(pred_y == 1)
        p2 = np.where(pred_y == 2)
        p3 = np.where(pred_y == 3)
        ax.scatter(X[p0[0], 0], X[p0[0], 1], c='b')
        ax.scatter(X[p1[0], 0], X[p1[0], 1], c='k')
        ax.scatter(X[p2[0], 0], X[p2[0], 1], c='g')
        ax.scatter(X[p3[0], 0], X[p3[0], 1], c='r')
    
    if nclusters == 6:
        p0 = np.where(pred_y == 0)
        p1 = np.where(pred_y == 1)
        p2 = np.where(pred_y == 2)
        p3 = np.where(pred_y == 3)
        p4 = np.where(pred_y == 4)
        p5 = np.where(pred_y == 5)
        ax.scatter(X[p0[0], 0], X[p0[0], 1], c='cyan')
        ax.scatter(X[p1[0], 0], X[p1[0], 1], c='g')
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
    rgb_array[array == 1] = [0, 255, 0]    # Rojo para 1
    rgb_array[array == 2] = [0, 0, 255]    # Verde para 2
    rgb_array[array == 3] = [255, 0, 0]    # Azul para 3
    
    return rgb_array
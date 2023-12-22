import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import colorsys
from skimage.filters import median
from matplotlib.widgets import Cursor



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
    dc = np.where(
        dc != 0, dc, np.NaN
    )  # Set NaN where there is a 0 division

    g = data[harmonic].real
    g /= dc
    s = data[harmonic].imag
    s /= -dc
    avg = np.mean(image_stack, axis=0)
    return avg, g, s


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
    return fig


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

# -------------------------
#   INTERACTIVE FUNCTIONS 
# -------------------------

def interactive(dc, g, s, Ro, nbit):
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
    nbit = 2**nbit
    fig, ax = plt.subplots(2, 2, figsize=(20, 12))

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
    img_new = np.zeros(dc.shape)  # todo si uso esto escrive sobre una img de fondo negro

    indices1 = np.where(M1 < 0)
    indices2 = np.where(M2 < 0)
    indices3 = np.where(M3 < 0)

    cmap = plt.cm.gray
    norm = plt.Normalize(img_new.min(), img_new.max())
    rgba = cmap(norm(img_new))
    # Set the colors
    rgba[indices1[0], indices1[1], :3] = 1, 0, 0  # blue
    rgba[indices2[0], indices2[1], :3] = 0, 1, 0  # green
    rgba[indices3[0], indices3[1], :3] = 0, 0, 1  # red

    return rgba
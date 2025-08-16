# Part 4 of the paper 
# Fluorescence microscopy phasor analysis with cursors

import numpy as np
import matplotlib.pyplot as plt

from phasorpy.plot import PhasorPlot, plot_image
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal, phasor_threshold
from tools import add_scale_bar

from phasorpy.cursors import (
    mask_from_elliptic_cursor,
    pseudo_color,
)

from tools4 import (
    build_thresholded_rgb,
    apply_plot_style, 
    rgb2bgr,
    phasor_center_of_mass,
    remap_rgb_channels,
    increase_brightness,
    )


# Apply the plotting style
apply_plot_style()

# import data from local file
path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/"
path2 = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/figures/sample3/"
# --- ..................................................... ---
# --- Part 1 Apply the phasor analysis with pure components ---
# --- ..................................................... ---

# Load the component images
blue = plt.imread(path + "component_dapi.tif")
green = plt.imread(path + "component_tubulin_488.tif")
red = plt.imread(path + "component_laminin_555.tif")

# Channel thresholding
# Thresholds are set to remove noise and retain significant features
changes = True
if changes:
    blue = build_thresholded_rgb(
        blue, thresholds=[0, 255, 0, 255, 0, 255], plotty=False)
    green = build_thresholded_rgb(
        green, thresholds=[0, 255, 0, 255, 0, 255], plotty=False)
    red = build_thresholded_rgb(
        red, thresholds=[60, 255, 0, 255, 0, 255], plotty=False)

# Calculate phasor for each component
avgb, realb, imagb = phasor_from_signal(
    rgb2bgr(blue)[0:, 850:1250, 750:1250], axis=0)
avgg, realg, imagg = phasor_from_signal(
    rgb2bgr(green)[0:, 200:1400, 600:1600], axis=0)
avgr, realr, imagr = phasor_from_signal(
    rgb2bgr(red)[0:, 500:1200, 1100:1600], axis=0)

# Threshold the phasor components
avgb, realb, imagb = phasor_threshold(avgb, realb, imagb, mean_min=20)
avgg, realg, imagg = phasor_threshold(avgg, realg, imagg, mean_min=5)
avgr, realr, imagr = phasor_threshold(avgr, realr, imagr, mean_min=25)

# Calculate the center of mass for each component
realb_cm, imagb_cm  = phasor_center_of_mass(realb, imagb, avgb)
realg_cm, imagg_cm  = phasor_center_of_mass(realg, imagg, avgg)
realr_cm, imagr_cm  = phasor_center_of_mass(realr, imagr, avgr)

path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/"
image = plt.imread(path + "sample3.tif")
threshold = [0, 100, 1, 80, 1, 95]
rang = np.array([[0.3, 0.8], [0.05, 1], [0.2, 0.5]])
threshold_bkg = 5


# Build the thresholded RGB image
rgb_composed = build_thresholded_rgb(image, 
                    thresholds=threshold, plotty=False)

img = rgb2bgr(rgb_composed) # Convert RGB to BGR
# img = rgb2bgr(image) # Convert RGB to BGR
avg, real, imag = phasor_from_signal(img, axis=0)
avg, real, imag = phasor_threshold(avg, real, imag, 
                                   mean_min=threshold_bkg)

cursors_real = [0.45, -0.14, -0.2]
cursors_imag = [0.20, 0.65, -0.3]

radius = [0.7, 0.1, 0.7] 
radius_minor = [0.35, 0.45, 0.35]

angles = angle=[np.pi/3, np.pi / 3, np.pi/3]

elliptic_mask = mask_from_elliptic_cursor(
    real,
    imag,
    cursors_real,
    cursors_imag,
    radius=radius,
    radius_minor=radius_minor,
    angle = angles,
)
color = ["b", "g", "r"]
plot = PhasorPlot(allquadrants=True, title='Phasor Plot with cursors')
plot.hist2d(real, imag, cmap='RdYlGn_r')
for i in range(3):
    plot.cursor(
        cursors_real[i],
        cursors_imag[i],
        radius=radius[i],
        radius_minor=radius_minor[i],
        color=color[i],
        linestyle='-',
        angle=angles[i],
    )

    pseudo_color_image = pseudo_color(*elliptic_mask, intensity=avg)

plt.savefig(path2 + "phasor_plot.png", 
            dpi=300, bbox_inches="tight")

new_mask = remap_rgb_channels(pseudo_color_image)

fig, ax = plt.subplots()
ax.imshow(new_mask)
ax.axis("off")
ax.set_title("Pseudocolor Image")

add_scale_bar(ax,
              length_px=625, height=10, color='white', linewidth=5,
              fontsize=12, label=None, pad=10)

plt.savefig(path2 + "pseudocolor_cursors.png", 
            dpi=300, bbox_inches="tight")

# ##########################################################################
# Parte 2 ploteo de canales separados 
# ##########################################################################
part2 = True
if part2:
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/"
    image = plt.imread(path + "sample3.tif")
    threshold = [0, 100, 1, 80, 1, 95]
    rang = np.array([[0.3, 0.8], [0.05, 1], [0.2, 0.5]])
    mean_threshold = 5

    arr = np.array(image)
    # Separar canales
    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]

    # Mostrar imagen original
    fig, ax = plt.subplots()
    ax.imshow(increase_brightness(arr))
    ax.axis("off")
    ax.set_title("Original RGB")
    add_scale_bar(ax,
                length_px=625, height=10, color='white', linewidth=5,
                fontsize=12, label=None, pad=10)

    plt.savefig(path2 + "rgb.png", 
                dpi=300, bbox_inches="tight")


    # Mostrar canal R en escala de grises con colorbar
    fig, ax = plt.subplots()
    im = plt.imshow(R, cmap="gray", vmin=0, vmax=100)
    ax.set_title("Red Channel")
    ax.axis("off")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="Intensity")
    add_scale_bar(ax,
            length_px=625, height=10, color='white', linewidth=5,
            fontsize=12, label=None, pad=10)

    plt.savefig(path2 + "red_channel.png", 
            dpi=300, bbox_inches="tight")


    # Mostrar canal G en escala de grises con colorbar
    fig, ax = plt.subplots()
    im = plt.imshow(G, cmap="gray", vmin=0, vmax=100)
    ax.set_title("Green Channel")
    ax.axis("off")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="Intensity")
    add_scale_bar(ax,
            length_px=625, height=10, color='white', linewidth=5,
            fontsize=12, label=None, pad=10)
    plt.savefig(path2 + "gren_channel.png", 
            dpi=300, bbox_inches="tight")


    # Mostrar canal B en escala de grises con colorbar
    fig, ax = plt.subplots()
    im = plt.imshow(B, cmap="gray", vmin=0, vmax=100)
    ax.set_title("Blue Channel")
    ax.axis("off")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="Intensity")
    add_scale_bar(ax,
            length_px=625, height=10, color='white', linewidth=5,
            fontsize=12, label=None, pad=10)
    plt.savefig(path2 + "blue_channel.png", 
            dpi=300, bbox_inches="tight")
    
    plt.show()
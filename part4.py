# Part 4 of the paper 
# Fluorescence microscopy phasor analysis
# This script processes fluorescence microscopy images,
# performs unmixing, and visualizes the results.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal, phasor_threshold
from phasorpy.components import phasor_component_fit

from tools4 import (
    build_thresholded_rgb,
    apply_plot_style, 
    rgb2bgr,
    phasor_center_of_mass,
    threshold_by_range,
    apply_unmixing_to_rgb,
    increase_brightness,
    photon_fraction_maps,
    plot_photon_unmixing,
    save_figure
    )


# Apply the plotting style
apply_plot_style()

# Set to True to save figures
savefig = False
formatfig = 'jpg'  # 'pdf', 'png', 'jpg'

# import data from local file
path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/"

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

if not print:
    # Print the center of mass for each component
    print(f"Blue center of mass: {realb_cm}, {imagb_cm}")
    print(f"Green center of mass: {realg_cm}, {imagg_cm}")
    print(f"Red center of mass: {realr_cm}, {imagr_cm}")

# Plot the phasor components
# plot histograms for each channel and print the mean values
# values: b=20, g=5, r=25
plotty = True
if plotty:
    # Create figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # --- Top row: Images ---
    axes[0, 0].imshow(blue, interpolation='nearest', cmap='Blues')
    axes[0, 0].set_title("Blue Channel")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(green, interpolation='nearest', cmap='Greens')
    axes[0, 1].set_title("Green Channel")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(red, interpolation='nearest', cmap='Reds')
    axes[0, 2].set_title("Red Channel")
    axes[0, 2].axis("off")

    # --- Bottom row: Histograms ---
    hist_b, bins_b = np.histogram(avgb, bins=256, range=(0, 255))
    axes[1, 0].plot(bins_b[:-1], hist_b, color='blue')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title("Histogram (Blue)")
    axes[1, 0].set_xlim(0, 255)

    hist_g, bins_g = np.histogram(avgg, bins=256, range=(0, 255))
    axes[1, 1].plot(bins_g[:-1], hist_g, color='green')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_title("Histogram (Green)")
    axes[1, 1].set_xlim(0, 255)

    hist_r, bins_r = np.histogram(avgr, bins=256, range=(0, 255))
    axes[1, 2].plot(bins_r[:-1], hist_r, color='red')
    axes[1, 2].set_yscale('log')
    axes[1, 2].set_title("Histogram (Red)")
    axes[1, 2].set_xlim(0, 255)

    plt.tight_layout()
    if savefig: save_figure(fig, path + "components_and_histograms", format=formatfig)

    plot = PhasorPlot(allquadrants=True, title='Components Phasor Plot')
    plot.hist2d(realb.flatten(), imagb.flatten(), cmap="RdYlGn_r")
    plt.plot(realb_cm, imagb_cm, color='blue', markersize=9, marker='o', 
                linestyle='None', label='Blue component')
    
    plot.hist2d(realg.flatten(), imagg.flatten(), cmap="RdYlGn_r")
    plt.plot(realg_cm, imagg_cm, color='green', markersize=9, marker='p', 
                linestyle='None', label='Green component')
    
    plot.hist2d(realr.flatten(), imagr.flatten(), cmap="RdYlGn_r")
    plt.plot(realr_cm, imagr_cm, color='red', markersize=9, marker='*', 
                linestyle='None', label='Red component')
    plt.legend()
    if savefig: save_figure(plot.fig, path + "components_phasor_plot", format=formatfig)

    # plt.show()


# --- .............................................................. ---
# --- Part 2 Apply the phasor unmixing to the RGB experimental image ---
# --- .............................................................. ---

sample1 = False  #single cell image configuration
sample2 = False  #single cell image configuration
if sample1:
    image = plt.imread(path + "sample1.tif")[150:1800, 600:2050]
    threshold = [5, 120, 5, 100, 5, 130]
    rang = np.array([[0.15, 0.95], [0, 0.8], [0.03, 0.4]])
    mean_threshold = 10 

elif sample2:
    image = plt.imread(path + "sample2.tif")[350:2000, 50:1600]
    threshold = [0, 130, 1, 80, 1, 100]
    rang = np.array([[0, 0.7], [0, 1], [0.2, 0.5]])
    mean_threshold = 2

else:
    # paper image configuration
    image = plt.imread(path + "sample3.tif")
    threshold = [0, 100, 1, 80, 1, 95]
    rang = np.array([[0.3, 0.8], [0.05, 1], [0.2, 0.5]])
    mean_threshold = 5
    

# Build the thresholded RGB image
rgb_composed = build_thresholded_rgb(image, thresholds=threshold, plotty=False)

img = rgb2bgr(rgb_composed) # Convert RGB to BGR
# img = rgb2bgr(image) # Convert RGB to BGR
avg, real, imag = phasor_from_signal(img, axis=0)
avg, real, imag = phasor_threshold(avg, real, imag, mean_min=mean_threshold)

# Components center of mass
gs = np.array([realb_cm, realg_cm, realr_cm])
ss = np.array([imagb_cm, imagg_cm, imagr_cm])

# Unmix the phasor components
frac = phasor_component_fit(avg, real, imag, gs, ss)

fracb = threshold_by_range(frac[0], rang[0][0], rang[0][1])
fracg = threshold_by_range(frac[1], rang[1][0], rang[1][1])
fracr = threshold_by_range(frac[2], rang[2][0], rang[2][1])

rgb_adjusted = apply_unmixing_to_rgb(image, fracr, fracg, fracb)
rgb_adjusted = increase_brightness(rgb_adjusted, factor=2)

R_photons, G_photons, B_photons = photon_fraction_maps(image, fracr, fracg, fracb)

plotty = True
if plotty:

    # Plot the original RGB image
    fig = plt.figure()
    plt.imshow(image, interpolation="nearest")
    plt.title("Original RGB Image")
    plt.axis("off") 
    plt.tight_layout()
    if savefig: save_figure(fig, path + "original_image", format=formatfig)

    # Plot the intensity image
    # This shows the sum of the RGB channels as a grayscale image
    fig, ax = plt.subplots()
    im = ax.imshow(np.sum(image, axis=2), cmap="gray", interpolation="nearest")
    ax.set_title("Intensity image")
    ax.axis("off")
    # Add a colorbar to the intensity image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Photon count")
    plt.tight_layout()
    if savefig: save_figure(fig, path + "intensity_image", format=formatfig)

    # Figure Phasor Plot
    plot = PhasorPlot(allquadrants=True, title='Phasor plot of the image')
    plot.hist2d(real.flatten(), imag.flatten(), cmap="RdYlGn_r")
    plt.plot(realb_cm, imagb_cm, color='blue', markersize=10, marker='o', 
                linestyle='None', label='Blue channel')
    plt.plot(realg_cm, imagg_cm, color='green', markersize=10, marker='p', 
                linestyle='None', label='Green channel')
    plt.plot(realr_cm, imagr_cm, color='red', markersize=10, marker='*', 
                linestyle='None', label='Red channel')
    plt.legend()
    plt.tight_layout()
    if savefig: save_figure(plot.fig, path + "phasor_plot", format=formatfig)


    # Plot Histograms
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Subplot 1: Complete Histograms
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

    # Subplot 2: Thresholded Histograms
    axs[1].hist(fracb.flatten()[fracb.flatten() != 0],
                bins=256, color='blue', alpha=0.5, label='Blue channel', log=True)
    axs[1].hist(fracg.flatten()[fracg.flatten() != 0],
                bins=256, color='green', alpha=0.5, label='Green channel', log=True)
    axs[1].hist(fracr.flatten()[fracr.flatten() != 0],
                bins=256, color='red', alpha=0.5, label='Red channel', log=True)
    axs[1].set_title("Thresholded fractions histogram")
    axs[1].set_xlabel("Fraction value")
    axs[1].set_ylabel("Frequency")
    axs[1].set_xlim(0, 1.2)
    axs[1].legend()
    plt.tight_layout()
    if savefig: save_figure(fig, path + "histograms", format=formatfig)


    # Plot the photon unmixing results
    fig = plot_photon_unmixing(fracr, fracg, fracb, R_photons, G_photons, B_photons)
    if savefig: save_figure(fig, path + "photon_unmixing", format=formatfig)


    # Plot the unmixed RGB image
    fig = plt.figure()
    plt.imshow(rgb_adjusted, interpolation="nearest")
    plt.title("Unmixed RGB Image")
    plt.axis("off")
    plt.tight_layout()
    if savefig: save_figure(fig, path + "unmixed_image", format=formatfig)

    plt.show()
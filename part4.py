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
    save_figure,
    draw_dashed_line_on_figures
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
plotty = False
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

sample1 = False  # single cell image configuration
sample2 = False  # single cell image configuration
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

rgb_unmixed = apply_unmixing_to_rgb(image, fracr, fracg, fracb)
rgb_adjusted = increase_brightness(rgb_unmixed, factor=2) # For plot version

R_photons, G_photons, B_photons = photon_fraction_maps(image, fracr, fracg, fracb)

plotty = True
if plotty:

    # Plot the original RGB image
    fig1 = plt.figure()
    plt.imshow(image, interpolation="nearest")
    plt.title("Original RGB Image")
    plt.axis("off") 
    plt.tight_layout()
    if savefig: save_figure(fig1, path + "original_image", format=formatfig)

    # Plot the intensity image
    # This shows the sum of the RGB channels as a grayscale image
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(np.sum(image, axis=2), cmap="gray", interpolation="nearest")
    ax2.set_title("Intensity image")
    ax2.axis("off")
    # Add a colorbar to the intensity image
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Photon count")
    plt.tight_layout()
    if savefig: save_figure(fig2, path + "intensity_image", format=formatfig)

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
    fig3 = plot.fig
    plt.tight_layout()
    if savefig: save_figure(fig3, path + "phasor_plot", format=formatfig)

    # Plot Histograms
    # Plot Complete Histograms
    fig4, ax4 = plt.subplots()
    ax4.hist(np.sum(image, axis=2).flatten(), bins=256, color='gray', alpha=0.8, log=True)
    ax4.set_title("Intensity Histogram")
    ax4.set_xlabel("Intensity")
    ax4.set_ylabel("Frequency")
    plt.tight_layout()
    if savefig:
        save_figure(fig4, path + "histogram_intensity", format=formatfig)

    # Plot Thresholded Fractions Histogram
    fig5, ax5 = plt.subplots()
    ax5.hist(fracb.flatten()[fracb.flatten() != 0],
            bins=256, color='blue', alpha=0.5, label='Blue channel', log=True)
    ax5.hist(fracg.flatten()[fracg.flatten() != 0],
            bins=256, color='green', alpha=0.5, label='Green channel', log=True)
    ax5.hist(fracr.flatten()[fracr.flatten() != 0],
            bins=256, color='red', alpha=0.5, label='Red channel', log=True)
    ax5.set_title("Thresholded fractions histogram")
    ax5.set_xlabel("Fraction value")
    ax5.set_ylabel("Frequency")
    ax5.set_xlim(0, 1.2)
    ax5.legend()
    plt.tight_layout()
    if savefig: save_figure(fig5, path + "histograms", format=formatfig)

    # Plot the photon unmixing results
    fig6 = plot_photon_unmixing(fracr, fracg, fracb, R_photons, G_photons, B_photons)
    if savefig: save_figure(fig6, path + "photon_unmixing", format=formatfig)

    # Plot the unmixed RGB image
    fig7 = plt.figure()
    plt.imshow(rgb_adjusted, interpolation="nearest")
    plt.title("Unmixed RGB Image")
    plt.axis("off")
    plt.tight_layout()
    if savefig: save_figure(fig7, path + "unmixed_image", format=formatfig)

    plt.show()
    plt.close('all')


# --- .......................... ---
# --- Part 3 Create Paper Figure ---
# --- .......................... ---

create_paper_figure = False
if create_paper_figure:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from PIL import Image

    # Figure 1 Original RGB Image
    fig1 = plt.figure()
    plt.imshow(image, interpolation="nearest")
    plt.title("Original RGB Image")
    plt.axis("off") 
    plt.tight_layout()

    # Draw a dashed line on the figure
    # Define the start and end points of the dashed line
    # These points are in pixel coordinates from the interactive function below
    start_point = (940, 1337)
    end_point = (1672, 1634)
    points = (start_point, end_point)
    draw_dashed_line_on_figures(fig1, points, color='white', linestyle='--', linewidth=1.5)

    # Figure 2 Phasor Plot
    plot = PhasorPlot(allquadrants=True, title='Phasor plot of the image')
    plot.hist2d(real.flatten(), imag.flatten(), cmap="RdYlGn_r")
    plt.plot(realb_cm, imagb_cm, color='blue', markersize=10, marker='o', 
                linestyle='None', label='Blue channel')
    plt.plot(realg_cm, imagg_cm, color='green', markersize=10, marker='p', 
                linestyle='None', label='Green channel')
    plt.plot(realr_cm, imagr_cm, color='red', markersize=10, marker='*', 
                linestyle='None', label='Red channel')
    plt.legend()
    fig2 = plot.fig
    plt.tight_layout()

    # Figure 6 Histograms
    fig6, ax6 = plt.subplots()
    ax6.hist(fracb.flatten()[fracb.flatten() != 0],
            bins=256, color='blue', alpha=0.5, label='Blue channel', log=True)
    ax6.hist(fracg.flatten()[fracg.flatten() != 0],
            bins=256, color='green', alpha=0.5, label='Green channel', log=True)
    ax6.hist(fracr.flatten()[fracr.flatten() != 0],
            bins=256, color='red', alpha=0.5, label='Red channel', log=True)
    ax6.set_title("Thresholded fractions histogram")
    ax6.set_xlabel("Fraction value")
    ax6.set_ylabel("Frequency")
    ax6.set_xlim(0, 1.2)
    ax6.legend()
    plt.tight_layout()

    # Figure 7 Unmixed RGB Image
    fig7 = plt.figure()
    plt.imshow(rgb_adjusted, interpolation="nearest")
    plt.title("Unmixed RGB Image")
    plt.axis("off")
    plt.tight_layout()

    draw_dashed_line_on_figures(fig7, points, color='white', linestyle='--', linewidth=1.5)


    # Figure 3, 4, 5, 8, 9 and 10 Photon Estimates

    # --- Reemplazar NaNs por 0 ---
    R_frac = np.nan_to_num(fracr, nan=0.0)
    G_frac = np.nan_to_num(fracg, nan=0.0)
    B_frac = np.nan_to_num(fracb, nan=0.0)
    R_photons = np.nan_to_num(R_photons, nan=0.0)
    G_photons = np.nan_to_num(G_photons, nan=0.0)
    B_photons = np.nan_to_num(B_photons, nan=0.0)

    # --- Normalización ---
    vmax_frac = 1.0
    vmax_prod = np.max([R_photons, G_photons, B_photons])

    def plot_image_only(img, title, vmin, vmax, cmap='inferno'):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        return fig

    # --- Crear figuras de fracciones ---
    fig3 = plot_image_only(R_frac, "Fractions red", vmin=0, vmax=vmax_frac)
    fig4 = plot_image_only(G_frac, "Fractions green", vmin=0, vmax=vmax_frac)
    fig5 = plot_image_only(B_frac, "Fractions blue", vmin=0, vmax=vmax_frac)


    # --- Crear figuras de fotones ---
    fig8 = plot_image_only(R_photons, "Photons red", vmin=0, vmax=vmax_prod)
    fig9 = plot_image_only(G_photons, "Photons green", vmin=0, vmax=vmax_prod)
    fig10 = plot_image_only(B_photons, "Photons blue", vmin=0, vmax=vmax_prod)

    # --- Convertir figura a imagen ---
    def fig_to_img_array(fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())
        plt.close(fig)
        return img

    # --- Redimensionar manteniendo aspecto ---
    def resize_preserve_aspect(img, target_width):
        h, w = img.shape[:2]
        scale = target_width / w
        new_height = int(h * scale)
        img_pil = Image.fromarray(img)
        resized = img_pil.resize((target_width, new_height), Image.Resampling.LANCZOS)
        return np.array(resized)

    # --- Padding para igualar altura ---
    def pad_to_height(img, target_height):
        h, w, c = img.shape
        if h == target_height:
            return img
        pad_top = (target_height - h) // 2
        pad_bottom = target_height - h - pad_top
        padded = np.pad(img, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant')
        return padded

    # --- Convertir todas las figuras a arrays ---
    figs = [fig1, fig2, fig3, fig4, fig5,
            fig6, fig7, fig8, fig9, fig10]
    images = [fig_to_img_array(f) for f in figs]

    # --- Redimensionar todas al mismo ancho manteniendo aspecto ---
    target_width = 600
    images_resized = [resize_preserve_aspect(im, target_width) for im in images]

    # --- Obtener la altura máxima ---
    target_height = max(im.shape[0] for im in images_resized)

    # --- Aplicar padding para unificar altura ---
    images_padded = [pad_to_height(im, target_height) for im in images_resized]

    # --- Crear figura final con 2 filas y 6 columnas ---
    fig_final, axs = plt.subplots(2, 5, figsize=(24, 9))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images_padded[i])
        ax.axis("off")
        ax.text(-0.05, 1.05, chr(65 + i), transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')

    plt.tight_layout()

    # --- Guardar figura final ---
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/figures/sample3/"
    fig_final.savefig(path + "figure_part4.pdf", dpi=600)
    fig_final.savefig(path + "figure_part4.jpg", dpi=600)

    plt.close(fig_final)


# --- ....................... ---
# --- Line intensity Analysis ---
# --- ....................... ---

line_analysis = False  # Set to True to run the interactive line analysis
if line_analysis:
    from skimage.measure import profile_line
    from skimage.draw import line

    def interactive_rgb_line_analysis(original_img, unmixed_img, 
                                      savefig=False, formatfig='pdf', path=""):
        """
        Allows the user to click two points on the original RGB image.
        Then:
        1. Draws a white dashed line over both the original and unmixed image.
        2. Extracts RGB intensity profiles along that line.
        3. Plots original vs unmixed profiles (one channel per subplot, vertically stacked).
        4. Plots all three unmixed channel profiles together with colored lines.
        
        Parameters:
        - original_img: np.ndarray (H, W, 3), uint8 RGB image.
        - unmixed_img: np.ndarray (H, W, 3), float (0–1) RGB image with possible NaNs.
        """
        # Step 1: User selects two points
        plt.imshow(original_img)
        plt.title("Click two points to define the line")
        points = plt.ginput(2)
        plt.close()

        if len(points) < 2:
            raise ValueError("Two points are required.")
        
        start = tuple(map(int, points[0]))
        end = tuple(map(int, points[1]))
        print("star point:", start, "end point", end)

        # Step 2: Preprocess unmixed image (replace NaNs, scale to 0–255)
        unmixed_clean = np.nan_to_num(unmixed_img, nan=0.0)
        unmixed_scaled = np.clip(unmixed_clean * 255, 0, 255).astype(np.uint8)

        # Step 3: Show both images with the line
        fig_img, axes_img = plt.subplots(1, 2, figsize=(12, 5))
        titles = ["Original RGB Image", "Unmixed RGB Image"]
        images = [original_img, unmixed_scaled]

        for i in range(2):
            axes_img[i].imshow(images[i])
            axes_img[i].set_title(titles[i])
            axes_img[i].axis("off")
            axes_img[i].plot([start[0], end[0]], [start[1], end[1]],
                            color='white', linestyle='--', linewidth=1.5)

        plt.tight_layout()

        # Step 4: Extract intensity profiles
        profiles_orig = [profile_line(original_img[:, :, c], start, end) for c in range(3)]
        profiles_unmix = [profile_line(unmixed_scaled[:, :, c], start, end) for c in range(3)]

        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']

        # Step 5: Plot original and unmixed per channel (stacked vertically)
        fig1, axes = plt.subplots(3, 1, figsize=(8, 10))  # 3 rows, 1 column
        for i in range(3):
            axes[i].plot(profiles_orig[i], color=colors[i], label=f"{labels[i]}")
            axes[i].plot(profiles_unmix[i], color='gray', label="Unmixed")
            axes[i].set_title(f"{labels[i]} Channel")
            axes[i].set_xlabel("Section")
            axes[i].set_ylabel("Intensity")
            axes[i].legend()

        fig1.suptitle("Channel Intensity Profiles (Original + Unmixed)", fontsize=14)
        plt.tight_layout()

        # Step 6: Plot all unmixed profiles together (colored)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for i in range(3):
            ax2.plot(profiles_unmix[i], color=colors[i], label=f"{labels[i]}")
        ax2.set_title("Intensity Profiles of Unmixed Channels")
        ax2.set_xlabel("Section")
        ax2.set_ylabel("Intensity")
        ax2.legend()
        plt.tight_layout()
        plt.show()

        if savefig:
            save_figure(fig_img, path + "line_analysis_images", format=formatfig)
            save_figure(fig1, path + "line_analysis_profiles", format=formatfig)
            save_figure(fig2, path + "line_analysis_combined_profiles", format=formatfig)

    path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/figures/sample3/"
    # Call the interactive analysis function
    interactive_rgb_line_analysis(image, rgb_unmixed, savefig=True, formatfig="jpg", path=path)
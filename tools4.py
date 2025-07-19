import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu


def apply_plot_style():
    """    Applies a consistent plotting style for the figures.
    This function sets the font sizes, family, 
    and other aesthetic parameters
    to ensure a uniform appearance across all plots.
    It is called at the beginning of the script to set the style 
    before any plots are created
    """
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


def save_figure(fig, name, format='pdf'):
    fig.savefig(f"{name}.{format}", bbox_inches='tight')
    plt.close(fig)


def build_thresholded_rgb(rgb, thresholds=None, plotty=False):
    """
    Applies per-channel thresholding (manual or Otsu), optionally plots diagnostics,
    and returns a new RGB image with only valid pixels retained.

    Parameters:
        rgb: np.ndarray - Input RGB image of shape (H, W, 3)
        thresholds: list or array [min_R, max_R, min_G, max_G, min_B, max_B]
                    If None, uses Otsu for min_* and 255 for max_*
        plotty: bool - If True, displays figures. If False, runs silently

    Returns:
        np.ndarray - RGB image with thresholded channels
    """
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray_avg = np.mean(rgb, axis=-1).astype(np.uint8)

    # Parse thresholds
    if thresholds is not None:
        min_R, max_R, min_G, max_G, min_B, max_B = thresholds
    else:
        min_R = min_G = min_B = 0
        max_R = max_G = max_B = 255

    # Otsu if no manual min threshold
    thresh_R = min_R if min_R > 0 else threshold_otsu(R)
    thresh_G = min_G if min_G > 0 else threshold_otsu(G)
    thresh_B = min_B if min_B > 0 else threshold_otsu(B)

    # Thresholding with NaN outside range
    R_thr = np.where((R > thresh_R) & (R <= max_R), R, np.nan)
    G_thr = np.where((G > thresh_G) & (G <= max_G), G, np.nan)
    B_thr = np.where((B > thresh_B) & (B <= max_B), B, np.nan)

    if plotty:
        # Figure 1: RGB, grayscale, grayscale histogram
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 0.6]})
        axes[0, 0].imshow(rgb, interpolation='nearest')
        axes[0, 0].set_title("Original RGB")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(gray_avg, cmap='gray', interpolation='nearest')
        axes[0, 1].set_title("Grayscale Average")
        axes[0, 1].axis("off")

        hist_gray, bins_gray = np.histogram(gray_avg.ravel(), bins=np.arange(257))
        ax_hist = plt.subplot2grid((2, 2), (1, 0), colspan=2, fig=fig)
        ax_hist.plot(bins_gray[1:-1], np.log1p(hist_gray[1:]), color='gray')
        ax_hist.set_title("Grayscale Histogram (log scale)")
        ax_hist.set_xlim(0, 255)
        ax_hist.set_xlabel("Intensity")
        ax_hist.set_ylabel("log(Count + 1)")
        plt.tight_layout()

        # Figure 2: Original RGB histograms
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        for ax, data, color, label in zip(
            axes2, [R, G, B], ['red', 'green', 'blue'], ['R', 'G', 'B']
        ):
            hist, bins = np.histogram(data, bins=np.arange(257))
            ax.plot(bins[1:-1], np.log1p(hist[1:]), color=color)
            ax.set_title(f"{label} Histogram (log)")
            ax.set_xlim(0, 255)
            ax.set_xlabel("Intensity")
            ax.set_ylabel("log(Count + 1)")
        plt.tight_layout()

        # Figure 3: Thresholded images + histograms
        fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
        for ax, data, cmap, title in zip(
            axes3[0],
            [R_thr, G_thr, B_thr],
            ['Reds', 'Greens', 'Blues'],
            [f"Thresholded R ({thresh_R}–{max_R})",
             f"Thresholded G ({thresh_G}–{max_G})",
             f"Thresholded B ({thresh_B}–{max_B})"]
        ):
            ax.imshow(data, cmap=cmap, interpolation='nearest')
            ax.set_title(title)
            ax.axis("off")

        for ax, data, color, label in zip(
            axes3[1], [R_thr, G_thr, B_thr],
            ['red', 'green', 'blue'], ['R', 'G', 'B']
        ):
            data_clean = np.nan_to_num(data, nan=0)
            hist, bins = np.histogram(data_clean, bins=np.arange(257))
            ax.plot(bins[1:-1], np.log1p(hist[1:]), color=color)
            ax.set_title(f"{label} Thresholded Histogram (log)")
            ax.set_xlim(0, 255)
        plt.tight_layout()

    # Compose new RGB image
    R_clean = np.nan_to_num(R_thr, nan=0).astype(np.uint8)
    G_clean = np.nan_to_num(G_thr, nan=0).astype(np.uint8)
    B_clean = np.nan_to_num(B_thr, nan=0).astype(np.uint8)
    rgb_composed = np.stack([R_clean, G_clean, B_clean], axis=-1)

    if plotty:
        # Final figure: original vs thresholded
        fig4, axes4 = plt.subplots(1, 2, figsize=(12, 6))
        axes4[0].imshow(rgb, interpolation='nearest')
        axes4[0].set_title("Original RGB")
        axes4[0].axis("off")

        axes4[1].imshow(rgb_composed, interpolation='nearest')
        axes4[1].set_title("Thresholded RGB")
        axes4[1].axis("off")
        plt.tight_layout()
        plt.show()

    return rgb_composed


def rgb2bgr(im):
    """
    Converts an RGB image to BGR by reordering the channels.
    
    Parameters
    ----------
    im : np.ndarray
        RGB image of shape (H, W, 3)
    
    Returns
    -------
    np.ndarray
        BGR image of shape (H, W, 3)
    """
    return im[:, :, ::-1].transpose(2, 0, 1)


def phasor_center_of_mass(g, s, avg):
    """
    Computes the center of mass of the phasor plot using G (real), 
    S (imaginary), and an intensity/average map.

    Parameters:
    - g: np.ndarray (N, M), G coordinates (real part)
    - s: np.ndarray (N, M), S coordinates (imaginary part)
    - avg: np.ndarray (N, M), average intensity (used as weight)

    Returns:
    - (g_com, s_com): tuple with the coordinates of the center of mass
    """
    total_weight = np.nansum(avg)
    g_com = np.nansum(g * avg) / total_weight
    s_com = np.nansum(s * avg) / total_weight
    return g_com, s_com


def threshold_by_range(channel, min_val, max_val):
    """
    Mask values outside the valid range with np.nan.

    Parameters:
    - channel: np.ndarray, input image channel
    - min_val: float or int, minimum threshold value
    - max_val: float or int, maximum threshold value

    Returns:
    - np.ndarray: float32 array with values outside the range replaced by NaN
    """
    mask = (channel >= min_val) & (channel <= max_val)
    output = np.where(mask, channel, np.nan)
    return output.astype(np.float32)


def apply_unmixing_to_rgb(rgb_image, R_unmix, G_unmix, B_unmix):
    """
    Apply unmixing weights to the original RGB image to modulate each channel.

    Args:
        rgb_image (np.ndarray): Original RGB image, shape (H, W, 3), uint8 or float.
        R_unmix, G_unmix, B_unmix (np.ndarray): Abundance maps (H, W) with values in [0, 1].

    Returns:
        np.ndarray: RGB image (H, W, 3), float32, values in [0, 1].
    """
    # Ensure float32 image for multiplication
    rgb_float = rgb_image.astype(np.float32)
    if rgb_float.max() > 1.0:
        rgb_float /= 255.0  # Normalize if needed

    # Separate channels
    R_orig = rgb_float[..., 0]
    G_orig = rgb_float[..., 1]
    B_orig = rgb_float[..., 2]

    # Multiply each channel by its unmixing abundance
    R_new = R_orig * R_unmix
    G_new = G_orig * G_unmix
    B_new = B_orig * B_unmix

    # Stack back into RGB image
    rgb_unmixed = np.stack([R_new, G_new, B_new], 
                           axis=-1).astype(np.float32)
    
    return rgb_unmixed


def increase_brightness(rgb_image, factor=1.5):
    """
    Increase brightness of an RGB image by multiplying pixel values.

    Args:
        rgb_image (np.ndarray): RGB image (H, W, 3), uint8 or float.
        factor (float): Brightness factor (>1 increases brightness).

    Returns:
        np.ndarray: Brightness-enhanced image, uint8.
    """
    img = rgb_image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0  # normalize to [0, 1] if needed

    img_bright = np.clip(img * factor, 0, 1)

    return (img_bright * 255).astype(np.uint8)



def photon_fraction_maps(rgb_image, R_frac, G_frac, B_frac):
    """
    Generate photon fraction-weighted images from an RGB image and fractional maps.

    Args:
        rgb_image (np.ndarray): RGB image (H, W, 3), uint8 or float32.
        R_frac, G_frac, B_frac (np.ndarray): Fraction maps (H, W) with values in [0, 1].

    Returns:
        gray_sum (np.ndarray): Grayscale sum image.
        R_photons, G_photons, B_photons (np.ndarray): Weighted photon images.
    """
    # Convert to float32 and normalize if necessary
    rgb = rgb_image.astype(np.float32)

    # Grayscale sum (simulated total photons per pixel)
    gray_sum = np.sum(rgb, axis=-1)

    # Multiply fraction maps by grayscale sum
    R_photons = R_frac * gray_sum
    G_photons = G_frac * gray_sum
    B_photons = B_frac * gray_sum

    return R_photons, G_photons, B_photons


def plot_photon_unmixing(R_frac, G_frac, B_frac, R_photons, G_photons, B_photons):

    """    Plot the photon fraction maps and their products."
    Parameters:
        R_frac, G_frac, B_frac: Fraction maps for each channel (H, W).
        R_photons, G_photons, B_photons: Photon maps for             
        each channel (H, W).
    """

    # Replace NaNs with 0
    R_frac = np.nan_to_num(R_frac, nan=0.0)
    G_frac = np.nan_to_num(G_frac, nan=0.0)
    B_frac = np.nan_to_num(B_frac, nan=0.0)
    R_photons = np.nan_to_num(R_photons, nan=0.0)
    G_photons = np.nan_to_num(G_photons, nan=0.0)
    B_photons = np.nan_to_num(B_photons, nan=0.0)

    # Normalize photon maps to [0, 1] for RGB visualization
    max_val = np.max([R_photons, G_photons, B_photons])
    # Shared colormap limits
    vmax_frac = 1.0
    vmax_prod = max_val

    # 2. Fracciones y productos
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), gridspec_kw={'wspace': 0.05})

    # --- Fracciones ---
    im_fracs = []
    for ax, img, title in zip(
        axes[0],
        [R_frac, G_frac, B_frac],
        ["Fractions red", "Fractions green", "Fractions blue"]
    ):
        im = ax.imshow(img, cmap='inferno', interpolation='nearest', vmin=0, vmax=vmax_frac)
        ax.set_title(title)
        ax.axis("off")
        im_fracs.append(im)

    fig.colorbar(
        im_fracs[0], ax=axes[0],
        orientation='vertical', fraction=0.02, pad=0.08,
        location='left', label="Fraction (0–1)"
    )

    # --- Productos ---
    im_prods = []
    for ax, img, title in zip(
        axes[1],
        [R_photons, G_photons, B_photons],
        ["Photons red", "Photons green", "Photons blue"]
    ):
        im = ax.imshow(img, cmap='inferno', interpolation='nearest', vmin=0, vmax=vmax_prod)
        ax.set_title(title)
        ax.axis("off")
        im_prods.append(im)

    fig.colorbar(
        im_prods[0], ax=axes[1],
        orientation='vertical', fraction=0.02, pad=0.08,
        location='left', label="Photon Estimate"
    )
    plt.tight_layout()
    return fig


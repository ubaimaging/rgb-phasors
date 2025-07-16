from skimage import io, img_as_float
from scipy.ndimage import gaussian_filter
import numpy as np
from skimage.restoration import richardson_lucy
from phasorpy.phasor import phasor_from_signal
import tools
import matplotlib.pyplot as plt

def calculate_weights(g, s, sigma=0.1):
    """
    Calculate weights based on the distance of (g, s) from the origin.
    """
    distance = np.sqrt(g**2 + s**2)
    weights = np.exp(-distance**2 / (2 * sigma**2))  # Gaussian weighting
    return weights


def selective_deconvolution(image, psf, weights, iterations=10):
    """
    Perform selective deconvolution based on phasor weights.
    """
    deconvolved = np.zeros_like(image)
    for c in range(3):  # Process each channel (R, G, B)
        # Apply weighted deconvolution
        deconvolved[..., c] = richardson_lucy(image[..., c] * weights, psf, iterations)
    return deconvolved

# Load the image
image = io.imread(
    "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/janelia_sample_3channes_roi5.tif")
image = img_as_float(image)  # Normalize to range [0, 1]
bgr = tools.rgb2bgr(image)

# Generate a Gaussian PSF
def generate_psf(size, sigma):
    x = np.arange(-(size // 2), (size // 2) + 1)
    y = np.arange(-(size // 2), (size // 2) + 1)
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf /= psf.sum()  # Normalize
    return psf

psf = generate_psf(size=9, sigma=2.0)

# Step 1: Calculate phasor components
_, g, s = phasor_from_signal(bgr, axis=0)

# Step 2: Calculate weights
weights = calculate_weights(g, s, sigma=0.5)

# Step 3: Perform selective deconvolution
deconvolved_image = selective_deconvolution(image, psf, weights, iterations=20)

# Save the result
io.imsave(
    "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/phasor_deconvolved_janelia_sample_3channes_roi5.tif", 
    np.clip(deconvolved_image * 255, 0, 255).astype(np.uint8))

image_conv = io.imread(
    "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/phasor_deconvolved_janelia_sample_3channes_roi5.tif")

plt.figure()
plt.imshow(image_conv)
plt.show()
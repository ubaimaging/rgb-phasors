import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.restoration import richardson_lucy
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def load_image(path):
    """Load an 8-bit RGB image and convert it to a float array."""
    image = io.imread(path)
    return img_as_float(image)  # Convert to range [0, 1]

def generate_psf(size, sigma):
    """
    Generate a simulated Gaussian Point Spread Function (PSF).
    - size: Kernel size (should be odd).
    - sigma: Standard deviation of the Gaussian PSF.
    """
    x = np.arange(-(size // 2), (size // 2) + 1)
    y = np.arange(-(size // 2), (size // 2) + 1)
    X, Y = np.meshgrid(x, y)
    psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    psf /= psf.sum()  # Normalize the PSF
    return psf

def deconvolve_rgb(image, psf, iterations=10):
    """
    Apply Richardson-Lucy deconvolution to each RGB channel.
    - image: RGB image as a float array in range [0, 1].
    - psf: Point Spread Function (PSF).
    - iterations: Number of Richardson-Lucy iterations.
    """
    deconvolved = np.zeros_like(image)
    for c in range(3):  # Process each channel (R, G, B)
        deconvolved[..., c] = richardson_lucy(image[..., c], psf, iterations)
    return deconvolved

def save_image(image, path):
    """Save the deconvolved image as an 8-bit file."""
    io.imsave(path, img_as_ubyte(image))  # Convert from [0, 1] to [0, 255]

# Example Usage
apply_dec = False
if apply_dec:
    if __name__ == "__main__":
        # Input and output file paths
        input_path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/janelia_sample_3channes_roi5.tif"
        output_path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/deconvolved_janelia_sample_3channes_roi5.tif"

        # Load the image
        image = load_image(input_path)
        # Generate a simulated PSF
        psf = generate_psf(size=7, sigma=2.0)  # Adjust 'size' and 'sigma' based on your microscope
        # Perform deconvolution
        deconvolved_image = deconvolve_rgb(image, psf, iterations=20)

        # Save the deconvolved image
        save_image(deconvolved_image, output_path)
        print(f"Deconvolved image saved at: {output_path}")

plotty = True
if plotty:
    image = load_image(
        "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/janelia_sample_3channes_roi5.tif")
    dec_image = load_image(
        "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig5/data/deconvolved_janelia_sample_3channes_roi5.tif")

    plt.figure(1)
    plt.imshow(image, interpolation="none")

    plt.figure(2)
    plt.imshow(dec_image, interpolation="none")
    plt.show()
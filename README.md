# RGB Phasor Analysis and Spectral Unmixing in Widefield Fluorescence

This module provides tools for applying **phasor-based analysis** and **spectral unmixing** to RGB images acquired from **widefield fluorescence microscopy**. The method enables decomposing fluorescence signals into meaningful components using phasor coordinates computed from RGB intensities, without requiring spectral detectors.

## 🔬 Overview

We introduce a pipeline that:

- Converts RGB fluorescence images to phasor space (G, S coordinates)
- Uses known pure components to build a phasor basis
- Fits mixed pixels via least squares or direct projection
- Generates unmixed RGB images and photon fraction maps
- Enables enhanced visualization of underlying fluorophore distributions

This approach enables **label separation**, **quantification**, and **improved interpretability** of RGB data acquired with standard widefield fluorescence systems.

## 📁 Folder Structure

```
.
├── part4.py                # Main script for RGB phasor analysis and unmixing
├── paper/
│   └── fig5/
│       └── data/           # Component and sample images (.tif)
├── tools4.py              # Helper functions: RGB conversion, thresholding, phasor computation, etc.
├── output/                # Automatically saved figures (optional)
```

## ⚙️ How It Works

1. **Load Component Images**: DAPI, Laminin-555, Tubulin-488, etc.
2. **Threshold and Preprocess**: Clean noise and select meaningful pixels.
3. **Compute Phasor Coordinates**: For each channel, based on RGB → BGR.
4. **Estimate Pure Component Centers**: Using phasor center-of-mass.
5. **Process Mixed Image**: Compute phasor coordinates, apply unmixing.
6. **Generate Outputs**:
   - Photon fraction maps per component
   - Reconstructed RGB image weighted by unmixed fractions
   - Grayscale intensity and histograms
   - Phasor plot with component markers

## 📦 Dependencies

- Python ≥ 3.7  
- NumPy  
- Matplotlib  
- Scikit-image (optional)  
- `PhasorPy` (for the `PhasorPlot` class)  
- `mpl_toolkits.axes_grid1` (for styled colorbars)

## 🚀 Run the Code

Edit `part4.py` to set the image paths and parameters. Then run:

```bash
python part4.py
```

Figures will be saved automatically if `savefig = True`.

## 🧪 Applications

- Widefield epifluorescence microscopy with standard RGB camera
- Component separation in 3-channel immunofluorescence
- Visualization of mixed-label samples
- Fast analysis without spectral detectors

## 📌 Notes

- Intensity thresholds and phasor ranges may need tuning depending on the fluorophores used.
- The pipeline can be extended to perform **phasor clustering**, **graph-based segmentation**, or **deep learning**.

## 📄 License

MIT License (or update based on your preference)

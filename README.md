# RGB Phasor Analysis for Microscopy Applications

This repository provides tools for applying **phasor-based spectral analysis** and **spectral unmixing** to RGB images obtained from standard microscopy. Originally developed for widefield fluorescence microscopy, the methodology is here extended to three major imaging domains: **multicolor fluorescence**, **label-free autofluorescence**, and **brightfield histological staining (H&E)**.

## 🔬 Overview

Phasor analysis is a model-free, geometric method commonly used in hyperspectral and FLIM microscopy. In this work, we adapt it to conventional RGB images, demonstrating that meaningful spectral features can be extracted even with reduced spectral resolution. By transforming RGB signals into the phasor (G, S) space, we enable:

- Visualization of spectral heterogeneity
- Unsupervised segmentation via phasor clustering
- Spectral unmixing and fluorophore separation
- Quantitative feature extraction (entropy, PCA, morphometry)

## 🧪 Applications

We demonstrate the method across three biological imaging modalities:

1. **Fluorescence microscopy**:  
   RGB images of fixed cells labeled with DAPI, Laminin-488, and NucRed were unmixed into component contributions using phasor coordinates. This enabled high-resolution visualization and quantification of nuclear and cytoskeletal structures.

2. **Autofluorescence imaging (label-free)**:  
   RGB images of pigmented skin lesions (nevus and melanoma) acquired under autofluorescence were analyzed in phasor space to segment regions with different metabolic or structural profiles. Parameters such as phasor entropy and principal component dispersion were used to characterize spectral heterogeneity.

3. **Brightfield histology (H&E)**:  
   RGB images of H&E-stained lung tissue were processed via phasor transformation. Clustering in phasor space allowed segmentation of tissue compartments and quantification of airspace collapse in experimental models of lung injury.

## 📁 Folder Structure

```
.
├── part4.py                # Main script for RGB phasor analysis and unmixing
├── paper/
│   └── fig5/
│       └── data/           # Component and sample images (.tif, .jpg)
├── tools4.py              # Helper functions: RGB conversion, thresholding, phasor computation, etc.
├── output/                # Automatically saved figures
```

## ⚙️ How It Works

1. **Load RGB image(s)**  
2. **Convert to Phasor Space (G, S)**  
3. **Optionally define pure components or phasor cursors**  
4. **Apply clustering or unmixing (e.g., least squares)**  
5. **Generate output maps, histograms, pseudocolor images**

## 📦 Dependencies

- Python ≥ 3.7  
- NumPy  
- Matplotlib  
- Pillow  
- Scikit-image (optional)  
- `PhasorPy` (for phasor plotting)  
- `mpl_toolkits.axes_grid1` (for styled colorbars)

## 🚀 Run the Code

Edit `part4.py` to configure your image inputs and analysis parameters. Then run:

```bash
python part4.py
```

Outputs will be saved automatically in the configured folder.

## 📌 Notes

- Intensity thresholds and phasor scaling may need tuning depending on image type and modality.
- The code can be extended to support superpixel segmentation, entropy analysis, or graph-based learning.
- A combined `figure_part4.jpg` is generated for use in publications.

## 📄 License

MIT License (or update based on your preference)

# Script for Figure 3 Autofluorescence 

import numpy
import matplotlib.pyplot as plt

from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.cursors import mask_from_circular_cursor

import tifffile

from tools import(
    plot_phasor_analysis
)


part_rgb = True
if part_rgb:

    # Analysis with RGB data
    imm = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/nev-mel/18852_10x_r1.tif")
    
    imn = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/nev-mel/16252_10x_r1.tif")

    # Phasor Analysis for Melanoma
    fig9, fig10, fig11, fig4_aux, mean_spectra1, gm_rgb, sm_rgb = plot_phasor_analysis(
        image=imm,
        threshold=30,
        filttime=2,
        cursors_real=[0.42, 0.1, -0.24],
        cursors_imag=[0.22, 0.15, 0.1],
        r=0.18,
        width_hist = 0.05
    )

    #Phasor Analysis for Nevus
    fig1, fig2, fig3, fig8_aux, mean_spectra2, gn_rgb, sn_rgb = plot_phasor_analysis(
        image=imn,
        threshold=60,
        filttime=2,
        cursors_real=[0.42, 0.1, -0.24],
        cursors_imag=[0.22, 0.15, 0.1],
        r=0.18,
        width_hist = 0.05
    )

    # plt.show()

part_hsi = True
if part_hsi:
    # Analysis with HSI data
    imn = tifffile.imread(
            "/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/sp/sp_16556_r2.lsm")
    imm = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/sp/sp_18852_r4.lsm")

    imm = numpy.rot90(imm, k=-1, axes=(1, 2))[:, 84:940]

    imn = numpy.rot90(imn, k=-1, axes=(1, 2))[:, 84:940]
    imn = numpy.rot90(imn, k=-1, axes=(1, 2))
    imn = numpy.rot90(imn, k=-1, axes=(1, 2))

    # Phasor Analysis for Melanoma
    fig13, fig14, fig15, fig12_aux, mean_spectra3, gm_hsi, sm_hsi = plot_phasor_analysis(
        image=imm,
        threshold=4,
        filttime=3,
        cursors_real=[-0.03, -0.23, -0.40],
        cursors_imag=[0.62, 0.36, 0.10],
        r=0.17,
        wavelengths=numpy.linspace(423, 723, imm.shape[0]),
        vmin=0,
        vmax=110
    )

    # Phasor Analysis for Nevus
    fig5, fig6, fig7, fig16_aux, mean_spectra4, gn_hsi, sn_hsi  = plot_phasor_analysis(
        image=imn,
        threshold=4,
        filttime=3,
        cursors_real=[-0.03, -0.23, -0.40],
        cursors_imag=[0.62, 0.36, 0.10],
        r=0.17,
        wavelengths=numpy.linspace(423, 723, imn.shape[0]),
        vmin=0,
        vmax=130
    )


import matplotlib.pyplot as plt
import numpy as np

# Creamos una gran figura con 3 filas (cursores) y 2 columnas (nevus/melanoma)
fig4, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)

# Configuración común
wavelengths_hsi = np.linspace(423, 723, len(mean_spectra3[0]))
rgb_ranges = [(420, 495), (495, 570), (570, 690)]
rgb_centers = [np.mean(r) for r in rgb_ranges]
rgb_widths = [r[1] - r[0] for r in rgb_ranges]
rgb_colors_light = ['#ADD8E6', '#90EE90', '#FFB6C1']
cursor_colors = ['blue', 'green', 'red']
cursor_labels = ['Cursor 1', 'Cursor 2', 'Cursor 3']

for i in range(3):  # Cursor loop
    # ---- NEVUS (columna 0) ----
    ax = axs[i, 0]
    if mean_spectra2[i] is not None:  # RGB Nevus
        rgb = mean_spectra2[i] / np.max(mean_spectra2[i])
        ax.bar(rgb_centers, rgb, width=rgb_widths,
               color=rgb_colors_light[i], edgecolor='black', alpha=0.6, label='RGB')

    if mean_spectra4[i] is not None:  # HSI Nevus
        hsi = mean_spectra4[i] / np.max(mean_spectra4[i])
        ax.plot(wavelengths_hsi, hsi, '-', color=cursor_colors[i], linewidth=2, label='HSI')

    ax.set_xlim(400, 710)
    ax.set_ylim(0, 1.05)
    if i == 2:
        ax.set_xlabel("Wavelength (nm)")
    if i == 0:
        ax.set_title("Nevus", fontsize=14)
    ax.set_ylabel(f"{cursor_labels[i]}\nNormalized Intensity")
    ax.grid(True)

    # ---- MELANOMA (columna 1) ----
    ax = axs[i, 1]
    if mean_spectra1[i] is not None:  # RGB Melanoma
        rgb = mean_spectra1[i] / np.max(mean_spectra1[i])
        ax.bar(rgb_centers, rgb, width=rgb_widths,
               color=rgb_colors_light[i], edgecolor='black', alpha=1.0, hatch='///', label='RGB')

    if mean_spectra3[i] is not None:  # HSI Melanoma
        hsi = mean_spectra3[i] / np.max(mean_spectra3[i])
        ax.plot(wavelengths_hsi, hsi, '--', color=cursor_colors[i], linewidth=2, label='HSI')

    ax.set_xlim(400, 710)
    ax.set_ylim(0, 1.05)
    if i == 2:
        ax.set_xlabel("Wavelength (nm)")
    if i == 0:
        ax.set_title("Melanoma", fontsize=14)
    ax.grid(True)

# plt.suptitle("RGB vs HSI Spectra – Nevus vs Melanoma", fontsize=16, y=0.98)
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.show()

##############################################################################################################

# PCA Analysis and Ellipses plot

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca_ellipse_only(g, s, label, color, ax):
    valid = ~np.isnan(g) & ~np.isnan(s)
    g_valid = g[valid].flatten()
    s_valid = s[valid].flatten()
    X = np.vstack((g_valid, s_valid)).T

    pca = PCA(n_components=2)
    pca.fit(X)

    center = np.mean(X, axis=0)
    components = pca.components_
    explained = pca.explained_variance_

    # 95% confidence ellipse
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    scale = np.sqrt(5.991)  # 95% CI for 2D Gaussian
    ellipse = center[:, None] + scale * (components.T @ (np.sqrt(explained)[:, None] * circle))

    ax.plot(ellipse[0], ellipse[1], color=color, label=label, linewidth=2)

# === Graficar elipses ===
plot = PhasorPlot(allquadrants=True, title='')
fig8 = plot.fig
ax = plot.ax
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Dibujar cada elipse

plot_pca_ellipse_only(gm_rgb, sm_rgb, "Melanoma RGB", "red", ax)
plot_pca_ellipse_only(gm_hsi, sm_hsi, "Melanoma HSI", "blue", ax)
plot_pca_ellipse_only(gn_rgb, sn_rgb, "Nevus RGB", "green", ax)
plot_pca_ellipse_only(gn_hsi, sn_hsi, "Nevus HSI", "orange", ax)

ax.legend()

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from math import pi

# 1. Cargar y filtrar datos de phasor
def get_valid_phasor(g, s):
    valid = ~np.isnan(g) & ~np.isnan(s)
    return np.column_stack((g[valid], s[valid]))

# Extraer datos
phasor_rgb_mel = get_valid_phasor(gm_rgb, sm_rgb)
phasor_rgb_nev = get_valid_phasor(gn_rgb, sn_rgb)
phasor_hsi_mel = get_valid_phasor(gm_hsi, sm_hsi)
phasor_hsi_nev = get_valid_phasor(gn_hsi, sn_hsi)

# 2. Agrupar
phasors = {
    ("Melanoma", "RGB"): phasor_rgb_mel,
    ("Melanoma", "HSI"): phasor_hsi_mel,
    ("Nevus", "RGB"): phasor_rgb_nev,
    ("Nevus", "HSI"): phasor_hsi_nev,
}

# 3. Calcular parámetros
rows = []

for (tipo, datos), phasor_data in phasors.items():
    pca = PCA(n_components=2)
    pca.fit(phasor_data)

    var1 = pca.explained_variance_[0]
    var2 = pca.explained_variance_[1]
    elongation = var1 / var2 if var2 > 0 else np.nan

    # Orientación del primer componente
    angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    angle_deg = np.degrees(angle_rad)

    # Área de la elipse: π·√λ1·√λ2
    area = pi * np.sqrt(var1) * np.sqrt(var2)

    rows.append({
        "Tipo": tipo,
        "Datos": datos,
        "Elongación": elongation,
        "Ángulo (°)": angle_deg,
        "Varianza 1": var1,
        "Varianza 2": var2,
        "Área elipse": area
    })

# 4. Mostrar como tabla
df = pd.DataFrame(rows)
print(df.to_markdown(index=False))

import matplotlib.pyplot as plt
import numpy as np

# Datos extraídos de la tabla
labels = ["Melanoma RGB", "Melanoma HSI", "Nevo RGB", "Nevo HSI"]
elongation = [21.8054, 13.6123, 2.1732, 3.4451]
var1 = [0.0303736, 0.0196559, 0.00206271, 0.00252572]
var2 = [0.00139294, 0.00144398, 0.000949161, 0.000733132]
area = [0.0204345, 0.0167369, 0.00439581, 0.00427497]

fig12, axes = plt.subplots(2, 2, figsize=(7, 5))
fig12.subplots_adjust(hspace=0.5, wspace=0.3)

# Paleta de colores suaves
colors = ['#5B8FF9', '#61DDAA', '#65789B', '#F6BD16']

# Gráfico de elongación
axes[0, 0].bar(labels, elongation, color=colors)
axes[0, 0].set_title("Elongation")
axes[0, 0].set_ylabel("Var1 / Var2")
axes[0, 0].tick_params(axis='x', rotation=20)

# Gráfico de varianza 1
axes[0, 1].bar(labels, var1, color=colors)
axes[0, 1].set_title("Variance 1 (Major axis)")
axes[0, 1].set_ylabel("Variance")
axes[0, 1].tick_params(axis='x', rotation=20)

# Gráfico de varianza 2
axes[1, 0].bar(labels, var2, color=colors)
axes[1, 0].set_title("Variance 2 (Minor axis)")
axes[1, 0].set_ylabel("Variance")
axes[1, 0].tick_params(axis='x', rotation=20)

# Gráfico de área de elipse
axes[1, 1].bar(labels, area, color=colors)
axes[1, 1].set_title("Ellipse Area")
axes[1, 1].set_ylabel("Area")
axes[1, 1].tick_params(axis='x', rotation=20)


################## Entropy ################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def compute_phasor_entropy(g, s, bins=50):
    """Compute entropy of phasor distribution without returning the histogram."""
    valid = ~np.isnan(g) & ~np.isnan(s)
    g_valid, s_valid = g[valid], s[valid]

    hist2d, _, _ = np.histogram2d(g_valid, s_valid, bins=bins,
                                  range=[[-1, 1], [-1, 1]])
    hist_flat = hist2d.flatten()
    hist_flat = hist_flat[hist_flat > 0]
    prob = hist_flat / np.sum(hist_flat)
    return entropy(prob)

# Nombres y datos
labels = ["Melanoma RGB", "Melanoma HSI", "Nevus RGB", "Nevus HSI"]
phasor_data = [ (gm_rgb, sm_rgb), (gm_hsi, sm_hsi), (gn_rgb, sn_rgb), (gn_hsi, sn_hsi) ]

# Calcular entropías
entropies = [compute_phasor_entropy(g, s) for g, s in phasor_data]

# Colores elegantes para el paper
colors = ["#E92218", '#2980B9', '#27AE60', "#F38D07"]

# Crear figura
fig16, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, entropies, color=colors, edgecolor='black')

# Anotar valores encima de cada barra
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.002,
            f"{yval:.3f}", ha='center', va='bottom', fontsize=9)

# Ejes y estilo
ax.set_ylabel("Spectral Entropy")
ax.set_ylim(0, max(entropies) + 0.02)
ax.tick_params(axis='x', rotation=15)

print("Mel RGB", "Mel HSI", "Nevus RGB", "Nevus HSI")
print(entropies)

# plt.tight_layout()

##########################################################################################

print_cm = True
if print_cm:
    from matplotlib.gridspec import GridSpec
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import string

    def figure_to_array(fig, dpi=150):
        fig.set_dpi(dpi)
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = canvas.get_width_height()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
        plt.close(fig)
        return img

    def plot_figure_grid(figures, titles=None, figsize=(16, 12), dpi=150, save_path=None):
        """
        Plots 16 figures in a 4x4 grid using FigureCanvasAgg.
        fig12 goes to [2, 3] and fig16 to [3, 3].
        """
        if len(figures) != 16:
            raise ValueError("You must provide exactly 16 figures.")
        if titles and len(titles) != 16:
            raise ValueError("Titles list must have 16 elements.")

        images = [figure_to_array(f, dpi=dpi) for f in figures]
        labels = list(string.ascii_uppercase[:16])  # ['A', 'B', ..., 'P']

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 4, figure=fig, wspace=0.01, hspace=0.01)

        index = 0

        def plot_ax(ax, img, label, title=None):
            ax.imshow(img)
            ax.axis("off")
            ax.text(0.02, 0.95, label, transform=ax.transAxes,
                    fontsize=12, fontweight='bold', va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
            if title:
                ax.set_title(title, fontsize=10, pad=4)

        # Fila 0: fig1 – fig4
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            plot_ax(ax, images[index], labels[index], titles[index] if titles else None)
            index += 1

        # Fila 1: fig5 – fig8
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            plot_ax(ax, images[index], labels[index], titles[index] if titles else None)
            index += 1

        # Fila 2: fig9 – fig12
        for i in range(4):
            ax = fig.add_subplot(gs[2, i])
            plot_ax(ax, images[index], labels[index], titles[index] if titles else None)
            index += 1

        # Fila 3: fig13 – fig16
        for i in range(4):
            ax = fig.add_subplot(gs[3, i])
            plot_ax(ax, images[index], labels[index], titles[index] if titles else None)
            index += 1

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

    # Tus figuras de matplotlib: fig1 a fig15
    figs = [
        fig1,  fig2,  fig3,  fig4,   # fila 1
        fig5,  fig6,  fig7,  fig8,   # fila 2
        fig9,  fig10, fig11, fig12,  # fila 3 
        fig13, fig14, fig15, fig16   # fila 4
    ]

    # Títulos opcionales
    titles = [
        "RGB Nevus", "Phasor Plot", "Pseudocolor", "Histogram RGB vs HSI",
        "HSI Nevus", "", "", "Phasor PCA Ellipses (95% CI)",
        "RGB Melanoma", "", "", "Average Spectra",
        "HSI Melanoma", "", "", "Phasor Entropy across Conditions"
    ]

    # Generar figura compuesta con labels + títulos
    plot_figure_grid(
        figs, titles=titles,
        save_path="/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig4/nev-mel/phasor_figure_grid.png",
        dpi=600)
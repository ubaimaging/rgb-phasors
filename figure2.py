import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tifffile as tiff
import tools
from phasorpy.plot import PhasorPlot


# Lee una imagen RGB calucla y plotea el phasor y el clusterized phasor
part1 = True
if part1: 
        impath1 = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/92-21B6ND09.tif"
        impath2 = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/066-21B6ND_INS08.tif"
        im1 = plt.imread(impath1)
        im2 = plt.imread(impath2)


        img1 = tools.rgb2bgr(im1)
        img2 = tools.rgb2bgr(im2)

        _, g1, s1 = tools.phasor(img1)
        g1 = tools.median_filter(g1, 2)
        s1 = tools.median_filter(s1, 2)

        _, g2, s2 = tools.phasor(img2)
        g2 = tools.median_filter(g2, 2)
        s2 = tools.median_filter(s2, 2)

        # Clustering segmentation
        nclusters = 4
        
        X1 = np.asarray([g1.flatten(), s1.flatten()]).transpose()
        kmeans = KMeans(n_clusters=nclusters, random_state=42, n_init="auto").fit(X1)
        pred_y1 = kmeans.fit_predict(X1)
        imp1 = pred_y1.reshape(g1.shape)

        X2 = np.asarray([g2.flatten(), s2.flatten()]).transpose()
        kmeans = KMeans(n_clusters=nclusters, random_state=0, n_init="auto").fit(X2)
        pred_y2 = kmeans.fit_predict(X2)
        imp2 = pred_y2.reshape(g2.shape)

        # Read the binarized images
        im1bin = plt.imread(
            "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/segmentations/phasor_nd/phasor_92-21B6ND09_binary.tif")
        im2bin = plt.imread(
            "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/segmentations/phasor_inst/phasor_066-21B6ND_INS08_binary.tif")

        plt.figure()
        plt.imshow(im1)
        plt.axis('off')
        plt.figure()
        plt.imshow(im2)
        plt.axis('off')

        plt.figure()
        plt.imshow(tools.map_mask_to_colors(imp1,[0, 2, 1, 3]))
        plt.title("Segmented ND")
        plt.axis('off')

        tools.cluster_phasor_plot_4_clusters(X1, pred_y1)

        plt.figure()
        plt.imshow(tools.map_mask_to_colors(imp2, [2, 3, 0, 1]))
        plt.title("Segmented ND inst")
        plt.axis('off')

        tools.cluster_phasor_plot_4_clusters(X2, pred_y2, colors=["r", "b", "k", "lime"])

        plt.figure()
        plt.imshow(im1bin)
        plt.axis('off')
        plt.figure()
        plt.imshow(im2bin)
        plt.axis('off')

        # Phasor Plot
        plot = PhasorPlot(allquadrants=True, title='Phasor plot')
        plot.hist2d(g1.flatten(), s1.flatten(), cmap="RdYlGn_r")

        plot = PhasorPlot(allquadrants=True, title='Phasor plot')
        plot.hist2d(g2.flatten(), s2.flatten(), cmap="RdYlGn_r")

        plt.show()

# Part 2
# Perform statistical analysis
part2 = False
if part2:
    import os
    import numpy as np
    import tifffile as tiff
    import matplotlib.pyplot as plt

    def calculate_areas(folder_path):
        """
        Calcula el porcentaje del área con valor 255 en cada máscara de una carpeta,
        leyendo las máscaras en orden alfabético.

        Parámetros:
        - folder_path (str): Ruta a la carpeta que contiene las máscaras TIFF.

        Retorno:
        - np.array: Porcentajes del área con valor 255 para cada máscara.
        """
        areas = []
        files = sorted([f for f in os.listdir(folder_path) if f.endswith((".tif", ".tiff"))])
        for file in files:
            mask = tiff.imread(os.path.join(folder_path, file))
            total_pixels = mask.size
            region_pixels = np.sum(mask == 255)
            areas.append((region_pixels / total_pixels) * 100)
        return np.array(areas)

    def group_areas(areas, group_size=10):
        """
        Agrupa los porcentajes en grupos de tamaño especificado.

        Parámetros:
        - areas (np.array): Array de porcentajes del área.
        - group_size (int): Tamaño de cada grupo.

        Retorno:
        - np.array: Array agrupado.
        """
        num_groups = len(areas) // group_size
        grouped = areas[:num_groups * group_size].reshape(num_groups, group_size)
        return grouped

    def plot_separated_boxplots_and_violin(d, boxplot_linewidth=2.5):
        """
        Genera una figura con boxplots y violin plots.

        Parámetros:
        - d: np.array, un array de tamaño (2, 100).
        - boxplot_linewidth: float, grosor de los trazos de los boxplots.
        """
        # Verificar que los datos tengan la forma esperada
        if d.shape != (2, 100):
            raise ValueError("El array de entrada debe tener forma (2, 100)")

        # Reorganizar a (2, 10, 10)
        d_reshaped = d.reshape(2, 10, 10)

        # Figura 1: Boxplots
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        box_data = [group for sub in d_reshaped for group in sub]
        box_labels = [
            f"Group-{i + 1} ({'ND' if idx == 0 else 'Inst'})"
            for idx, sub in enumerate(d_reshaped)
            for i in range(10)
        ]
        # Crear el boxplot y capturar los objetos
        boxplot_elements = ax1.boxplot(box_data, labels=box_labels)

        # Ajustar el grosor de los elementos del boxplot
        for element in ['boxes', 'whiskers', 'caps', 'medians']:
            for line in boxplot_elements[element]:
                line.set_linewidth(boxplot_linewidth)

        # Configurar el título y etiquetas
        ax1.set_title("Boxplots of Groups", fontsize=16, weight="bold")
        ax1.set_xlabel("Groups", fontsize=14)
        ax1.set_ylabel("Percent Area", fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Figura 2: Violin Plots
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.violinplot(
            [d_reshaped[0].flatten(), d_reshaped[1].flatten()],
            showmeans=True,
            showmedians=True
        )
        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(["ND", "Inst"])
        ax2.set_title("Violin Plot of Groups", fontsize=16, weight="bold")
        ax2.set_xlabel("Condition", fontsize=14)
        ax2.set_ylabel("Percent Area", fontsize=14)
        plt.tight_layout()

        plt.show()

    # Ruta al folder principal que contiene las carpetas ND y Inst
    folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/binarized phasor/"
    subfolders = {"ND": os.path.join(folder, "nd"), "Inst": os.path.join(folder, "inst")}

    # Verificar que ambas carpetas existan
    assert os.path.isdir(subfolders["ND"]), "No se encontró la carpeta ND."
    assert os.path.isdir(subfolders["Inst"]), "No se encontró la carpeta Inst."

    # Calcular áreas para ambas carpetas
    areas_nd = calculate_areas(subfolders["ND"])
    areas_inst = calculate_areas(subfolders["Inst"])

    # Agrupar las áreas en 10 grupos de 10 máscaras cada uno
    grouped_nd = group_areas(areas_nd, group_size=10)
    grouped_inst = group_areas(areas_inst, group_size=10)

    # Combinar los datos en una sola matriz
    data_combined = np.array([grouped_nd.flatten(), grouped_inst.flatten()])

    # Aplicar la función plot_separated_boxplots_and_violin con trazos más gruesos
    plot_separated_boxplots_and_violin(data_combined, boxplot_linewidth=3.0)
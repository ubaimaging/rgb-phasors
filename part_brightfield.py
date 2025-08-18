import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import tifffile as tiff
import os

from phasorpy.plot import PhasorPlot
from phasorpy.phasor import phasor_from_signal

from tools import (
    rgb2bgr,
    cluster_phasor_plot_4_clusters,
    median_filter,
    map_mask_to_colors,
    apply_plot_style
)

plt.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "figure.figsize": (8, 8),
})


apply_plot_style()

# Lee una imagen RGB calcula y plotea el phasor y el clusterized phasor
part1 = True
if part1: 
        impath1 = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/92-21B6ND09.tif"
        impath2 = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/066-21B6ND_INS08.tif"
        im1 = plt.imread(impath1)
        im2 = plt.imread(impath2)

        _, g1, s1 = phasor_from_signal(rgb2bgr(im1), axis=0)
        _, g2, s2 = phasor_from_signal(rgb2bgr(im2), axis=0)
        
        g1 = median_filter(g1, 2)
        s1 = median_filter(s1, 2)

        g2 = median_filter(g2, 2)
        s2 = median_filter(s2, 2)

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

        from tools4 import  increase_brightness
        from tools import add_scale_bar


        fig1 = plt.figure(figsize=(8, 8))
        ax1 = plt.gca()
        ax1.imshow(increase_brightness(im1, 2))
        ax1.axis('off')
        add_scale_bar(ax1, length_px=500, color='white')
        # plt.title("Normal Diet RGB")

        fig5 = plt.figure(figsize=(8, 8))
        ax5 = plt.gca()
        ax5.imshow(increase_brightness(im2, 2))
        ax5.axis('off')
        add_scale_bar(ax5, length_px=500, color='white')
        # plt.title("Normal Diet Inst RGB")

        fig4 = plt.figure(figsize=(8, 8))
        plt.imshow(map_mask_to_colors(imp1,[0, 2, 1, 3]))
        # plt.title("Segmentation (ND)")
        plt.axis('off')
        ax4 = plt.gca()
        add_scale_bar(ax4, length_px=500, color='white')

        tam = 6
        fig3 = cluster_phasor_plot_4_clusters(X1, pred_y1)
        fig3.set_size_inches(tam, tam)

        fig8 = plt.figure(figsize=(8, 8))
        plt.imshow(map_mask_to_colors(imp2, [2, 3, 0, 1]))
        plt.axis('off')
        ax8 = plt.gca()
        add_scale_bar(ax8, length_px=500, color='white')

        fig7 = cluster_phasor_plot_4_clusters(X2, pred_y2, 
                                              colors=["r", "b", "k", "lime"])
        fig7.set_size_inches(tam, tam)

        plotmask = False
        if plotmask:
            plt.figure()
            plt.imshow(im1bin)
            plt.axis('off')
            plt.figure()
            plt.imshow(im2bin)
            plt.axis('off')

        # Phasor Plot
        plot = PhasorPlot(allquadrants=True, title='')
        plot.hist2d(g1.flatten(), s1.flatten(), cmap="RdYlGn_r")
        fig2 = plot.fig
        fig2.set_size_inches(tam, tam)
        fig2.tight_layout()
        ax2 = plot.ax
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)

        plot = PhasorPlot(allquadrants=True, title='')
        plot.hist2d(g2.flatten(), s2.flatten(), cmap="RdYlGn_r")
        fig6 = plot.fig
        fig6.set_size_inches(tam, tam)
        fig6.tight_layout()
        ax6 = plot.ax
        ax6.set_xlim(-1, 1)
        ax6.set_ylim(-1, 1)

        # plt.show()

# Part 2
# Perform statistical analysis with Area of Alveolar Space
part2 = True
if part2:
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


    def plot_separated_boxplots_and_violin(d, boxplot_linewidth=1.5):
        """
        Genera una figura con boxplots y violin plots con estilo profesional.

        Parámetros:
        - d: np.array, un array de tamaño (2, 100).
        - boxplot_linewidth: float, grosor de los trazos de los boxplots.
        """
        if d.shape != (2, 100):
            raise ValueError("El array de entrada debe tener forma (2, 100)")

        # Estilo global
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 12,
            'axes.linewidth': 1.2,
            'xtick.major.width': 1.1,
            'ytick.major.width': 1.1,
        })

        # Reorganizar a (2, 10, 10)
        d_reshaped = d.reshape(2, 10, 10)

        # === Figura 1: Boxplots ===
        fig9, ax1 = plt.subplots(figsize=(12, 4))
        box_data = [group for sub in d_reshaped for group in sub]

        # Etiquetas mejoradas (solo índices)
        box_labels = [f"{i+1}" for i in range(20)]

        # Colores y estilos personalizados
        boxprops = dict(facecolor='white', color='black', linewidth=boxplot_linewidth)
        whiskerprops = dict(color='black', linewidth=boxplot_linewidth)
        capprops = dict(color='black', linewidth=boxplot_linewidth)
        medianprops = dict(color='black', linewidth=boxplot_linewidth)

        boxplot_elements = ax1.boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            boxprops=boxprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
            medianprops=medianprops
        )

        ax1.set_title("Boxplots of Groups")
        ax1.set_xlabel("Group Index")
        ax1.set_ylabel("% Alveolar space area")
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        plt.xticks(rotation=0)
        plt.tight_layout()

        # === Figura 2: Violin Plot ===
        fig10, ax2 = plt.subplots(figsize=(4, 4))
        parts = ax2.violinplot(
            [d_reshaped[0].flatten(), d_reshaped[1].flatten()],
            showmeans=True,
            showmedians=True
        )

        for pc in parts['bodies']:
            pc.set_facecolor('#1f77b4')
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)

        # Estilo para las líneas de media/mediana
        for partname in ('cmeans', 'cmedians'):
            vp = parts.get(partname)
            if vp:
                vp.set_edgecolor('black')
                vp.set_linewidth(1.5)

        ax2.set_xticks([1, 2])
        ax2.set_xticklabels(["ND", "Inst"])
        ax2.set_title("Violin Plot of Groups")
        ax2.set_xlabel("Condition")
        ax2.set_ylabel("% Alveolar space area")
        ax2.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        return fig9, fig10

        # plt.show()

    # Ruta al folder principal que contiene las carpetas ND y Inst
    folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/binarized phasor/"
    subfolders = {"ND": os.path.join(folder, "nd"), "Inst": os.path.join(folder, "inst")}

    # Verificar que ambas carpetas existan
    assert os.path.isdir(subfolders["ND"]), "No se encontró la carpeta ND."
    assert os.path.isdir(subfolders["Inst"]), "No se encontró la carpeta Inst."

    # Calcular áreas de tejido
    areas_nd = calculate_areas(subfolders["ND"])
    areas_inst = calculate_areas(subfolders["Inst"])

    # Obtener áreas complementarias (espacio alveolar)
    alveolar_nd = 100 - areas_nd
    alveolar_inst = 100 - areas_inst

    # Agrupar las áreas en 10 grupos de 10 máscaras cada uno
    grouped_nd = group_areas(alveolar_nd, group_size=10)
    grouped_inst = group_areas(alveolar_inst, group_size=10)

    # Combinar los datos en una sola matriz
    data_combined = np.array([grouped_nd.flatten(), grouped_inst.flatten()])

    # Aplicar la función plot_separated_boxplots_and_violin con trazos más gruesos
    fig9, fig10 = plot_separated_boxplots_and_violin(data_combined, boxplot_linewidth=3.0)

    # Check if the data has normal distribution to parametric or non-parametric test
    from scipy.stats import shapiro

    stat_nd, p_nd = shapiro(grouped_nd.flatten())
    stat_inst, p_inst = shapiro(grouped_inst.flatten())

    print("p > 0.05, Normal distribution → Parametric test")
    print("p < 0.05, Not normal distribution → Non parametric test")

    print("Shapiro-Wilk test ND:", np.round(p_nd, 3))
    print("Shapiro-Wilk test Inst:", np.round(p_inst, 3))
    print( "ND normal distribution:", np.round(p_nd, 3) > 0.05)
    print( "Inst normal distribution:", np.round(p_inst, 3) > 0.05)

    # Perform T-test for independent samples
    from scipy.stats import ttest_ind

    nd_data = grouped_nd.flatten()
    inst_data = grouped_inst.flatten()

    stat, p_value = ttest_ind(nd_data, inst_data)
    print("T-test p-value:", p_value)
    if p_value < 0.05:
        print("Significant difference between ND and Inst groups (p < 0.05)")
    else:
        print("No significant difference between ND and Inst groups (p >= 0.05)")

    # Calculate Cohen's d for effect size
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + 
                              (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
        return (np.mean(x) - np.mean(y)) / pooled_std

    d = cohens_d(inst_data, nd_data)
    print("Cohen's d:", d)

    
    # Perform classification using logistic regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, f1_score, roc_auc_score,
        roc_curve, auc
    )

    # Crear etiquetas (0: ND, 1: INST)
    labels_nd = np.zeros_like(alveolar_nd)
    labels_inst = np.ones_like(alveolar_inst)

    # Combinar datos y etiquetas
    X = np.concatenate([alveolar_nd, alveolar_inst]).reshape(-1, 1)
    y = np.concatenate([labels_nd, labels_inst])

    # === Entrenar el clasificador ===
    model = LogisticRegression()
    model.fit(X, y)

    # === Predicción de probabilidades ===
    probs = model.predict_proba(X)[:, 1]

    # === Métricas de rendimiento ===
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_score = roc_auc_score(y, probs)

    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC: {auc_score:.3f}")

    # === Curva ROC ===
    fpr, tpr, thresholds = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    # === Umbral óptimo (Youden's Index) ===
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    print(f"Optimal threshold (Youden’s index): {best_threshold:.3f}")

    # === Matriz de confusión con mejor umbral ===
    y_opt = (probs >= best_threshold).astype(int)
    cm = confusion_matrix(y, y_opt)
    print("Confusion Matrix:")
    print(cm)

    # === Gráfico de la curva ROC ===
    fig11 = plt.figure(figsize=(12, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', linewidth=2)
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', 
                label=f'Youden Index (Thresh={best_threshold:.2f})', s=50)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    labels = ["ND", "Instilled"]

    # Estilo
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12
    })

    # Crear figura
    fig12 = plt.figure(figsize=(4, 4))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=1.5, linecolor='white', square=True,
                    annot_kws={"size": 16, "weight": "bold", "color": "black"})

    # Etiquetas y diseño
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)

    # ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
savefig = False
if savefig:
    res = 300
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig2y3/"
    fig1.savefig(path + "figure1.tiff", dpi=res, bbox_inches="tight")
    fig2.savefig(path + "figure2.tiff", dpi=res, bbox_inches="tight")
    fig3.savefig(path + "figure3.tiff", dpi=res, bbox_inches="tight")
    fig4.savefig(path + "figure4.tiff", dpi=res, bbox_inches="tight")
    fig5.savefig(path + "figure5.tiff", dpi=res, bbox_inches="tight")
    fig6.savefig(path + "figure6.tiff", dpi=res, bbox_inches="tight")
    fig7.savefig(path + "figure7.tiff", dpi=res, bbox_inches="tight")
    fig8.savefig(path + "figure8.tiff", dpi=res, bbox_inches="tight")
    fig9.savefig(path + "figure9.tiff", dpi=res, bbox_inches="tight")
    fig10.savefig(path + "figure10.tiff", dpi=res, bbox_inches="tight")
    fig11.savefig(path + "figure11.tiff", dpi=res, bbox_inches="tight")
    fig12.savefig(path + "figure12.tiff", dpi=res, bbox_inches="tight")

"""
This section explores the discriminative potential of various parameters 
derived directly from the phasor plot to classify between ND and ND_inst groups. 
Metrics such as the phasor center of mass (cm_g, cm_s), principal components (PCA), 
phasor area, elongation, and angular orientation were evaluated.

Although logistic regression models based on these features did not achieve high accuracy,
the analysis provides valuable insights into the spectral structure of the tissues and 
suggests that these parameters may be useful when combined with other approaches. 
All results are stored in .csv files, serving as a basis for further statistical evaluation 
or integration into more advanced models.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import tifffile as tiff
import os

from phasorpy.plot import PhasorPlot
from phasorpy.phasor import phasor_from_signal

from tools import (
    rgb2bgr,
)


# Part 3
# Process center of mass for ND and Instilled groups
part3 = False
if part3:
    import pandas as pd
    from skimage.io import imread


    def binarize_mask(mask):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return (mask > 0)

    def phasor_center_of_mass(g, s, avg, mask):
        g = g[mask]
        s = s[mask]
        avg = avg[mask]
        total_weight = np.nansum(avg)
        g_com = np.nansum(g * avg) / total_weight
        s_com = np.nansum(s * avg) / total_weight
        return g_com, s_com

    def process_group_cm(group_label, img_folder, mask_folder):
        img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(".tif")])
        results = []

        for img_file in img_files:
            try:
                img_path = os.path.join(img_folder, img_file)
                mask_name = f"phasor_{img_file.replace('.tif', '')}_binary.tif"
                mask_path = os.path.join(mask_folder, mask_name)

                img = imread(img_path)
                img_bgr = rgb2bgr(img)
                avg, g, s = phasor_from_signal(img_bgr, axis=0)

                mask = imread(mask_path)
                mask = binarize_mask(mask)

                if mask.shape != g.shape:
                    print(f"Dimensiones no coinciden para: {img_file}")
                    continue

                cm_g, cm_s = phasor_center_of_mass(g, s, avg, mask)
                if cm_g is None or cm_s is None:
                    continue

                basename = img_file.replace(".tif", "")
                results.append({
                    "image": basename,
                    "group": group_label,
                    "cm_g": cm_g,
                    "cm_s": cm_s
                })

            except Exception as e:
                print(f"Error en {img_file}: {e}")
                continue

        return pd.DataFrame(results)

    # Paths
    nd_img_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND"
    nd_mask_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/binarized phasor/nd"
    inst_img_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst"
    inst_mask_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/binarized phasor/inst"
    save_csv = "/Users/schutyb/Documents/Projects/rgb-phasors/results/phasor_cm_nd_inst.csv"

    # Process and save
    df_nd = process_group_cm("ND", nd_img_folder, nd_mask_folder)
    df_inst = process_group_cm("ND_inst", inst_img_folder, inst_mask_folder)
    df_cm = pd.concat([df_nd, df_inst], ignore_index=True)
    df_cm.to_csv(save_csv, index=False)
    print(f"\n✅ Guardado en: {save_csv}")
    print(df_cm.head())


# Part 4
# PCA Analysis of the Phasor Signal
# This part processes the phasor signals and computes PCA metrics for each image.
part4 = False
if part4:
    import os
    import numpy as np
    import pandas as pd
    from skimage.io import imread
    from phasorpy.phasor import phasor_from_signal
    from sklearn.decomposition import PCA
    from skimage.color import rgb2gray  # solo si necesitás

    def binarize_mask(mask):
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask > 0

    def compute_phasor_pca(g, s, mask):
        g_vals = g[mask]
        s_vals = s[mask]

        if len(g_vals) < 2 or len(s_vals) < 2:
            return None  # PCA requiere al menos 2 puntos

        X = np.vstack([g_vals, s_vals]).T
        pca = PCA(n_components=2)
        pca.fit(X)

        var1, var2 = pca.explained_variance_
        elongation = var1 / var2 if var2 != 0 else np.inf
        angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        angle_deg = np.degrees(angle_rad)
        area = np.pi * np.sqrt(var1) * np.sqrt(var2)

        return var1, var2, elongation, angle_deg, area

    def process_group_pca(group_label, img_folder, mask_folder):
        img_files = sorted([f for f in os.listdir(img_folder) if f.endswith(".tif")])
        results = []

        for img_file in img_files:
            try:
                img_path = os.path.join(img_folder, img_file)
                mask_path = os.path.join(mask_folder, f"phasor_{img_file.replace('.tif', '')}_binary.tif")

                img = imread(img_path)
                img_bgr = rgb2bgr(img)
                _, g, s = phasor_from_signal(img_bgr, axis=0)

                mask = imread(mask_path)
                mask = binarize_mask(mask)

                if mask.shape != g.shape:
                    print(f"⚠️ Dimensiones distintas: {img_file}")
                    continue

                pca_vals = compute_phasor_pca(g, s, mask)
                if pca_vals is None:
                    continue

                var1, var2, elongation, angle_deg, area = pca_vals
                results.append({
                    "image": img_file.replace(".tif", ""),
                    "group": group_label,
                    "var1": var1,
                    "var2": var2,
                    "elongation": elongation,
                    "angle_deg": angle_deg,
                    "area": area
                })

            except Exception as e:
                print(f"❌ Error en {img_file}: {e}")
                continue

        return pd.DataFrame(results)

    # === Paths ===
    nd_img_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND"
    nd_mask_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/binarized phasor/nd"
    inst_img_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst"
    inst_mask_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/binarized phasor/inst"
    save_csv = "/Users/schutyb/Documents/Projects/rgb-phasors/results/phasor_pca_metrics.csv"

    # === Ejecución ===
    df_nd = process_group_pca("ND", nd_img_folder, nd_mask_folder)
    df_inst = process_group_pca("ND_inst", inst_img_folder, inst_mask_folder)
    df_pca = pd.concat([df_nd, df_inst], ignore_index=True)
    df_pca.to_csv(save_csv, index=False)

    print(f"\n✅ CSV guardado en: {save_csv}")
    print(df_pca.head())

# Statistical analysis of PCA metrics
# This part reads the PCA metrics CSV and performs statistical tests on the elongation,
# area, and angle_deg variables.

part5 = False
if part5:
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import shapiro, ttest_ind, mannwhitneyu

    # Leer CSV
    df = pd.read_csv("/Users/schutyb/Documents/Projects/rgb-phasors/results/phasor_pca_metrics.csv")

    # Variables a analizar
    variables = ['elongation', 'area', 'angle_deg']

    # Comparar ND vs ND_inst
    for var in variables:
        print(f"\n===== Análisis de {var} =====")

        nd_vals = df[df['group'] == 'ND'][var].dropna()
        inst_vals = df[df['group'] == 'ND_inst'][var].dropna()

        # Normalidad
        p_nd = shapiro(nd_vals).pvalue
        p_inst = shapiro(inst_vals).pvalue
        print(f"Shapiro-Wilk p (ND): {p_nd:.3f}")
        print(f"Shapiro-Wilk p (ND_inst): {p_inst:.3f}")

        # Test estadístico
        if p_nd > 0.05 and p_inst > 0.05:
            stat, pval = ttest_ind(nd_vals, inst_vals)
            print(f"T-test p-value: {pval:.4e}")
        else:
            stat, pval = mannwhitneyu(nd_vals, inst_vals)
            print(f"Mann-Whitney U p-value: {pval:.4e}")

        # Boxplot
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x='group', y=var, palette=['skyblue', 'salmon'])
        sns.swarmplot(data=df, x='group', y=var, color='k', size=3, alpha=0.6)
        plt.title(f"{var} por grupo (ND vs Inst)")
        plt.tight_layout()
        plt.show()

# Part 6
# Logistic Regression on PCA metrics
# This part reads the PCA metrics CSV and performs logistic regression to classify ND vs ND_inst.
part6 = True
if part6:
    import pandas as pd
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    )

    # === Cargar datos ===
    csv_path = "/Users/schutyb/Documents/Projects/rgb-phasors/results/phasor_pca_metrics.csv"
    df = pd.read_csv(csv_path)

    # Convertir etiquetas a binario
    df['label'] = df['group'].map({'ND': 0, 'ND_inst': 1})

    # === Variables ===
    X = df[['area', 'angle_deg']].values
    y = df['label'].values

    # === Escalado ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === División train/test ===
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=42, test_size=0.25)

    # === Modelo ===
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # === Predicción ===
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # === Métricas ===
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    # === Umbral óptimo con Youden’s Index ===
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    j_best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[j_best_idx]

    # === Salida ===
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print(f"Optimal threshold (Youden’s index): {optimal_threshold:.3f}")
    print("Confusion Matrix:")
    print(cm)

    # === Plot decisión y distribución ===
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=df['group'], palette={'ND': 'blue', 'ND_inst': 'red'})
    plt.title("Distribución por Área y Ángulo")
    plt.xlabel("Área (normalizada)")
    plt.ylabel("Ángulo (normalizado)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Curva ROC ===
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title("Curva ROC")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
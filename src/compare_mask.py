import os
import numpy as np
import tifffile as tiff
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# Métricas de similitud pixel a pixel
def dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1 & mask2)
    return (2 * intersection) / (np.sum(mask1) + np.sum(mask2))

def jaccard_index(mask1, mask2):
    intersection = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return intersection / union

def accuracy(mask1, mask2):
    correct_pixels = np.sum(mask1 == mask2)
    total_pixels = mask1.size
    return correct_pixels / total_pixels

# Visualización de superposición: Comparación dos a dos
def plot_pairwise_overlapping_masks(mask1, mask2, method1, method2, filename):
    """
    Genera un gráfico de superposición entre dos métodos.
    """
    plt.figure(figsize=(10, 10))
    plt.contour(mask1, colors='red', linewidths=2, label=method1)
    plt.contour(mask2, colors='blue', linewidths=2, label=method2)
    plt.legend()
    plt.title(f"Overlapping Masks: {method1} vs {method2}")
    plt.savefig(filename)
    plt.show()

# Leer máscaras desde carpetas
def load_masks_from_folder(folder_path):
    """
    Lee máscaras desde subcarpetas de un directorio principal,
    asegurándose de que los nombres de archivo coincidan entre métodos.

    Parámetros:
    - folder_path (str): Ruta al directorio principal.

    Retorno:
    - dict: Diccionario con nombres de subcarpetas como claves y diccionarios de máscaras como valores.
    """
    masks_dict = {}
    filenames_set = None

    # Recorrer las subcarpetas
    for method in os.listdir(folder_path):
        method_path = os.path.join(folder_path, method)
        if os.path.isdir(method_path):
            masks_dict[method] = {}

            # Leer archivos de la carpeta
            files = [f for f in sorted(os.listdir(method_path)) if f.endswith((".tif", ".tiff", ".png"))]

            # Si es la primera carpeta, inicializar el conjunto de nombres comunes
            if filenames_set is None:
                filenames_set = set(files)
            else:
                # Intersección para garantizar que solo se usen nombres comunes
                filenames_set = filenames_set.intersection(files)

            # Cargar máscaras
            for file in files:
                if file in filenames_set:
                    mask_path = os.path.join(method_path, file)
                    masks_dict[method][file] = tiff.imread(mask_path) > 0  # Leer y binarizar

    return masks_dict

# Pipeline para comparar todas las combinaciones de métodos
def analyze_all_comparisons(folder_path):
    """
    Compara máscaras generadas por diferentes métodos en subcarpetas.

    Parámetros:
    - folder_path (str): Ruta al directorio principal con subcarpetas.

    Resultados:
    - Devuelve un DataFrame con los resultados comparativos.
    - Genera gráficos de superposición entre métodos para ejemplos seleccionados.
    - Muestra promedios y desviaciones estándar de las métricas.
    """
    masks_dict = load_masks_from_folder(folder_path)
    methods = list(masks_dict.keys())
    common_files = list(next(iter(masks_dict.values())).keys())
    results = []

    # Seleccionar una imagen como ejemplo para graficar superposición
    example_file = common_files[0]

    # Comparar todas las combinaciones de métodos
    for method1, method2 in combinations(methods, 2):
        mask1_example = masks_dict[method1][example_file]
        mask2_example = masks_dict[method2][example_file]

        # Graficar superposición para el ejemplo
        plot_pairwise_overlapping_masks(
            mask1_example,
            mask2_example,
            method1=method1,
            method2=method2,
            filename=f"{method1}_vs_{method2}_example.png"
        )

        for filename in common_files:
            mask1 = masks_dict[method1][filename]
            mask2 = masks_dict[method2][filename]

            # Calcular métricas de similitud
            dice = dice_coefficient(mask1, mask2)
            jaccard = jaccard_index(mask1, mask2)
            acc = accuracy(mask1, mask2)

            # Agregar resultados
            results.append({
                "Filename": filename,
                "Method 1": method1,
                "Method 2": method2,
                "Dice": dice,
                "Jaccard": jaccard,
                "Accuracy": acc
            })

    # Convertir resultados a DataFrame
    results_df = pd.DataFrame(results)

    # Calcular promedios y desviaciones estándar por grupo
    summary = results_df.groupby(["Method 1", "Method 2"]).agg(
        Dice_Mean=("Dice", "mean"),
        Dice_Std=("Dice", "std"),
        Jaccard_Mean=("Jaccard", "mean"),
        Jaccard_Std=("Jaccard", "std"),
        Accuracy_Mean=("Accuracy", "mean"),
        Accuracy_Std=("Accuracy", "std")
    ).reset_index()

    # Mostrar promedios y desviaciones estándar
    print("Summary of Means and Standard Deviations:")
    print(summary)

    return results_df, summary


calcular = False
if calcular:
    if __name__ == "__main__":
        # Ruta a la carpeta principal con subcarpetas: KMeans, MultiOtsu, Phasor
        folder_path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/segmentations/all mask for comparison"

        # Ejecutar el análisis
        results, summary = analyze_all_comparisons(folder_path)

        # Guardar resultados en CSV
        results.to_csv(folder_path + "/all_comparisons_results.csv", index=False)
        summary.to_csv(folder_path + "/all_comparisons_summary.csv", index=False)

        # Mostrar resumen en consola
        print(summary)


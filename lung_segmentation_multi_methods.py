import os
import tifffile as tiff
import numpy as np
from sklearn.cluster import KMeans
from skimage.filters import threshold_multiotsu
from skimage.color import rgb2gray
from tools import rgb2bgr, phasor, median_filter, binarize_images, calculate_mask_areas_with_stats
import matplotlib.pyplot as plt
import tools


# Function Definitions
def multiotsu_segmentation(image, classes=4):
    gray_image = rgb2gray(image)
    gray_image = (gray_image * 255).astype(np.uint8)
    thresholds = threshold_multiotsu(gray_image, classes=classes)
    mask = np.digitize(gray_image, bins=thresholds)
    return mask


def kmeans_segmentation(image, n_clusters=4, random_state=42):
    normalized_image = image / 255.0
    pixels = normalized_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(pixels)
    segmented_image = labels.reshape(image.shape[:2])
    return segmented_image


def segmentation_from_phasor_cluster(real, imag, n_clusters=4, random_state=42):
    x = np.asarray([real.flatten(), imag.flatten()]).transpose()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(x)
    labels = kmeans.predict(x)
    mask = labels.reshape(real.shape)
    return mask


def simple_segment_and_save(input_folder, output_folder, segmentation_method):
    """
    Reads TIFF images from a folder, applies a segmentation method, and saves the results.

    Args:
        input_folder (str): Path to the input folder containing TIFF images.
        output_folder (str): Path to the output folder to save segmented images.
        segmentation_method (str): The segmentation method to apply ('multiotsu', 'kmeans', 'phasor').
    """
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over TIFF files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".tif") or file_name.endswith(".tiff"):
            file_path = os.path.join(input_folder, file_name)
            image = tiff.imread(file_path)

            # Apply the selected segmentation method
            if segmentation_method == 'multiotsu':
                segmented_image = multiotsu_segmentation(image)
            elif segmentation_method == 'kmeans':
                segmented_image = kmeans_segmentation(image)
            elif segmentation_method == 'phasor':
                im = rgb2bgr(image)
                _, real, imag = phasor(im)
                real = median_filter(real, 2)
                imag = median_filter(imag, 2)
                segmented_image = segmentation_from_phasor_cluster(real, imag)
            else:
                raise ValueError(f"Unsupported segmentation method: {segmentation_method}")

            # Save the segmented image
            output_path = os.path.join(output_folder, f"{segmentation_method}_{file_name}")
            tiff.imwrite(output_path, segmented_image.astype(np.uint8))

cal_mask = False
if cal_mask:
    # multiotsu
    simple_segment_and_save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/",
                            "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/multiotsu/nd/", 
                            'multiotsu')
    simple_segment_and_save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/",
                            "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/multiotsu/inst/", 
                            'multiotsu')

    # kmeans nd
    simple_segment_and_save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/",
                            "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/kmeans/nd/", 
                            'kmeans')
    simple_segment_and_save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/",
                            "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/kmeans/inst/", 
                            'kmeans')

    # phasor nd
    simple_segment_and_save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/",
                            "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/", 
                            'phasor')
    simple_segment_and_save("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/",
                            "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/inst/", 
                            'phasor')


# Check image mask
plotty = False
if plotty:
    im = tiff.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/phasor_95-21B6ND10.tif")
    plt.imshow(im)
    plt.show()


# Now binarize mask 
bin_mask = False
if bin_mask:
    binarize_images("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/00/",
                     "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/binary/",
                     background_value=0)
    
    binarize_images("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/01/",
                     "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/binary/",
                     background_value=1)
    
    binarize_images("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/02/",
                     "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/binary/",
                     background_value=2)

    binarize_images("/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/03/",
                        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask/phasor/nd/binary/",
                        background_value=3)
    

# Main folder path
get_areas = False
if get_areas:
    main_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/segmentations/"
    # Calculate areas and create a DataFrame
    df = calculate_mask_areas_with_stats(main_folder)
    # Save the DataFrame to a CSV file (optional)
    df.to_csv(main_folder + "mask_areas_with_stats.csv", index=False)
    # Display the DataFrame
    print(df)

# Comapre the area obtain with 3 methods and plot boxplot
comapre_areas = False
if comapre_areas:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Datos de la tabla
    data = {
        "Method": ["KMeans", "KMeans", "MultiOtsu", "MultiOtsu", "Phasor", "Phasor"],
        "Condition": ["ND", "ND Inst", "ND", "ND Inst", "ND", "ND Inst"],
        "Mean": [34.96, 69.18, 34.81, 69.17, 32.84, 64.99],
        "Std": [5.93, 11.11, 5.99, 11.19, 5.78, 11.23]
    }

    # Convertir a DataFrame
    df = pd.DataFrame(data)

    # Preparar los datos para visualización simulando valores a partir de media y std
    import numpy as np

    plot_data = []
    for _, row in df.iterrows():
        values = np.random.normal(row["Mean"], row["Std"], size=100)  # Generar 100 valores simulados
        plot_data.extend([
            {"Method": row["Method"], "Condition": row["Condition"], "Value": v}
            for v in values
        ])

    plot_df = pd.DataFrame(plot_data)

    # Crear el boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=plot_df,
        x="Condition",
        y="Value",
        hue="Method",
        palette="Set2"
    )

    # Configurar el gráfico
    plt.title("Boxplot for Methods and Conditions", fontsize=16, weight="bold")
    plt.xlabel("Condition", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.legend(title="Method", fontsize=10)
    plt.tight_layout()
    plt.show()
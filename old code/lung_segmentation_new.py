import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from tools import cluster_phasor_plot, rgb2bgr, phasor, median_filter, invert_mask
import tools 
import tifffile
from phasorpy.plot import PhasorPlot
from sklearn.mixture import GaussianMixture


# Parte 0 analisis manual de mascaras
# Parte 1 calculo de mascaras
# Parte 2 invierto las mascaras que estan al reves 
# Parte 3 calculo de areas y estadísticos 
# Parte 4 ploteo de mascara y phasors
# Parte 5 grafico de boxplot y violins

def mask_from_cluster_phasor_kmeans(im, harmonic=1, nclusters=2):
    _, real, imag = phasor(im, harmonic=harmonic)
    #real = median_filter(real, 2)
    #imag = median_filter(imag, 2)
    x = np.asarray([real.flatten(), imag.flatten()]).transpose()

    kmeans = KMeans(n_clusters= nclusters, random_state=42, n_init="auto").fit(x)
    labels = kmeans.predict(x)
    mask = labels.reshape(real.shape)
    return mask

def mask_from_cluster_phasor_gmm(im, harmonic=1, nclusters=2):
    _, real, imag = phasor(im, harmonic=harmonic)
    real = median_filter(real, 5)
    imag = median_filter(imag, 5)

    gmm = GaussianMixture(n_components=nclusters, random_state=42, 
                          covariance_type="tied", init_params='kmeans')
    x = np.asarray([real.flatten(), imag.flatten()]).transpose()
    gmm.fit(x)
    labels = gmm.predict(x)
    mask = labels.reshape(real.shape)
    return mask

# Parte 0 
part0 = True
if part0:
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif"
    # path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/063-21B6ND_INS04.tif"
    im = tifffile.imread(path)
    im = rgb2bgr(im)

    # Aplicar la función
    mask1 = mask_from_cluster_phasor_kmeans(im, nclusters=4)
    mask2 = mask_from_cluster_phasor_gmm(im, nclusters=4)

    plt.figure(1)
    plt.imshow(mask1)
    
    plt.figure(2)
    plt.imshow(mask2)
    plt.show()

# Parte 1
# calculo de mascaras
part1 = False
if part1:
    # Carpeta de entrada y salida
    input_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND_inst/"
    output_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/mask_ND_inst"
    os.makedirs(output_folder, exist_ok=True)

    # Procesar todas las imágenes TIFF
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # Leer la imagen
            filepath = os.path.join(input_folder, filename)
            im = tifffile.imread(filepath)
            im = rgb2bgr(im)
            
            # Aplicar la función
            mask = mask_from_cluster_phasor(im, nclusters=2)
            
            # Guardar la imagen resultante
            path_mask = os.path.join(output_folder, f"mask_{filename}")
            tifffile.imwrite(path_mask, mask.astype(np.uint8))

    print("Procesamiento completo. Archivos guardados en:", output_folder)

# Parte 2
# Invierto las mascaras
part2 = False
if part2:
    input_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/phasor mask/nd_invertir/"
    output_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/phasor mask/mask_ND/"
    os.makedirs(output_folder, exist_ok=True)

    # Procesar todas las imágenes TIFF
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # Leer la imagen
            filepath = os.path.join(input_folder, filename)
            im = tifffile.imread(filepath)
            
            # Aplicar la función
            inv_mask = invert_mask(im)
            
            # Guardar la imagen resultante
            output_path = os.path.join(output_folder, f"{filename}")
            tifffile.imwrite(output_path, inv_mask.astype(np.uint8))  # Guardar como uint8

    print("Procesamiento completo. Archivos guardados en:", output_folder)


# Parte 4
# Ploteo de mascara y phasor 
part4 = False
if part4:
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif"
    # path = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig1/im_rgb.png"
    im = plt.imread(path)
    # im = tools.transform_array(im)
    # im = tools.replace_with_nan(im)
    im = rgb2bgr(im)

    _, real1, imag1 = phasor(im)
    _, real2, imag2 = phasor(im, harmonic=2)

    aux1 = real1 * imag1
    aux2 = real2 * imag2

    coord_r1 = real1.flatten()[~np.isnan(aux1.flatten())]
    coord_i1 = imag1.flatten()[~np.isnan(aux1.flatten())]

    coord_r2 = real2.flatten()[~np.isnan(aux2.flatten())]
    coord_i2 = imag2.flatten()[~np.isnan(aux2.flatten())]

    plot = PhasorPlot(allquadrants=True, title='Phasor plot 1st harmonic')
    plot.hist2d(coord_r1, coord_i1, cmap="RdYlGn_r")

    plot = PhasorPlot(allquadrants=True, title='Phasor plot 2nd harmonic')
    plot.hist2d(coord_r2, coord_i2, cmap="RdYlGn_r")

    x1 = np.asarray([coord_r1, coord_i1]).transpose()
    kmeans = KMeans(n_clusters= 3, random_state=42, n_init="auto").fit(x1)
    labels1 = kmeans.predict(x1)
    cluster_phasor_plot(x1, labels1)

    x2 = np.asarray([coord_r2, coord_i2]).transpose()
    kmeans = KMeans(n_clusters= 3, random_state=42, n_init="auto").fit(x2)
    labels2 = kmeans.predict(x2)
    cluster_phasor_plot(x2, labels2)

    plt.show()


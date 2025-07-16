# RGB sensor simulation with bayer matrix
import numpy as np
import matplotlib.pyplot as plt

def from_bayer_to_rgb(bayer_matrix):
    """
    Convierte una matriz Bayer n x n a una imagen RGB.

    Parámetros:
    bayer_matrix (numpy array): Matriz Bayer con intensidades en el formato Bayer:
                                G R
                                B G

    Retorna:
    numpy array: Imagen RGB de dimensiones (n//2, n//2, 3), con valores normalizados (0-255).
    """
    # Extraer las dimensiones de la matriz Bayer
    n, m = bayer_matrix.shape

    if n % 2 != 0 or m % 2 != 0:
        raise ValueError("La matriz Bayer debe tener dimensiones pares.")

    # Crear arrays para los canales R, G y B
    R = bayer_matrix[0::2, 1::2]  # Valores rojos
    B = bayer_matrix[1::2, 0::2]  # Valores azules
    G = (bayer_matrix[0::2, 0::2] + bayer_matrix[1::2, 1::2]) / 2  # Promedio de valores verdes

    # Construir la imagen RGB
    rgb_image = np.stack([R, G, B], axis=-1)

    # Normalización al rango 0-255
    max_val = np.max(rgb_image)
    if max_val > 0:
        rgb_image = (rgb_image / max_val) * 255

    return rgb_image.astype(np.uint8)


# Part 1 RGB sensor simulation with bayer matrix
part1 = True
if part1:
    # Matriz Bayer 6x6 que genera 3x3 píxeles RGB
    bayer_matrix = np.array([
        [100, 100, 80, 10, 10, 20],  # G R G R G R
        [100, 100, 10, 10, 100, 10],  # B G B G B G
        [0, 100, 95, 15, 10, 100],  # G R G R G R
        [20, 10, 30, 90, 10, 5], # B G B G B G
        [10, 10, 85, 35, 10, 10],  # G R G R G R
        [100, 10, 20, 85, 10, 10]   # B G B G B G
    ])

    # Convertir la matriz Bayer a imagen RGB
    rgb_image = from_bayer_to_rgb(bayer_matrix)

    # Mostrar resultados
    print("Matriz Bayer original:")
    print(bayer_matrix)
    print("Imagen RGB reconstruida (0-255):")
    print(rgb_image)

    # Visualizar la matriz Bayer y la imagen RGB reconstruida
    plt.figure(figsize=(8, 4))

    # Mostrar la matriz Bayer
    plt.subplot(1, 2, 1)
    plt.imshow(bayer_matrix, cmap='gray', interpolation='nearest')
    plt.colorbar(label='Intensidad')
    plt.title("Matriz Bayer")
    plt.axis('off')

    # Mostrar la imagen RGB reconstruida
    plt.subplot(1, 2, 2)
    plt.imshow(rgb_image)
    plt.title("Imagen RGB reconstruida")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Part 2 read raw image and convert to RGB

part2 = False
if part2:
    # Leer la imagen RAW
    raw_image = plt.imread(
        '/Users/schutyb/Documents/Projects/rgb-phasors/data/raw imaging/uprocessed.tif')
    
    # Dimensiones de la imagen RAW
    # n = raw_image.shape[0]
    # m = raw_image.shape[1]

    rgb_image = from_bayer_to_rgb(raw_image)
    
    # Visualizar la imagen RGB
    plt.imshow(raw_image)
    plt.title("Imagen RGB desde matriz Bayer")
    plt.axis('off')
    plt.show()
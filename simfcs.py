import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np


# Ruta a tu imagen .tif
ruta1 = "/Users/schutyb/Downloads/roi13.tif"
ruta2 = "/Users/schutyb/Downloads/roi13_stack.tif"

# Leer la imagen
imagen1 = tiff.imread(ruta1)
imagen2 = tiff.imread(ruta2)

# Mostrar las dimensiones
print("Dimensiones de la imagen 1:", imagen1.shape)
print("Dimensiones de la imagen 2:", imagen2.shape)

plt.figure()
plt.imshow(imagen1[1])

plt.figure()
plt.imshow(imagen1[1])

plt.show()
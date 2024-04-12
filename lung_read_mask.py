import numpy as np
import matplotlib.pyplot as plt


mask1 = plt.imread("/Users/schutyb/Downloads/46-21B6ND01.tif")
aux = np.sum(mask1, axis=-1)
hist_min = np.histogram(aux)[0][0]  # obtengo el porcentaje de ceros en la mascara 
tissue_area = hist_min / (mask1.shape[0] * mask1.shape[1])

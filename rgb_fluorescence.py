import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tools
import tifffile
import pandas as pd


"""
1 - Simulation of a RGB file with 3 spectral components, use Tamara simulators, and 
simulate three components like dapi, mitotracker and cell mask.
2 - Then create an RGB image, with different percetatge of each component in each pixel
then apply Alex methods to determine each component
3 - 
"""

# -------
# Part 1
# -------

# leer los txt de los espectos que descargue 
# formar una imagen con combinaciones lineales de esos espectros
# hacer el unmixing de Alex con las coordenadas de los espctros puros
# crear una imagen con los valores del unmixing. 

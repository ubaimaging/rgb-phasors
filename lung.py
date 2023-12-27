import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tools
import os
import hsipy

# Part 1: read all the files and obtain the cluster
# Part 2: segmentate all images and creat mask 
# Part 3: calculates the areas with the mask and stats analysis

# ------------
#    Part 1
# ------------

path = '/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/all/'
name = sorted(os.listdir(path))


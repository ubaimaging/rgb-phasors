import numpy as np
import matplotlib.pyplot as plt
import os


part1 = False
if part1:
    type = "default_dark"  # el 1 corresponde al tejido 
    # type = "otsu"
    # type = "mean"

    path1 = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/fiji mask/" + type + "/B6ND-INST_mask/"
    path2 = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/fiji mask/" + type + "/b6nd-10_mask/"

    path = [path1, path2]
    for j in range(2):
        name = sorted(os.listdir(path[j]))[1:]
        per = np.zeros(len(name))
        if type == "default_dark":
            for i in range(len(name)):
                mask = plt.imread(path[j] + name[i])
                aux = ((np.sum(mask, axis=-1) - 255) / (3*255) - 1) * (-1)
                per[i] = np.sum(aux) / (mask.shape[0] * mask.shape[1])
        else:
            for i in range(len(name)):
                mask = plt.imread(path[j] + name[i])
                aux = (np.sum(mask, axis=-1) - 255) / (3*255)
                per[i] = np.sum(aux) / (mask.shape[0] * mask.shape[1])
        per_avg = np.mean(per)
        per_std = np.std(per)
        print("Type:", type, "tissue area", per_avg, "standar deviation", per_std)


# Part 2 Compare the 4 methods for segmentation
# Using the Jaccard coefficient
part2 = True
if part2:

    def jaccard_coefficient(seg1, seg2, seg3, seg4):
        # Calcula la intersección entre los cuatro conjuntos
        intersection = np.sum(np.logical_and(seg1, np.logical_and(seg2, np.logical_and(seg3, seg4))))
        
        # Calcula la unión entre los cuatro conjuntos
        union = np.sum(np.logical_or(seg1, np.logical_or(seg2, np.logical_or(seg3, seg4))))
        
        # Calcula el coeficiente de Jaccard
        jaccard = intersection / union
        return jaccard

    path_mean = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/fiji mask/mean/B6ND-INST_mask/"
    name_mean = sorted(os.listdir(path_mean))[1:]

    path_dd = "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/fiji mask/default_dark/B6ND-INST_mask/"
    name_dd = sorted(os.listdir(path_dd))[1:]
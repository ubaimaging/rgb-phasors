import numpy as np
import matplotlib.pyplot as plt
import os


part1 = True
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


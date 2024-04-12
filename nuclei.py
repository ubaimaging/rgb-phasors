# Nuclei segmentation

import numpy as np
import tools
import matplotlib.pyplot as plt

nuclei = True
if nuclei:
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/nuclei/nuclei_rgb/"
    rgb1 = tools.rgb2bgr(plt.imread(path + "01_1.png"))
    bgr = tools.rgb2bgr(rgb1)
    dc1, g1, s1 = tools.phasor(rgb1)
    try1 = True
    if try1:
        for i in range(2):
            tools.interactive2(dc1, g1, s1, 0.1, nbit=8, filter=3)
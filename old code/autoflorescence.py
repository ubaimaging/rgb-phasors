# Autoflorescence and RGB phasors

import numpy as np
import tools
import matplotlib.pyplot as plt


liver = False
if liver:
    path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/higados/"
    rgb1 = tools.rgb2bgr(plt.imread(path + "Higado_ApoE_HFD_r4 (2).tif"))
    rgb2 = tools.rgb2bgr(plt.imread(path + "ApoE_ND_r13.tif"))
    rgb3 = tools.rgb2bgr(plt.imread(path + "higado_B6_HFD_r21.tif"))
    rgb4 = tools.rgb2bgr(plt.imread(path + "B6_ND_r1.tif"))

    dc1, g1, s1 = tools.phasor(rgb1)
    dc2, g2, s2 = tools.phasor(rgb2)
    dc3, g3, s3 = tools.phasor(rgb3)
    dc4, g4, s4 = tools.phasor(rgb4)

    tile = False
    if tile:
        im = np.zeros([3, 2 * rgb1.shape[1], 2 * rgb1.shape[2]])

        im[:, 0:rgb1.shape[1], 0:rgb1.shape[2]] = rgb1
        im[:, 0:rgb1.shape[1], rgb1.shape[2]:2*rgb1.shape[2]] = rgb2
        im[:, rgb1.shape[1]:2*rgb1[1], 0:rgb1.shape[2]] = rgb3
        im[:, rgb1.shape[1]:2*rgb1[1], rgb1.shape[2]:2*rgb1.shape[2]] = rgb4

        dc, g, s = tools.phasor(im)

    # Visualization
    try1 = True
    if try1:
        for i in range(2):
            tools.interactive2(dc1, g1, s1, 0.1, nbit=8, filter=3)
    


skin = True
if skin:
    rgbn = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig3/nev-mel/16252_10x_r1.tif")
    rgbm = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig3/nev-mel/18852_10x_r1.tif")

    bgrn = tools.rgb2bgr(rgbn)
    dcn, gn, sn = tools.phasor(bgrn)
    bgrm = tools.rgb2bgr(rgbm)
    dcm, gm, sm = tools.phasor(bgrm)

    # threshold each image separatly
    thresholding = True
    if thresholding:
        dcn = np.where(dcn > 80, dcn, np.zeros(dcn.shape))
        dcm = np.where(dcm > 30, dcm, np.zeros(dcm.shape))

    g = np.zeros([gn.shape[0], gn.shape[1] * 2])
    g[0:gn.shape[0], 0:gn.shape[1]] = gn
    g[0:gn.shape[0], gn.shape[1]:] = gm

    s = np.zeros([sn.shape[0], sn.shape[1] * 2])
    s[0:sn.shape[0], 0:sn.shape[1]] = sn
    s[0:sn.shape[0], sn.shape[1]:] = sm

    dc = np.zeros([dcn.shape[0], dcn.shape[1] * 2])
    dc[0:dcn.shape[0], 0:dcn.shape[1]] = dcn
    dc[0:dcn.shape[0], dcn.shape[1]:] = dcm

    # Visualization
    try1 = True
    if try1:
        for i in range(1):
            tools.interactive2(dcn, gn, sn, 0.15, nbit=8, filter=3)



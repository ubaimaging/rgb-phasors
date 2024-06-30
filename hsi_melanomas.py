import tools 
import tifffile
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy

imn = tifffile.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/hsi-nev-mel/sp_16556_r2.lsm")
imm = tifffile.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/hsi-nev-mel/sp_18852_r2.lsm")
dcm, gm, sm,  = tools.phasor(imm)
dcn, gn, sn,  = tools.phasor(imn)

dcm = numpy.rot90(dcm, k=-1)
gm = numpy.rot90(gm, k=-1)
sm = numpy.rot90(sm, k=-1)
dcn = numpy.rot90(dcn)
gn = numpy.rot90(gn)
sn = numpy.rot90(sn)

# Ploteo dc y phasor
plt.figure(1)
plt.imshow(dcm, cmap="gray")
plt.title("Melanoma")
maskm = dcm > 5
gm = tools.median_filter(gm, 5)
sm = tools.median_filter(sm, 5)

plt.figure(3)
plt.imshow(dcn, cmap="gray")
plt.title("Nevo")
maskn = dcn > 5
gn = tools.median_filter(gn, 5)
sn = tools.median_filter(sn, 5)

centers = [[-0.08, 0.6], [-0.26, 0.36], [-0.4, 0.11]]
r = 0.15

rgbn = tools.rgb_coloring(dcn, gn, sn, 2, centers, r)
plt.figure()
plt.imshow(rgbn)

rgbm = tools.rgb_coloring(dcm, gm, sm, 2, centers, r)
plt.figure()
plt.imshow(rgbm)

fign, axn = plt.subplots(1, 1, figsize=(6, 6))
axn = tools.phasor_figure(gn[maskn], sn[maskn], circle_plot=True)
ccolor = ['blue', 'green', 'red']
for i in range(3):
    circle = plt.Circle((centers[i][0], centers[i][1]), r, color=ccolor[i], fill=False, linewidth=3)
    axn.add_patch(circle)


figm, axm = plt.subplots(1, 1, figsize=(6, 6))
axm = tools.phasor_figure(gm[maskm], sm[maskm], circle_plot=True)
ccolor = ['blue', 'green', 'red']
for i in range(3):
    circle = plt.Circle((centers[i][0], centers[i][1]), r, color=ccolor[i], fill=False, linewidth=3)
    axm.add_patch(circle)

plt.show()


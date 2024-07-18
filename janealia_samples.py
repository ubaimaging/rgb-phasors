import tools 
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy
from PIL import Image 
import tifffile


resto = False
if resto:    
    # image = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelias_samples/sample 60x values 111.tif")
    image = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelias_samples/rois/mel_roi1.tif")

    im = tools.rgb2bgr(image)

    #im = Image.fromarray(image)
    #im.save("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelias_samples/rois/mel_roi2.tif")

    dc, g, s,  = tools.phasor(im)

    # Ploteo dc y phasor
    plt.figure(1)
    # plt.imshow(dc, cmap="gray")
    plt.imshow(image)
    plt.title("Average image")

    plt.figure(2)
    plt.hist(dc.flatten(), bins=256, range=(0, 256))

    mask = dc > 0
    g = tools.median_filter(g, 3)
    s = tools.median_filter(s, 3)

    centers = [[-0.2, 0.53], [-0.07, 0.22], [0.06, -0.02]]
    centers = [[0.2, -0.28], [-0.15, 0.43], [-0.003, 0.03]]
    r = 0.2

    g = numpy.where(mask, g, numpy.zeros(g.shape))
    s = numpy.where(mask, s, numpy.zeros(s.shape))

    rgb = tools.rgb_coloring(dc, g, s, 15, centers, r)
    plt.figure(3)
    plt.imshow(rgb)

    fign, axn = plt.subplots(1, 1, figsize=(6, 6))
    axn = tools.phasor_figure(g[mask], s[mask], circle_plot=True)
    ccolor = ['blue', 'green', 'red']
    for i in range(3):
        circle = plt.Circle((centers[i][0], centers[i][1]), r, color=ccolor[i], fill=False, linewidth=3)
        axn.add_patch(circle)

    plt.show()


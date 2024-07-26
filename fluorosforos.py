import tools 

import numpy as np
import matplotlib.pyplot as plt
import phasorpy

from phasorpy.cursors import (
    mask_from_circular_cursor,
    pseudo_color,
)
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal


path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/fluorosforos/"

parte1 = True
if parte1:
    imb = plt.imread(path + "f_r6g_13.tif")
    blue = tools.rgb2bgr(imb)
    dcb, gb, sb,  = tools.phasor(blue)
    plt.figure()
    plt.title("Blue")
    plt.imshow(imb)
    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(gb.flatten(), sb.flatten(), cmap="RdYlGn_r")

    aga = False
    if aga:
        img = plt.imread(path + "f+r6g_12b.tif")
        green = tools.rgb2bgr(img)
        dcg, gg, sg,  = tools.phasor(green)
        plt.figure()
        plt.title("Green")
        plt.imshow(img)
        plot = PhasorPlot(allquadrants=True, title='Phasor plot')
        plot.hist2d(gg.flatten(), sg.flatten(), cmap="RdYlGn_r")

        imr = plt.imread(path + "fluoresceina_frgb.tif")
        red = tools.rgb2bgr(imr)
        dcr, gr, sr,  = tools.phasor(red)
        plt.figure()
        plt.title("Red")
        plt.imshow(imr)
        plot = PhasorPlot(allquadrants=True, title='Phasor plot')
        plot.hist2d(gr.flatten(), sr.flatten(), cmap="RdYlGn_r")


        mix = plt.imread(path + "rho6g_frgb.tif")
        mix2 = tools.rgb2bgr(mix)
        dcm, gm, sm,  = tools.phasor(mix2)
        plt.figure()
        plt.title("Blue")
        plt.imshow(mix)
        plot = PhasorPlot(allquadrants=True, title='Phasor plot')
        plot.hist2d(gm.flatten(), sm.flatten(), cmap="RdYlGn_r")

        mix = plt.imread(path + "rho110_frgb.tif")
        mix2 = tools.rgb2bgr(mix)
        dcm, gm, sm,  = tools.phasor(mix2)
        plt.figure()
        plt.title("Blue")
        plt.imshow(mix)
        plot = PhasorPlot(allquadrants=True, title='Phasor plot')
        plot.hist2d(gm.flatten(), sm.flatten(), cmap="RdYlGn_r")

    plt.show()
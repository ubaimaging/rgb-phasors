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


path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/acetatos/25-05/"

# --------------------------
# Parte 1 
# --------------------------

# Lectura de los casos base RGB puros y de la luz basal
parte1 = True
if parte1:
    # basal = plt.imread(path + "acetato_pure_blue.tif")
    basal = 0

    imb = plt.imread(path + "acetato_pure_blue.tif") + basal
    blue = tools.rgb2bgr(imb)
    dcb, gb, sb,  = tools.phasor(blue)
    plt.figure()
    plt.title("Blue")
    plt.imshow(imb)
    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(gb.flatten(), sb.flatten(), cmap="RdYlGn_r")


    img = plt.imread(path + "acetato_pure_green.tif") + basal
    green = tools.rgb2bgr(img)
    dcg, gg, sg,  = tools.phasor(green)
    plt.figure()
    plt.title("Green")
    plt.imshow(img)
    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(gg.flatten(), sg.flatten(), cmap="RdYlGn_r")

    imr = plt.imread(path + "acetato_pure_red.tif") + basal
    red = tools.rgb2bgr(imr)
    dcr, gr, sr,  = tools.phasor(red)
    plt.figure()
    plt.title("Red")
    plt.imshow(imr)
    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(gr.flatten(), sr.flatten(), cmap="RdYlGn_r")


    mix = plt.imread(path + "acetato_r+b.tif")
    mix2 = tools.rgb2bgr(mix)
    dcm, gm, sm,  = tools.phasor(mix2)
    plt.figure()
    plt.title("Blue")
    plt.imshow(mix)
    plot = PhasorPlot(allquadrants=True, title='Phasor plot')
    plot.hist2d(gm.flatten(), sm.flatten(), cmap="RdYlGn_r")

    plt.show()

# -----------------------
# Parte 2
# -----------------------
# Mezcla de acetatos con diferentes cantidades

parte2 = False
if parte2: 
    # leer imagenes
    # 1 - Sacar los cm de las componentes puras
    # 2 - Crear la matriz A de componentes 

    if False:
        # leo los 3 componentes puros 
        imb = plt.imread(path + "blue")
        imb = tools.rgb2bgr(imb)
        img = plt.imread(path + "blue")
        img = tools.rgb2bgr(img)
        imr = plt.imread(path + "blue")
        imr = tools.rgb2bgr(imr)

        _, gb, sg = phasor_from_signal(imb)
        _, gg, sg = phasor_from_signal(img)
        _, gr, sr = phasor_from_signal(imr)
        imb = plt.imread(path + "acetato_pure_blue.tif")

    # centros de masa






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


basal = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/acetatos/acetato_camera_light_10x_2024-05-23T16-39-19.485.tif")
basal = 0

imb = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/acetatos/acetato_blue_more_int.tif") + basal
blue = tools.rgb2bgr(imb)
dcb, gb, sb,  = tools.phasor(blue)
plt.figure()
plt.title("Blue")
plt.imshow(imb)
plot = PhasorPlot(allquadrants=True, title='Phasor plot')
plot.hist2d(gb.flatten(), sb.flatten(), cmap="RdYlGn_r")


img = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/acetatos/acetato_green_more_int.tif") + basal
green = tools.rgb2bgr(img)
dcg, gg, sg,  = tools.phasor(img)
plt.figure()
plt.title("Green")
plt.imshow(img)
plot = PhasorPlot(allquadrants=True, title='Phasor plot')
plot.hist2d(gg.flatten(), sg.flatten(), cmap="RdYlGn_r")

imr = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/acetatos/acetato_red_more_int.tif") + basal
red = tools.rgb2bgr(imr)
dcr, gr, sr,  = tools.phasor(red)
plt.figure()
plt.title("Red")
plt.imshow(imr)
plot = PhasorPlot(allquadrants=True, title='Phasor plot')
plot.hist2d(gr.flatten(), sr.flatten(), cmap="RdYlGn_r")

plt.show()




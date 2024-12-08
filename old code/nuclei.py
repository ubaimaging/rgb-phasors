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


path = "/Users/schutyb/Documents/Projects/rgb-phasors/data/nuclei/he_nuclei/"

im = plt.imread(path + "riñon_60x_02.tif")
bgr = tools.rgb2bgr(im)
dc, g, s,  = tools.phasor(bgr)
plt.figure()
plt.title("Blue")
plt.imshow(im)
plot = PhasorPlot(allquadrants=True, title='Phasor plot')
plot.hist2d(g.flatten(), s.flatten(), cmap="RdYlGn_r")

cursors_real = [0.17, 0, -0.03]
cursors_imag = [0, -0.15, 0.07]

cursors_real = [0.16, -0.03, -0.06]
cursors_imag = [0, -0.15, 0.07]
r = 0.13
circular_mask = mask_from_circular_cursor(
    g, s, cursors_real, cursors_imag, radius=r)

# Plot cursors Blue, Green, Red
plot.cursor(
    cursors_real[0],
    cursors_imag[0],
    radius=r,
    color=CATEGORICAL[1],
    linestyle='-',
)

plot.cursor(
    cursors_real[1],
    cursors_imag[1],
    radius=r,
    color=CATEGORICAL[2],
    linestyle='-',
)

plot.cursor(
    cursors_real[2],
    cursors_imag[2],
    radius=r,
    color=CATEGORICAL[0],
    linestyle='-',
)
bgr = [CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[0]] 
segmented_image = pseudo_color(dc, circular_mask, colors=bgr)

fig, ax = plt.subplots()
ax.set_title('Segmented image with circular cursors')
ax.imshow(segmented_image)
plt.show()
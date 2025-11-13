import numpy as np
import matplotlib.pyplot as plt
import tools
from tools import phasor, cluster_phasor_plot, rgb2bgr, map_to_rgb

from phasorpy.cursors import mask_from_circular_cursor
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.phasor import phasor_from_signal


im = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/paper/extra_figure/manzanas.jpg")

plt.figure()
plt.imshow(im)
plt.axis("off")

avg, g, s = phasor_from_signal(im)

cursors = True
if cursors:
    cursors_real = [0.7, 0.39, 0.1]
    cursors_imag = [0.0, 0.2, 0.03]

    plot = PhasorPlot(allquadrants=True, title='')
    plot.hist2d(g.flatten(), s.flatten(), cmap="RdYlGn_r")
    fig4 = plot.fig
    fig4 = plot.fig
    plot.fig.set_size_inches(5, 5) 
    plot.ax.set_aspect('equal')  

    # Plot cursors Blue, Green, Red
    plot.cursor(
        cursors_real[0],
        cursors_imag[0],
        radius=0.2,
        color=CATEGORICAL[1],
        linestyle='-',
    )

    plot.cursor(
        cursors_real[1],
        cursors_imag[1],
        radius=0.2,
        color=CATEGORICAL[2],
        linestyle='-',
    )

    plot.cursor(
        cursors_real[2],
        cursors_imag[2],
        radius=0.2,
        color=CATEGORICAL[0],
        linestyle='-',
    )

    cursors_mask = mask_from_circular_cursor(
        g, s, cursors_real, cursors_imag, radius=0.5)
    
    auxmask = np.transpose(cursors_mask, (1, 2, 0)).astype(int)

    auxx = map_to_rgb(auxmask)

    fig7 = plt.figure(figsize=(5, 5))
    plt.imshow(auxx)
    plt.axis("off")

    img_rgb = np.transpose(cursors_mask, (1, 2, 0))

    # Plotear
    plt.figure()
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title('RGB Image')
    plt.show()

plt.show()
import tools 
import tifffile
import matplotlib.pyplot as plt


# imb = tifffile.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/acetatos/acetato_blue_more_int.tif")
blue = tools.rgb2bgr(plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/acetatos/acetato_blue_more_int.tif"))
dc, g, s,  = tools.phasor(blue)
# tools.interactive(dc, g, s, 0.1, 8, filter=2)
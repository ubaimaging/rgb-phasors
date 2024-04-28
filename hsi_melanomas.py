import tools 
import tifffile
import matplotlib.pyplot as plt


im = tifffile.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/hsi-nev-mel/sp_16556_r1.lsm")
dc, g, s,  = tools.phasor(im)
tools.interactive(dc, g, s, 0.1, 8, filter=3)

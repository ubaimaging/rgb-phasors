import tools 
import tifffile
import matplotlib.pyplot as plt


im = tifffile.imread("/Users/schutyb/Documents/fotos Daniela/SP_xxx_r3.lsm")
dc, _, _ = tools.phasor(im)

plt.imshow(dc, cmap="Spectral")
plt.show()
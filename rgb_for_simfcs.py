import numpy 
import matplotlib.pyplot as plt
import tools
import tifffile as tiff



image = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelia 2025/b/b_roi1.tif")
image = plt.imread('/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelia 2025/g/g_roi4.tif')
image = plt.imread('/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelia 2025/r/r_roi1.tif')
image = plt.imread('/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelia 2025/rgb/roi9.tif')



bgr = tools.rgb2bgr(image)[:, 500:1524, 500:1524]

tiff.imwrite(
    '/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/janelia 2025/cut_for_simfcs/roi9.tif',
    bgr)


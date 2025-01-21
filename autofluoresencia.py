import numpy
import matplotlib.pyplot as plt
import tools
from phasorpy.plot import PhasorPlot
from phasorpy.color import CATEGORICAL
from phasorpy.cursors import mask_from_circular_cursor
import tifffile


# Part 1 is for to RGB imaging
part1 = False
if part1:
    # Open the RGB images 
    imm = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/nev-mel/18852_10x_r1.tif")
    imn = plt.imread("/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/nev-mel/16252_10x_r1.tif")

    # Plot RGB original and unmixed channels
    plt.figure()
    plt.imshow(imm)
    plt.axis("off")

    plt.figure()
    plt.imshow(imn)
    plt.axis("off")

    rgbm = tools.rgb2bgr(imm)
    rgbn = tools.rgb2bgr(imn)

    # Unmixed channels
    plt.figure()
    plt.imshow(rgbm[0], cmap="Blues")
    plt.axis("off")
    plt.figure()
    plt.imshow(rgbm[1], cmap="Greens")
    plt.axis("off")
    plt.figure()
    plt.imshow(rgbm[2], cmap="Reds")
    plt.axis("off")

    plt.figure()
    plt.imshow(rgbn[0], cmap="Blues")
    plt.axis("off")
    plt.figure()
    plt.imshow(rgbn[1], cmap="Greens")
    plt.axis("off")
    plt.figure()
    plt.imshow(rgbn[2], cmap="Reds")
    plt.axis("off")

    # calculate the phasor
    dcm, gm, sm = tools.phasor(rgbm)
    dcn, gn, sn = tools.phasor(rgbn)

    # Threshold
    thresholdm = 40
    thresholdn = 60
    gm = numpy.where(dcm > thresholdm, gm, numpy.NaN)
    sm = numpy.where(dcm > thresholdm, sm, numpy.NaN)
    gn = numpy.where(dcn > thresholdn, gn, numpy.NaN)
    sn = numpy.where(dcn > thresholdn, sn, numpy.NaN)

    # median filter
    gm = tools.median_filter(gm, 2)
    sm = tools.median_filter(sm, 2)
    gn = tools.median_filter(gn, 2)
    sn = tools.median_filter(sn, 2)

    # PLOTs
    # intensity
    plt.figure()
    plt.imshow(dcm, cmap="gray")
    plt.axis('off')
    plt.figure()
    plt.imshow(dcn, cmap="gray")
    plt.axis('off')

    # Phasor Plot
    # Add 3 cursors and create pseudocolor
    cursors_real = [0.4, 0.1, -0.21]
    cursors_imag = [0.22, 0.15, 0.1]
    r = 0.15 # radius

    plot1 = PhasorPlot(allquadrants=True, title='Phasor plot melanoma')
    plot1.hist2d(gm.flatten(), sm.flatten(), cmap="RdYlGn_r")
    # Plot cursors Blue, Green, Red
    plot1.cursor(
        cursors_real[0],
        cursors_imag[0],
        radius=r,
        color=CATEGORICAL[1],
        linestyle='-',
    )

    plot1.cursor(
        cursors_real[1],
        cursors_imag[1],
        radius=r,
        color=CATEGORICAL[2],
        linestyle='-',
    )

    plot1.cursor(
        cursors_real[2],
        cursors_imag[2],
        radius=r,
        color=CATEGORICAL[0],
        linestyle='-',
    )

    plot2 = PhasorPlot(allquadrants=True, title='Phasor plot nevus')
    plot2.hist2d(gn.flatten(), sn.flatten(), cmap="RdYlGn_r")
    # Plot cursors Blue, Green, Red
    plot2.cursor(
        cursors_real[0],
        cursors_imag[0],
        radius=r,
        color=CATEGORICAL[1],
        linestyle='-',
    )

    plot2.cursor(
        cursors_real[1],
        cursors_imag[1],
        radius=r,
        color=CATEGORICAL[2],
        linestyle='-',
    )

    plot2.cursor(
        cursors_real[2],
        cursors_imag[2],
        radius=r,
        color=CATEGORICAL[0],
        linestyle='-',
    )

    # pseudocolor melanoma
    cursors_mask = mask_from_circular_cursor(gm, sm, cursors_real, cursors_imag, radius=0.15)
    auxmask = numpy.transpose(cursors_mask, (1, 2, 0)).astype(int)
    auxm = tools.map_to_rgb(auxmask)

    # pseudocolor nevo
    cursors_mask = mask_from_circular_cursor(gn, sn, cursors_real, cursors_imag, radius=0.15)
    auxmask = numpy.transpose(cursors_mask, (1, 2, 0)).astype(int)
    auxn = tools.map_to_rgb(auxmask)

    plt.figure(figsize=(6, 6))
    plt.imshow(auxm)
    plt.title("Pseudocolor melanoma")

    plt.figure(figsize=(6, 6))
    plt.imshow(auxn)
    plt.title("Pseudocolor nevus")
    
    # plt.show()


# Part 2 is for HSI imaging
part2 = True
if part2:
    imn = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/sp/sp_16556_r2.lsm")
    imm = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/autofluorescencia/sp/sp_18852_r1.lsm")
    
    dcm, gm, sm,  = tools.phasor(imm)
    dcn, gn, sn,  = tools.phasor(imn)

    dcm = numpy.rot90(dcm, k=-1)[:850, :970]
    gm = numpy.rot90(gm, k=-1)[:850, :970]
    sm = numpy.rot90(sm, k=-1)[:850, :970]
    dcn = numpy.rot90(dcn)[:850, :970]
    gn = numpy.rot90(gn)[:850, :970]
    sn = numpy.rot90(sn)[:850, :970]

    # median filter
    n = 3
    gm = tools.median_filter(gm, n)
    sm = tools.median_filter(sm, n)
    gn = tools.median_filter(gn, n)
    sn = tools.median_filter(sn, n)

    t = 5
    gm = numpy.where(dcm > t, gm, numpy.NaN)
    sm = numpy.where(dcm > t, sm, numpy.NaN)
    gn = numpy.where(dcn > t, gn, numpy.NaN)
    sn = numpy.where(dcn > t, sn, numpy.NaN)

    # PLOTs
    # intensity
    plt.figure()
    plt.imshow(dcm, cmap="gray")
    plt.axis('off')
    plt.figure()
    plt.imshow(dcn, cmap="gray")
    plt.axis('off')

    # Phasor Plot
    # Add 3 cursors and create pseudocolor
    cursors_real = [-0.05, -0.27, -0.41]
    cursors_imag = [0.55, 0.34, 0.06]
    r = 0.165 # radius

    plot1 = PhasorPlot(allquadrants=True, title='Phasor plot melanoma')
    plot1.hist2d(gm, sm, cmap="RdYlGn_r")
    # Plot cursors Blue, Green, Red
    plot1.cursor(
        cursors_real[0],
        cursors_imag[0],
        radius=r,
        color=CATEGORICAL[1],
        linestyle='-',
    )

    plot1.cursor(
        cursors_real[1],
        cursors_imag[1],
        radius=r,
        color=CATEGORICAL[2],
        linestyle='-',
    )

    plot1.cursor(
        cursors_real[2],
        cursors_imag[2],
        radius=r,
        color=CATEGORICAL[0],
        linestyle='-',
    )

    plot2 = PhasorPlot(allquadrants=True, title='Phasor plot nevus')
    plot2.hist2d(gn, sn, cmap="RdYlGn_r")
    # Plot cursors Blue, Green, Red
    plot2.cursor(
        cursors_real[0],
        cursors_imag[0],
        radius=r,
        color=CATEGORICAL[1],
        linestyle='-',
    )

    plot2.cursor(
        cursors_real[1],
        cursors_imag[1],
        radius=r,
        color=CATEGORICAL[2],
        linestyle='-',
    )

    plot2.cursor(
        cursors_real[2],
        cursors_imag[2],
        radius=r,
        color=CATEGORICAL[0],
        linestyle='-',
    )

    # pseudocolor melanoma
    cursors_mask = mask_from_circular_cursor(gm, sm, cursors_real, cursors_imag, radius=r)
    auxmask = numpy.transpose(cursors_mask, (1, 2, 0)).astype(int)
    auxm = tools.map_to_rgb(auxmask)

    # pseudocolor nevo
    cursors_mask = mask_from_circular_cursor(gn, sn, cursors_real, cursors_imag, radius=r)
    auxmask = numpy.transpose(cursors_mask, (1, 2, 0)).astype(int)
    auxn = tools.map_to_rgb(auxmask)

    plt.figure(figsize=(6, 6))
    plt.imshow(auxm)
    plt.title("Pseudocolor melanoma")

    plt.figure(figsize=(6, 6))
    plt.imshow(auxn)
    plt.title("Pseudocolor nevus")
    plt.show()

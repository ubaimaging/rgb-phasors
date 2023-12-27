import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tools
import tifffile
import pandas as pd


# -------
# Part 1
# -------

# leer los txt de los espectos que descargue 
# formar una imagen con combinaciones lineales de esos espectros
# hacer el unmixing de Alex con las coordenadas de los espctros puros
# crear una imagen con los valores del unmixing.

# leo los componentes descargados de https://www.chroma.com/spectra-viewer
components = False 
if components:
    dapi = pd.read_csv("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/components/dapi_sp.txt", sep="\t")
    mito = pd.read_csv("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/components/mitotraker_orange_sp.txt", sep="\t")
    alexa = pd.read_csv("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/components/alexa_594_sp.txt", sep="\t")

    x = np.arange(350, 760)
    d = np.zeros(410)
    d[25:250] = dapi.i
    m = np.zeros(410)
    m[185:351] = mito.i
    a = np.zeros(410)
    a[230:401] = alexa.i

    # armar un data frame con x, d, m, a
    data = {'w': x, 'dapi': d, 'mito': m, 'alexa': a}
    df = pd.DataFrame(data=data)
    df.to_csv("components.csv", index=False)

plotty = False
if plotty:
    data = pd.read_csv("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/components/components.csv", sep=",")
    plt.figure(1)
    plt.plot(data.w, data.dapi, "b", label="dapi") 
    plt.plot(data.w, data.mito, "orange", label="mitrotraker orange") 
    plt.plot(data.w, data.alexa, "r", label="alexa 594") 
    plt.legend()
    plt.ylim([0.1, 1.1])
    plt.show()

# Crear la imagen con CL de los componentes puros
plotty2 = False
if plotty2:
    # hay que reducir el espectro a 3 componentes que contienen BGR entonces podeos suamr franjas 
    # en el proceso de medida el sensor esta sumando fotones en un rango de long de onda
    # defino tres rangos para sumar los valores 350-460, 460-580, 580-760
    # los tres puntos en x son  460, 580, 620
    data = pd.read_csv("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/components/components.csv", sep=",")
    dapi = np.zeros(3)
    dapi[0] = np.sum(data.dapi[0:110])
    dapi[1] = np.sum(data.dapi[111:231])
    dapi[2] = np.sum(data.dapi[232:len(data.dapi)])

    mito = np.zeros(3)
    mito[0] = np.sum(data.mito[0:110])
    mito[1] = np.sum(data.mito[111:231])
    mito[2] = np.sum(data.mito[232:len(data.alexa)])

    alexa = np.zeros(3)
    alexa[0] = np.sum(data.alexa[0:110])
    alexa[1] = np.sum(data.alexa[111:231])
    alexa[2] = np.sum(data.alexa[232:len(data.alexa)])
    x = np.asarray([460, 580, 620])

    # calculo los phasor de los espectros RGB puros
    phd = tools.phasor(dapi)
    phm = tools.phasor(mito)
    pha = tools.phasor(alexa)

    # Ploteo los tres componentes en RGB y el phasor de cada uno de ellos
    plt.figure(1)
    plt.stem(x, dapi, "b", label="dapi") 
    plt.stem(x + 2, mito, "g", label="mitrotraker orange") 
    plt.stem(x + 4, alexa, "r", label="alexa 594") 
    plt.legend()

    fig, ax = plt.subplots()
    circle1 = plt.Circle((phd[1], phd[2]), 0.05, color='b')
    circle2 = plt.Circle((phm[1], phm[2]), 0.05, color='g')
    circle3 = plt.Circle((pha[1], pha[2]), 0.05, color='r')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    tools.phasor_circle(ax)
    plt.show()

# Leo los datos de los tres espectros y armo comb lineales con los tres espectros puros
data = pd.read_csv("/Users/schutyb/Documents/Projects/rgb-phasors/data/fluoresence/components/components.csv", sep=",")

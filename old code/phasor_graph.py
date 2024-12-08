import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from tools import rgb2bgr, phasor, median_filter
import tifffile
import matplotlib.pyplot as plt

# Función para calcular la diferencia angular
def angular_difference(phase1, phase2):
    delta_phase = np.abs(phase1 - phase2)
    return np.minimum(delta_phase, 2 * np.pi - delta_phase)

# Función para calcular la probabilidad de conexión
def connection_probability(coords_block_i, coords_block_j, phases_block_i, phases_block_j, sigma_dist, sigma_fase):
    # Calcular distancias euclidianas
    diff_coords = coords_block_i[:, np.newaxis, :] - coords_block_j[np.newaxis, :, :]
    distances = np.linalg.norm(diff_coords, axis=2)
    
    # Calcular diferencias angulares
    phase_diff = np.abs(phases_block_i[:, np.newaxis] - phases_block_j[np.newaxis, :])
    phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
    
    # Calcular la probabilidad combinada
    P_matrix = np.exp(-distances**2 / sigma_dist**2) * np.exp(-phase_diff**2 / sigma_fase**2)
    return P_matrix

# Función para procesar un bloque
def process_block(i_start, j_start, block_size, coordinates, phases, sigma_dist, sigma_fase, threshold):
    i_end = min(i_start + block_size, len(coordinates))
    j_end = min(j_start + block_size, len(coordinates))

    coords_block_i = coordinates[i_start:i_end]
    coords_block_j = coordinates[j_start:j_end]
    phases_block_i = phases[i_start:i_end]
    phases_block_j = phases[j_start:j_end]

    # Calcular probabilidades
    P_matrix = connection_probability(coords_block_i, coords_block_j, phases_block_i, phases_block_j, sigma_dist, sigma_fase)

    # Filtrar conexiones por umbral
    i_indices, j_indices = np.where(P_matrix > threshold)
    edges = [
        (i_start + i, j_start + j, P_matrix[i, j])
        for i, j in zip(i_indices, j_indices)
        if i_start + i < j_start + j  # Evitar duplicados
    ]
    return edges

# Función principal para procesar con multiprocesamiento
def process_blocks_with_multiprocessing(coordinates, phases, sigma_dist, sigma_fase, threshold=0.9, block_size=1000, num_workers=4):
    N = len(coordinates)
    tasks = []
    G = nx.Graph()

    # Configurar el multiprocesamiento
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i_start in range(0, N, block_size):
            for j_start in range(i_start + 1, N, block_size):
                tasks.append(executor.submit(
                    process_block, i_start, j_start, block_size,
                    coordinates, phases, sigma_dist, sigma_fase, threshold
                ))

        # Recoger los resultados
        for task in tasks:
            edges = task.result()
            G.add_weighted_edges_from(edges)

    return G

# Punto de entrada principal
if __name__ == '__main__':
    # Leer y procesar la imagen
    im = tifffile.imread(
        "/Users/schutyb/Documents/Projects/rgb-phasors/data/lung/B6ND/52-21B6ND05.tif")[450:500, 1350:1400]
    im = rgb2bgr(im)
    _, real, imag = phasor(im)
    real = median_filter(real, 3)
    imag = median_filter(imag, 3)

    # Calcular coordenadas y fases
    coordinates = np.asarray([real.flatten(), imag.flatten()]).transpose().astype(np.float16)
    phases = (np.angle(real + 1j * imag) + np.pi).flatten().astype(np.float16)

    # Parámetros
    sigma_dist = 0.2
    sigma_fase = np.pi / 4
    threshold = 0.9
    block_size = 1000
    num_workers = 8  # Ajustar según tu CPU

    # Procesar por bloques con multiprocesamiento
    G = process_blocks_with_multiprocessing(
        coordinates, phases, sigma_dist, sigma_fase, threshold, block_size, num_workers)

    # Dibujar el grafo
    plt.figure(figsize=(10, 10))  # Tamaño del gráfico
    nx.draw(G, node_size=10, edge_color="gray", with_labels=False)
    plt.title("Visualización del Grafo")
    plt.show()
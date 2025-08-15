#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io, morphology
from skimage.util import img_as_float
from skimage.filters import threshold_otsu, gaussian
from skimage.color import rgb2hed

def read_rgb(path):
    """Lee TIFF y devuelve RGB float en [0,1]."""
    img = io.imread(path)
    if img.ndim == 3 and img.shape[-1] == 4:  # Quitar alfa
        img = img[..., :3]
    if img.ndim == 2:  # Gris -> RGB
        img = np.stack([img]*3, axis=-1)
    img = img_as_float(img)
    return np.clip(img, 0, 1)

def alveolar_percent(rgb, blur_sigma=1.0, min_object=500):
    """Segmenta tejido vs fondo y devuelve % de fondo."""
    hed = rgb2hed(rgb)
    H, E = hed[..., 0], hed[..., 1]
    stain_total = H + E
    stain_s = gaussian(stain_total, sigma=blur_sigma, preserve_range=True)
    thr = threshold_otsu(stain_s)
    tissue_mask = stain_s > thr
    tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=min_object)
    alveolar_mask = ~tissue_mask
    h, w, _ = rgb.shape
    return 100.0 * (alveolar_mask.sum() / (h * w))

def process_folder(folder_path, exts):
    """
    Procesa imágenes y devuelve lista de tuplas (nombre_archivo, porcentaje).
    """
    paths = sorted([p for p in folder_path.rglob("*") if p.suffix in exts])
    results = []
    for p in paths:
        try:
            rgb = read_rgb(p)
            perc = alveolar_percent(rgb)
            results.append((p.name, round(float(perc), 4)))
            print(f"OK: {p}")
        except Exception as e:
            print(f"ERROR: {p} -> {e}")
    return results

def main():
    parser = argparse.ArgumentParser(
        description="CSV con nombres de imagen y % de fondo (espacio alveolar) por carpeta."
    )
    parser.add_argument("root_dir", type=str, help="Directorio raíz con subcarpetas de imágenes TIFF")
    parser.add_argument("--out_csv", type=str, default="alveolar_percent_by_folder.csv",
                        help="Ruta del CSV de salida")
    args = parser.parse_args()

    root = Path(args.root_dir)
    assert root.is_dir(), f"No existe el directorio: {root}"

    image_exts = {".tif", ".tiff", ".TIF", ".TIFF"}
    subfolders = sorted([p for p in root.iterdir() if p.is_dir()])

    if not subfolders:
        print("No se encontraron subcarpetas en el directorio raíz.")
        return

    # Recolectar nombres de todas las imágenes
    all_names = set()
    folder_data = {}
    for folder in subfolders:
        results = process_folder(folder, image_exts)
        folder_data[folder.name] = {name: perc for name, perc in results}
        all_names.update(name for name, _ in results)

    all_names = sorted(all_names)

    # Construir DataFrame con primera columna = nombre imagen
    df = pd.DataFrame({"Imagen": all_names})
    for folder in subfolders:
        folder_name = folder.name
        df[folder_name] = [folder_data[folder_name].get(name, np.nan) for name in all_names]

    df.to_csv(args.out_csv, index=False)
    print(f"\nListo. CSV guardado en: {args.out_csv}")
    print("Primera columna = nombre de imagen, luego una columna por carpeta.")

if __name__ == "__main__":
    main()
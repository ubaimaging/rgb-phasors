import os
import numpy as np
import tifffile as tiff


part1 = False
if part1:
    # Directorio donde están almacenados los archivos TIFF
    input_directory = "/Users/schutyb/Documents/Projects/rgb-phasors/data/psf/exp2/" # Cambia esta ruta según tu configuración
    output_file = "/Users/schutyb/Documents/Projects/rgb-phasors/data/psf/z_stack_exp2.tiff"  # Nombre del archivo de salida

    # Listar los archivos TIFF en el directorio
    file_list = sorted([f for f in os.listdir(input_directory) if f.endswith(".tiff") or f.endswith(".tif")],
                    key=lambda x: int(os.path.splitext(x)[0]))  # Ordenar por número (1.tiff, 2.tiff, ...)

    # Inicializar una lista para almacenar las imágenes cargadas
    z_stack = []

    # Leer y apilar las imágenes
    for file_name in file_list:
        file_path = os.path.join(input_directory, file_name)
        print(f"Cargando: {file_name}")
        image = tiff.imread(file_path)
        z_stack.append(image)

    # Convertir a un array 3D (z, y, x)
    z_stack_array = np.stack(z_stack, axis=0)

    # Guardar el z-stack como un archivo TIFF multipágina
    tiff.imwrite(output_file, z_stack_array, imagej=True)

    print(f"Z-stack creado y guardado como: {output_file}")

part2 = True
if part2:
    import numpy as np
    import tifffile as tiff
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # Función gaussiana para ajustar la PSF
    def gaussian(x, a, x0, sigma):
        return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Cargar z-stack en formato TIFF (RGB)
    z_stack_file = "/Users/schutyb/Documents/Projects/rgb-phasors/data/psf/z_stack_exp2.tiff"   # Cambia esta ruta
    z_stack = tiff.imread(z_stack_file)[:, 500:1000, 900:1400]  # z_stack con dimensiones (z, y, x, 3)

    # Seleccionar una ROI pequeña para el análisis de la PSF
    z_center = z_stack.shape[0] // 2
    x_center = z_stack.shape[2] // 2
    y_center = z_stack.shape[1] // 2
    roi_size = 20  # Tamaño de la región de interés (en píxeles)

    # Extraer intensidades para cada canal (R, G, B) en el ROI
    psf_profiles = {}
    channels = {"R": 0, "G": 1, "B": 2}
    for color, channel in channels.items():
        roi_intensities = []
        for z in range(z_stack.shape[0]):
            roi = z_stack[z, y_center - roi_size // 2:y_center + roi_size // 2,
                        x_center - roi_size // 2:x_center + roi_size // 2, channel]
            roi_intensities.append(np.sum(roi))  # Sumar intensidades en el ROI
        psf_profiles[color] = np.array(roi_intensities)

    # Ajustar la PSF para cada canal
    psf_fit = {}
    fwhm = {}
    for color, profile in psf_profiles.items():
        z_positions = np.arange(len(profile))
        
        # Validar datos del perfil
        profile = np.maximum(profile, 0)  # Evitar valores negativos
        
        # Parámetros iniciales dinámicos
        p0 = [profile.max(), np.argmax(profile), len(profile) / 10]
        
        try:
            # Ajustar la función gaussiana
            popt, _ = curve_fit(gaussian, z_positions, profile, p0=p0, maxfev=10000)
            psf_fit[color] = popt
            fwhm[color] = 2.355 * popt[2]  # FWHM = 2.355 * sigma
        except RuntimeError as e:
            print(f"Error al ajustar el canal {color}: {e}")
            psf_fit[color] = [0, 0, 0]
            fwhm[color] = None

    # Visualizar los perfiles PSF y los ajustes
    plt.figure(figsize=(12, 6))
    for i, (color, profile) in enumerate(psf_profiles.items()):
        z_positions = np.arange(len(profile))
        plt.plot(z_positions, profile, 'o', label=f"{color} Datos")
        if color in psf_fit and fwhm[color] is not None:
            plt.plot(z_positions, gaussian(z_positions, *psf_fit[color]), '-', label=f"{color} Ajuste (FWHM={fwhm[color]:.2f} píxeles)")

    plt.xlabel("Posición Z (píxeles)")
    plt.ylabel("Intensidad (suma en ROI)")
    plt.title("PSF en Z para cada canal RGB")
    plt.legend()
    plt.grid()
    plt.show()

    # Imprimir resultados
    for color, params in psf_fit.items():
        if fwhm[color] is not None:
            print(f"Canal {color}: Amplitud = {params[0]:.2f}, Centro = {params[1]:.2f}, Sigma = {params[2]:.2f}, FWHM = {fwhm[color]:.2f}")
        else:
            print(f"Canal {color}: No se pudo ajustar el perfil.")

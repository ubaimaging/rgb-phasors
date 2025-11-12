from PIL import Image
import os

# === CONFIGURACIÓN ===
input_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig/input_image/"   # carpeta donde están las imágenes
output_folder = "/Users/schutyb/Documents/Projects/rgb-phasors/paper/fig/output_image/"     # carpeta donde se guardarán los PDFs

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Extensiones permitidas
valid_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

# Buscar todas las imágenes
image_files = sorted([
    f for f in os.listdir(input_folder)
    if f.lower().endswith(valid_exts)
])

if not image_files:
    raise ValueError("No se encontraron imágenes en la carpeta especificada.")

# === CONVERSIÓN ===
for file in image_files:
    input_path = os.path.join(input_folder, file)
    output_name = os.path.splitext(file)[0] + ".pdf"
    output_path = os.path.join(output_folder, output_name)

    img = Image.open(input_path)

    # Convertir a RGB si es necesario (requerido por PDF)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Guardar como PDF con resolución alta
    img.save(output_path, "PDF", resolution=300.0)

    print(f"✅ {file} → {output_name}")

print(f"\n🎉 Conversión completa. Archivos PDF guardados en: {output_folder}")
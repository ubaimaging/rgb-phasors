import pandas as pd

# Cargar el archivo Excel
archivo_excel = "/Users/schutyb/Downloads/pVHLWT_060922.xlsx"  # Cambia esto por la ruta de tu archivo

# Leer todas las hojas del Excel
dfs = pd.read_excel(archivo_excel, sheet_name=None)  # Lee todas las hojas

# Mostrar cada hoja en forma de tabla
for nombre_hoja, df in dfs.items():
    print(f"\n Hoja: {nombre_hoja}")
    print(df.head())  # Muestra las primeras filas de cada hoja
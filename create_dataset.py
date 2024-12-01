import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


# Función para procesar un archivo .mat
def process_mat_file(file_path):
    # Cargar el archivo .mat
    data = loadmat(file_path)

    # Matrices y sus clases correspondientes
    matrices = [("hook", 0), ("lat", 1), ("spher", 2)]
    combined_data = []

    for matrix, label in matrices:
        # Concatenar los canales 1 y 2 horizontalmente
        ch1 = data[f"{matrix}_ch1"]
        ch2 = data[f"{matrix}_ch2"]
        concatenated = np.hstack((ch1, ch2))

        # Añadir la etiqueta de clase al final de cada fila
        labeled_data = np.hstack(
            (concatenated, np.full((concatenated.shape[0], 1), label))
        )
        combined_data.append(labeled_data)

    return np.vstack(combined_data)


def get_mat_files(directory):
    return [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".mat")
    ]


directory = "./dataset/raw/"
files = get_mat_files(directory)

print("Archivos disponibles:")
for idx, file in enumerate(files):
    print(f"{idx}: {file}")

selected_indices = input(
    "Selecciona los archivos que deseas procesar (índices separados por coma): "
)
selected_files = [files[int(idx)] for idx in selected_indices.split(",")]

all_data = []
for file_path in selected_files:
    print(f"Procesando archivo: {file_path}")
    all_data.append(process_mat_file(file_path))

final_data = np.vstack(all_data)
print(final_data.shape)

output_csv = "output_data.csv"
pd.DataFrame(final_data).to_csv(output_csv, index=False, header=False)
print(f"Archivo CSV generado: {output_csv}")

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np




# VISUALIZACIÓN DE DATOS:

# Selecciona la acción (por ejemplo, Tesla - TSLA)
ticker = "TSLA"

# Descarga los datos históricos (por ejemplo, últimos 5 años)
data = yf.download(ticker, start="2018-01-01", end="2024-01-01")

# Mostrar las primeras filas del dataset
print(data.head())

# Visualizar el precio de cierre
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title(f'Precio de Cierre de {ticker} (2018-2023)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre en USD')
plt.show()








# PREPROCESAMIENTO DE DATOS:

# Seleccionar valores relevantes:
# Seleccionar la columna 'Close' del dataframe
close_data = data[['Close']]

# Mostrar las primeras filas
print(close_data.head())



# Manejo de valores faltantes:
# Verificar si hay valores faltantes
missing_values = close_data.isnull().sum()
print(f"Valores faltantes: \n{missing_values}")

# Eliminar filas con valores faltantes (si es necesario)
close_data.dropna(inplace=True)



# Normalización de los datos:
# Escalar los datos al rango [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_data)

# Mostrar una vista previa de los datos escalados
print(scaled_data[:5])




# Creación de secuencias temporales:
# Definir la ventana de tiempo (por ejemplo, 60 días)
window_size = 60

# Crear listas vacías para las secuencias de entrenamiento
x_train = []
y_train = []

# Llenar las listas con secuencias de entrenamiento
for i in range(window_size, len(scaled_data)):
    x_train.append(scaled_data[i-window_size:i, 0])  # Secuencia de 60 días anteriores
    y_train.append(scaled_data[i, 0])  # El valor del día a predecir

# Convertir las listas a arrays de numpy
x_train, y_train = np.array(x_train), np.array(y_train)

# Mostrar el tamaño de los datos de entrenamiento
print(f"Tamaño de los datos de entrenamiento: {x_train.shape}, {y_train.shape}")





#Ajustar las dimensiones de los datos para el modelo LSTM:
# Ajustar las dimensiones de x_train para que sea compatible con LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Mostrar la nueva forma de x_train
print(f"Nuevo tamaño de x_train: {x_train.shape}")


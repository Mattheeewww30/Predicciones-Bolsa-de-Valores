import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from datetime import timedelta




# 1. VISUALIZACIÓN DE DATOS:

# Selecciona la acción (por ejemplo, Tesla - TSLA)
ticker = "TSLA"

# Descarga los datos históricos (por ejemplo, últimos 5 años)
data = yf.download(ticker, start="2020-01-01", end="2024-01-01")

# Mostrar las primeras filas del dataset
print(data.head())

# Visualizar el precio de cierre
plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
plt.title(f'Precio de Cierre de {ticker} (2018-2023)')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre en USD')
plt.show()








# 2. PREPROCESAMIENTO DE DATOS:

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





# 3. CONSTRUCIÓN DEL MODELO:
# Crear el modelo LSTM
# Crear el modelo LSTM optimizado
model = Sequential()

# Primera capa LSTM con Dropout
model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

# Segunda capa LSTM
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))

# Capa Dense con 50 neuronas
model.add(Dense(units=50))

# Capa final Dense con 1 neurona (predicción del precio)
model.add(Dense(units=1))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenar el modelo con datos optimizados
model.fit(x_train, y_train, batch_size=32, epochs=30)




close_data = data[['Close']]

# Dividimos los datos en entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# Crear los datos de entrenamiento a partir del conjunto escalado
train_data = scaled_data[0:training_data_len, :]

# Obtener los datos de prueba
test_data = scaled_data[training_data_len - window_size:, :]

# Crear los conjuntos de x_test e y_test
x_test = []
y_test = close_data[training_data_len:].values

for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i, 0])

# Convertir x_test en un array de numpy
x_test = np.array(x_test)

# Redimensionar x_test para que sea compatible con el modelo LSTM
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Hacer las predicciones
predictions = model.predict(x_test)

# Desescalar las predicciones
predictions = scaler.inverse_transform(predictions)

# Visualizar los resultados
train = close_data[:training_data_len]
valid = close_data[training_data_len:]
valid['Predictions'] = predictions

# Graficar los datos
plt.figure(figsize=(16, 8))
plt.title('Modelo LSTM: Predicción de Precios de Acciones')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre en USD')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Entrenamiento', 'Valor Real', 'Predicción'], loc='lower right')
plt.show()



# CÁLCULO DE EFICIENCIA DEL MODELO:
precio_promedio = y_test.mean()
print(f'El precio promedio de las acciones es: {precio_promedio}')
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')
error_relativo = (rmse / precio_promedio) * 100
print(f'Error relativo: {error_relativo}%')





# GRÁFICO 3:

# Supongamos que tienes 60 días anteriores
# Datos más recientes (últimos 60 días) que serán usados como base para las predicciones futuras
last_sequence = scaled_data[-window_size:]

# Número de días a predecir (resto de 2024 + primer semestre de 2025)
dias_a_predecir = 180  # Aproximadamente 6 meses

# Lista para guardar las predicciones futuras
futuras_predicciones = []

# Generar predicciones futuras iterativamente
for i in range(dias_a_predecir):
    # Usar la secuencia más reciente para hacer la predicción
    input_sequence = last_sequence.reshape((1, window_size, 1))
    predicted_price = model.predict(input_sequence)

    # Guardar la predicción
    futuras_predicciones.append(predicted_price[0, 0])

    # Actualizar la secuencia: descartar el primer valor y añadir la nueva predicción al final
    last_sequence = np.append(last_sequence[1:], predicted_price)

# Desescalar las predicciones a su valor original
futuras_predicciones = np.array(futuras_predicciones).reshape(-1, 1)
futuras_predicciones_desescaladas = scaler.inverse_transform(futuras_predicciones)

# Crear fechas para las predicciones futuras
ultima_fecha = pd.to_datetime(data.index[-1])  # Última fecha real en el dataset
fechas_futuras = pd.date_range(ultima_fecha + timedelta(days=1), periods=dias_a_predecir, freq='D')

# Graficar los resultados
plt.figure(figsize=(16, 8))
plt.plot(data['Close'], label='Datos Reales', color='blue')
plt.plot(fechas_futuras, futuras_predicciones_desescaladas, label='Predicciones Futuras (2024-2025)', linestyle='--', color='orange')  # Línea discontinua

# Asegurar que el eje X muestre las fechas de 2025-01 y 2025-07
plt.title('Predicciones de Precios de Acciones para el resto de 2024 y primer semestre de 2025')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre en USD')

# Configuración de los ticks del eje X para incluir las fechas clave
plt.xticks(pd.date_range(start='2024-01-01', end='2025-07-01', freq='6M'), rotation=45)

plt.legend()
plt.tight_layout()  # Ajustar el gráfico para evitar superposición de etiquetas
plt.show()

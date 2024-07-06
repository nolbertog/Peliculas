import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# ## 1. Introducción
# El objetivo de este proyecto es desarrollar un sistema de recomendación de películas utilizando 
# redes neuronales. Este sistema recomendará películas basándose en las preferencias de género de 
# los usuarios. Para lograr esto, se utilizará una base de datos de películas y se entrenará una red 
# neuronal que aprenderá a predecir las preferencias de los usuarios a partir de sus calificaciones anteriores.

print("----------------------------------------------------------------------------------------------------------1")

# ## 2. Recolección de Datos
# El proyecto utilizará el conjunto de datos de themoviedb, que es una colección bien conocida 
# y ampliamente utilizada en la investigación sobre sistemas de recomendación. Este conjunto 
# de datos contiene información sobre películas, géneros y calificaciones de usuarios, lo que proporciona 
# una base sólida para el desarrollo del sistema de recomendación.

# Cargamos los datos
movies = pd.read_csv('movies.csv')

# Visualizamos los primeros registros del dataset
print("Primeros registros del dataset: \n", movies.head())

# Visualizamos la cantidad de registros y columnas del dataset
print("Cantidad de registros y columnas: ", movies.shape)

# Visualizamos los tipos de datos de las columnas
print("Tipos de datos de las columnas: \n", movies.dtypes)

print("----------------------------------------------------------------------------------------------------------2")

# ## 3. Preprocesamiento de Datos
# Antes de entrenar el modelo, los datos deben ser preprocesados. Este paso incluye la limpieza de datos, 
# como el manejo de valores nulos. Además, se realizará la codificación de 
# los géneros de las películas en un formato que pueda ser interpretado por la red neuronal. Esto implica 
# convertir las etiquetas de texto de los géneros en vectores binarios mediante un proceso llamado binarización.

# Eliminamos las columnas que no son necesarias
movies = movies.drop(columns=['movie_id'])

# Convertimos la columna 'genres' en una lista de géneros
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# Binarizamos los géneros de las películas
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_)
movies = pd.concat([movies, genres_encoded], axis=1)

# Eliminamos la columna original 'genres'
movies = movies.drop(columns=['genres'])

# Visualizamos los primeros registros del dataset preprocesado
print("Primeros registros del dataset preprocesado: \n", movies.head())

# Visualizamos la cantidad de registros y columnas del dataset preprocesado
print("Cantidad de registros y columnas del dataset preprocesado: ", movies.shape)

# Visualizamos los tipos de datos de las columnas del dataset preprocesado
print("Tipos de datos de las columnas del dataset preprocesado: \n", movies.dtypes)

print("Columnas: ", movies.columns)

print("----------------------------------------------------------------------------------------------------------3")

# ## 4. Construcción del Modelo
# El corazón del sistema de recomendación será una red neuronal construida utilizando TensorFlow y Keras. La red 
# neuronal se diseñará con varias capas densamente conectadas que permitirán al modelo aprender las complejas 
# relaciones entre los géneros de las películas. Se seleccionará una arquitectura 
# adecuada y se configurarán los parámetros del modelo, como el número de neuronas en cada capa y las funciones de activación.

# División de los datos en características y etiquetas
X = movies.drop(columns=['title', 'release_date'])
y = movies[mlb.classes_]

# Construcción del Modelo
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(y.shape[1], activation='sigmoid'))

# Compilación del Modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# Visualización del Modelo
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

print("Columnas: ", movies.columns)

print("----------------------------------------------------------------------------------------------------------4")

# ## 5. Entrenamiento del Modelo
# El modelo se entrenará utilizando los datos de Durante el entrenamiento, el 
# modelo ajustará sus pesos para minimizar el error en la predicción de las preferencias de los usuarios. El 
# conjunto de datos se dividirá en conjuntos de entrenamiento y prueba para evaluar el rendimiento del modelo y 
# evitar el sobreajuste. Se utilizarán técnicas de validación cruzada para asegurar la robustez del modelo.

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir datos de entrada a flotante
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=2)

print("----------------------------------------------------------------------------------------------------------5")

# ## 6. Generación de Recomendaciones
# Una vez que el modelo esté entrenado, se utilizará para generar recomendaciones de películas basadas en 
# las preferencias de género del usuario. El usuario podrá ingresar sus géneros favoritos, y el modelo predecirá 
# las películas que probablemente le gusten. Estas recomendaciones se presentarán en orden de probabilidad, 
# permitiendo al usuario descubrir nuevas películas alineadas con sus intereses.

def recommend_movies(user_genres, model, mlb, movies):
    user_genres_encoded = mlb.transform([user_genres])
    predictions = model.predict(user_genres_encoded)
    movie_scores = np.dot(predictions, movies[mlb.classes_].T)
    recommended_movie_indices = np.argsort(movie_scores[0])[::-1]
    recommended_movies = movies.iloc[recommended_movie_indices]
    return recommended_movies

# Uso
user_genres = ['war']

# Ejemplos de generos: ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 
# 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
# 'Sci-Fi', 'Thriller', 'War', 'Western']

recommended_movies = recommend_movies(user_genres, model, mlb, movies)
print("Películas recomendadas: \n", recommended_movies.head())

print("----------------------------------------------------------------------------------------------------------6")

# ## 7. Generación del Reporte Inicial
# Se generará un reporte inicial que describirá en detalle los pasos realizados hasta el momento, incluyendo:

# - Descripción del conjunto de datos utilizado.
# - Métodos de preprocesamiento de datos aplicados.
# - Diseño y configuración de la red neuronal.
# - Resultados preliminares del entrenamiento del modelo.
# - Ejemplos de recomendaciones generadas.

# Resumen
reporte = {
    "Descripción del conjunto de datos": movies.describe(),
    "Métodos de preprocesamiento de datos aplicados": "Binarización de géneros, eliminación de columnas no necesarias.",
    "Diseño y configuración de la red neuronal": model.summary(),
    "Resultados preliminares del entrenamiento del modelo": history.history,
    "Ejemplos de recomendaciones generadas": recommended_movies.head()
}

print("Reporte Inicial: \n", reporte)

print("----------------------------------------------------------------------------------------------------------7")

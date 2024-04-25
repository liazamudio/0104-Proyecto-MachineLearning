# IMPORTACIONES

# Librerías generales
import numpy as np
import pandas as pd
from scipy import stats
import MLUtilities as utils                             # Para importar funciones de archivo MLUtilities

# Librerías de Machine learning
from sklearn.model_selection import train_test_split    # Librería para hacer la separación del dataframe para entrenar  modelos
from sklearn.model_selection import KFold               # Librería empleada en la validación cruzada
from sklearn.metrics import confusion_matrix            # Librería requerida para la Matriz de confusión
from scipy import stats                                 # Para la separación de datos

# Clasificadores
from sklearn.cluster import KMeans                      # Trabajar con el modelo KMeans
from sklearn.naive_bayes import GaussianNB              # Naive Bayes
from sklearn.svm import SVC                             # Support Vector Classifier
from sklearn.neural_network import MLPClassifier        # Neural Network
from sklearn.ensemble import RandomForestClassifier     # Random Forest Classifier 
import lazypredict
from lazypredict import LazyClassifier                  # LazyClassifier


# DEFINICIÓN DE FUNCIONES

# Función para generar particiones (Separación de datos)
def particionar(entradas, salidas, porcentaje_entrenamiento, porcentaje_validacion, porcentaje_prueba):
    temp_size = porcentaje_validacion + porcentaje_prueba
    print(temp_size)
    x_train, x_temp, y_train, y_temp = train_test_split(entradas, salidas, test_size =temp_size)
    if(porcentaje_validacion > 0):
        test_size = porcentaje_prueba/temp_size
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = test_size)
    else:
        return [x_train, None, x_temp, y_train, None, y_temp]
    return [x_train, x_val, x_test, y_train, y_val, y_test]

# MÉTRICAS DE MACHINE LEARNING

def calcularAccuracy(TP, TN, FP, FN):                       # Precisión o Accurancy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy * 100
    return accuracy

def calcularSensibilidad(TP, TN, FP, FN):                   # Sensibilidad
    sensibilidad = TP / (TP + FN)
    sensibilidad = sensibilidad * 100
    return sensibilidad

def calcularEspecificidad(TP, TN, FP, FN):                  # Especificidad
    especificidad = TN / (TN + FP)
    especificidad = especificidad * 100
    return especificidad

# Función que calcula la distancia eucilidiana

def distEuclidiana(muestra, dataset):
    distancias = np.zeros((dataset.shape[0],1))
    for counter in range(0,dataset.shape[0]):
        distancias[counter] = np.linalg.norm(muestra-dataset[counter])
    return distancias

# Función que encuentra el centroide mas cercano

def centroideCercano(muestra, listaCentroides):
    listaDistancias = distEuclidiana(muestra, listaCentroides)
    centroideCercano = np.argmin(listaDistancias)
    return centroideCercano

# Función que clasifica por centroides

def clasificarPorCentroides(muestras, centroides):
    resultado = np.zeros((muestras.shape[0],1))
    for counter in range(0, muestras.shape[0]):
        resultado[counter] = centroideCercano(muestras[counter], centroides)
    return resultado

# Función que separa datos dependiendo de la etiqueta de valor esperado que tengan

def separarDatos(muestras, valoresEsperados, valorAFiltrar):
    indices = np.where(valoresEsperados == valorAFiltrar)
    return muestras[indices], valoresEsperados[indices]

# Función para obtener la moda

def obtenerModa(resultados):
    moda = (stats.mode(resultados)[0]).reshape(-1)
    return moda[0]

# Función para obtener la accuracy de una muestra con K medias

def obtenerAccuracy_kmedias(muestras, centroides):
    numMuestras = muestras.shape[0]
    resultados = clasificarPorCentroides(muestras, centroides)
    moda = obtenerModa(resultados)
    indicesErrores = np.where(resultados != moda)
    cantidadErrores = len(resultados[indicesErrores])
    accuracy = ((numMuestras - cantidadErrores) / numMuestras) *100
    return accuracy

# Función que recomienda productos

def recomiendameProductos(listaDeProductos, datosProductos,productosEjemplo,centroides):
    clasificacionDeseada = utils.centroideCercano(productosEjemplo, centroides)     # Vamos a buscar el centroide mas cercano (con MLUtilities ;))
    clasificaciones = utils.clasificarPorCentroides(datosProductos, centroides)     # Luego, vamos a clasificar todas las peliculas por centroides
    indices = np.where(clasificaciones == clasificacionDeseada)[0]                  # Finalmente, sacaremos los indices que hacen match entre clasificaciones
    return listaDeProductos[indices]                                                # Y regresamos la lista de peliculas.


# Funciones que determinan métricas de la Matriz de confusión

def calcularAccuracy(TP, TN, FP, FN):                                               # Función para obtener la accuracy por medio de la Matriz de confusión
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = accuracy * 100
    return accuracy

def calcularSensibilidad(TP, TN, FP, FN):                                           # Función para obtener la sensibilidad por medio de la Matriz de confusión
    sensibilidad = TP / (TP + FN)
    sensibilidad = sensibilidad * 100
    return sensibilidad


def calcularEspecificidad(TP, TN, FP, FN):                                          # Función para obtener la especificidad por medio de la Matriz de confusión
    especificidad = TN / (TN + FP)
    especificidad = especificidad * 100
    return especificidad

# Función que nos despliega los resultados de las evaluaciones de los clasificadores
def evaluar(y_test, y_pred):
    # Obtención de los resultados de una matriz de confusión de 4x4
    resultado = confusion_matrix(y_test, y_pred)
    print(resultado)
    print(resultado.shape)
    
    # Extraemos la flattened_array
    flattened_array = resultado.ravel()
    
    # Definir posiciones para cada valor
    TN_pos = 0                                                                      # Esquina superior izquierda
    FP_pos = TN_pos + resultado.shape[1] - 1                                        # Esquina superior derecha
    # FN_pos = FP_pos + resultado.shape[0] * resultado.shape[1] - 1                   # Esquina inferior izquierda
    FN_pos = FP_pos + resultado.shape[0] + resultado.shape[1] + 1                   # Esquina inferior izquierda
    TP_pos = FN_pos + resultado.shape[1] - 1                                        # Esquina inferior derecha
    
    # Extremos valores de TN, FP, FN, TP 
    TN = flattened_array[TN_pos]
    FP = flattened_array[FP_pos]
    FN = flattened_array[FN_pos]
    TP = flattened_array[TP_pos]

    # Imprimir los valores obtenidos
    # print("True positives: "+str(TP))             # Ejemplo de otra forma de desplegar los resultados    
    print("True positives:", TP)
    print("True negatives:", TN)
    print("False positives:", FP)
    print("False negative:", FN)

    # Determinación de los indicadores
    acc = round((calcularAccuracy(TP, TN, FP, FN)),3)
    sen = round((calcularSensibilidad(TP, TN, FP, FN)),3)
    spec = round((calcularEspecificidad(TP, TN, FP, FN)),3)

    print(f'Precision: {acc} %.')
    print(f'Sensibilidad: {sen} %.')
    print(f'Especificidad: {spec} %.')

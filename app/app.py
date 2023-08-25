
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from domain.SupportVectorRegression import SupportVectorRegression
from logger.logger import logger
import os
from domain.DataTreat import DataTreat
import numpy as np
from domain.ExtremeLearningMachine import ExtremeLearningMachine
from domain.DeepNeuralNetwork import DeepNeuralNetwork
############################# Borrame #############################
from sklearn import preprocessing


def select_dataset(input_directory):
    # Listado de todos los ficheros en el directorio especificado
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    
    # Comprobación de que hay ficheros en el directorio
    if not files:
        print("No hay datasets en el directorio especificado.")
        return None

    # Se muestra la lista de datasets disponibles
    print("Please choose a dataset:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")

    # Entrada del usuario para seleccionar un dataset
    selection = -1
    while selection < 1 or selection > len(files):
        try:
            selection = int(input("Introduce el número del dataset: "))
            if selection < 1 or selection > len(files):
                print(f"El número debe encontrarse entre 1 y {len(files)}.")
        except ValueError:
            print("Por favor, introduce un número.")
    
    # Se devuelve el path del fichero seleccionado
    path = os.path.join(input_directory, files[selection-1])
    return path 

def select_model(model_name, input_dim):
    if model_name.upper() == "SVM":
        logger.info("El modelo elegido para la predicción es SVM")
        return SupportVectorRegression(kernel='rbf')
    elif model_name.upper() == "ELM":
        logger.info("El modelo elegido para la predicción es ELM")
        return ExtremeLearningMachine() 
    elif model_name.upper() == "DNN":
        logger.info("El modelo elegido para la predicción es DNN")
        return DeepNeuralNetwork(input_dim) 

def train_model(model, x_train, x_test, y_train, y_test ):
    
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    return model, score

if __name__ == "__main__":
    
    # Modelo
    print("Elegir un algoritmo: SVM, ELM, DNN")
    model_name = input()
    
    # Dataset
    input_directory = "resources/4_ready/"
    dataset = select_dataset(input_directory)
    data_handler = DataTreat(dataset)
    data_handler.process_data()

    model = select_model(model_name,data_handler.get_input_dimension())
    
    # Entrenamiento
    trained_model, score = train_model(model, data_handler.x_train, data_handler.x_test, data_handler.y_train, data_handler.y_test)
    print("RMSE: ", trained_model.calculate_rmse(data_handler.x_test, data_handler.y_test))
    print("MAE: ", trained_model.calculate_mae(data_handler.x_test, data_handler.y_test))
    print("R2: ", trained_model.calculate_r2(data_handler.x_test, data_handler.y_test))
    print("Time: ", trained_model.prediction_time(data_handler.x_test))
    
    # Predicción
    
    ############################# Borrame #############################
    #model.grid_search(data_handler.x_train, data_handler.y_train.ravel())
    # Valores originales
    min_max_scaler_x = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()
    #y_train = min_max_scaler_y.inverse_transform(data_handler.y_train)
    #y_test = min_max_scaler_y.inverse_transform(data_handler.y_test)

    # Predicción svm
    svm_predictions = model.predict(data_handler.x_train)
    svm_predictions_test = model.predict(data_handler.x_test)
    #svm_predictions = model.predict(data_handler.x_train)
    #svm_predictions_test = model.predict(data_handler.x_test)
    #svm_predictions = min_max_scaler_y.inverse_transform([svm_predictions])
    #svm_predictions_test = min_max_scaler_y.inverse_transform([svm_predictions_test])

    # Mostrar gráfico training
    model.show_graph(data_handler.y_train,svm_predictions,'results/svm_train.jpg')
    # Mostrar gráfico testing
    model.show_graph(data_handler.y_test,svm_predictions_test,'results/svm_test.jpg')
    # Mostrar resultado
    #model.show_graph()
    
    """     print("Especificar el porcentaje para pruebas (como un decimal, por ejemplo, 0.2 para 20%):")
        porcentaje_test = float(input()) """

from domain.MultilayerPerceptron import MultilayerPerceptron
from domain.SupportVectorRegression import SupportVectorRegression
from logger.logger import logger
import os
from domain.DataTreat import DataTreat
from domain.ExtremeLearningMachine import ExtremeLearningMachine
from domain.ExtremeLearningMachine import optimize_hyperparameters
import numpy as np
from datetime import datetime
from sklearn import preprocessing


def select_dataset(input_directory):
    # Listado de todos los ficheros en el directorio especificado
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
    
    # Comprobación de que hay ficheros en el directorio
    if not files:
        print("No hay datasets en el directorio especificado.")
        logger.error("No hay datasets en el directorio especificado.")
        return None

    # Se muestra la lista de datasets disponibles
    print("Elige un dataset:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")

    # Entrada del usuario para seleccionar un dataset
    selection = -1
    while selection < 1 or selection > len(files):
        try:
            selection = int(input("Introduce el número del dataset: "))
            if selection < 1 or selection > len(files):
                print(f"El número debe encontrarse entre 1 y {len(files)}.")
                logger.error(f"El número debe encontrarse entre 1 y {len(files)}.")
        except ValueError:
            print("Por favor, introduce un número.")
            logger.error("Se introdujo un valor no numérico.")
    
    # Se devuelve el path del fichero seleccionado
    path = os.path.join(input_directory, files[selection-1])
    print(f"Dataset seleccionado: {path}")
    logger.info(f"Dataset seleccionado: {path}")
    return path 

def select_model(model_name, input_shape):
    if model_name.upper() == "SVM":
        # Para SVM, podríamos pedir el kernel y el parámetro C
        kernel = input("Introduce el kernel para SVM (rbf/linear/poly/sigmoid): ")
        C = float(input("Introduce el parámetro C para SVM: "))
        print("El modelo elegido para la predicción es SVM")
        logger.info("El modelo elegido para la predicción es SVM")
        return SupportVectorRegression(kernel=kernel, C=C)
    elif model_name.upper() == "ELM":
        # Solicitar hiperparámetros específicos para ELM
        print("El modelo elegido para la predicción es ELM")
        logger.info("El modelo elegido para la predicción es ELM")
        seed = int(input("Elige una semilla para la aleatoriedad: "))
        print("La semilla elegida es: " + str(seed))
        logger.info("La semilla elegida es: " + str(seed))

        grid_search = input("¿Desea realizar una búsqueda de hiperparámetros para ELM? (s/n): ")
        if grid_search.lower() == "s":
            neuron_range = range(10, 101, 10)  # De 10 a 100 neuronas, en pasos de 10
            activation_functions = [np.tanh, sigmoid, relu, linear]  # Ejemplo de funciones de activación
            print("Realizando búsqueda de hiperparámetros para ELM...")
            print("Se va a probar con el rango de neuronas de 10 a 100, en pasos de 10. Y con las funciones de activación: sigmoid, tanh, relu y linear.")
            logger.info("Realizando búsqueda de hiperparámetros para ELM...")
            logger.info("Se va a probar con el rango de neuronas de 10 a 100, en pasos de 10. Y con las funciones de activación: sigmoid, tanh, relu y linear.")
            hyperparameters, score = optimize_hyperparameters(data_handler.x_train, data_handler.y_train.ravel(), neuron_range, activation_functions, seed)
            num_neurons = hyperparameters["neurons"]
            activation_function = hyperparameters["activation_function"]
            
            print("El resultado ha sido el siguiente:")
            print(f"El número de neuronas óptimo es: {num_neurons}")
            print(f"La función de activación óptima es: {activation_function}")
            print(f"El score obtenido ha sido: {score}")
            logger.info("El resultado ha sido el siguiente:")
            logger.info(f"El número de neuronas óptimo es: {num_neurons}")
            logger.info(f"La función de activación óptima es: {activation_function}")
            logger.info(f"El score obtenido ha sido: {score}")
        else:
            num_neurons = int(input("Introduce el número de neuronas en la capa oculta para ELM: "))
            activation_function = input("Introduce la función de activación para ELM (sigmoid/tanh/relu/linear): ")
            
        print("El número de neuronas en la capa oculta es: "+  str(num_neurons))
        print("La función de activación es: "+  str(activation_function))
        logger.info("El número de neuronas en la capa oculta es: "+  str(num_neurons))
        logger.info("La función de activación es: "+ str(activation_function))
        return ExtremeLearningMachine(num_neurons=num_neurons, activation_function=activation_function, seed=seed) 
    elif model_name.upper() == "MLP":
        # Solicitar hiperparámetros específicos para MLP
        num_layers = int(input("Introduce el número de capas para MLP: "))
        neurons_per_layer = [int(input(f"Introduce el número de neuronas para la capa (Recuerda que la última tiene que ser 1) {i+1}: ")) for i in range(num_layers)]
        learning_rate = float(input("Introduce la tasa de aprendizaje para MLP: "))
        activation_functions = [input(f"Introduce la función de activación para la capa {i+1} (sigmoid/tanh/relu): ") for i in range(num_layers)]
        optimizer = input("Introduce el optimizador para MLP (adam/sgd/rmsprop): ")
        loss = input("Introduce la función de pérdida para MLP (mean_squared_error/mean_absolute_error/categorical_crossentropy): ")
        
        print("El modelo elegido para la predicción es MLP")
        print("El número de capas es: "+ str(num_layers))
        print("El learning rate es: "+ str(learning_rate))
        print("El optimizador es: "+ str(optimizer))
        print("La función de pérdida es: "+ str(loss))
        logger.info("El modelo elegido para la predicción es MLP")
        logger.info("El número de capas es: "+ str(num_layers))
        logger.info("El learning rate es: "+ str(learning_rate))
        logger.info("El optimizador es: "+ str(optimizer))
        logger.info("La función de pérdida es: "+ str(loss))
        for i, neurons in enumerate(neurons_per_layer):
            print(f"El número de neuronas en la capa {i+1} es: {neurons}")
            print(f"La función de activación en la capa {i+1} es: {activation_functions[i]}")
            logger.info(f"El número de neuronas en la capa {i+1} es: {neurons}")
            logger.info(f"La función de activación en la capa {i+1} es: {activation_functions[i]}")
            
        return MultilayerPerceptron(input_shape, neurons_per_layer=neurons_per_layer, activation_functions=activation_functions, optimizer=optimizer, loss=loss)

def train_model(model, x_train, x_test, y_train, y_test ):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    logger.info(f"El score del modelo es: {score}")
    return model, score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

if __name__ == "__main__":
    logger.info("Inicio de la aplicación")
    # Modelo
    print("Elige un algoritmo: SVM, ELM, MLP")
    model_name = input().strip()

    # Dataset
    input_directory = "resources/4_ready/"
    dataset = select_dataset(input_directory)
    if dataset is not None:
        
        data_handler = DataTreat(dataset)
        data_handler.process_data()

        model = select_model(model_name, data_handler.get_input_dimension())
        
        # Entrenamiento
        trained_model, score = train_model(model, data_handler.x_train, data_handler.x_test, data_handler.y_train, data_handler.y_test)
        print(f"RMSE: {trained_model.calculate_rmse(data_handler.x_test, data_handler.y_test)}")
        print(f"MAE: {trained_model.calculate_mae(data_handler.x_test, data_handler.y_test)}")
        print(f"R2: {trained_model.calculate_r2(data_handler.x_test, data_handler.y_test)}")
        print(f"Time: {trained_model.prediction_time(data_handler.x_test)}")
        logger.info(f"El RMSE del modelo es: {trained_model.calculate_rmse(data_handler.x_test, data_handler.y_test)}")
        logger.info(f"El MAE del modelo es: {trained_model.calculate_mae(data_handler.x_test, data_handler.y_test)}")
        logger.info(f"El R2 del modelo es: {trained_model.calculate_r2(data_handler.x_test, data_handler.y_test)}")
        logger.info(f"El tiempo de predicción del modelo es: {trained_model.prediction_time(data_handler.x_test)}")
    
    # Predicción
    # Valores originales
    min_max_scaler_x = preprocessing.MinMaxScaler()
    min_max_scaler_y = preprocessing.MinMaxScaler()
    
    predictions = model.predict(data_handler.x_train)
    predictions_test = model.predict(data_handler.x_test)

    current_date = datetime.now().strftime("%Y%m%d")


    # Mostrar gráfico training
    predictions = data_handler.desnormalize_data(predictions, "y")
    data_handler.y_train = data_handler.desnormalize_data(data_handler.y_train, "y")
    model.show_graph(data_handler.y_train,predictions,'results/TRAIN_GRAPH_' + model_name.upper() + '_' + current_date + '.jpg')
    # Mostrar gráfico testing
    predictions_test = data_handler.desnormalize_data(predictions_test, "y")
    data_handler.y_test = data_handler.desnormalize_data(data_handler.y_test, "y")
    model.show_graph(data_handler.y_test,predictions_test,'results/TEST_GRAPH_' + model_name.upper() + '_' + current_date + '.jpg')


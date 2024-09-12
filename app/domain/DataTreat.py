import pandas as pd
import numpy as np
from sklearn import preprocessing

class DataTreat:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dataset = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.min_max_scaler_x = preprocessing.MinMaxScaler()
        self.min_max_scaler_y = preprocessing.MinMaxScaler()

    def load_data(self):
        self.dataset = pd.read_excel(self.filepath)

    def prepare_inputs_outputs(self):
        # Extracción de las columnas del dataset
        festive_today = self.dataset['Festivo Hoy']
        weekday_today = self.dataset['DiaSemana Hoy']
        temperature_today = self.dataset['Temperatura Hoy']
        precipitation_today = self.dataset['Precipitacion Hoy']
        
        festive_tomorrow = self.dataset['Festivo Mannana']
        weekday_tomorrow = self.dataset['DiaSemana Mannana']

        users_yesterday = self.dataset['CantidadUsuarios Hoy-1']
        festive_yesterday = self.dataset['Festivo Hoy-1']
        weekday_yesterday = self.dataset['DiaSemana Hoy-1']

        users_2days_ago = self.dataset['CantidadUsuarios Hoy-2']
        festive_2days_ago = self.dataset['Festivo Hoy-2']
        weekday_2days_ago = self.dataset['DiaSemana Hoy-2']

        users_3days_ago = self.dataset['CantidadUsuarios Hoy-3']
        festive_3days_ago = self.dataset['Festivo Hoy-3']
        weekday_3days_ago = self.dataset['DiaSemana Hoy-3']

        users_4days_ago = self.dataset['CantidadUsuarios Hoy-4']
        festive_4days_ago = self.dataset['Festivo Hoy-4']
        weekday_4days_ago = self.dataset['DiaSemana Hoy-4']
        
        outputs = self.dataset['CantidadUsuarios Hoy']

        # Combinación de las columnas en un array
        inputs = np.column_stack([
            festive_today, weekday_today, temperature_today, precipitation_today,
            festive_tomorrow, weekday_tomorrow, users_yesterday, festive_yesterday,
            weekday_yesterday, users_2days_ago, festive_2days_ago, weekday_2days_ago,
            users_3days_ago, festive_3days_ago, weekday_3days_ago, users_4days_ago,
            festive_4days_ago, weekday_4days_ago
        ])
        
        return inputs, outputs.values

    def split_data(self, inputs, outputs, test_size=0.2):
        # Determinar el número de muestras totales
        total_samples = len(inputs)
        
        # Calcular el número de muestras para el conjunto de prueba
        test_samples = int(total_samples * test_size)
        
        # Crear índices fijos para entrenamiento y prueba
        train_indices = range(test_samples, total_samples)
        test_indices = range(test_samples)
        
        # Dividir los datos utilizando los índices fijos
        self.x_train = inputs[train_indices]
        self.x_test = inputs[test_indices]
        self.y_train = outputs[train_indices]
        self.y_test = outputs[test_indices]

    def normalize_data(self):
        self.x_train = self.min_max_scaler_x.fit_transform(self.x_train)
        self.x_test = self.min_max_scaler_x.transform(self.x_test)
    
        self.y_train = self.y_train.reshape(-1, 1)
        self.y_train = self.min_max_scaler_y.fit_transform(self.y_train)
        self.y_test = self.y_test.reshape(-1, 1)
        self.y_test = self.min_max_scaler_y.transform(self.y_test)
        
    def desnormalize_data(self, data, type):
        if type == "x":
            return self.min_max_scaler_x.inverse_transform(data)
        else:
            return self.min_max_scaler_y.inverse_transform(data)
        

    def process_data(self):
        self.load_data()
        inputs, outputs = self.prepare_inputs_outputs()
        self.split_data(inputs, outputs)
        self.normalize_data()

    def get_input_dimension(self):
        return self.x_train.shape[1]

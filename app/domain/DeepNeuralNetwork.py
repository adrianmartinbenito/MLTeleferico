import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

class DeepNeuralNetwork:
    def __init__(self, input_dim, hidden_layers=[16,16], activation='relu', output_activation='linear'):
        """
        Constructor de la clase.
        
        Parámetros:
        - input_dim: dimensión del vector de entrada.
        - hidden_layers: una lista con el número de neuronas en cada capa oculta.
        - activation: función de activación para las capas ocultas.
        - output_activation: función de activación para la capa de salida.
        """
        self.model = Sequential()

        # Agregar capa de entrada
        self.model.add(Dense(hidden_layers[0], activation=activation, input_dim=input_dim))

        # Agregar capas ocultas
        for neurons in hidden_layers[1:]:
            self.model.add(Dense(neurons, activation=activation))
        
        # Agregar capa de salida
        self.model.add(Dense(1, activation=output_activation))

        # Compilar el modelo
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def fit(self, x, y, epochs=30, batch_size=128, validation_split=0.2, verbose=1):
        """Entrenar el modelo."""
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=verbose)
        return history

    def predict(self, x):
        """Predicción con el modelo."""
        return self.model.predict(x)

    # TODO - Implementar un método score para mostrar los resultados
    def score(self, x, y):
        """Evaluar el modelo y devolver la pérdida y otras métricas."""
        return self.model.evaluate(x, y)

    def summary(self):
        """Imprimir un resumen del modelo."""
        return self.model.summary()
    
    # Hay que adaptar esto eh
    def show_graph(self, y, y_predict, save_location):
    
        plt.figure(figsize=(30,18))
        plt.plot(y[:],"g")
        plt.plot(y_predict[:],"r")
        plt.legend(['Actual','Predicted'])
        plt.savefig(save_location,dpi=500)
        plt.show()

    def calculate_rmse(self, x, y_true):
        y_pred = self.predict(x)
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def calculate_mae(self, x, y_true):
        y_pred = self.predict(x)
        return mean_absolute_error(y_true, y_pred)

    def calculate_r2(self, x, y_true):
        y_pred = self.predict(x)
        return r2_score(y_true, y_pred)
    
    def prediction_time(self, x):
        start_time = time.time()
        self.predict(x)
        end_time = time.time()
        return end_time - start_time

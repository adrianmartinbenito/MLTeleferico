import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from numpy.random import default_rng


class ExtremeLearningMachine:
    def __init__(self, num_neurons, activation_function, seed=None):
        self.num_neurons = num_neurons
        switcher = {
            'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
            'relu': lambda x: np.maximum(0, x),
            'tanh': lambda x: np.tanh(x),
            'linear': lambda x: x
        }
        self.activation_function = switcher.get(activation_function, lambda x: x)
        self.input_weights = None
        self.bias_weights = None
        self.output_weights = None
        self.scaler = StandardScaler()
        self.rng = default_rng(seed)


    def _initialize_weights(self, input_size):
        """Inicializa los pesos de entrada y bias aleatoriamente."""
        self.input_weights = self.rng.standard_normal((input_size, self.num_neurons))
        self.bias_weights = self.rng.standard_normal(self.num_neurons)

    def fit(self, x, y):
        """Entrenamiento del modelo ELM."""
        x = self.scaler.fit_transform(x)
        
        if self.input_weights is None or self.bias_weights is None:
            self._initialize_weights(x.shape[1])
        
        hidden_layer_output = self.activation_function(x @ self.input_weights + self.bias_weights)
        self.output_weights = np.linalg.pinv(hidden_layer_output) @ y

    def predict(self, x):
        """Predicción del modelo ELM."""
        x = self.scaler.transform(x)
        hidden_layer_output = self.activation_function(x @ self.input_weights + self.bias_weights)
        return hidden_layer_output @ self.output_weights

    def score(self, x, y):
        """Devuelve el coeficiente de determinación R^2 de la predicción."""
        y_pred = self.predict(x)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u/v)
    
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

def optimize_hyperparameters(x, y, neuron_range, activation_functions, test_size=0.2, random_state=None):
    """
    Optimiza los hiperparámetros para el modelo ELM.
    
    Parámetros:
    - X: Características de entrada.
    - y: Etiquetas objetivo.
    - neuron_range: Rango de neuronas en la capa oculta para probar.
    - activation_functions: Lista de funciones de activación para probar.
    - test_size: Proporción del conjunto de datos a utilizar como conjunto de prueba.
    - random_state: Semilla para la generación de números aleatorios.
    
    Retorna:
    - best_hyperparameters: Mejores hiperparámetros encontrados.
    - best_score: Mejor puntuación R^2 obtenida.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    best_score = -np.inf
    best_hyperparameters = None
    
    for neurons in neuron_range:
        for activation_function in activation_functions:
            model = ExtremeLearningMachine(neurons, activation_function)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            score = r2_score(y_test, y_pred)
            
            if score > best_score:
                best_score = score
                best_hyperparameters = {'neurons': neurons, 'activation_function': activation_function}
    
    return best_hyperparameters, best_score

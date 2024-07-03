import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

class ExtremeLearningMachine:
    def __init__(self, hidden_units=70, activation_function=np.tanh):
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.input_weights = None
        self.bias_weights = None
        self.output_weights = None
        self.scaler = StandardScaler()

    def _initialize_weights(self, input_size):
        """Inicializa los pesos de entrada y bias aleatoriamente."""
        self.input_weights = np.random.randn(input_size, self.hidden_units)
        self.bias_weights = np.random.randn(self.hidden_units)

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

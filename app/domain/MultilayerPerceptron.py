import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

class MultilayerPerceptron:
    def __init__(self, input_shape, neurons_per_layer, activation_functions, optimizer, loss):
        self.model = Sequential()
        self.model.add(Dense(neurons_per_layer[0], activation=activation_functions[0], input_shape=(input_shape,)))
        for layer_size, activation in zip(neurons_per_layer[1:], activation_functions[1:]):
            self.model.add(Dense(layer_size, activation=activation))
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    def fit(self, x_train, y_train, epochs=100, batch_size=32, validation_data=None):
        return self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def summary(self):
        return self.model.summary()

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
    
    def score(self, x, y_true):
        y_pred = self.predict(x)
        return r2_score(y_true, y_pred)

    def predict(self, x):
        return self.model.predict(x)

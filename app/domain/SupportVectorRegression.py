from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time


class SupportVectorRegression:
    # Ejemplo de uso:
    # svr = SupportVectorRegression(kernel='rbf')
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # svr.fit(x_train, y_train)
    # predictions = svr.predict(x_test)

    def __init__(self, kernel='linear'):
        self.kernel = kernel
        #self.model = SVR(kernel=self.kernel)
        self.model = SVR(kernel=self.kernel, C=10000, epsilon=0.01, gamma=0.03)
        self.scaler = StandardScaler()

    def fit(self, x, y):
        """Entranamiento del modelo SVR."""
        x = self.scaler.fit_transform(x)
        self.model.fit(x, y.ravel())
    def score(self, x, y):
        """Devuelve el score del modelo SVR."""
        return self.model.score(x, y.ravel())

    def predict(self, x):
        """Predicción del modelo SVR."""
        return self.model.predict(x)
    
    def set_params(self, **params):
        """Define los parámetros del modelo SVR."""
        return self.model.set_params(**params)
    
    def grid_search(self, x_train, y_train ):
        """Realiza una búsqueda de parámetros del modelo SVR."""
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'gamma': [0.01, 0.1, 1],  # Si usas kernel='rbf'
            #'degree': [2, 3, 4],     # Si usas kernel='poly'
            #'coef0': [0, 1, 2]       # Si usas kernel='poly' o 'sigmoid'
        }
        grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)  # Ajusta 'kernel' según tu elección
        grid_search.fit(x_train, y_train)

        best_svr = grid_search.best_estimator_
        print('Best C:', best_svr.C)
        print('Best epsilon:', best_svr.epsilon)
        print('Best Gamma:', best_svr.gamma)
        print('Best Kernel:', best_svr.kernel)
        print('Best Degree:', best_svr.degree)
        print('Best Coef0:', best_svr.coef0)
        print('Best Score:', grid_search.best_score_)
        print('Best Params:', grid_search.best_params_)
        print('Best Estimator:', grid_search.best_estimator_)
        print('Best Index:', grid_search.best_index_)

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



###################### Librerias ############################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers,models
from sklearn import preprocessing
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from scipy import linalg



###################### Leer datos ###########################

dataset = pd.read_excel('resources/dataset_completo.xlsx')

SVM =True
DNN =False
ELM =False

####################### Funciones ###########################


    
###################### Tratar datos #########################

date = dataset['Fecha']
festive = dataset['Festivo']
weekday = dataset['DiaSemana']
temperature = dataset['Temperatura']
rainfall = dataset['Precipitacion']
amount_users = dataset.iloc[:,[5]].values

# Se definen las entradas
input = festive,weekday,temperature,rainfall
input = np.array(input)
input = input.transpose()
output = np.array(amount_users)

# Definir valores de training y testing 
x_train, x_test, y_train, y_test, date_train, date_test = train_test_split(input, output, date, test_size=0.2, random_state= 71) #shuffle=False
date_train = date_train.astype(str)
date_test = date_test.astype(str)

# Normalizar entre 0 y 1
min_max_scaler_x = preprocessing.MinMaxScaler()
x_train = min_max_scaler_x.fit_transform(x_train)
x_test = min_max_scaler_x.transform(x_test)

min_max_scaler_y = preprocessing.MinMaxScaler() 
y_train = min_max_scaler_y.fit_transform(y_train)
y_test = min_max_scaler_y.transform(y_test)

# Valores originales
y_train = min_max_scaler_y.inverse_transform(y_train)
y_test = min_max_scaler_y.inverse_transform(y_test)
    
# Predicción dnn    
dnn_predictions = model.predict(x_train)
dnn_predictions_test = model.predict(x_test)
dnn_predictions = min_max_scaler_y.inverse_transform(dnn_predictions)
dnn_predictions_test = min_max_scaler_y.inverse_transform(dnn_predictions_test)

# Mostrar gráfico training
showGraph(date_train[:,0],y_train,dnn_predictions,date_train[::20,0],'results/dnn_train.jpg')
# Mostrar gráfico testing
showGraph(date_train[:,0],y_test,dnn_predictions_test,date_test[::10,0],'results/dnn_test.jpg')

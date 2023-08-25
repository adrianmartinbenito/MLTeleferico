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

def showGraph(date,y,y_predict,date_show,save_location):
    
    plt.figure(figsize=(30,18))
    plt.plot(date,y[:],"g")
    plt.plot(date,y_predict[0],"r")
    plt.xticks(date_show) #rotation ='vertical'
    plt.legend(['Actual','Predicted'])
    plt.savefig(save_location,dpi=500)
    plt.show()
    
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

input_size = 4
hidden_size = 1000
input_weights = abs(np.random.normal(size=[input_size,hidden_size]))
biases = abs(np.random.normal(size=[hidden_size]))

def relu(x):
    return np.maximum(x, 0, x)

def hidden_nodes(x):
    G = np.dot(x, input_weights)
    G = G + biases
    H = relu(G)
    return H

output_weights = np.dot(linalg.pinv(hidden_nodes(x_train)), y_train)

def predict(x):
    out = hidden_nodes(x)
    out = np.dot(out, output_weights)
    return out

prediction = predict(x_test)

""" correct = 0
total = x_test.shape[0]
for i in range(total):
    predicted = np.argmax(prediction[i])
    actual = np.argmax(y_test[i])
    correct += 1 if predicted == actual else 0
accuracy = correct/total
print("###########################")
print('Accuracy for ', hidden_size, ' hidden nodes: ', accuracy) """

###################### Prediciones ##########################

# Valores originales
y_train = min_max_scaler_y.inverse_transform(y_train)
y_test = min_max_scaler_y.inverse_transform(y_test)

# Predicción elm
elm_predictions = predict(x_train)
elm_predictions_test = predict(x_test)
elm_predictions = min_max_scaler_y.inverse_transform(elm_predictions)
elm_predictions_test = min_max_scaler_y.inverse_transform(elm_predictions_test)

# Mostrar gráfico training
showGraph(date_train[:,0],y_train,elm_predictions,date_train[::20,0],'results/elm_train.jpg')
# Mostrar gráfico testing
showGraph(date_train[:,0],y_test,elm_predictions_test,date_test[::10,0],'results/elm_test.jpg')

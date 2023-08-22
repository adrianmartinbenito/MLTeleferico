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

########################## SVM ##############################

if SVM:
    
    K = 5 #Cross variations
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    parameters = [{'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,3],'degree' : [3,8],'coef0' : [0.01,0.1,0.5],'gamma' : ('auto','scale')}]
    model = GridSearchCV(SVR( epsilon=0.1), parameters , cv= K, scoring='mean_squared_error' ,n_jobs = -1, verbose = 1)
    
    #model = SVR(epsilon=0.01,kernel='rbf',gamma= 'scale', C=5, coef0=0.01, degree=3 ,verbose=1)
    model.fit(x_train, y_train.ravel())
    print(model.score(x_test,y_test.ravel()))
    
    # Comprobar el score de cada parámetro
    print(model.best_params_)
    
    print(model.best_score_)
    
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score'] 
    for mean, std, params in zip(means, stds, model.cv_results_['params']): 
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    
    
 
########################## ELM ##############################

if ELM:
    input_size = 4
    hidden_size = 1000
    input_weights = abs(np.random.normal(size=[input_size,hidden_size]))
    biases = abs(np.random.normal(size=[hidden_size]))

    def relu(x):
        return np.maximum(x, 0, x)

    def hidden_nodes(X):
        G = np.dot(X, input_weights)
        G = G + biases
        H = relu(G)
        return H

    output_weights = np.dot(linalg.pinv(hidden_nodes(x_train)), y_train)

    def predict(X):
        out = hidden_nodes(X)
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

########################## RED ##############################

if DNN:
    epochs = 100    # Número de epochs para el entrenamiento del modelo
    batch_size = 16     # Batch size para el entrenamiento del modelo
    learning_rate = 0.001
    input_shape = 4
    LAGS = 200

    model = models.Sequential()

    model.add(layers.Conv1D(batch_size, 2, activation="relu", input_shape = (LAGS,input_shape)))
    model.add(layers.AveragePooling1D(pool_size=27, strides=1, padding='valid'))
    model.add(layers.LSTM(units = 12,activation="sigmoid", return_sequences=False))
    model.add(layers.Dense(units = 1,input_shape = (LAGS,input_shape)))
        

    # Compilamos el modelo
    model.compile(optimizer="RMSProp",
        loss= "mae",
        metrics=[tf.keras.metrics.MeanAbsoluteError(),tf.keras.metrics.RootMeanSquaredError(),"MAPE"])

    print(model.summary())

    history = model.fit(x=x_train, y=y_train, epochs=epochs,batch_size=batch_size, verbose=1)


###################### Prediciones ##########################

# Valores originales
y_train = min_max_scaler_y.inverse_transform(y_train)
y_test = min_max_scaler_y.inverse_transform(y_test)

if DNN:
    # Predicción dnn    
    dnn_predictions = model.predict(x_train)
    dnn_predictions_test = model.predict(x_test)
    dnn_predictions = min_max_scaler_y.inverse_transform(dnn_predictions)
    dnn_predictions_test = min_max_scaler_y.inverse_transform(dnn_predictions_test)
    
    # Mostrar gráfico training
    showGraph(date_train[:,0],y_train,dnn_predictions,date_train[::20,0],'results/dnn_train.jpg')
    # Mostrar gráfico testing
    showGraph(date_train[:,0],y_test,dnn_predictions_test,date_test[::10,0],'results/dnn_test.jpg')


if SVM:
    # Predicción svm
    svm_predictions = model.predict(x_train)
    svm_predictions_test = model.predict(x_test)
    svm_predictions = min_max_scaler_y.inverse_transform([svm_predictions])
    svm_predictions_test = min_max_scaler_y.inverse_transform([svm_predictions_test])
    
    # Mostrar gráfico training
    showGraph(date_train,y_train,svm_predictions,date_train[::40],'results/svm_train.jpg')
    # Mostrar gráfico testing
    showGraph(date_test,y_test,svm_predictions_test,date_test[::10],'results/svm_test.jpg')

if ELM:
    # Predicción elm
    elm_predictions = predict(x_train)
    elm_predictions_test = predict(x_test)
    elm_predictions = min_max_scaler_y.inverse_transform(elm_predictions)
    elm_predictions_test = min_max_scaler_y.inverse_transform(elm_predictions_test)
    
    # Mostrar gráfico training
    showGraph(date_train[:,0],y_train,elm_predictions,date_train[::20,0],'results/elm_train.jpg')
    # Mostrar gráfico testing
    showGraph(date_train[:,0],y_test,elm_predictions_test,date_test[::10,0],'results/elm_test.jpg')



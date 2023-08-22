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

dataset = pd.read_excel('resources/dataset_completo.xlsx')

#######################################################################
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

#######################################################################

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

#######################################################################
# Valores originales
y_train = min_max_scaler_y.inverse_transform(y_train)
y_test = min_max_scaler_y.inverse_transform(y_test)

# Predicción svm
svm_predictions = model.predict(x_train)
svm_predictions_test = model.predict(x_test)
svm_predictions = min_max_scaler_y.inverse_transform([svm_predictions])
svm_predictions_test = min_max_scaler_y.inverse_transform([svm_predictions_test])

# Mostrar gráfico training
showGraph(date_train,y_train,svm_predictions,date_train[::40],'results/svm_train.jpg')
# Mostrar gráfico testing
showGraph(date_test,y_test,svm_predictions_test,date_test[::10],'results/svm_test.jpg')

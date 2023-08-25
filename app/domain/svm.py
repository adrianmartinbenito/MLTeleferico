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

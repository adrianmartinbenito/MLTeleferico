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
from sklearn.metrics import explained_variance_score


    

###################### Leer datos ###########################


####################### Funciones ###########################


    
###################### Tratar datos #########################


########################## SVM ##############################


 
########################## ELM ##############################

""" if ELM:
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

    prediction = predict(x_test) """

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


###################### Prediciones ##########################

# Valores originales

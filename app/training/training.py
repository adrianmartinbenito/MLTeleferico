
from sklearn.svm import SVR
from keras import layers,models

def svm_train_model(args): 
    #K = 50 #Cross variations
    #scorer = make_scorer(mean_squared_error, greater_is_better=False)
    #parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]
    #model = GridSearchCV(SVR( epsilon=0.01), parameters , cv= K, scoring=scorer)
    #model = SVR(kernel='rbf',gamma=1, C=10, epsilon = 0.01)
    model = SVR(kernel=args.kernel,gamma=args.gamma, C=args.c, epsilon = args.epsilon)
    model.fit(x_train, y_train.ravel())
    # Comprobar el score de cada parámetro
    #print("Grid scores on training set:")
    #means = model.cv_results_['mean_test_score']
    #stds = model.cv_results_['std_test_score'] 
    #for mean, std, params in zip(means, stds, model.cv_results_['params']):
        #    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

def dnn_train_model(args):
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

def elm_train_model(args):
    print("elm")
    
def train_model(args,parser):
    if args.model == "svm":
        svm_train_model(args)
    elif args.model == "elm":
        svm_train_model(args)
    elif args.model == "dnn" :
        svm_train_model(args)
    else:
        svm_train_model(args)

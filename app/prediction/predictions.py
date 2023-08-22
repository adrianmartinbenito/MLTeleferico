
def showGraph(date,y,y_predict,date_show,save_location):
    
    plt.figure(figsize=(30,18))
    plt.plot(date,y[:],"g")
    plt.plot(date,y_predict[0],"r")
    plt.xticks(date_show) #rotation ='vertical'
    plt.legend(['Actual','Predicted'])
    plt.savefig(save_location,dpi=500)
    plt.show()
    
def dnn_predictions(args):
    # Predicción dnn    
    dnn_predictions = model.predict(x_train)
    dnn_predictions_test = model.predict(x_test)
    dnn_predictions = min_max_scaler_y.inverse_transform(dnn_predictions)
    dnn_predictions_test = min_max_scaler_y.inverse_transform(dnn_predictions_test)
    # Mostrar gráfico training
    showGraph(date_train[:,0],y_train,dnn_predictions,date_train[::20,0],'results/dnn_train.jpg')
    # Mostrar gráfico testing
    showGraph(date_train[:,0],y_test,dnn_predictions_test,date_test[::10,0],'results/dnn_test.jpg')


def svm_predictions(args):
    # Predicción svm
    svm_predictions = model.predict(x_train)
    svm_predictions_test = model.predict(x_test)
    svm_predictions = min_max_scaler_y.inverse_transform([svm_predictions])
    svm_predictions_test = min_max_scaler_y.inverse_transform([svm_predictions_test])
    # Mostrar gráfico training
    showGraph(date_train,y_train,svm_predictions,date_train[::40],'results/svm_train.jpg')
    # Mostrar gráfico testing
    showGraph(date_test,y_test,svm_predictions_test,date_test[::10],'results/svm_test.jpg')

def elm_predictions(args):
    # Predicción elm
    elm_predictions = predict(x_train)
    elm_predictions_test = predict(x_test)
    elm_predictions = min_max_scaler_y.inverse_transform(elm_predictions)
    elm_predictions_test = min_max_scaler_y.inverse_transform(elm_predictions_test)
    # Mostrar gráfico training
    showGraph(date_train[:,0],y_train,elm_predictions,date_train[::20,0],'results/elm_train.jpg')
    # Mostrar gráfico testing
    showGraph(date_train[:,0],y_test,elm_predictions_test,date_test[::10,0],'results/elm_test.jpg')

def get_data(args):
    y_train = min_max_scaler_y.inverse_transform(y_train)
    y_test = min_max_scaler_y.inverse_transform(y_test)

def predictions(args,parser):
    if args.model == "SVM":
        svm_train_model(args)
    elif args.model == "ELM":
        svm_train_model(args)
    elif args.model == "DNN" :
        svm_train_model(args)
    else:
        svm_train_model(args)

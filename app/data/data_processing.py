
def read_dataset():
    dataset = pd.read_excel('resources/dataset_completo.xlsx')

def data_treatment():
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
    x_train, x_test, y_train, y_test, date_train, date_test = train_test_split(input, output, date, test_size=0.2, random_state= 0) #shuffle=False
    date_train = date_train.astype(str)
    date_test = date_test.astype(str)

    # Normalizar entre 0 y 1
    min_max_scaler_x = preprocessing.MinMaxScaler()
    x_train = min_max_scaler_x.fit_transform(x_train)
    x_test = min_max_scaler_x.transform(x_test)

    min_max_scaler_y = preprocessing.MinMaxScaler() 
    y_train = min_max_scaler_y.fit_transform(y_train)
    y_test = min_max_scaler_y.transform(y_test)

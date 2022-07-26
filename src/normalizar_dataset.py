
import pandas as pd
import numpy as np
from sklearn import preprocessing
import xlsxwriter

###################### Leer datos ###########################

usersDB = pd.read_excel('resources/Usuarios_Teleferico.xlsx')   #Usuarios_Teleférico.xlsx
meteorDB = pd.read_excel('resources/Madrid_Meteorologia.xlsx')  #Madrid_datos_bici.xlsx
festiveDB = pd.read_csv('resources/Tipo_Dia.csv', sep = ';') #datos_limpio.csv

####################### Funciones ###########################

def controlNullPeople(date,date2,amount_users):
    newAmount = []
    for i in range(len(date2)):
        if(date2[i] in date):
            newAmount.append(amount_users[np.where(date == date2[i])])
        else:
            newAmount.append((newAmount[i-1]+newAmount[i-2]+newAmount[i-3]+newAmount[i-4]+newAmount[i-5])/5)
    return newAmount       

def convertToList(array):
    newList = []
    for i in array:
        newList.append(i[0])
        
    return newList   
###################### Tratar datos #########################

# Primer excel
usersDB.dropna(subset = ['Fecha'], inplace=True)
usersDB = usersDB.groupby('Fecha',as_index =False).sum()
date = usersDB.iloc[:,[0]].values
amount_users = usersDB.iloc[:,[2]].values

# Segundo excel
meteorDB.dropna(subset = ['Fecha'], inplace=True)
meteorDB = meteorDB[(meteorDB.Fecha >= date[0,0]) & (meteorDB.Fecha <= date[date.size-1,0])]
date2 = meteorDB.iloc[:,[0]].values
weekday = meteorDB.iloc[:,[1]].values 
temperature = meteorDB.iloc[:,[4]].values 
rainfall = meteorDB.iloc[:,[5]].values 

# Tercer excel
festiveDB = festiveDB[(pd.to_datetime(festiveDB['Dia'], dayfirst=True) >= date[0,0]) & (pd.to_datetime(festiveDB['Dia'], dayfirst=True) <= date[date.size-1,0])]
festive = festiveDB.iloc[:,[3]].values
# Pasar strings a numericos
encoder = preprocessing.LabelEncoder()
encoder.fit(festive.ravel())
festive = [encoder.transform(festive.ravel())]
festive = np.transpose(festive)

# Se introducen valores nuevos que antes eran nulos
amount_users_modified = controlNullPeople(date,date2,amount_users)

# Se genera el nuevo dataset con todos los datos necesarios
writer = pd.ExcelWriter('resources/dataset_completo.xlsx', engine='xlsxwriter')
date2 = convertToList(date2)
festive = convertToList(festive)
weekday = convertToList(weekday)
temperature = convertToList(temperature)
rainfall = convertToList(rainfall)
amount_users_modified = convertToList(amount_users_modified)

df = pd.DataFrame({'Fecha':date2,
                   'Festivo':festive,
                   'DiaSemana':weekday,
                   'Temperatura':temperature,
                   'Precipitacion':rainfall,
                   'CantidadUsuarios':amount_users_modified})

df.to_excel(writer, sheet_name='teleferico', index=False)
writer.save()
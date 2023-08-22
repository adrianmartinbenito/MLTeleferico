from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd

# Carga de datasets (como ejemplo)
def cargar_dataset(nombre_dataset):
    # Aquí puedes agregar lógica para cargar diferentes conjuntos de datos.
    if nombre_dataset == "dataset1":
        print("dataset1")
        return pd.read_csv("resources/completed/dataset_completo_1.xlsx")
    elif nombre_dataset == "dataset2":
        print("dataset2")
        return pd.read_csv("resources/completed/dataset_completo_2.xlsx")
    elif nombre_dataset == "dataset3":
        print("dataset3")
        return pd.read_csv("resources/completed/dataset_completo_3.xlsx")

def elegir_modelo(nombre_modelo):
    if nombre_modelo == "svm":
        print("svm")
    elif nombre_modelo == "elm":
        print("elm")
    elif nombre_modelo == "dnn":
        print("dnn")

""" def entrenar_modelo(modelo, dataset, porcentaje_test):
    X = dataset.drop('target', axis=1)  # Asumiendo que 'target' es la columna objetivo
    y = dataset['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=porcentaje_test)
    modelo.fit(X_train, y_train)
    
    score = modelo.score(X_test, y_test)
    return modelo, score
 """
if __name__ == "__main__":
    print("Elegir un algoritmo: logistic_regression, random_forest, svm")
    nombre_modelo = input()
    modelo = elegir_modelo(nombre_modelo)

    print("Elegir un dataset: dataset1, dataset2, dataset3")
    nombre_dataset = input()
    dataset = cargar_dataset(nombre_dataset)

    print("Especificar el porcentaje para pruebas (como un decimal, por ejemplo, 0.2 para 20%):")
    porcentaje_test = float(input())

    
    
    
"""     modelo_entrenado, score = entrenar_modelo(modelo, dataset, porcentaje_test)
    
    print(f"Modelo entrenado con una precisión de {score*100:.2f}% en el conjunto de pruebas.") """

    # Aquí puedes agregar funcionalidad para hacer predicciones si lo deseas.

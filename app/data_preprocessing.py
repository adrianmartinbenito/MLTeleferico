""" 
Este módulo tiene como objetivo recoger los datos de entrada y realizar un preprocesamiento de esos
datos para poder analizarlos de la manera más óptima y sencilla posible.

"""

from pathlib import Path
from pandas import DataFrame, Series, ExcelWriter
import pandas as pd

def data_reorganization(input_file: Path, output_file: Path):
    """ 
    Esta función permite generar un nuevo dataset mostrando la relación con los días más próximos,
    por cada día, se miran los cuatro días anteriores obteniendo su cantidad de usuarios, festividad y dia de la semana,
    mientras que también se muestra el día de mañana pero sin coger la cantidad de usuarios.
    
    """
    
    reader: DataFrame = pd.read_excel(input_file)

    festive: Series = reader["Festivo"]
    day_of_the_week: Series = reader["DiaSemana"]
    temperature: Series = reader["Temperatura"]
    precipitation: Series = reader["Precipitacion"]
    number_users: Series = reader["CantidadUsuarios"]

    writer = ExcelWriter(output_file, engine='xlsxwriter')

    final_dataset = DataFrame({
        'Festivo Hoy': festive.values.tolist(),
        'DiaSemana Hoy': day_of_the_week.values.tolist(),
        'Temperatura Hoy': temperature.values.tolist(),
        'Precipitacion Hoy': precipitation.values.tolist(),
        'Festivo Mannana': festive.shift(-1).values.tolist(),
        'DiaSemana Mannana': day_of_the_week.shift(-1).values.tolist(),
        'CantidadUsuarios Hoy-1': number_users.shift(1).values.tolist(),
        'Festivo Hoy-1': festive.shift(1).values.tolist(),
        'DiaSemana Hoy-1': day_of_the_week.shift(1).values.tolist(),
        'CantidadUsuarios Hoy-2': number_users.shift(2).values.tolist(),
        'Festivo Hoy-2': festive.shift(2).values.tolist(),
        'DiaSemana Hoy-2': day_of_the_week.shift(2).values.tolist(),
        'CantidadUsuarios Hoy-3': number_users.shift(3).values.tolist(),
        'Festivo Hoy-3': festive.shift(3).values.tolist(),
        'DiaSemana Hoy-3': day_of_the_week.shift(3).values.tolist(),
        'CantidadUsuarios Hoy-4': number_users.shift(4).values.tolist(),
        'Festivo Hoy-4': festive.shift(4).values.tolist(),
        'DiaSemana Hoy-4': day_of_the_week.shift(4).values.tolist(),
        'CantidadUsuarios Hoy': number_users.values.tolist(),
    })

    final_dataset = final_dataset.iloc[4:].astype(float)
    
    final_dataset.to_excel(writer, sheet_name='teleferico', index=False)
    writer.save()


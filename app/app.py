from pathlib import Path
import sys
import json
import argparse
from typing import Dict
from training.training import train_model
from config.config import env_config
from logger.logger import logger

def parse_args(args):
    parser = argparse.ArgumentParser(description="Entrenamiento y comparación de modelos de machine learning")

    # Argumentos obligatorios
    parser.add_argument("--data_path", help="ruta del archivo con los datos de entrenamiento")

    # Argumentos obligatorios para modelos de machine learning
    parser.add_argument("--model", help="tipo de modelo 1 a entrenar", default="svm")

    # Argumentos opcionales para hiperparámetros de modelos de machine learning
    parser.add_argument("--hyper_params1", help="hiperparámetros del modelo 1", default="{\"alpha\": 0.1}")
    """ parser.add_argument("--hyper_params2", help="hiperparámetros del modelo 2", default="{\"n_estimators\": 100}")
    parser.add_argument("--hyper_params3", help="hiperparámetros del modelo 3", default="{\"hidden_layer_sizes\": (50, 50)}")
 """
    # Argumentos opcionales para entrenamiento y validación
    parser.add_argument("--test_size", type=float, help="proporción de los datos para validación", default=0.2)
    parser.add_argument("--shuffle", help="barajar los datos antes de hacer el entrenamiento y la validación", action="store_true")
    parser.add_argument("--random_seed", type=int, help="semilla aleatoria para reproducir los resultados", default=None)

    # Argumentos opcionales para guardar modelos y resultados
    parser.add_argument("--save_models", help="guardar los modelos entrenados", action="store_true")
    parser.add_argument("--output_path", help="ruta donde guardar los modelos entrenados", default="models/")
    parser.add_argument("--evaluation_metric", help="métrica de evaluación para comparar modelos", default="accuracy")

    return parser.parse_args(args), parser

def main(options):
    logger.info(f"Begin training. Parms: {options}")
    args, parser = parse_args(options)
    train_model(args,parser)

if __name__ == "__main__":
    main(sys.argv[1:])

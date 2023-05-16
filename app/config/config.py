# Al importar setting realiza carga por defecto de ficheros 
# de configuración. En este caso importa las variables de 
# entorno que se encuentra en el fichero .env del
# directorio de trabajo.

from dynaconf import Dynaconf
env_config = Dynaconf(
    envvar_prefix="MACHINELEARNING",
    load_dotenv=True,
    dotenv_verbose=True
)


import logging
from pathlib import Path

from config.config import env_config

#Crear el directorio de logs si no existe
path = Path(env_config['log']['fichero_log']).resolve()
path.parent.mkdir(parents=True, exist_ok=True) 

"""
    Código extraído de:
       - https://github.com/tiangolo/fastapi/issues/1276
          
          No saca en el log las peticiones http
          No saca el payload indicado en la traza


       - Intento aplicar https://github.com/tiangolo/fastapi/issues/1276#issuecomment-663781706 pero no tiene efecto

       - Revisar https://medium.com/1mgofficial/how-to-override-uvicorn-logger-in-fastapi-using-loguru-124133cdcd4e


"""

import sys
from pprint import pformat

from loguru import logger
from loguru._defaults import LOGURU_FORMAT

class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentaion.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def format_log(record: dict) -> str:
    """
    Custom format for loguru loggers.
    Uses pformat for log any data like request/response body during debug.
    Works with logging if loguru handler it.

    Example:
    >>> payload = [{"users":[{"name": "Nick", "age": 87, "is_active": True}, {"name": "Alex", "age": 27, "is_active": True}], "count": 2}]
    >>> logger.bind(payload=).debug("users payload")
    >>> [   {   'count': 2,
    >>>         'users': [   {'age': 87, 'is_active': True, 'name': 'Nick'},
    >>>                      {'age': 27, 'is_active': True, 'name': 'Alex'}]}]
    """
    format_string = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "\
                    "<level>{level: <8}</level> | "\
                    "<level>{process}</level> | "\
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "\
                    "<level>{message}</level>"

    if record["extra"].get("payload") is not None:
        record["extra"]["payload"] = pformat(
            record["extra"]["payload"], indent=4, compact=True, width=88
        )
        format_string += "\n<level>{extra[payload]}</level>"

    format_string += "{exception}\n"
    return format_string

def format_stats(record: dict) -> str:
    """
    Custom format for statistics.
    
    Example:
    >>> payload = [{"users":[{"name": "Nick", "age": 87, "is_active": True}, {"name": "Alex", "age": 27, "is_active": True}], "count": 2}]
    >>> logger.bind(payload=).debug("users payload")
    >>> [   {   'count': 2,
    >>>         'users': [   {'age': 87, 'is_active': True, 'name': 'Nick'},
    >>>                      {'age': 27, 'is_active': True, 'name': 'Alex'}]}]
    """
    format_string = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "\
                    "<level>{process}</level> | "\
                    "<level>{message}</level>"

    if record["extra"].get("payload") is not None:
        record["extra"]["payload"] = pformat(
            record["extra"]["payload"], indent=4, compact=True, width=88
        )
        format_string += "\n<level>{extra[payload]}</level>"
    return format_string


# set loguru format for root logger
logging.getLogger().handlers = [InterceptHandler()]
# set format
logger.configure(
    handlers=[{"sink": env_config['log']['fichero_log'], 
               "level": logging.DEBUG, 
               #"enqueue": True,
               "rotation": "00:00",
               "retention": env_config['log']['num_max_ficheros_log'],
               "format": format_log}]
)

import logging
import os


def init_logger(log_file=""):
    """ Initialize a logger

    :param log_file: the file path to save logs
    :type log_file: str
    :return: a logger that binds the console and the file
    :rtype: logging.RootLogger
    """
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def close_logger(logger):
    """ Close the logger safely

    :param logger: a logger to close
    :type logger: logging.RootLogger
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
import logging

class Logger:

    def __init__(self):
        pass

    def log_tabular(self):
        pass


def get_logger(log_file_path, name="Unknown"):
    logger = logging.getLogger(name)
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path)
    c_handler.setLevel(logging.WARNING)
    f_handler.setLevel(logging.WARNING)

    c_format = logging.Formatter("%(levelname)s - %(message)s")
    f_format = logging.Formatter("%(levelname)s - %(message)s")

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger


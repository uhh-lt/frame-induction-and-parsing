import logging
import sys
import os


def create_logger(name, log_dir, level=logging.WARNING):
    logger = logging.getLogger(name)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fhandler = logging.FileHandler(filename=os.path.join(log_dir, name+'.log'), mode='a')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(level)
    
    return logger

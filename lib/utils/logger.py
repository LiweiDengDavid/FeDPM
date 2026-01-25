import os
import sys
import logging

def get_logger(log_dir, name, log_filename, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)

    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print('Log directory:', log_dir)
    
    return logger


def get_color_logger(log_dir, name, log_filename, level=logging.INFO):
    import colorlog 
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)


    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(file_formatter)


    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(message)s',
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        print('Log directory:', log_dir)
    
    return logger
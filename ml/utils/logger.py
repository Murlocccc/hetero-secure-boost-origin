import logging
import time
import os


class LOGGER_V1:


    mask = {
        'info': True,
        'debug': True,
        'debug_data': False,
        'warning': False,
    }

    @classmethod
    def info(cls, str: str):
        if cls.mask['info']:
            print('info: ', str)

    @classmethod
    def debug(cls, str: str):
        if cls.mask['debug']:
            print('debug: ', str)

    @classmethod
    def debug_data(cls, str: str):
        if cls.mask['debug_data']:
            print('debug_data: ', str)
    
    @classmethod
    def warning(cls, str: str):
        if cls.mask['warning']:
            print('warning: ', str)

class MyLoggerFactory:
    logger = logging.getLogger()
    @classmethod
    def build(cls,user_name:str="default"):
        # cls.logger.set
        cls.logger.setLevel(logging.DEBUG)
        if not os.path.exists('log'):
            os.mkdir("log")
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [ %(filename)s:%(lineno)s]\t%(message)s")
        log_file_name = f"log/{user_name}_{time.strftime('%Y-%m-%d-%H_%M_%S')}.log"
        
        fh = logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
        fh.setLevel(cls.logger.level)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(cls.logger.level)
        ch.setFormatter(formatter)

        cls.logger.addHandler(fh)
        cls.logger.addHandler(ch)
        return cls.logger

    @classmethod
    def get_logger(cls):
        return LOGGER_V1 if cls.logger is None else cls.logger


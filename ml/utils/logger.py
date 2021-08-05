import logging



class LOGGER:

    @classmethod
    def basic_config(cls, **kwargs):
        logging.basicConfig(**kwargs)

    @classmethod
    def info(cls, info: str):
        logging.info(info)

    @classmethod
    def debug(cls, debug: str):
        logging.debug(debug)

    @classmethod
    def warning(cls, warning: str):
        logging.warning(warning)

    # mask = {
    #     'info': True,
    #     'debug': True,
    #     'debug_data': False,
    #     'warning': False,
    # }

    # @classmethod
    # def info(cls, str: str):
    #     if cls.mask['info']:
    #         print('info: ', str)

    # @classmethod
    # def debug(cls, str: str):
    #     if cls.mask['debug']:
    #         print('debug: ', str)

    # @classmethod
    # def debug_data(cls, str: str):
    #     if cls.mask['debug_data']:
    #         print('debug_data: ', str)
    
    # @classmethod
    # def warning(cls, str: str):
    #     if cls.mask['warning']:
    #         print('warning: ', str)

class MyLoggerFactory:
    logger = logging.getLogger()
    @classmethod
    def build(cls,user_name:str="default"):
        cls.logger = logging.getLogger(user_name)
        cls.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [ %(filename)s:%(lineno)s]\t%(message)s")
        log_file_name = f"{user_name}.log"
        
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
        return LOGGER if cls.logger is None else cls.logger


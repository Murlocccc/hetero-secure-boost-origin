
class LOGGER:

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
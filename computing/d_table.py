from computing._computing import CTableABC
import typing
import copy

class DTable(CTableABC):
    
    def __init__(self, include_key: bool, data: typing.List=None) -> None:
        super().__init__()

        if data is None:
            self.__data = {}
        else:
            if not isinstance(include_key, bool):
                raise TypeError('the param inculde_key should be a bool !')
            elif not isinstance(data, typing.List):
                raise TypeError('the param data should be a List, while the input type is {} !'.format(type(data)))
            else:
                if include_key == True:
                    self.__data = {}
                    for item in data:
                        # 下面这句话跑不通不知道问什么，只好换个写法
                        # if isinstance(item, typing.Tuple[typing.Any, typing.Any]):
                        if isinstance(item, typing.Tuple) and len(item) == 2:
                            if item[0] in self.__data:
                                raise ValueError('the param data has duplicate key: {}'.format(item[0]))
                            else:
                                self.__data[item[0]] = item[1]
                        else:
                            raise TypeError('every element in data should be a Tuper[Object, Object] while include_key = True !')
                else:  # include_key == False
                    self.__data = {}
                    for i,item in enumerate(data):
                        self.__data[i] = item

    def __str__(self) -> str:
        return str(self.schema) + '\n' + str(self.__data)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def collect(self, **kargs) -> typing.Iterator:
        return self.__data.items()

    def take(self, n: int=1, **kargs) -> list:
        ret_val = []
        for i, k in enumerate(self.__data.keys()):
            if i >= n:
                break
            else:
                ret_val.append((k, self.__data[k]))
        return ret_val

    # 这个写的是真的丑，有空再想想怎么改
    def first(self, **kargs):
        if len(self.__data) == 0:
            return None
        else:
            k = list(self.__data.keys())[0]
            return self.__data[k]

    def count(self) -> int:
        return len(self.__data)

    def map(self, func: typing.Callable) -> 'CTableABC':
        new_table = []
        for k, v in self.__data.items():
            new_table.append((k, func(k, v)))
        return DTable(True, new_table)
    
    def mapValues(self, func: typing.Callable):
        new_table = []
        for k, v in self.__data.items():
            new_table.append((k, func(v)))
        return DTable(True, new_table)

    def mapPartitions(self, func: typing.Callable):
        new_table = func(self.__data.items())
        return DTable(False, new_table)

    def mapReducePartitions(self, mapper: typing.Callable, reducer: typing.Callable, **kargs):
        map_list = mapper(self.__data.items())
        new_dict = {}
        for k, v in map_list:
            if k not in new_dict:
                new_dict[k] = v
            else:
                new_dict[k] = reducer(new_dict[k], v)
        return DTable(True, list(new_dict.items()))

    def reduce(self, func: typing.Callable):
        ret_val = None
        for v in self.__data.values():
            if ret_val is None:
                ret_val = v
            else:
                ret_val = func(ret_val, v)
        return ret_val

    def filter(self, func: typing.Callable):
        new_table = []
        for k, v in self.__data.items():
            if func(k, v) == True:
                new_table.append((k, v))
        return DTable(True, new_table)

    def join(self, other, func: typing.Callable=lambda v1, v2: v1):
        new_table = []
        for k, v in other.get_kvs():
            if k in self.__data:
                new_table.append((k, func(self.__data[k], v)))
        return DTable(True, new_table)

    def union(self, other, func: typing.Callable=lambda v1, v2: v1):
        new_dict = copy.deepcopy(self.__data)
        for k, v in other.get_kvs():
            if k in new_dict:
                new_dict[k] = func(new_dict[k], v)
            else:
                new_dict[k] = v
        return DTable(True, list(new_dict.items()))

    def subtractByKey(self, other):
        new_table = []
        for k, v in self.__data.items():
            if k not in other.get_keys():
                new_table.append((k, v))
        return DTable(True, new_table)
    
    def get_keys(self) -> typing.KeysView:
        return self.__data.keys()
    
    def get_kvs(self) -> typing.ItemsView:
        return self.__data.items()

    def get_values(self) ->typing.ValuesView:
        return self.__data.values()
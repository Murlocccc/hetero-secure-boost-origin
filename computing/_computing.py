import abc
import typing


class CTableABC(abc.ABC):

    @abc.abstractmethod
    def collect(self, **kargs) -> typing.Iterator:
        """
        Returns
        -------
        generator
           generator of data
        """
        ...

    @abc.abstractmethod
    def take(self, n: int =1, **kargs) -> list:
        """
        take ``n`` data from table

        Parameters
        ----------
        n: int
          number of data to take

        Returns
        -------
        list
           a list of ``n`` data

        Notes
        ------
        no order guarantee
        """
        ...

    @abc.abstractmethod
    def first(self, **kargs):
        """
        take one data from table

        Returns
        -------
        object
          a data from table

        Notes
        ------
        no order guarantee
        """
        ...
    
    @abc.abstractmethod
    def count(self) -> int:
        """
        number of data in table

        Returns
        -------
        int
           number of data
        """
        ...
    
    @abc.abstractmethod
    def map(self, func: typing.Callable) -> 'CTableABC':
        
        ...
    
    @abc.abstractmethod
    def mapValues(self, func: typing.Callable):
        ...
    
    @abc.abstractmethod
    def mapPartitions(self, func:typing.Callable):
        ...

    @abc.abstractmethod
    def mapReducePartitions(self, mapper: typing.Callable, reducer: typing.Callable, **kargs):
        ...

    @abc.abstractmethod
    def reduce(self, func: typing.Callable):
        ...

    @abc.abstractmethod
    def filter(self, func: typing.Callable):
        ...

    @abc.abstractmethod
    def join(self, other, func: typing.Callable):
        ...

    @abc.abstractmethod
    def union(self, other, func: typing.Callable=lambda v1, v2: v1):
       ...

    @abc.abstractmethod
    def subtractByKey(self, other):
        ...

    @property
    def schema(self):
        if not hasattr(self, '_schema'):
            setattr(self, '_schema', {})
        return getattr(self, '_schema')
    
    @schema.setter
    def schema(self, value: typing.Dict):
        setattr(self, '_schema', value)
    pass
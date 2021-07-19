import functools
import math

from ml.utils.logger import LOGGER
from ml.feature.sparse_vector import SparseVector
from ml.statistic import data_overview
import numpy as np
from ml.param.feature_binning_param import FeatureBinningParam
from computing.d_table import DTable

class Binning:
    """
    This is use for discrete data so that can transform data or use information for feature selection.

    Parameters
    ----------
    params : FeatureBinningParam object,
             Parameters that user set.

    Attributes
    ----------
    cols_dict: dict
        Record key, value pairs where key is cols' name, and value is cols' index. This is use for obtain correct
        data from a data_instance

    """

    def __init__(self, params: FeatureBinningParam, party_name: str, abnormal_list: list=None) -> None:
        self.params = params
        self.bin_num = params.bin_num
        self.cols_index = params.cols
        self.cols = []
        self.cols_dict = {}
        self.party_name = party_name
        self.header = None
        if abnormal_list is None:
            self.abnormal_list = []
        else:
            self.abnormal_list = abnormal_list
        self.iv_result = None
        self.splite_points = None
    
    def _init_cols(self, data_instances: DTable):

        # Already initialized
        if len(self.cols_dict) != 0:
            return
        
        header = data_overview.get_header(data_instances)
        self.header = header
        if self.cols_index == -1:
            self.cols = header
            self.cols_index = [i for i in range(len(header))]
        else:
            cols = []
            for idx in self.cols_index:
                try:
                    idx = int(idx)
                except:
                    raise ValueError("In binning module, selected index: {} is not integer".format(idx))
                
                if idx >= len(header):
                    raise ValueError(
                        "In binning module, selected index: {} exceed length of data dimension".format(idx))
                cols.append(header[idx])
            self.cols = cols
        
        self.cols_dict = {}
        for col in self.cols:
            col_index = header.index(col)
            self.cols_dict[col] = col_index
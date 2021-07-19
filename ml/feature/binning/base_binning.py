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

    def convert_feature_to_bin(self, data_instances, transform_cols_idx=-1, split_points=None):
        self._init_cols(data_instances)
        if transform_cols_idx is None:
            return data_instances, None, None

        if transform_cols_idx == -1:
            transform_cols_idx = self.cols_index
        else:
            assert isinstance(transform_cols_idx, (list, tuple))
            for col in transform_cols_idx:
                if col not in self.cols_index:
                    raise RuntimeError("Binning Transform cols: {} should be fit before transform".format(col))

        transform_cols_idx = list(map(int, transform_cols_idx))
        if split_points is None:
            split_points = self.split_points

        is_sparse = data_overview.is_sparse_data(data_instances)
        if is_sparse:
            f = functools.partial(self._convert_sparse_data,
                                  transform_cols_idx=transform_cols_idx,
                                  split_points_dict=split_points,
                                  header=self.header)
            new_data = data_instances.mapValues(f)
        else:
            f = functools.partial(self._convert_dense_data,
                                  transform_cols_idx=transform_cols_idx,
                                  split_points_dict=split_points,
                                  header=self.header)
            new_data = data_instances.mapValues(f)
        new_data.schema = {"header": self.header}
        bin_sparse = self.get_sparse_bin(transform_cols_idx, split_points)
        split_points_result = []
        for idx, col_name in enumerate(self.header):
            if col_name not in self.split_points:
                continue
            s_ps = self.split_points[col_name]
            s_ps = np.array(s_ps)
            split_points_result.append(s_ps)
        split_points_result = np.array(split_points_result)
        return new_data, split_points_result, bin_sparse

    @staticmethod
    def _convert_sparse_data(instances, transform_cols_idx, split_points_dict, header):
        all_data = instances.features.get_all_data()
        data_shape = instances.features.get_shape()
        indice = []
        sparse_value = []
        # print("In _convert_sparse_data, transform_cols_idx: {}, header: {}, split_points_dict: {}".format(
        #     transform_cols_idx, header, split_points_dict
        # ))
        for col_idx, col_value in all_data:
            if col_idx in transform_cols_idx:
                col_name = header[col_idx]
                split_points = split_points_dict[col_name]
                bin_num = Binning.get_bin_num(col_value, split_points)
                indice.append(col_idx)
                sparse_value.append(bin_num)
            else:
                indice.append(col_idx)
                sparse_value.append(col_value)

        sparse_vector = SparseVector(indice, sparse_value, data_shape)
        instances.features = sparse_vector
        return instances

    @staticmethod
    def _convert_dense_data(instances, transform_cols_idx, split_points_dict, header):
        features = instances.features
        for col_idx, col_value in enumerate(features):
            if col_idx in transform_cols_idx:
                col_name = header[col_idx]
                split_points = split_points_dict[col_name]
                bin_num = Binning.get_bin_num(col_value, split_points)
                features[col_idx] = bin_num

        instances.features = features
        return instances

    @staticmethod
    def get_bin_num(value, split_points):
        col_bin_num = len(split_points)
        for bin_num, split_point in enumerate(split_points):
            if value <= split_point:
                col_bin_num = bin_num
                break
        col_bin_num = int(col_bin_num)
        return col_bin_num

    def get_sparse_bin(self, transform_cols_idx, split_points_dict):
            """
            Get which bins the 0 located at for each column.
            """
            result = {}
            for col_idx in transform_cols_idx:
                col_name = self.header[col_idx]
                split_points = split_points_dict[col_name]
                sparse_bin_num = self.get_bin_num(0, split_points)
                result[col_idx] = sparse_bin_num
            return result


from ml.utils.logger import LOGGER
from ml.feature.binning.base_binning import Binning
from ml.param.feature_binning_param import FeatureBinningParam
from computing.d_table import DTable
from ml.statistic import data_overview
from ml.feature.quantile_summaries import QuantileSummaries, SparseQuantileSummaries
import functools

class QuantileBinning(Binning):
    def __init__(self, params: FeatureBinningParam, party_name: str='Base', abnormal_list: list=None) -> None:
        super().__init__(params, party_name, abnormal_list=abnormal_list)
        self.summary_dict = None
    
    def fit_split_points(self, data_instances: DTable):
        """
        Apply the binning method

        Parameters
        ----------
        data_instances : DTable
            The input data

        Returns
        -------
        split_points : dict.
            Each value represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = {'x1': [0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            'x2': [1, 2, 3, 4, ...],           # The second feature
                            ...]                         # Other features

        """
        self._init_cols(data_instances)
        percentile_value = 1.0 / self.bin_num

        # calculate the split points
        percentile_rates = [i * percentile_value for i in range (1, self.bin_num)]  # 举个例子，bin_num = 4，那么rate = [0.25,0.50,0.75]
        is_sparse = data_overview.is_sparse_data(data_instances)

        if self.summary_dict is None:
            f = functools.partial(self.approxiQuantile,
                                  params=self.params,
                                  abnormal_list=self.abnormal_list,
                                  cols_dict=self.cols_dict,
                                  header=self.header,
                                  is_sparse=is_sparse)
            summary_dict = data_instances.mapPartitions(f)
            summary_dict = summary_dict.reduce(self.merge_summary_dict)
            if is_sparse:
                total_count = data_instances.count()
                for _, summary_obj in summary_dict.items():
                    summary_obj.set_total_count(total_count)
                
            self.summary_dict = summary_dict
        else:
            summary_dict = self.summary_dict
        
        split_points = {}
        for col_name, summary in summary_dict.items():
            split_point = []
            for percentile_rate in percentile_rates:
                s_p = summary.query(percentile_rate)
                if s_p not in split_point:
                    split_point.append(s_p)
            split_points[col_name] = split_point
        self.split_points = split_points

    @staticmethod
    def approxiQuantile(data_instances, params, cols_dict, abnormal_list, header, is_sparse) -> dict:
        """
        Calculates each quantile information

        Parameters
        ----------
        data_instances : DTable
            The input data

        cols_dict: dict
            Record key, value pairs where key is cols' name, and value is cols' index.

        params : FeatureBinningParam object,
                Parameters that user set.

        abnormal_list: list, default: None
            Specify which columns are abnormal so that will not static when traveling.

        header: list,
            Storing the header information.

        is_sparse: bool
            Specify whether data_instance is in sparse type

        Returns
        -------
        summary_dict: dict
            {'col_name1': summary1,
             'col_name2': summary2,
             ...
             }

        """

        summary_dict = {}
        if not is_sparse:
            # feature_nums = len(one_piece.features)
            for col_name, _ in cols_dict.items():
                quantile_summaries = QuantileSummaries(compress_thres=params.compress_thres,
                                                       head_size=params.head_size,
                                                       error=params.error,
                                                       abnormal_list=abnormal_list)
                summary_dict[col_name] = quantile_summaries
        else:

            for col_name, _ in cols_dict.items():
                quantile_summaries = SparseQuantileSummaries(compress_thres=params.compress_thres,
                                                             head_size=params.head_size,
                                                             error=params.error,
                                                             abnormal_list=abnormal_list)
                # quantile_summaries.set_zeros_num(total_len)
                summary_dict[col_name] = quantile_summaries

        QuantileBinning.insert_datas(data_instances, summary_dict, cols_dict, header, is_sparse)
        for _, summary_obj in summary_dict.items():
            summary_obj.compress()

        return [summary_dict]

    @staticmethod
    def insert_datas(data_instances, summary_dict, cols_dict, header, is_sparse):

        # print('type is ', type(data_instances))
        for _, instance in data_instances:
            if not is_sparse:
                if type(instance).__name__ == 'Instance':
                    features = instance.features
                else:
                    features = instance
                for col_name, summary in summary_dict.items():
                    col_index = cols_dict[col_name]
                    summary.insert(features[col_index])
            else:
                data_generator = instance.features.get_all_data()
                for col_idx, col_value in data_generator:
                    col_name = header[col_idx]
                    summary = summary_dict[col_name]
                    summary.insert(col_value)

    @staticmethod
    def merge_summary_dict(s_dict1, s_dict2):
        if s_dict1 is None and s_dict2 is None:
            return None
        if s_dict1 is None:
            return s_dict2
        if s_dict2 is None:
            return s_dict1

        new_dict = {}
        for col_name, summary1 in s_dict1.items():
            summary2 = s_dict2.get(col_name)
            summary1.merge(summary2)
            new_dict[col_name] = summary1
        return new_dict

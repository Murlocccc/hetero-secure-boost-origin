
import warnings
from ml.utils.logger import LOGGER
from ml.tree.creterion import XgboostCriterion
from computing.d_table import DTable
from ml.utils import consts
from ml.tree.node import SplitInfo

class Splitter():
    def __init__(self, criterion_method, criterion_params=[0, 1], min_impurity_split=1e-2, min_sample_split=2,
                 min_leaf_node=1) -> None:
        LOGGER.info("splitter init!")
        if not isinstance(criterion_method, str):
            raise TypeError("criterion_method type should be str, but %s find" % (type(criterion_method).__name__))

        if criterion_method == "xgboost": 
            if not criterion_params:
                self.criterion = XgboostCriterion()
            else:
                try:
                    reg_lambda = float(criterion_params[0])
                    self.criterion = XgboostCriterion(reg_lambda)
                except:
                    warnings.warn("criterion_params' first criterion_params should be numeric")
                    self.criterion = XgboostCriterion()

        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node

    def find_split_single_histogram_guest(self, histogram, valid_features):
        best_fid = None
        best_gain = self.min_impurity_split - consts.FLOAT_ZERO
        best_bid = None
        best_sum_grad_l = None
        best_sum_hess_l = None
        for fid in range(len(histogram)):
            if valid_features[fid] is False:
                continue
            bin_num = len(histogram[fid])
            if bin_num == 0:
                continue
            sum_grad = histogram[fid][bin_num - 1][0]
            sum_hess = histogram[fid][bin_num - 1][1]
            node_cnt = histogram[fid][bin_num - 1][2]

            if node_cnt < self.min_sample_split:
                break

            for bid in range(bin_num):
                sum_grad_l = histogram[fid][bid][0]
                sum_hess_l = histogram[fid][bid][1]
                node_cnt_l = histogram[fid][bid][2]

                sum_grad_r = sum_grad - sum_grad_l
                sum_hess_r = sum_hess - sum_hess_l
                node_cnt_r = node_cnt - node_cnt_l

                if node_cnt_l >= self.min_leaf_node and node_cnt_r >= self.min_leaf_node:
                    gain = self.criterion.split_gain([sum_grad, sum_hess],
                                                     [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])

                    if gain > self.min_impurity_split and gain > best_gain:
                        best_gain = gain
                        best_fid = fid
                        best_bid = bid
                        best_sum_grad_l = sum_grad_l
                        best_sum_hess_l = sum_hess_l

        splitinfo = SplitInfo(sitename=consts.GUEST, best_fid=best_fid, best_bid=best_bid,
                              gain=best_gain, sum_grad=best_sum_grad_l, sum_hess=best_sum_hess_l)

        return splitinfo

    def find_split(self, histograms, valid_features):
        LOGGER.info("splitter find split of raw data")
        histogram_table = DTable(False, histograms)
        splitinfo_table = histogram_table.mapValues(lambda sub_hist:
                                                    self.find_split_single_histogram_guest(sub_hist, valid_features))
        tree_node_splitinfo = [splitinfo[1] for splitinfo in splitinfo_table.collect()]

        return tree_node_splitinfo

    def find_split_single_histogram_host(self, histogram, valid_features, sitename):
        node_splitinfo = []
        node_grad_hess = []
        for fid in range(len(histogram)):
            if valid_features[fid] is False:
                continue
            bin_num = len(histogram[fid])
            if bin_num == 0:
                continue
            node_cnt = histogram[fid][bin_num - 1][2]

            if node_cnt < self.min_sample_split:
                break

            for bid in range(bin_num):
                sum_grad_l = histogram[fid][bid][0]
                sum_hess_l = histogram[fid][bid][1]
                node_cnt_l = histogram[fid][bid][2]

                node_cnt_r = node_cnt - node_cnt_l

                if node_cnt_l >= self.min_leaf_node and node_cnt_r >= self.min_leaf_node:
                    splitinfo = SplitInfo(sitename=sitename, best_fid=fid,
                                          best_bid=bid, sum_grad=sum_grad_l, sum_hess=sum_hess_l)

                    node_splitinfo.append(splitinfo)
                    node_grad_hess.append((sum_grad_l, sum_hess_l))

        return node_splitinfo, node_grad_hess

    def find_split_host(self, histograms, valid_features, sitename=consts.HOST):
        LOGGER.info("splitter find split of host")
        histogram_table = DTable(False, histograms)
        host_splitinfo_table = histogram_table.mapValues(lambda hist:
                                                         self.find_split_single_histogram_host(hist, valid_features, sitename))

        tree_node_splitinfo = []
        encrypted_node_grad_hess = []

        for _, splitinfo in host_splitinfo_table.collect():
            tree_node_splitinfo.append(splitinfo[0])
            encrypted_node_grad_hess.append(splitinfo[1])

        return tree_node_splitinfo, encrypted_node_grad_hess

    def node_gain(self, grad, hess):
        return self.criterion.node_gain(grad, hess)

    def node_weight(self, grad, hess):
        return self.criterion.node_weight(grad, hess)
    
    def split_gain(self, sum_grad, sum_hess, sum_grad_l, sum_hess_l, sum_grad_r, sum_hess_r):
        gain = self.criterion.split_gain([sum_grad, sum_hess], \
                                         [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])
        return gain

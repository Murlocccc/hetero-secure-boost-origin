
from computing.d_table import DTable
from ml.tree.decision_tree import DecisionTree
from ml.param.boosting_tree_param import BoostingTreeParam
from ml.utils.logger import MyLoggerFactory
from ml.tree.splitter import Splitter
from ml.utils import consts
from ml.tree.node import Node
from ml.tree.feateur_histogram import FeatureHistogram
from arch.api.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta, CriterionMeta
from arch.api.boosting_tree_model_param_pb2 import DecisionTreeModelParam
import copy
import functools
import numpy as np
import os
import time
import random

LOGGER = MyLoggerFactory().get_logger()

class HeteroDecisionTreeGuest(DecisionTree):
    def __init__(self, tree_param: BoostingTreeParam, is_first=False):
        LOGGER = MyLoggerFactory().get_logger()
        LOGGER.info('hetero decision tree guest init!')
        super().__init__(tree_param)
        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node)
        self.data_bin = None
        self.grad_and_hess = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.data_bin_with_node_dispatch = None
        self.node_dispatch = None
        self.infos = None
        self.valid_features = None
        self.encrypter = None
        self.encrypted_mode_calculator = None
        self.best_splitinfo_guest = None
        self.tree_node_queue = None
        self.cur_split_nodes = None
        self.tree_ = []
        self._nodeid_purity_list = []
        self.tree_node_num = 0
        self.split_maskdict = {}
        # self.transfer_inst = HeteroDecisionTreeTransferVariable()
        self.predict_weights = None
        self.runtime_idx = 0
        self.feature_importances_ = {}
        self.y = None
        self.is_first = is_first
        self.node_dispatch_log_path = 'node_dispatch_log/'
        if self.node_dispatch_log_path!= None and not os.path.exists(self.node_dispatch_log_path):
            os.mkdir(self.node_dispatch_log_path)
        LOGGER.debug("It's the first tree" if self.is_first else "It isn't the first tree")
        
    def set_inputinfo(self, data_bin=None, grad_and_hess=None, bin_split_points=None, bin_sparse_points=None):
        LOGGER.info("set input info")
        self.data_bin = data_bin
        self.grad_and_hess = grad_and_hess
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def set_valid_features(self, valid_features=None):
        LOGGER.info("set valid features")
        self.valid_features = valid_features

    def set_encrypter(self, encrypter):
        LOGGER.info("set encrypter")
        self.encrypter = encrypter
    
    def set_encrypted_mode_calculator(self, encrypted_mode_calculator):
        self.encrypted_mode_calculator = encrypted_mode_calculator

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        # self.transfer_inst.set_flowid(flowid)
        # TODO

    def get_grad_hess_sum(self, grad_and_hess_table):
        LOGGER.info("calculate the sum of grad and hess")
        sum_grad, sum_hess = grad_and_hess_table.reduce(
            lambda value1, value2: (value1[0] + value2[0], value1[1] + value2[1]))
        return sum_grad, sum_hess

    def dispatch_all_node_to_root(self, root_id=0):
        LOGGER.info("dispatch all node to root")
        self.node_dispatch = self.data_bin.mapValues(lambda data_inst: (1, root_id))

    def update_feature_importance(self, splitinfo):
        if self.feature_importance_type == "split":
            inc = 1
        elif self.feature_importance_type == "gain":
            inc = splitinfo.gain
        else:
            raise ValueError("feature importance type {} not support yet".format(self.feature_importance_type))

        sitename = splitinfo.sitename
        fid = splitinfo.best_fid

        if (sitename, fid) not in self.feature_importances_:
            self.feature_importances_[(sitename, fid)] = 0

        self.feature_importances_[(sitename, fid)] += inc

    def update_tree_node_queue(self, splitinfos, max_depth_reach):
        LOGGER.info("update tree node, splitlist length is {}, tree node queue size is {}".format(
            len(splitinfos), len(self.tree_node_queue)))
        new_tree_node_queue = []
        for i in range(len(self.tree_node_queue)):
            sum_grad = self.tree_node_queue[i].sum_grad
            sum_hess = self.tree_node_queue[i].sum_hess
            if max_depth_reach or splitinfos[i].gain <= \
                    self.min_impurity_split + consts.FLOAT_ZERO:
                self.tree_node_queue[i].is_leaf = True
                # 从当前剩余的实例里筛除和这个结点关联的实例
                inst_belong_to_the_leaf = self.node_dispatch.filter(lambda k, value: value[1] == self.tree_node_queue[i].id)
                # 找到实例的标签值（join 的默认回调函数为lambda a, b: a）
                p_n_in_the_leaf = self.y.join(inst_belong_to_the_leaf)
                # 累加标签值，得到该节点正样本数量（这个只在0-1二分类有意义）
                num_positive = p_n_in_the_leaf.reduce(lambda a, b: a + b)
                # 该节点样本总量
                num_totle = p_n_in_the_leaf.count()
                # 计算纯净度（考虑 除以0 的异常）
                purity = 0 if num_totle == 0 else max(num_totle - num_positive, num_positive) / num_totle
                # 暂存起来，用以计算相关的统计量
                self._nodeid_purity_list.append((self.tree_node_queue[i].id, purity, num_totle))
                # LOGGER.debug('node with id = {} is a leaf, its purity = {}'.format(self.tree_node_queue[i].id, purity))
            else:
                self.tree_node_queue[i].left_nodeid = self.tree_node_num + 1
                self.tree_node_queue[i].right_nodeid = self.tree_node_num + 2
                self.tree_node_num += 2

                left_node = Node(id=self.tree_node_queue[i].left_nodeid,
                                 sitename=consts.GUEST,
                                 sum_grad=splitinfos[i].sum_grad,
                                 sum_hess=splitinfos[i].sum_hess,
                                 weight=self.splitter.node_weight(splitinfos[i].sum_grad, splitinfos[i].sum_hess))
                right_node = Node(id=self.tree_node_queue[i].right_nodeid,
                                  sitename=consts.GUEST,
                                  sum_grad=sum_grad - splitinfos[i].sum_grad,
                                  sum_hess=sum_hess - splitinfos[i].sum_hess,
                                  weight=self.splitter.node_weight( \
                                      sum_grad - splitinfos[i].sum_grad,
                                      sum_hess - splitinfos[i].sum_hess))

                new_tree_node_queue.append(left_node)
                new_tree_node_queue.append(right_node)

                self.tree_node_queue[i].sitename = splitinfos[i].sitename
                if self.tree_node_queue[i].sitename == consts.GUEST:
                    self.tree_node_queue[i].fid = self.encode("feature_idx", splitinfos[i].best_fid)
                    self.tree_node_queue[i].bid = self.encode("feature_val", splitinfos[i].best_bid,
                                                              self.tree_node_queue[i].id)
                else:
                    self.tree_node_queue[i].fid = splitinfos[i].best_fid
                    self.tree_node_queue[i].bid = splitinfos[i].best_bid

                self.update_feature_importance(splitinfos[i])
            self.tree_.append(self.tree_node_queue[i])

        self.tree_node_queue = new_tree_node_queue

    def get_histograms(self, node_map={}):
        LOGGER.info("start to get node histograms")
        histograms = FeatureHistogram.calculate_histogram(
            self.data_bin_with_node_dispatch, self.grad_and_hess,
            self.bin_split_points, self.bin_sparse_points,
            self.valid_features, node_map)
        acc_histograms = FeatureHistogram.accumulate_histogram(histograms)
        return acc_histograms

    def encrypt(self, val):
        return self.encrypter.encrypt(val)

    def decrypt(self, val):
        return self.encrypter.decrypt(val)

    def encode(self, etype="feature_idx", val=None, nid=None):
        if etype == "feature_idx":
            return val

        if etype == "feature_val":
            self.split_maskdict[nid] = val
            return None

        raise TypeError("encode type %s is not support!" % (str(etype)))

    @staticmethod
    def decode(dtype="feature_idx", val=None, nid=None, split_maskdict=None):
        if dtype == "feature_idx":
            return val

        if dtype == "feature_val":
            if nid in split_maskdict:
                return split_maskdict[nid]
            else:
                raise ValueError("decode val %s cause error, can't reconize it!" % (str(val)))

        return TypeError("decode type %s is not support!" % (str(dtype)))

    def sync_encrypted_splitinfo_host(self, dep=-1, batch=-1):
        LOGGER.info("get encrypted splitinfo of depth {}, batch {}".format(dep, batch))
        
        encrypted_splitinfo_host = self.transfer_inst.recv_data_from_hosts(-1)

        # encrypted_splitinfo_host = federation.get(name=self.transfer_inst.encrypted_splitinfo_host.name,
        #                                           tag=self.transfer_inst.generate_transferid(
        #                                               self.transfer_inst.encrypted_splitinfo_host, dep, batch),
        #                                           idx=-1)
        return encrypted_splitinfo_host

    def sync_federated_best_splitinfo_host(self, federated_best_splitinfo_host, dep=-1, batch=-1, idx=-1):
        LOGGER.info("send federated best splitinfo of depth {}, batch {}".format(dep, batch))
        
        self.transfer_inst.send_data_to_hosts(federated_best_splitinfo_host, idx)

        # federation.remote(obj=federated_best_splitinfo_host,
        #                   name=self.transfer_inst.federated_best_splitinfo_host.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.federated_best_splitinfo_host,
        #                                                              dep,
        #                                                              batch),
        #                   role=consts.HOST,
        #                   idx=idx)

    def find_host_split(self, value):
        cur_split_node, encrypted_splitinfo_host = value
        sum_grad = cur_split_node.sum_grad
        sum_hess = cur_split_node.sum_hess
        # best_gain = self.min_impurity_split - consts.FLOAT_ZERO
        best_gain = -100000000
        best_idx = -1

        for i in range(len(encrypted_splitinfo_host)):

            sum_grad_l, sum_hess_l, sum_grad_r, sum_hess_r = encrypted_splitinfo_host[i]
            sum_grad_l = self.decrypt(sum_grad_l)
            sum_hess_l = self.decrypt(sum_hess_l)
            sum_grad_r = self.decrypt(sum_grad_r)
            sum_hess_r = self.decrypt(sum_hess_r)


            # gain = self.splitter.split_gain(sum_grad, sum_hess, sum_grad_l,
            #                                 sum_hess_l, sum_grad_r, sum_hess_r)
            gain = self.splitter.split_gain_host(sum_grad, sum_hess, sum_grad_l,
                                            sum_hess_l, sum_grad_r, sum_hess_r)

            if gain > best_gain:
                best_gain = gain
                best_idx = i

        random_eta = random.random() + 1e-1
        best_gain = best_gain * random_eta
        if best_idx==-1:
            assert abs(best_gain+100000000) < 1e-5
        best_gain = self.encrypt(best_gain)
        
        return best_idx, best_gain, random_eta

    def federated_find_split(self, dep=-1, batch=-1):
        LOGGER.info("federated find split of depth {}, batch {}".format(dep, batch))
        encrypted_splitinfo_host = self.sync_encrypted_splitinfo_host(dep, batch)
        random_eta = []
        for i in range(len(encrypted_splitinfo_host)):
            encrypted_splitinfo_host_table = DTable(
                False, list(zip(self.cur_split_nodes, encrypted_splitinfo_host[i])))
            # splitinfos (index,(encrypt(best_idx, encrypy(best_gain))) -> 每个host都会得到自己最佳的idx，
            splitinfos = encrypted_splitinfo_host_table.mapValues(self.find_host_split).collect()
            random_eta.append([splitinfo[1][2] for splitinfo in splitinfos])
            best_splitinfo_host = [splitinfo[1][:2] for splitinfo in splitinfos]
            # -> 每个host都会得到自己最佳的idx
            self.sync_federated_best_splitinfo_host(best_splitinfo_host, dep, batch, i)
        return random_eta

    def sync_final_split_host(self, dep=-1, batch=-1):
        LOGGER.info("get host final splitinfo of depth {}, batch {}".format(dep, batch))
        
        final_splitinfo_host = self.transfer_inst.recv_data_from_hosts(-1)

        # final_splitinfo_host = federation.get(name=self.transfer_inst.final_splitinfo_host.name,
        #                                       tag=self.transfer_inst.generate_transferid(
        #                                           self.transfer_inst.final_splitinfo_host, dep, batch),
        #                                       idx=-1)

        return final_splitinfo_host

    def sync_tree_node_queue(self, tree_node_queue, dep=-1):
        LOGGER.info("send tree node queue of depth {}".format(dep))
        mask_tree_node_queue = copy.deepcopy(tree_node_queue)
        for i in range(len(mask_tree_node_queue)):
            mask_tree_node_queue[i] = Node(id=mask_tree_node_queue[i].id)

        self.transfer_inst.send_data_to_hosts(mask_tree_node_queue, -1)
        
        # federation.remote(obj=mask_tree_node_queue,
        #                   name=self.transfer_inst.tree_node_queue.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree_node_queue, dep),
        #                   role=consts.HOST,
        #                   idx=-1)

    def sync_encrypted_grad_and_hess(self):
        LOGGER.info("send encrypted grad and hess to host")
        encrypted_grad_and_hess = self.encrypt_grad_and_hess()
        self.transfer_inst.send_data_to_hosts(encrypted_grad_and_hess, -1)
        # federation.remote(obj=encrypted_grad_and_hess,
        #                   name=self.transfer_inst.encrypted_grad_and_hess.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.encrypted_grad_and_hess),
        #                   role=consts.HOST,
        #                   idx=-1)
    
    def sync_node_positions(self, dep):
        LOGGER.info("send node positions of depth {}".format(dep))
        
        self.transfer_inst.send_data_to_hosts(self.node_dispatch, -1)
        # federation.remote(obj=self.node_dispatch,
        #                   name=self.transfer_inst.node_positions.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.node_positions, dep),
        #                   role=consts.HOST,
        #                   idx=-1)

    def sync_dispatch_node_host(self, dispatch_guest_data, dep=-1):
        LOGGER.info("send node to host to dispath, depth is {}".format(dep))
        
        self.transfer_inst.send_data_to_hosts(dispatch_guest_data, -1)

        # federation.remote(obj=dispatch_guest_data,
        #                   name=self.transfer_inst.dispatch_node_host.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.dispatch_node_host, dep),
        #                   role=consts.HOST,
        #                   idx=-1)

    def sync_dispatch_node_host_result(self, dep=-1):
        LOGGER.info("get host dispatch result, depth is {}".format(dep))
        
        dispatch_node_host_result = self.transfer_inst.recv_data_from_hosts(-1)

        # dispatch_node_host_result = federation.get(name=self.transfer_inst.dispatch_node_host_result.name,
        #                                            tag=self.transfer_inst.generate_transferid(
        #                                                self.transfer_inst.dispatch_node_host_result, dep),
        #                                            idx=-1)

        return dispatch_node_host_result

    def sync_tree(self):
        LOGGER.info("sync tree to host")

        self.transfer_inst.send_data_to_hosts(self.tree_, -1)

        # federation.remote(obj=self.tree_,
        #                   name=self.transfer_inst.tree.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree),
        #                   role=consts.HOST,
        #                   idx=-1)

    def sync_predict_finish_tag(self, finish_tag, send_times):
        LOGGER.info("send the {}-th predict finish tag {} to host".format(finish_tag, send_times))
        
        self.transfer_inst.send_data_to_hosts(finish_tag, -1)

        # federation.remote(obj=finish_tag,
        #                   name=self.transfer_inst.predict_finish_tag.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_finish_tag, send_times),
        #                   role=consts.HOST,
        #                   idx=-1)

    def sync_predict_data(self, predict_data, send_times):
        LOGGER.info("send predict data to host, sending times is {}".format(send_times))
        
        self.transfer_inst.send_data_to_hosts(predict_data, -1)

        # federation.remote(obj=predict_data,
        #                   name=self.transfer_inst.predict_data.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_data, send_times),
        #                   role=consts.HOST,
        #                   idx=-1)

    def convert_bin_to_real(self):
        LOGGER.info("convert tree node bins to real value")
        for i in range(len(self.tree_)):
            if self.tree_[i].is_leaf is True:
                continue
            if self.tree_[i].sitename == consts.GUEST:
                fid = self.decode("feature_idx", self.tree_[i].fid, split_maskdict=self.split_maskdict)
                bid = self.decode("feature_val", self.tree_[i].bid, self.tree_[i].id, self.split_maskdict)
                real_splitval = self.encode("feature_val", self.bin_split_points[fid][bid], self.tree_[i].id)
                self.tree_[i].bid = real_splitval

    @staticmethod
    def dispatch_node(value, tree_=None, decoder=None,
                      split_maskdict=None, bin_sparse_points=None):
        unleaf_state, nodeid = value[1]

        if tree_[nodeid].is_leaf is True:
            return tree_[nodeid].weight
        else:
            if tree_[nodeid].sitename == consts.GUEST:
                fid = decoder("feature_idx", tree_[nodeid].fid, split_maskdict=split_maskdict)
                bid = decoder("feature_val", tree_[nodeid].bid, nodeid, split_maskdict)
                if value[0].features.get_data(fid, bin_sparse_points[fid]) <= bid:
                    return (1, tree_[nodeid].left_nodeid)
                else:
                    return (1, tree_[nodeid].right_nodeid)
            else:
                return (1, tree_[nodeid].fid, tree_[nodeid].bid, tree_[nodeid].sitename,
                        nodeid, tree_[nodeid].left_nodeid, tree_[nodeid].right_nodeid)

    def redispatch_node(self, dep=-1):
        LOGGER.info("redispatch node of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.dispatch_node,
                                                 tree_=self.tree_,
                                                 decoder=self.decode,
                                                 split_maskdict=self.split_maskdict,
                                                 bin_sparse_points=self.bin_sparse_points)
        dispatch_guest_result = self.data_bin_with_node_dispatch.mapValues(dispatch_node_method)
        tree_node_num = self.tree_node_num
        LOGGER.info("remask dispatch node result of depth {}".format(dep))
        
        dispatch_to_host_result = dispatch_guest_result.filter(lambda key, value: isinstance(value, tuple) and len(value) > 2)
       
        dispatch_guest_result = dispatch_guest_result.subtractByKey(dispatch_to_host_result)
        leaf = dispatch_guest_result.filter(lambda  key, value: isinstance(value, tuple) is False) 
        if self.predict_weights is None:
            self.predict_weights = leaf
        else:
            self.predict_weights = self.predict_weights.union(leaf)

        dispatch_guest_result = dispatch_guest_result.subtractByKey(leaf)
        
        self.sync_dispatch_node_host(dispatch_to_host_result, dep)
        dispatch_node_host_result = self.sync_dispatch_node_host_result(dep)

        self.node_dispatch = None
        for idx in range(len(dispatch_node_host_result)):
            if self.node_dispatch is None:
                self.node_dispatch = dispatch_node_host_result[idx]
            else:
                self.node_dispatch = self.node_dispatch.join(dispatch_node_host_result[idx], \
                                                                 lambda unleaf_state_nodeid1, unleaf_state_nodeid2: \
                                                                 unleaf_state_nodeid1 if len(
                                                                 unleaf_state_nodeid1) == 2 else unleaf_state_nodeid2)

        if self.node_dispatch is None:
            self.node_dispatch = dispatch_guest_result
        else:
            self.node_dispatch = self.node_dispatch.union(dispatch_guest_result)

    def encrypt_grad_and_hess(self):
        LOGGER.info("start to encrypt grad and hess")
        encrypted_grad_and_hess = self.encrypted_mode_calculator.encrypt(self.grad_and_hess)
        return encrypted_grad_and_hess

    def splitinfo_host_remove_random_eta(self, splitinfo_host, random_eta):
        for i in range(len(splitinfo_host)):
            for j in range(len(splitinfo_host[i])):
                splitinfo_host[i][j].gain = self.decrypt(splitinfo_host[i][j].gain)
                splitinfo_host[i][j].gain = splitinfo_host[i][j].gain / random_eta[i][j]
        return splitinfo_host

    def find_best_split_guest_and_host(self, splitinfo_guest_host):
        best_gain_host = -1000000000
        best_gain_host_idx = 0
        for i in range(1, len(splitinfo_guest_host)):
            # gain_host_i = self.decrypt(splitinfo_guest_host[i].gain)
            gain_host_i = splitinfo_guest_host[i].gain
            if best_gain_host < gain_host_i:
                best_gain_host = gain_host_i
                best_gain_host_idx = i

        LOGGER.debug(f"best_gain_host={best_gain_host} best_gain_host_idx={best_gain_host_idx} splitinfo_guest_gain={splitinfo_guest_host[0].gain}")
        if splitinfo_guest_host[0].gain >= best_gain_host - consts.FLOAT_ZERO:
            best_splitinfo = splitinfo_guest_host[0]
        else:
            LOGGER.debug(f"type splitinfo_guest_host[best_gain_host_idx].sum_grad = {type(splitinfo_guest_host[best_gain_host_idx].sum_grad)}")
            LOGGER.debug(f"type splitinfo_guest_host[best_gain_host_idx].best_fid = {splitinfo_guest_host[best_gain_host_idx].best_fid}")
            best_splitinfo = splitinfo_guest_host[best_gain_host_idx]
            best_splitinfo.sum_grad = self.decrypt(best_splitinfo.sum_grad)
            best_splitinfo.sum_hess = self.decrypt(best_splitinfo.sum_hess)
            best_splitinfo.gain = best_gain_host

        return best_splitinfo

    def merge_splitinfo(self, splitinfo_guest, splitinfo_host):
        LOGGER.info("merge splitinfo")
        merge_infos = []
        for i in range(len(splitinfo_guest)):
            splitinfo = [splitinfo_guest[i]]
            for j in range(len(splitinfo_host)):
                splitinfo.append(splitinfo_host[j][i])

            merge_infos.append(splitinfo)

        splitinfo_guest_host_table = DTable(False, merge_infos)
        best_splitinfo_table = splitinfo_guest_host_table.mapValues(self.find_best_split_guest_and_host)
        best_splitinfos = [best_splitinfo[1] for best_splitinfo in best_splitinfo_table.collect()]

        return best_splitinfos

    def fit(self):
        LOGGER.info("begin to fit guest decision tree")

        self.sync_encrypted_grad_and_hess()

        root_sum_grad, root_sum_hess = self.get_grad_hess_sum(self.grad_and_hess)
        root_node = Node(id=0, sitename=consts.GUEST, sum_grad=root_sum_grad, sum_hess=root_sum_hess,
                         weight=self.splitter.node_weight(root_sum_grad, root_sum_hess))
        self.tree_node_queue = [root_node]

        self.dispatch_all_node_to_root()

        for dep in range(self.max_depth):
            LOGGER.info("start to fit depth {}, tree node queue size is {}".format(dep, len(self.tree_node_queue)))

            self.sync_tree_node_queue(self.tree_node_queue, dep)
            if len(self.tree_node_queue) == 0:
                break
                
            self.sync_node_positions(dep)

            self.data_bin_with_node_dispatch = self.data_bin.join(self.node_dispatch,
                                                                  lambda data_inst, dispatch_info: (
                                                                      data_inst, dispatch_info))

            batch = 0
            splitinfos = []
            for i in range(0, len(self.tree_node_queue), self.max_split_nodes):
                self.cur_split_nodes = self.tree_node_queue[i: i + self.max_split_nodes]

                node_map = {}
                node_num = 0
                for tree_node in self.cur_split_nodes:
                    node_map[tree_node.id] = node_num
                    node_num += 1
                
                acc_histograms = self.get_histograms(node_map=node_map)

                self.best_splitinfo_guest = self.splitter.find_split(acc_histograms, self.valid_features)

                randome_eta = self.federated_find_split(dep, batch)
                final_splitinfo_host = self.sync_final_split_host(dep, batch)
                if self.is_first:
                    final_splitinfo_host = []
                # 去掉随机数
                final_splitinfo_host_without_eta = self.splitinfo_host_remove_random_eta(final_splitinfo_host, randome_eta)

                cur_splitinfos = self.merge_splitinfo(self.best_splitinfo_guest, final_splitinfo_host_without_eta)
                splitinfos.extend(cur_splitinfos)

                batch += 1

            max_depth_reach = True if dep + 1 == self.max_depth else False
            self.update_tree_node_queue(splitinfos, max_depth_reach)
            self.save_node_dispatch()
            self.redispatch_node(dep)
        
        self.save_node_dispatch()
        self.sync_tree()

        # LOGGER.debug('len of tree_ is {}'.format(len(self.tree_)))
        self.convert_bin_to_real()
        tree_ = self.tree_
        LOGGER.info("tree node num is %d" % len(tree_))
        # 把树结构输出到文件
        file_name = self.node_dispatch_log_path + "Tree_struction" + time.strftime("_%Y-%m-%d-%H_%M_%S".format(self.runtime_idx)) + '.log'
        f = open(file_name, 'w', newline='', encoding='utf-8')
        for node in self.tree_:
            f.write(str(node))
        f.write("\n\n\n")
        f.close()

        purity_list = []
        count_list = []
        for nodeid_purity_count in self._nodeid_purity_list:
            nodeid, purity, count = nodeid_purity_count
            purity_list.append(purity)
            count_list.append(count)
            LOGGER.debug('node with id = {} is a leaf, its purity = {}, count is {}'.format(nodeid, purity, count))
        LOGGER.debug('the average of the purity is {}'.format(np.average(purity_list)))
        LOGGER.debug('the maximum of the purity is {}'.format(np.max(purity_list)))
        LOGGER.debug('the minimun of the purity is {}'.format(np.min(purity_list)))
        LOGGER.debug('the weighted average of the purity is {}'.format(np.average(purity_list, weights=count_list)))

        LOGGER.info("end to fit guest decision tree")
        
    def get_model(self):
        model_meta = self.get_model_meta()
        model_param = self.get_model_param()

        return model_meta, model_param

    def get_model_meta(self):
        model_meta = DecisionTreeModelMeta()
        model_meta.criterion_meta.CopyFrom(CriterionMeta(criterion_method=self.criterion_method,
                                                         criterion_param=self.criterion_params))

        model_meta.max_depth = self.max_depth
        model_meta.min_sample_split = self.min_sample_split
        model_meta.min_impurity_split = self.min_impurity_split
        model_meta.min_leaf_node = self.min_leaf_node

        return model_meta

    def get_model_param(self):
        model_param = DecisionTreeModelParam()
        for node in self.tree_:
            model_param.tree_.add(id=node.id,
                                  sitename=node.sitename,
                                  fid=node.fid,
                                  bid=node.bid,
                                  weight=node.weight,
                                  is_leaf=node.is_leaf,
                                  left_nodeid=node.left_nodeid,
                                  right_nodeid=node.right_nodeid)

        model_param.split_maskdict.update(self.split_maskdict)

        return model_param
    
    def get_feature_importance(self):
        return self.feature_importances_

    def load_model(self, model_meta=None, model_param=None):
        LOGGER.info("load tree model")
        self.set_model_meta(model_meta)
        self.set_model_param(model_param)

    def set_model_meta(self, model_meta):
        self.max_depth = model_meta.max_depth
        self.min_sample_split = model_meta.min_sample_split
        self.min_impurity_split = model_meta.min_impurity_split
        self.min_leaf_node = model_meta.min_leaf_node
        self.criterion_method = model_meta.criterion_meta.criterion_method
        self.criterion_params = list(model_meta.criterion_meta.criterion_param)

    def set_model_param(self, model_param):
        self.tree_ = []
        for node_param in model_param.tree_:
            _node = Node(id=node_param.id,
                         sitename=node_param.sitename,
                         fid=node_param.fid,
                         bid=node_param.bid,
                         weight=node_param.weight,
                         is_leaf=node_param.is_leaf,
                         left_nodeid=node_param.left_nodeid,
                         right_nodeid=node_param.right_nodeid)

            self.tree_.append(_node)

        self.split_maskdict = dict(model_param.split_maskdict)

    def sync_data_predicted_by_host(self, send_times):

        predict_data = self.transfer_inst.recv_data_from_hosts(-1)

        LOGGER.info("get predicted data by host, recv times is {}".format(send_times))
        # predict_data = federation.get(name=self.transfer_inst.predict_data_by_host.name,
        #                               tag=self.transfer_inst.generate_transferid(
        #                                   self.transfer_inst.predict_data_by_host, send_times),
        #                               idx=-1)
        return predict_data

    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, split_maskdict=None):
        nid, tag = predict_state

        while tree_[nid].sitename == consts.GUEST:
            if tree_[nid].is_leaf is True:
                return tree_[nid].weight

            fid = decoder("feature_idx", tree_[nid].fid, split_maskdict=split_maskdict)
            bid = decoder("feature_val", tree_[nid].bid, nid, split_maskdict)

            if data_inst.features.get_data(fid, 0) <= bid:
                nid = tree_[nid].left_nodeid
            else:
                nid = tree_[nid].right_nodeid

        return nid, 1

    def predict(self, data_instances):
        LOGGER.info("start to predict!")
        predict_data = data_instances.mapValues(lambda data_inst: (0, 1))
        site_host_send_times = 0
        predict_result = None

        while True:
            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_,
                                              decoder=self.decode,
                                              split_maskdict=self.split_maskdict)
            predict_data = predict_data.join(data_instances, traverse_tree)
            predict_leaf = predict_data.filter(lambda key, value: isinstance(value, tuple) is False)
            if predict_result is None:
                predict_result = predict_leaf
            else:
                predict_result = predict_result.union(predict_leaf)

            predict_data = predict_data.subtractByKey(predict_leaf)

            unleaf_node_count = predict_data.count()

            if unleaf_node_count == 0:
                self.sync_predict_finish_tag(True, site_host_send_times)
                break
                
            self.sync_predict_finish_tag(False, site_host_send_times)
            self.sync_predict_data(predict_data, site_host_send_times)

            predict_data_host = self.sync_data_predicted_by_host(site_host_send_times)
            for i in range(len(predict_data_host)):
                predict_data = predict_data.join(predict_data_host[i],
                                                 lambda state1_nodeid1, state2_nodeid2:
                                                 state1_nodeid1 if state1_nodeid1[
                                                                          1] == 0 else state2_nodeid2)

            site_host_send_times += 1

        LOGGER.info("predict finish!")
        return predict_result

    @staticmethod
    def traverse_tree_v2(data_inst, tree_=None,
                      decoder=None, split_maskdict=None):
        def traverse_tree_dfs(nid, flag):
            nonlocal tree_
            if tree_[nid].is_leaf:
                return [(flag, tree_[nid].weight)]
            if tree_[nid].sitename == consts.GUEST:
                fid = decoder("feature_idx", tree_[nid].fid, split_maskdict=split_maskdict)
                bid = decoder("feature_val", tree_[nid].bid, nid, split_maskdict)
                left_flag, right_flag = False, False
                if data_inst.features.get_data(fid, 0) <= bid:
                    left_flag = True
                else:
                    right_flag = True
                return traverse_tree_dfs(tree_[nid].left_nodeid, left_flag and flag) \
                    + traverse_tree_dfs(tree_[nid].right_nodeid, right_flag and flag) 
            return traverse_tree_dfs(tree_[nid].left_nodeid, flag) + traverse_tree_dfs(tree_[nid].right_nodeid, flag) 
        predict_leaf_result = traverse_tree_dfs(0, True)
        return predict_leaf_result


    def sync_data_predicted_by_host_v2(self, send_times):

        predict_data = self.transfer_inst.recv_data_from_hosts(-1)

        LOGGER.info("get predicted data by host, recv times is {}".format(send_times))
        # predict_data = federation.get(name=self.transfer_inst.predict_data_by_host.name,
        #                               tag=self.transfer_inst.generate_transferid(
        #                                   self.transfer_inst.predict_data_by_host, send_times),
        #                               idx=-1)
        return predict_data

    @staticmethod
    def merge_predict_result(predict_leaf_result1, predict_leaf_result2):
        assert len(predict_leaf_result1) == len(predict_leaf_result2)
        n = len(predict_leaf_result1) 
        return [(predict_leaf_result1[i][0] and predict_leaf_result2[i], predict_leaf_result1[i][1]) for i in range(n)]

    def predict_v2(self, data_instances):
        LOGGER.info("start to predict!")
        predict_data = data_instances.mapValues(lambda data_inst: (0, 1))
        site_host_send_times = 0
        predict_result = None

        traverse_tree = functools.partial(self.traverse_tree_v2,
                                            tree_=self.tree_,
                                            decoder=self.decode,
                                            split_maskdict=self.split_maskdict)
        predict_data = data_instances.mapValues(traverse_tree)

        predict_data_host = self.sync_data_predicted_by_host_v2(site_host_send_times)
        for i in range(len(predict_data_host)):
            predict_data = predict_data.join(predict_data_host[i], self.merge_predict_result)
        predict_result = predict_data.mapValues(lambda pre_list: sum([value for flag,value in pre_list if flag is True]))
        LOGGER.info("predict_v2 finish!")
        return predict_result

    def set_y(self, y: DTable):
        self.y = y

    def save_node_dispatch(self):
        # LOGGER.debug(f"node_dispatch !! = [{str(self.node_dispatch)}]")
        now_dep_node_dispatch = self.node_dispatch.mapValues(lambda x:x[1]).collect()
        for k,v in now_dep_node_dispatch:
            self.tree_[v].node_dispatch.append(k)
        pass
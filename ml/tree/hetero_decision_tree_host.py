
from ml.tree.decision_tree import DecisionTree
from ml.utils.logger import LOGGER, MyLoggerFactory
from ml.tree.splitter import SplitInfo, Splitter
from ml.utils import consts
from ml.tree.node import Node
from arch.api.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from arch.api.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from ml.tree.feateur_histogram import FeatureHistogram
import functools

class HeteroDecisionTreeHost(DecisionTree):
    def __init__(self, tree_param):
        self.logger = MyLoggerFactory().get_logger()
        self.logger.info('hetero decision tree host init!')
        super(HeteroDecisionTreeHost, self).__init__(tree_param)

        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node)

        self.data_bin = None
        self.data_bin_with_position = None
        self.grad_and_hess = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.infos = None
        self.valid_features = None
        self.pubkey = None
        self.privakey = None
        self.tree_id = None
        self.encrypted_grad_and_hess = None
        self.tree_node_queue = None
        self.cur_split_nodes = None
        self.split_maskdict = {}
        self.tree_ = None
        self.runtime_idx = 0
        self.sitename = consts.HOST

    def set_flowid(self, flowid=0):
        self.logger.info("set flowid, flowid is {}".format(flowid))
        # self.transfer_inst.set_flowid(flowid)

    def set_runtime_idx(self, runtime_idx):
        self.runtime_idx = runtime_idx
        self.sitename = ".".join([consts.HOST, str(self.runtime_idx)])
        self.logger.info("runtime idx is {}, sitename is {}".format(self.runtime_idx, self.sitename))

    def set_inputinfo(self, data_bin=None, grad_and_hess=None, bin_split_points=None, bin_sparse_points=None):
        self.logger.info("set input info")
        self.data_bin = data_bin
        self.grad_and_hess = grad_and_hess
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def set_valid_features(self, valid_features=None):
        self.logger.info("set valid features")
        self.valid_features = valid_features

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

    def sync_encrypted_grad_and_hess(self):
        self.logger.info("get encrypted grad and hess")
        self.grad_and_hess = self.transfer_inst.recv_data_from_guest()
        # self.grad_and_hess = federation.get(name=self.transfer_inst.encrypted_grad_and_hess.name,
        #                                     tag=self.transfer_inst.generate_transferid(
        #                                         self.transfer_inst.encrypted_grad_and_hess),
        #                                     idx=0)

    def sync_tree_node_queue(self, dep=-1):
        self.logger.info("get tree node queue of depth {}".format(dep))
        self.tree_node_queue = self.transfer_inst.recv_data_from_guest()
        
        # self.tree_node_queue = federation.get(name=self.transfer_inst.tree_node_queue.name,
        #                                       tag=self.transfer_inst.generate_transferid(
        #                                           self.transfer_inst.tree_node_queue, dep),
        #                                       idx=0)

    def sync_node_positions(self, dep=-1):
        self.logger.info("get tree node queue of depth {}".format(dep))

        node_positions = self.transfer_inst.recv_data_from_guest()
        # node_positions = federation.get(name=self.transfer_inst.node_positions.name,
        #                                 tag=self.transfer_inst.generate_transferid(self.transfer_inst.node_positions,
        #                                                                            dep),
        #                                 idx=0)
        return node_positions

    def get_histograms(self, node_map={}):
        self.logger.info("start to get node histograms")
        # self.data_bin_with_position = self.data_bin.join(node_positions, lambda v1, v2: (v1, v2))
        histograms = FeatureHistogram.calculate_histogram(
            self.data_bin_with_position, self.grad_and_hess,
            self.bin_split_points, self.bin_sparse_points,
            self.valid_features, node_map)
        self.logger.info("begin to accumulate histograms")
        acc_histograms = FeatureHistogram.accumulate_histogram(histograms)
        self.logger.info("acc histogram shape is {}".format(len(acc_histograms)))
        return acc_histograms

    def sync_encrypted_splitinfo_host(self, encrypted_splitinfo_host, dep=-1, batch=-1):
        self.logger.info("send encrypted splitinfo of depth {}, batch {}".format(dep, batch))
        
        self.transfer_inst.send_data_to_guest(encrypted_splitinfo_host)

        # federation.remote(obj=encrypted_splitinfo_host,
        #                   name=self.transfer_inst.encrypted_splitinfo_host.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.encrypted_splitinfo_host, dep,
        #                                                              batch),
        #                   role=consts.GUEST,
        #                   idx=-1)

    def sync_federated_best_splitinfo_host(self, dep=-1, batch=-1):
        self.logger.info("get federated best splitinfo of depth {}, batch {}".format(dep, batch))
        
        federated_best_splitinfo_host = self.transfer_inst.recv_data_from_guest()

        # federated_best_splitinfo_host = federation.get(name=self.transfer_inst.federated_best_splitinfo_host.name,
        #                                                tag=self.transfer_inst.generate_transferid(
        #                                                    self.transfer_inst.federated_best_splitinfo_host, dep,
        #                                                    batch),
        #                                                idx=0)
        return federated_best_splitinfo_host

    def sync_final_splitinfo_host(self, splitinfo_host, federated_best_splitinfo_host, dep=-1, batch=-1):
        self.logger.info("send host final splitinfo of depth {}, batch {}".format(dep, batch))
        final_splitinfos = []
        for i in range(len(splitinfo_host)):
            best_idx, best_gain = federated_best_splitinfo_host[i]
            if best_idx != -1:
                assert splitinfo_host[i][best_idx].sitename == self.sitename
                splitinfo = splitinfo_host[i][best_idx]
                splitinfo.best_fid = self.encode("feature_idx", splitinfo.best_fid)
                assert splitinfo.best_fid is not None
                splitinfo.best_bid = self.encode("feature_val", splitinfo.best_bid, self.cur_split_nodes[i].id)
                splitinfo.gain = best_gain
            else:
                splitinfo = SplitInfo(sitename=self.sitename, best_fid=-1, best_bid=-1, gain=best_gain)

            final_splitinfos.append(splitinfo)

        self.transfer_inst.send_data_to_guest(final_splitinfos)

        # federation.remote(obj=final_splitinfos,
        #                   name=self.transfer_inst.final_splitinfo_host.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.final_splitinfo_host, dep,
        #                                                              batch),
        #                   role=consts.GUEST,
        #                   idx=-1)

    def sync_dispatch_node_host(self, dep):
        self.logger.info("get node from host to dispath, depth is {}".format(dep))
        
        dispatch_node_host = self.transfer_inst.recv_data_from_guest()

        # dispatch_node_host = federation.get(name=self.transfer_inst.dispatch_node_host.name,
        #                                     tag=self.transfer_inst.generate_transferid(
        #                                         self.transfer_inst.dispatch_node_host, dep),
        #                                     idx=0)
        return dispatch_node_host

    @staticmethod
    def dispatch_node(value1, value2, sitename=None, decoder=None,
                      split_maskdict=None, bin_sparse_points=None,):

        unleaf_state, fid, bid, node_sitename, nodeid, left_nodeid, right_nodeid = value1
        if node_sitename != sitename:
            return value1

        fid = decoder("feature_idx", fid, split_maskdict=split_maskdict)
        bid = decoder("feature_val", bid, nodeid, split_maskdict=split_maskdict)
        if value2.features.get_data(fid, bin_sparse_points[fid]) <= bid:
            return unleaf_state, left_nodeid
        else:
            return unleaf_state, right_nodeid

    def sync_dispatch_node_host_result(self, dispatch_node_host_result, dep=-1):
        self.logger.info("send host dispatch result, depth is {}".format(dep))
        
        self.transfer_inst.send_data_to_guest(dispatch_node_host_result)

        # federation.remote(obj=dispatch_node_host_result,
        #                   name=self.transfer_inst.dispatch_node_host_result.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.dispatch_node_host_result, dep),
        #                   role=consts.GUEST,
        #                   idx=-1)

    def find_dispatch(self, dispatch_node_host, dep=-1):
        self.logger.info("start to find host dispath of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.dispatch_node,
                                                 sitename=self.sitename,
                                                 decoder=self.decode,
                                                 split_maskdict=self.split_maskdict,
                                                 bin_sparse_points=self.bin_sparse_points)
        dispatch_node_host_result = dispatch_node_host.join(self.data_bin, dispatch_node_method)
        self.sync_dispatch_node_host_result(dispatch_node_host_result, dep)

    def sync_tree(self):
        self.logger.info("sync tree from guest")
        
        self.tree_ = self.transfer_inst.recv_data_from_guest()

        # self.tree_ = federation.get(name=self.transfer_inst.tree.name,
        #                             tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree),
        #                             idx=0)

    def remove_duplicated_split_nodes(self, split_nid_used):
        self.logger.info("remove duplicated nodes from split mask dict")
        duplicated_nodes = set(self.split_maskdict.keys()) - set(split_nid_used)
        for nid in duplicated_nodes:
            del self.split_maskdict[nid]

    def convert_bin_to_real(self):
        self.logger.info("convert tree node bins to real value")
        split_nid_used = []
        for i in range(len(self.tree_)):
            if self.tree_[i].is_leaf is True:
                continue

            if self.tree_[i].sitename == self.sitename:
                fid = self.decode("feature_idx", self.tree_[i].fid, split_maskdict=self.split_maskdict)
                bid = self.decode("feature_val", self.tree_[i].bid, self.tree_[i].id, self.split_maskdict)
                real_splitval = self.encode("feature_val", self.bin_split_points[fid][bid], self.tree_[i].id)
                self.tree_[i].bid = real_splitval

                split_nid_used.append(self.tree_[i].id)

        self.remove_duplicated_split_nodes(split_nid_used)

    @staticmethod
    def traverse_tree(predict_state, data_inst, tree_=None,
                      decoder=None, split_maskdict=None, sitename=consts.HOST):

        nid, _ = predict_state
        if tree_[nid].sitename != sitename:
            return predict_state

        while tree_[nid].sitename == sitename:
            fid = decoder("feature_idx", tree_[nid].fid, split_maskdict=split_maskdict)
            bid = decoder("feature_val", tree_[nid].bid, nid, split_maskdict)

            if data_inst.features.get_data(fid, 0) <= bid:
                nid = tree_[nid].left_nodeid
            else:
                nid = tree_[nid].right_nodeid

        return (nid, 0)

    def sync_predict_finish_tag(self, recv_times):
        self.logger.info("get the {}-th predict finish tag from guest".format(recv_times))
        
        finish_tag = self.transfer_inst.recv_data_from_guest()

        # finish_tag = federation.get(name=self.transfer_inst.predict_finish_tag.name,
        #                             tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_finish_tag,
        #                                                                        recv_times),
        #                             idx=0)
        return finish_tag

    def sync_predict_data(self, recv_times):
        self.logger.info("srecv predict data to host, recv times is {}".format(recv_times))
        
        predict_data = self.transfer_inst.recv_data_from_guest()

        # predict_data = federation.get(name=self.transfer_inst.predict_data.name,
        #                               tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_data,
        #                                                                          recv_times),
        #                               idx=0)

        return predict_data

    def sync_data_predicted_by_host(self, predict_data, send_times):
        self.logger.info("send predicted data by host, send times is {}".format(send_times))
        
        self.transfer_inst.send_data_to_guest(predict_data)

        # federation.remote(obj=predict_data,
        #                   name=self.transfer_inst.predict_data_by_host.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.predict_data_by_host,
        #                                                              send_times),
        #                   role=consts.GUEST,
        #                   idx=0)

    def fit(self):
        self.logger.info("begin to fit host decision tree")
        self.sync_encrypted_grad_and_hess()

        for dep in range(self.max_depth):
            self.sync_tree_node_queue(dep)
            if len(self.tree_node_queue) == 0:
                break

            node_positions = self.sync_node_positions(dep)
            self.data_bin_with_position = self.data_bin.join(node_positions, lambda v1, v2: (v1, v2))

            batch = 0
            for i in range(0, len(self.tree_node_queue), self.max_split_nodes):
                self.cur_split_nodes = self.tree_node_queue[i: i + self.max_split_nodes]
                node_map = {}
                node_num = 0
                for tree_node in self.cur_split_nodes:
                    node_map[tree_node.id] = node_num
                    node_num += 1

                acc_histograms = self.get_histograms(node_map=node_map)

                splitinfo_host, encrypted_splitinfo_host = self.splitter.find_split_host(acc_histograms,
                                                                                         self.valid_features,
                                                                                         self.sitename)

                self.sync_encrypted_splitinfo_host(encrypted_splitinfo_host, dep, batch)
                federated_best_splitinfo_host = self.sync_federated_best_splitinfo_host(dep, batch)
                self.sync_final_splitinfo_host(splitinfo_host, federated_best_splitinfo_host, dep, batch)

                batch += 1
            
            dispatch_node_host = self.sync_dispatch_node_host(dep)
            self.find_dispatch(dispatch_node_host, dep)

        self.sync_tree()
        # self.logger.debug('len of tree_ is {}'.format(len(self.tree_)))
        self.convert_bin_to_real()

        self.logger.info("end to fit guest decision tree")

    def predict(self, data_inst):
        self.logger.info("start to predict!")
        site_guest_send_times = 0
        while True:
            finish_tag = self.sync_predict_finish_tag(site_guest_send_times)
            if finish_tag is True:
                break
            
            predict_data = self.sync_predict_data(site_guest_send_times)

            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_,
                                              decoder=self.decode,
                                              split_maskdict=self.split_maskdict,
                                              sitename=self.sitename)
            predict_data = predict_data.join(data_inst, traverse_tree)

            self.sync_data_predicted_by_host(predict_data, site_guest_send_times)

            site_guest_send_times += 1
        
        self.logger.info("predict finish!")

    def get_model(self):
        model_meta = self.get_model_meta()
        model_param = self.get_model_param()

        return model_meta, model_param

    def get_model_meta(self):
        model_meta = DecisionTreeModelMeta()

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

    def load_model(self, model_meta=None, model_param=None):
        self.logger.info("load tree model")
        self.set_model_meta(model_meta)
        self.set_model_param(model_param)

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
    
    def set_model_meta(self, model_meta):
        self.max_depth = model_meta.max_depth
        self.min_sample_split = model_meta.min_sample_split
        self.min_impurity_split = model_meta.min_impurity_split
        self.min_leaf_node = model_meta.min_leaf_node

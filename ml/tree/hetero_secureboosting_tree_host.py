

from ml.tree.boosting_tree import BoostingTree
from ml.utils.logger import MyLoggerFactory
from ml.utils import consts
from ml.feature.binning.quantile_binning import QuantileBinning
from ml.param.feature_binning_param import FeatureBinningParam
from ml.tree.hetero_decision_tree_host import HeteroDecisionTreeHost
from numpy import random

LOGGER = MyLoggerFactory().get_logger()

class HeteroSecureBoostingTreeHost(BoostingTree):
    def __init__(self):
        super(HeteroSecureBoostingTreeHost, self).__init__()

        # self.flowid = 0
        self.tree_dim = None
        self.feature_num = None
        self.trees_ = []
        self.tree_meta = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.data_bin = None
        self.runtime_idx = 0
        self.role = consts.HOST

    def set_runtime_idx(self, runtime_idx):
        self.runtime_idx = runtime_idx

    def convert_feature_to_bin(self, data_instances):
        LOGGER.info("convert feature to bins")
        param_obj = FeatureBinningParam(bin_num=self.bin_num)
        binning_obj = QuantileBinning(param_obj) 
        binning_obj.fit_split_points(data_instances)
        self.data_bin, self.bin_split_points, self.bin_sparse_points = binning_obj.convert_feature_to_bin(data_instances)
        # LOGGER.debug('len of bin_sparse_points is {}'.format(len(self.bin_sparse_points)))
        LOGGER.info("convert feature to bins over")

    def sample_valid_features(self):
        LOGGER.info("sample valid features")
        if self.feature_num is None:
            self.feature_num = self.bin_split_points.shape[0]

        choose_feature = random.choice(range(0, self.feature_num), \
                                       max(1, int(self.subsample_feature_rate * self.feature_num)), replace=False)

        valid_features = [False for i in range(self.feature_num)]
        for fid in choose_feature:
            valid_features[fid] = True
        return valid_features

    def gen_feature_fid_mapping(self, schema):
        header = schema.get("header")
        for i in range(len(header)):
            self.feature_name_fid_mapping[header[i]] = i

    def generate_flowid(self, round_num, tree_num):
        LOGGER.info("generate flowid, flowid {}".format(self.flowid))
        return ".".join(map(str, [self.flowid, round_num, tree_num]))

    def sync_tree_dim(self):
        LOGGER.info("sync tree dim from guest")
        self.tree_dim = self.transfer_inst.recv_data_from_guest()
        # self.tree_dim = federation.get(name=self.transfer_inst.tree_dim.name,
        #                                tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree_dim),
        #                                idx=0)
        LOGGER.info("tree dim is %d" % (self.tree_dim))

    def sync_stop_flag(self, num_round):
        LOGGER.info("sync stop flag from guest, boosting round is {}".format(num_round))
        stop_flag = self.transfer_inst.recv_data_from_guest()
        
        # stop_flag = federation.get(name=self.transfer_inst.stop_flag.name,
        #                            tag=self.transfer_inst.generate_transferid(self.transfer_inst.stop_flag, num_round),
        #                            idx=0)

        return stop_flag

    def fit(self, data_instances):
        LOGGER.info("begin to train secureboosting guest model")
        self.gen_feature_fid_mapping(data_instances.schema)
        LOGGER.debug("schema is {}".format(data_instances.schema))
        data_instances = self.data_alignment(data_instances)
        self.convert_feature_to_bin(data_instances)
        self.sync_tree_dim()

        for i in range(self.num_trees):
            # n_tree = []
            for tidx in range(self.tree_dim):
                LOGGER.info('============TREE_{}.{} START=============='.format(i, tidx))
                tree_inst = HeteroDecisionTreeHost(self.tree_param)

                tree_inst.set_inputinfo(data_bin=self.data_bin, bin_split_points=self.bin_split_points,
                                        bin_sparse_points=self.bin_sparse_points)

                valid_features = self.sample_valid_features()
                tree_inst.set_flowid(self.generate_flowid(i, tidx))
                tree_inst.set_runtime_idx(self.runtime_idx)
                tree_inst.set_valid_features(valid_features)
                tree_inst.set_transfer_inst(self.transfer_inst)

                tree_inst.fit()
                tree_meta, tree_param = tree_inst.get_model()
                self.trees_.append(tree_param)
                if self.tree_meta is None:
                    self.tree_meta = tree_meta
                # n_tree.append(tree_inst.get_tree_model())

            # self.trees_.append(n_tree)

            if self.n_iter_no_change is True:
                stop_flag = self.sync_stop_flag(i)
                if stop_flag:
                    break

        LOGGER.info("end to train secureboosting guest model")

    def predict(self, data_instances, predict_param=None):
        LOGGER.info("start predict")
        data_instances = self.data_alignment(data_instances)
        rounds = len(self.trees_) // self.tree_dim
        for i in range(rounds):
            # n_tree = self.trees_[i]
            for tidx in range(self.tree_dim):
                tree_inst = HeteroDecisionTreeHost(self.tree_param)
                tree_inst.load_model(self.tree_meta, self.trees_[i * self.tree_dim + tidx])
                # tree_inst.set_tree_model(self.trees_[i * self.tree_dim + tidx])
                tree_inst.set_flowid(self.generate_flowid(i, tidx))
                tree_inst.set_runtime_idx(self.runtime_idx)
                tree_inst.set_transfer_inst(self.transfer_inst)

                tree_inst.predict(data_instances)

        LOGGER.info("end predict")

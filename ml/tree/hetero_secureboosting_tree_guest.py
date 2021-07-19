from computing.d_table import DTable
from ml.tree.boosting_tree import BoostingTree
from ml.utils import consts
from ml.utils.logger import LOGGER
from ml.loss.cross_entropy import SigmoidBinaryCrossEntropyLoss, SoftmaxCrossEntropyLoss
from ml.feature.instance import Instance
from ml.param.feature_binning_param import FeatureBinningParam
from ml.feature.binning.quantile_binning import QuantileBinning


class HeteroSecureBoostingTreeGuest(BoostingTree):
    def __init__(self):
        super().__init__()

        self.convegence = None
        self.y = None
        self.F = None
        self.data_bin = None
        self.loss = None
        self.init_score = None
        self.classes_dict = {}
        self.classes_ = []
        self.num_classes = 0
        self.classify_target = "binary"
        self.feature_num = None
        self.encrypter = None
        self.grad_and_hess = None
        # self.flowid = 0
        self.tree_dim = 1
        self.tree_meta = None
        self.trees_ = []
        self.history_loss = []
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.encrypted_mode_calculator = None
        self.runtime_idx = 0
        self.feature_importances_ = {}
        self.role = consts.GUEST

    def set_loss(self, objective_param):
        loss_type = objective_param.objective
        params = objective_param.params
        LOGGER.info("set objective, objective is {}".format(loss_type))
        if self.task_type == consts.CLASSIFICATION:
            if loss_type == "cross_entropy":
                if self.num_classes == 2:
                    self.loss = SigmoidBinaryCrossEntropyLoss()
                else:
                    self.loss = SoftmaxCrossEntropyLoss()
            else:
                raise NotImplementedError("objective %s not supported yet" % (loss_type))
        elif self.task_type == consts.REGRESSION:
            raise NotImplementedError("task_type REGRESSION not supported yet")
            # if loss_type == "lse":
            #     self.loss = LeastSquaredErrorLoss()
            # elif loss_type == "lae":
            #     self.loss = LeastAbsoluteErrorLoss()
            # elif loss_type == "huber":
            #     self.loss = HuberLoss(params[0])
            # elif loss_type == "fair":
            #     self.loss = FairLoss(params[0])
            # elif loss_type == "tweedie":
            #     self.loss = TweedieLoss(params[0])
            # elif loss_type == "log_cosh":
            #     self.loss = LogCoshLoss()
            # else:
            #     raise NotImplementedError("objective %s not supported yet" % (loss_type))
        else:
            raise NotImplementedError("objective %s not supported yet" % (loss_type))

    def convert_feature_to_bin(self, data_instances:DTable):
        LOGGER.info("convert feature to bins")
        LOGGER.debug('bin_num = {}'.format(self.bin_num))
        param_obj = FeatureBinningParam(bin_num=self.bin_num)
        binning_obj = QuantileBinning(param_obj)
        binning_obj.fit_split_points(data_instances)

    def fit(self, data_instances:DTable):
        LOGGER.info("begin to train secureboosting guest model")
        self.gen_feature_fid_mapping(data_instances.schema)
        LOGGER.debug("schema is {}".format(data_instances.schema))
        LOGGER.debug("feature_name_fid_mapping is {}".format(self.feature_name_fid_mapping))
        data_instances = self.data_alignment(data_instances)
        LOGGER.debug_data(data_instances)
        self.convert_feature_to_bin(data_instances)
    
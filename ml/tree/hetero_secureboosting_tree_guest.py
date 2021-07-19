from computing.d_table import DTable
from ml.tree.boosting_tree import BoostingTree
from ml.utils import consts
from ml.utils.logger import LOGGER
from ml.loss.cross_entropy import SigmoidBinaryCrossEntropyLoss, SoftmaxCrossEntropyLoss
from ml.feature.instance import Instance
from ml.param.feature_binning_param import FeatureBinningParam
from ml.feature.binning.quantile_binning import QuantileBinning
from ml.utils.classfiy_label_checker import ClassifyLabelChecker, RegressionLabelChecker
from ml.secureprotol.encrypt import PaillierEncrypt
from ml.secureprotol.encrypt_mode import EncryptModeCalculator
import functools

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

    def update_f_value(self, new_f=None, tidx=-1):
        LOGGER.info("update tree f value, tree idx is {}".format(tidx))
        if self.F is None:
            if self.tree_dim > 1:
                self.F, self.init_score = self.loss.initialize(self.y, self.tree_dim)
            else:
                self.F, self.init_score = self.loss.initialize(self.y)
        else:
            accumuldate_f = functools.partial(self.accumulate_f,
                                              lr=self.learning_rate,
                                              idx=tidx)

            self.F = self.F.join(new_f, accumuldate_f)

    def convert_feature_to_bin(self, data_instances:DTable):
        LOGGER.info("convert feature to bins")
        param_obj = FeatureBinningParam(bin_num=self.bin_num)
        binning_obj = QuantileBinning(param_obj)
        binning_obj.fit_split_points(data_instances)
        self.data_bin, self.bin_split_points, self.bin_sparse_points = binning_obj.convert_feature_to_bin(data_instances)  # 分别是DTable，ndarray，dict
        LOGGER.info("convert feature to bins over")
        # print('data_bin(type: {}) is \n'.format(type(self.data_bin)), self.data_bin)
        # print('data_bin count is {}'.format(self.data_bin.count()))
        # print('data_bin header is {}'.format(self.data_bin.schema.get('header')))
        # print('bin_split_points(type: {}) is \n'.format(type(self.bin_split_points)), self.bin_split_points)
        # print('bin_sparse_points(type: {}) is \n'.format(type(self.bin_sparse_points)), self.bin_sparse_points)

    def set_y(self):
        LOGGER.info("set label from data and check label")
        self.y = self.data_bin.mapValues(lambda instance: instance.label)
        self.check_label()
    
    def generate_encrypter(self):
        LOGGER.info("generate encrypter")
        if self.encrypt_param.method == consts.PAILLIER:
            self.encrypter = PaillierEncrypt()
            self.encrypter.generate_key(self.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yes!!!")

        self.encrypted_calculator = EncryptModeCalculator(self.encrypter, self.calculated_mode, self.re_encrypted_rate)

    def check_label(self):
        LOGGER.info("check label")
        if self.task_type == consts.CLASSIFICATION:
            self.num_classes, self.classes_ = ClassifyLabelChecker.validate_y(self.y)
            if self.num_classes > 2:
                self.classify_target = "multinomial"
                self.tree_dim = self.num_classes

            range_from_zero = True
            for _class in self.classes_:
                try:
                    if _class >= 0 and _class < range_from_zero and isinstance(_class, int):
                        continue
                    else:
                        range_from_zero = False
                        break
                except:
                    range_from_zero = False

            self.classes_ = sorted(self.classes_)
            if not range_from_zero:
                class_mapping = dict(zip(self.classes_, range(self.num_classes)))
                self.y = self.y.mapValues(lambda _class: class_mapping[_class])

        else:
            RegressionLabelChecker.validate_y(self.y)

        self.set_loss(self.objective_param)

    def compute_grad_and_hess(self):
        LOGGER.info('compute grad and hess')
        loss_method = self.loss
        if self.task_type == consts.CLASSIFICATION:
            self.grad_and_hess = self.y.join(self.F, lambda y, f_val: \
                (loss_method.compute_grad(y, loss_method.predict(f_val)),
                loss_method.compute_hess(y, loss_method.predict(f_val))))
        else:
            self.grad_and_hess = self.y.join(self.F, lambda y, f_val:
            (loss_method.compute_grad(y, f_val),
             loss_method.compute_hess(y, f_val)))

    def sync_tree_dim(self):
        # TODO
        pass

    def fit(self, data_instances:DTable):
        LOGGER.info("begin to train secureboosting guest model")
        self.gen_feature_fid_mapping(data_instances.schema)
        LOGGER.debug("schema is {}".format(data_instances.schema))
        LOGGER.debug("feature_name_fid_mapping is {}".format(self.feature_name_fid_mapping))
        data_instances = self.data_alignment(data_instances)
        LOGGER.debug_data(data_instances)
        self.convert_feature_to_bin(data_instances)
        self.set_y()
        self.update_f_value()
        self.generate_encrypter()

        self.sync_tree_dim()

        for i in range(self.num_trees):
            self.compute_grad_and_hess()
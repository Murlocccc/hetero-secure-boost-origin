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
from ml.tree.hetero_decision_tree_guest import HeteroDecisionTreeGuest
from ml.optim.convergence import DiffConverge
from ml.loss.regression_loss import LeastAbsoluteErrorLoss, LeastSquaredErrorLoss, HuberLoss, FairLoss, TweedieLoss, LogCoshLoss
import numpy as np
import functools
import copy
from numpy import random

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
            if loss_type == "lse":
                self.loss = LeastSquaredErrorLoss()
            elif loss_type == "lae":
                self.loss = LeastAbsoluteErrorLoss()
            elif loss_type == "huber":
                self.loss = HuberLoss(params[0])
            elif loss_type == "fair":
                self.loss = FairLoss(params[0])
            elif loss_type == "tweedie":
                self.loss = TweedieLoss(params[0])
            elif loss_type == "log_cosh":
                self.loss = LogCoshLoss()
            else:
                raise NotImplementedError("objective %s not supported yet" % (loss_type))
        else:
            raise NotImplementedError("objective %s not supported yet" % (loss_type))

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

    def check_convergence(self, loss):
        LOGGER.info("check convergence")
        if self.convegence is None:
            self.convegence = DiffConverge(eps=self.tol)

        return self.convegence.is_converge(loss)

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

    def compute_loss(self):
        LOGGER.info("compute loss")
        if self.task_type == consts.CLASSIFICATION:
            loss_method = self.loss
            y_predict = self.F.mapValues(lambda val: loss_method.predict(val))
            loss = loss_method.compute_loss(self.y, y_predict)
        elif self.task_type == consts.REGRESSION:
            if self.objective_param.objective in ["lse", "lae", "logcosh", "tweedie", "log_cosh", "huber"]:
                loss_method = self.loss
                loss = loss_method.compute_loss(self.y, self.F)
            else:
                loss_method = self.loss
                y_predict = self.F.mapValues(lambda val: loss_method.predict(val))
                loss = loss_method.compute_loss(self.y, y_predict)

        return float(loss)

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
        LOGGER.info("sync tree dim to host")

        self.transfer_inst.send_data_to_hosts(self.tree_dim, -1)

        # federation.remote(obj=self.tree_dim,
        #                   name=self.transfer_inst.tree_dim.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.tree_dim),
        #                   role=consts.HOST,
        #                   idx=-1)

    def sync_stop_flag(self, stop_flag, num_round):
        LOGGER.info("sync stop flag to host, boosting round is {}".format(num_round))
        self.transfer_inst.send_data_to_hosts(stop_flag, -1)
        
        # federation.remote(obj=stop_flag,
        #                   name=self.transfer_inst.stop_flag.name,
        #                   tag=self.transfer_inst.generate_transferid(self.transfer_inst.stop_flag, num_round),
        #                   role=consts.HOST,
        #                   idx=-1)

    def generate_flowid(self, round_num, tree_num):
        LOGGER.info("generate flowid, flowid {}".format(self.flowid))
        return ".".join(map(str, [self.flowid, round_num, tree_num]))

    def get_grad_and_hess(self, tree_idx):
        LOGGER.info("get grad and hess of tree {}".format(tree_idx))
        grad_and_hess_subtree = self.grad_and_hess.mapValues(
            lambda grad_and_hess: (grad_and_hess[0][tree_idx], grad_and_hess[1][tree_idx]))
        return grad_and_hess_subtree

    @staticmethod
    def accumulate_f(f_val, new_f_val, lr=0.1, idx=0):
        f_val[idx] += lr * new_f_val
        return f_val

    def update_feature_importance(self, tree_feature_importance):
        for fid in tree_feature_importance:
            if fid not in self.feature_importances_:
                self.feature_importances_[fid] = 0

            self.feature_importances_[fid] += tree_feature_importance[fid]

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

    def fit(self, data_instances:DTable):
        LOGGER.info("begin to train secureboosting guest model")
        self.gen_feature_fid_mapping(data_instances.schema)
        # LOGGER.debug("schema is {}".format(data_instances.schema))
        data_instances = self.data_alignment(data_instances)
        # LOGGER.debug_data(data_instances)
        self.convert_feature_to_bin(data_instances)
        self.set_y()
        self.update_f_value()
        self.generate_encrypter()

        self.sync_tree_dim()

        for i in range(self.num_trees):
            # n_tree = []
            self.compute_grad_and_hess()
            for tidx in range(self.tree_dim):
                tree_inst = HeteroDecisionTreeGuest(self.tree_param)

                tree_inst.set_inputinfo(self.data_bin, self.get_grad_and_hess(tidx), self.bin_split_points,
                                        self.bin_sparse_points)
                
                valid_features = self.sample_valid_features()
                tree_inst.set_valid_features(valid_features)
                tree_inst.set_encrypter(self.encrypter)
                tree_inst.set_encrypted_mode_calculator(self.encrypted_calculator)
                tree_inst.set_flowid(self.generate_flowid(i, tidx))
                tree_inst.set_transfer_inst(self.transfer_inst)
                
                tree_inst.fit()

                tree_meta, tree_param = tree_inst.get_model()
                self.trees_.append(tree_param)
                if self.tree_meta is None:
                    self.tree_meta = tree_meta
                # n_tree.append(tree_inst.get_tree_model())
                self.update_f_value(new_f=tree_inst.predict_weights, tidx=tidx)
                self.update_feature_importance(tree_inst.get_feature_importance())
                
            # self.trees_.append(n_tree)
            loss = self.compute_loss()
            self.history_loss.append(loss)
            LOGGER.info("round {} loss is {}".format(i, loss))

            if self.n_iter_no_change is True:
                if self.check_convergence(loss):
                    self.sync_stop_flag(True, i)
                    break
                else:
                    self.sync_stop_flag(False, i)

            LOGGER.debug("history loss is {}".format(min(self.history_loss)))

            LOGGER.info("end to train secureboosting guest model")

    def predict_f_value(self, data_instances:DTable):
        LOGGER.info("predict tree f value, there are {} trees".format(len(self.trees_)))
        tree_dim = self.tree_dim
        init_score = self.init_score
        self.F = data_instances.mapValues(lambda v: copy.deepcopy(init_score))
        rounds = len(self.trees_) // self.tree_dim
        for i in range(rounds):
            for tidx in range(self.tree_dim):
                tree_inst = HeteroDecisionTreeGuest(self.tree_param)
                tree_inst.load_model(self.tree_meta, self.trees_[i * self.tree_dim + tidx])
                # tree_inst.set_tree_model(self.trees_[i * self.tree_dim + tidx])
                tree_inst.set_flowid(self.generate_flowid(i, tidx))
                tree_inst.set_transfer_inst(self.transfer_inst)

                predict_data = tree_inst.predict(data_instances)
                self.update_f_value(new_f=predict_data, tidx=tidx)

    def predict(self, data_instances:DTable):
        LOGGER.info("start predict")
        data_instances = self.data_alignment(data_instances)
        self.predict_f_value(data_instances)
        if self.task_type == consts.CLASSIFICATION:
            loss_method = self.loss
            classes_ = self.classes_
            if self.num_classes == 2:
                # print(self.F)
                # print('---------------------------------------------------------------------------')
                predicts = self.F.mapValues(lambda f: float(loss_method.predict(f)))
                # print(predicts)
                # print('---------------------------------------------------------------------------')
                threshold = self.predict_param.threshold
                predict_result = data_instances.join(predicts, lambda inst, pred: [inst.label, classes_[1] if pred > threshold else classes_[0], pred, {"0": 1 - pred, "1": pred}])
            else:
                predicts = self.F.mapValues(lambda f: loss_method.predict(f).tolist())
                predict_label = predicts.mapValues(lambda preds: classes_[np.argmax(preds)])
                predict_result = data_instances.join(predicts, lambda inst, preds: [inst.label, classes_[np.argmax(preds)], np.max(preds), dict(zip(map(str, classes_), preds))])
        elif self.task_type == consts.REGRESSION:
            if self.objective_param.objective in ["lse", "lae", "huber", "log_cosh", "fair", "tweedie"]:
                predicts = self.F
            else:
                raise NotImplementedError("objective {} not supprted yet".format(self.objective_param.objective))
            predict_result = data_instances.join(predicts, lambda inst, pred: [inst.label, float(pred), float(pred), {"label": float(pred)}])
        else:
            raise NotImplementedError("task type {} not supported yet".format(self.task_type)) 
        
        LOGGER.info("end predict")

        return predict_result


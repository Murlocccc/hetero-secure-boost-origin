
from computing.d_table import DTable
import numpy as np
from ml.param.boosting_tree_param import BoostingTreeParam
from ml.utils import consts
from ml.feature.sparse_vector import SparseVector
from ml.model_base import ModelBase
from ml.utils import abnormal_detection
from federation.transfer_inst import TransferInstGuest, TransferInstHost


class BoostingTree(ModelBase):
    def __init__(self):
        super(BoostingTree, self).__init__()
        self.tree_param = None
        self.task_type=None
        self.objective_param = None
        self.learning_rate = None
        self.num_trees = None
        self.subsample_feature_rate = None
        self.n_iter_no_change = None
        self.encrypt_param = None
        self.tol = 0.0
        self.bin_num = None
        self.calculated_mode = None
        self.re_encrypted_rate = None
        self.predict_param = None
        self.cv_param = None
        self.feature_name_fid_mapping = {}
        self.role = ''
        self.mode = consts.HETERO
        self.transfer_inst = None
        self.model_param = BoostingTreeParam(num_trees=5)

    def _init_model(self, boostingtree_param):
        self.tree_param = boostingtree_param.tree_param
        self.task_type = boostingtree_param.task_type
        self.objective_param = boostingtree_param.objective_param
        self.learning_rate = boostingtree_param.learning_rate
        self.num_trees = boostingtree_param.num_trees
        self.subsample_feature_rate = boostingtree_param.subsample_feature_rate
        self.n_iter_no_change = boostingtree_param.n_iter_no_change
        self.encrypt_param = boostingtree_param.encrypt_param
        self.tol = boostingtree_param.tol
        self.bin_num = boostingtree_param.bin_num
        self.calculated_mode = boostingtree_param.encrypted_mode_calculator_param.mode
        self.re_encrypted_rate = boostingtree_param.encrypted_mode_calculator_param.re_encrypted_rate
        self.predict_param = boostingtree_param.predict_param
        self.cv_param = boostingtree_param.cv_param

    @staticmethod
    def data_format_transform(row):
        if type(row.features).__name__ != consts.SPARSE_VECTOR:  # 说明是ndarray类型，需要转化为SPARSE_VECTOR
            feature_shape = row.features.shape[0]
            indices = []
            data = []

            for i in range(feature_shape):
                if np.abs(row.features[i]) < consts.FLOAT_ZERO:
                    continue

                indices.append(i)
                data.append(row.features[i])

            row.features = SparseVector(indices, data, feature_shape)

        return row

    def data_alignment(self, data_instances:DTable):
        abnormal_detection.empty_table_detection(data_instances)
        # TODO abnormal_detection.empty_feature_detection(data_instances)

        schema = data_instances.schema

        new_data_instances = data_instances.mapValues(lambda row: BoostingTree.data_format_transform(row))

        new_data_instances.schema = schema

        return new_data_instances

    def gen_feature_fid_mapping(self, schema):
        header = schema.get("header")
        for i in range(len(header)):
            self.feature_name_fid_mapping[header[i]] = i

    def fit(self, data_inst):
        pass

    def predict(self, data_inst):
        pass

    def _get_cv_param(self):
        self.model_param.cv_param.role = self.role
        self.model_param.cv_param.mode = self.mode
        return self.model_param.cv_param

    def predict_proba(self, data_inst):
        pass

    def load_model(self):
        pass

    def save_data(self):
        return self.data_output

    def save_model(self):
        pass

    def set_transfer_inst(self, transfer_inst):
        self.transfer_inst = transfer_inst

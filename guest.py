from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest
from i_o.utils import read_from_csv
from ml.feature.instance import Instance
from ml.utils.logger import LOGGER


def test_hetero_secure_boost_guest():

    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)

    # 从文件读取数据
    header, ids, features, lables = read_from_csv('breast_hetero_mini_guest.csv')
    instances = []
    for i, feature in enumerate(features):
        inst = Instance(inst_id=ids[i], features=feature, label=lables[i])
        instances.append(inst)

    # 生成DTable
    data_instances = DTable(False, instances)
    data_instances.schema['header'] = header

    # fit
    hetero_secure_boost_guest.fit(data_instances=data_instances)

if __name__ == '__main__':
    test_hetero_secure_boost_guest()

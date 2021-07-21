from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest
from i_o.utils import read_from_csv_with_lable
from ml.feature.instance import Instance
from ml.utils.logger import LOGGER
from federation.transfer_inst import TransferInstGuest
import random


def test_hetero_secure_boost_guest():

    # guest传输实体
    transfer_inst = TransferInstGuest()

    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)
    hetero_secure_boost_guest.set_transfer_inst(transfer_inst)

    # 从文件读取数据，并划分训练集和测试集
    # header, ids, features, lables = read_from_csv('data/breast_hetero_mini/breast_hetero_mini_guest.csv')
    header, ids, features, lables = read_from_csv_with_lable('data/breast_hetero/breast_hetero_guest.csv')
    instances = []
    for i, feature in enumerate(features):
        inst = Instance(inst_id=ids[i], features=feature, label=lables[i])
        instances.append(inst)
    
    train_instances, test_instances = data_split(instances, 0.8, True, 2)
    

    # 生成DTable
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header

    # fit
    hetero_secure_boost_guest.fit(data_instances=train_instances)

    # predict
    predict_result = hetero_secure_boost_guest.predict(test_instances)

    # print(predict_result)

    def func(kvs):
        correct_num = 0
        for _, v in kvs:
            if v[0] == v[1]:
                correct_num += 1
        return [correct_num / len(kvs)]

    accuracy = predict_result.mapPartitions(func).reduce(lambda a, b: a + b)

    print('accuracy is ', accuracy)

def data_split(full_list, ratio, shuffle=False, random_seed=None):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     返回的第一个列表的占比
    :param shuffle:   是否随机
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2

def heteto_sbt_guest():

    transfer_inst = TransferInstGuest()

    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)
    hetero_secure_boost_guest.set_transfer_inst(transfer_inst)

    # 从文件读取数据
    # header, ids, features, lables = read_from_csv('data/breast_hetero_mini/breast_hetero_mini_guest.csv')
    header, ids, features, lables = read_from_csv_with_lable('data/breast_hetero/breast_hetero_guest.csv')
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

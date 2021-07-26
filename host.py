# role：host
# start parameters：4
#   - the address of csv file
#   - the proportion of data divided
#   - the port of guest
#   - the run_time_idx, it should be different among hosts

# example
#   python .\host.py data/breast_hetero/breast_hetero_host.csv 0.8 10086 0

from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_host import HeteroSecureBoostingTreeHost
from i_o.utils import read_from_csv_with_no_lable
from ml.feature.instance import Instance
from ml.utils.logger import LOGGER
from federation.transfer_inst import TransferInstHost
import random
import sys

def getArgs():
    argv = sys.argv[1:]
    return argv

def test_hetero_seucre_boost_host():

    argv = getArgs()
    csv_address = argv[0]
    divided_proportion = float(argv[1])
    port = int(argv[2])
    run_time_idx = int(argv[3])

    # host传输实体
    transfer_inst = TransferInstHost(port=port)

    random_seed = transfer_inst.recv_data_from_guest()
    
    hetero_secure_boost_host = HeteroSecureBoostingTreeHost()
    hetero_secure_boost_host.model_param.subsample_feature_rate = 1
    hetero_secure_boost_host._init_model(hetero_secure_boost_host.model_param)
    hetero_secure_boost_host.set_transfer_inst(transfer_inst)
    hetero_secure_boost_host.set_runtime_idx(run_time_idx)

    # 从文件读取数据，并划分训练集和测试集
    # header, ids, features, lables = read_from_csv('data/breast_hetero_mini/breast_hetero_mini_guest.csv')
    header, ids, features= read_from_csv_with_no_lable(csv_address)


    instances = []
    for i, feature in enumerate(features):
        inst = Instance(inst_id=ids[i], features=feature)
        instances.append(inst)
    
    train_instances, test_instances = data_split(instances, divided_proportion, True, 2)
    # hetero_secure_boost_host.model_param.subsample_feature_rate = 1   

    # ids = [a.inst_id for a in train_instances]

    # print(sorted(ids))

    # return

    # 生成DTable
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header

    # fit
    hetero_secure_boost_host.fit(data_instances=train_instances)

    # predict
    hetero_secure_boost_host.predict(test_instances)

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

def heteto_sbt_host():

    # python host.py data/weather/weather_train_host0.csv data/weather/weather_test_host0.csv 10086 0
    # python host.py data/weather/weather_train_host1.csv data/weather/weather_test_host1.csv 10086 1

    # python host.py data/lr/lr_train_host0.csv data/lr/lr_test_host0.csv 10086 0
    # python host.py data/lr/lr_train_host1.csv data/lr/lr_test_host1.csv 10086 1

    argv = getArgs()
    train_csv_address = argv[0]
    test_csv_address = argv[1]
    port = int(argv[2])
    run_time_idx = int(argv[3])

    # host传输实体
    transfer_inst = TransferInstHost(port=port)
    
    hetero_secure_boost_host = HeteroSecureBoostingTreeHost()
    # hetero_secure_boost_host.model_param.subsample_feature_rate = 1
    hetero_secure_boost_host._init_model(hetero_secure_boost_host.model_param)
    hetero_secure_boost_host.set_transfer_inst(transfer_inst)
    hetero_secure_boost_host.set_runtime_idx(run_time_idx)

    # 从文件读取数据
    header1, ids1, features1 = read_from_csv_with_no_lable(train_csv_address)
    header2, ids2, features2 = read_from_csv_with_no_lable(test_csv_address)

    train_instances = []
    test_instances = []

    for i, feature in enumerate(features1):
        inst = Instance(inst_id=ids1[i], features=feature)
        train_instances.append(inst)
    
    for i, feature in enumerate(features2):
        inst = Instance(inst_id=ids2[i], features=feature)
        test_instances.append(inst)

    # 生成DTable
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header1
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header2

    LOGGER.info('length of train set is {}, schema is {}'.format(train_instances.count(), train_instances.schema))
    LOGGER.info('length of test set is {}, schema is {}'.format(test_instances.count(), test_instances.schema))

    # fit
    hetero_secure_boost_host.fit(data_instances=train_instances)

    # predict
    hetero_secure_boost_host.predict(test_instances)

if __name__ == '__main__':
    heteto_sbt_host()
# role：host
# start parameters：4
#   - the address of csv file
#   - the proportion of data divided
#   - the port of guest
#   - the run_time_idx, it should be different among hosts

from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_host import HeteroSecureBoostingTreeHost
from i_o.utils import read_from_csv_with_no_lable
from ml.feature.instance import Instance
from ml.utils.logger import MyLoggerFactory
from federation.transfer_inst import TransferInstHost
import random
import sys
import logging
from ml.utils.logger import LOGGER
import time


my_logger = MyLoggerFactory.build("host")

def getArgs():
    argv = sys.argv[1:]
    return argv

def test_hetero_seucre_boost_host():

    # python host.py data/weather/weather_train_host0.csv data/weather/weather_test_host0.csv 10086 0
    # python host.py data/weather/weather_train_host1.csv data/weather/weather_test_host1.csv 10086 1

    # 获取命令行参数
    argv = getArgs()
    train_csv_address = argv[0]
    test_csv_address = argv[1]
    port = int(argv[2])
    run_time_idx = int(argv[3])

    # 初始化 log 模块
    LOGGER.basic_config(filename=time.strftime("log/host{}_%Y-%m-%d-%H_%M_%S".format(run_time_idx))+'.log', level=logging.DEBUG)

    # 记录一些参数设置到日志
    LOGGER.info('here is the host_{}'.format(run_time_idx))
    LOGGER.info('train_file is {}'.format(train_csv_address))
    LOGGER.info('test_file is {}'.format(test_csv_address))

    # 实例化 hetero secure boost tree host 实体
    transfer_inst = TransferInstHost(port=port)
    
    # 设置 hetero secure boost tree host 的参数
    hetero_secure_boost_host = HeteroSecureBoostingTreeHost()

    # 使用默认参数，对 hetero secure boost tree host 进行初始化
    hetero_secure_boost_host._init_model(hetero_secure_boost_host.model_param)

    # 设置 hetero secure boost tree host 的运行时标记，用以区分各个 host
    hetero_secure_boost_host.set_runtime_idx(run_time_idx)

    # 给 hetero secure boost tree host 分配一个传输实体
    hetero_secure_boost_host.set_transfer_inst(transfer_inst)

    # 从训练集文件和测试集文件读取数据
    header1, ids1, features1 = read_from_csv_with_no_lable(train_csv_address)
    header2, ids2, features2 = read_from_csv_with_no_lable(test_csv_address)

    # 将读取的数据转化为 Instance 对象
    train_instances = []
    test_instances = []
    for i, feature in enumerate(features1):
        inst = Instance(inst_id=ids1[i], features=feature)
        train_instances.append(inst)
    for i, feature in enumerate(features2):
        inst = Instance(inst_id=ids2[i], features=feature)
        test_instances.append(inst)
    
    # 使用上面得到的 Instance 的列表转化为 DTable 对象
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header1
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header2

    # 记录数据集相关信息到日志
    LOGGER.info('length of train set is {}, schema is {}'.format(train_instances.count(), train_instances.schema))
    LOGGER.info('length of test set is {}, schema is {}'.format(test_instances.count(), test_instances.schema))

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

    transfer_inst = TransferInstHost()

    hetero_secure_boost_host = HeteroSecureBoostingTreeHost()
    hetero_secure_boost_host._init_model(hetero_secure_boost_host.model_param)
    hetero_secure_boost_host.set_transfer_inst(transfer_inst)

    # 从文件读取数据
    # header, ids, features, lables = read_from_csv('data/breast_hetero_mini/breast_hetero_mini_guest.csv')
    header, ids, features= read_from_csv_with_no_lable('data/breast_hetero/breast_hetero_guest.csv')
    instances = []
    for i, feature in enumerate(features):
        inst = Instance(inst_id=ids[i], features=feature)
        instances.append(inst)

    # 生成DTable
    data_instances = DTable(False, instances)
    data_instances.schema['header'] = header


    # fit
    hetero_secure_boost_host.fit(data_instances=data_instances)

if __name__ == '__main__':
    test_hetero_seucre_boost_host()
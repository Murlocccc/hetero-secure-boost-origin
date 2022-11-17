# role：guest
# start parameters：5
#   - the address of csv file
#   - the number of hosts
#   - the proportion of data divided
#   - the type of the task, only support 'CLASSIFICATION'
#   - the port for federation

from ml.utils.logger import MyLoggerFactory
from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest
from i_o.utils import read_from_csv_with_lable
from ml.feature.instance import Instance
from federation.transfer_inst import TransferInstGuest
from ml.utils import consts
import random
import sys
import time

my_logger = MyLoggerFactory.build("guest")

def getArgs():
    argv = sys.argv[1:]
    return argv

def test_hetero_secure_boost_guest():

    # python guest.py data/weather/weather_train_guest.csv data/weather/weather_test_guest.csv 2 CLASSIFICATION 10086

    # python guest.py data/lr/lr_train_guest.csv data/lr/lr_test_guest.csv 2 CLASSIFICATION 10086

    # python guest.py data/asd/train_guest.csv data/asd/test_guest.csv 1 CLASSIFICATION 10086

    # python guest.py data/breast_hetero_mini/breast_hetero_mini_train_guest.csv data/breast_hetero_mini/breast_hetero_mini_test_guest.csv 1 CLASSIFICATION 10086

    # python guest.py data/credit2/credit2_train_guest.csv data/credit2/credit2_test_guest.csv 1 CLASSIFICATION 10086

    # python guest.py data/breast_hetero/breast_hetero_train_guest.csv data/breast_hetero/breast_hetero_test_guest.csv 1 CLASSIFICATION 10086

    # python guest.py data/vehicle_scale_hetero/vehicle_scale_hetero_train_guest.csv data/vehicle_scale_hetero/vehicle_scale_hetero_test_guest.csv 1 CLASSIFICATION 10086

    # 获取命令行参数
    argv = getArgs()
    train_csv_address = argv[0]
    test_csv_address = argv[1]
    num_hosts = int(argv[2])
    task_type = argv[3]
    port = int(argv[4])

    # 初始化 log 模块

    # 记录一些参数设置到日志
    my_logger.info('here is the guest')
    my_logger.info('train_file is {}'.format(train_csv_address))
    my_logger.info('test_file is {}'.format(test_csv_address))
    my_logger.info('task_type is {}'.format(task_type))

    # 实例化 guest 传输实体
    transfer_inst = TransferInstGuest(port, num_hosts)

    # 实例化 hetero secure boost tree guest 实体
    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()

    # 设置 hetero secure boost tree guest 的参数
    hetero_secure_boost_guest.model_param.tree_param.max_depth=5
    if task_type == 'CLASSIFICATION':
        hetero_secure_boost_guest.model_param.task_type = consts.CLASSIFICATION
    # elif task_type == 'REGRESSION':
    #     hetero_secure_boost_guest.model_param.task_type = consts.REGRESSION
    else:
        raise ValueError('the value of param task_type wrong')

    # 使用设置的参数以及默认参数，对 hetero secure boost tree guest 进行初始化
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)

    # 给 hetero secure boost tree guest 分配一个传输实体
    hetero_secure_boost_guest.set_transfer_inst(transfer_inst)

    # 从训练集文件和测试集文件读取数据
    header1, ids1, features1, lables1 = read_from_csv_with_lable(train_csv_address)
    header2, ids2, features2, lables2 = read_from_csv_with_lable(test_csv_address)
    
    # 将读取的数据转化为 Instance 对象
    train_instances = []
    test_instances = []
    for i, feature in enumerate(features1):
        inst = Instance(inst_id=ids1[i], features=feature, label=lables1[i])
        train_instances.append(inst)
    for i, feature in enumerate(features2):
        inst = Instance(inst_id=ids2[i], features=feature, label=lables2[i])
        test_instances.append(inst)
    
    # 使用上面得到的 Instance 的列表转化为 DTable 对象
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header1
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header2

    # 记录数据集相关信息到日志
    my_logger.info('length of train set is {}, schema is {}'.format(train_instances.count(), train_instances.schema))
    my_logger.info('length of test set is {}, schema is {}'.format(test_instances.count(), test_instances.schema))

    # fit
    hetero_secure_boost_guest.fit(data_instances=train_instances)

    # predict
    predict_result = hetero_secure_boost_guest.predict(test_instances)

    # print(predict_result)

    # 得到二分类混淆矩阵的函数
    def cal_statistic(kvs):
        ret = [0, 0, 0, 0]
        for _, v in kvs:
            if v[0] == v[1]:
                if v[0] == 1:
                    ret[0] += 1
                else:
                    ret[1] += 1
            else:
                if v[0] == 1:
                    ret[3] += 1
                else:
                    ret[2] += 1
        return [ret]

    # 计算并输出混淆矩阵
    statistic = predict_result.mapPartitions(cal_statistic).reduce(lambda a, b: a + b)
    my_logger.info('(TP, TN, FP, FN) is {}'.format(statistic))

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

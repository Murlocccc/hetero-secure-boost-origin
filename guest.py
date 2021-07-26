# role：guest
# start parameters：5
#   - the address of csv file
#   - the number of hosts
#   - the proportion of data divided
#   - the type of the task, only support 'CLASSIFICATION'
#   - the port for federation


# example:
#   python .\guest.py data/breast_hetero/breast_hetero_guest.csv 1 0.8 CLASSIFICATION 10086


from numpy import positive
from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest
from i_o.utils import read_from_csv_with_lable
from ml.feature.instance import Instance
from ml.utils.logger import LOGGER
from federation.transfer_inst import TransferInstGuest
from ml.utils import consts
import random
import sys

def getArgs():
    argv = sys.argv[1:]
    return argv

def test_hetero_secure_boost_guest():

    argv = getArgs()
    csv_address = argv[0]
    num_hosts = int(argv[1])
    divided_proportion = float(argv[2])
    task_type = argv[3]
    port = int(argv[4])

    # guest传输实体
    transfer_inst = TransferInstGuest(port, num_hosts)

    random_seed = random.randint(1,1000)

    transfer_inst.send_data_to_hosts(random_seed, -1)

    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()
    if task_type == 'CLASSIFICATION':
        hetero_secure_boost_guest.model_param.task_type = consts.CLASSIFICATION
    # elif task_type == 'REGRESSION':
    #     hetero_secure_boost_guest.model_param.task_type = consts.REGRESSION
    else:
        raise ValueError('the value of param task_type wrong')

    hetero_secure_boost_guest.model_param.subsample_feature_rate = 1
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)
    
    hetero_secure_boost_guest.set_transfer_inst(transfer_inst)

    # 从文件读取数据，并划分训练集和测试集
    # header, ids, features, lables = read_from_csv('data/breast_hetero_mini/breast_hetero_mini_guest.csv')
    header, ids, features, lables = read_from_csv_with_lable(csv_address)
    instances = []

    for i, feature in enumerate(features):
        inst = Instance(inst_id=ids[i], features=feature, label=lables[i])
        instances.append(inst)
    
    train_instances, test_instances = data_split(instances, divided_proportion, True, 2)

    # 生成DTable
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header

    # fit
    hetero_secure_boost_guest.fit(data_instances=train_instances)

    # predict
    predict_result = hetero_secure_boost_guest.predict(test_instances)

    def func(kvs):
        correct_num = 0
        for i, v in kvs:
            if v[0] == v[1]:
                correct_num += 1
            # else:
            # (i, v)
        print(correct_num)
        print(len(kvs))
        return [correct_num / len(kvs)]

    # print(predict_result)

    accuracy = predict_result.mapPartitions(func).reduce(lambda a, b: a + b)

    # print('num is ', predict_result.count())
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

    # python guest.py data/weather/weather_train_guest.csv data/weather/weather_test_guest.csv 2 CLASSIFICATION 10086

    # python guest.py data/lr/lr_train_guest.csv data/lr/lr_test_guest.csv 2 CLASSIFICATION 10086

    argv = getArgs()
    train_csv_address = argv[0]
    test_csv_address = argv[1]
    num_hosts = int(argv[2])
    task_type = argv[3]
    port = int(argv[4])

    # guest传输实体
    transfer_inst = TransferInstGuest(port, num_hosts)

    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()
    if task_type == 'CLASSIFICATION':
        hetero_secure_boost_guest.model_param.task_type = consts.CLASSIFICATION
    # elif task_type == 'REGRESSION':
    #     hetero_secure_boost_guest.model_param.task_type = consts.REGRESSION
    else:
        raise ValueError('the value of param task_type wrong')
    # hetero_secure_boost_guest.model_param.subsample_feature_rate = 1
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)
    hetero_secure_boost_guest.set_transfer_inst(transfer_inst)

    # 从文件读取数据
    header1, ids1, features1, lables1 = read_from_csv_with_lable(train_csv_address)
    header2, ids2, features2, lables2 = read_from_csv_with_lable(test_csv_address)
    train_instances = []
    test_instances = []

    for i, feature in enumerate(features1):
        inst = Instance(inst_id=ids1[i], features=feature, label=lables1[i])
        train_instances.append(inst)

    for i, feature in enumerate(features2):
        inst = Instance(inst_id=ids2[i], features=feature, label=lables2[i])
        test_instances.append(inst)

    # 生成DTable
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header1
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header2

    LOGGER.info('length of train set is {}, schema is {}'.format(train_instances.count(), train_instances.schema))
    LOGGER.info('length of test set is {}, schema is {}'.format(test_instances.count(), test_instances.schema))

    # fit
    hetero_secure_boost_guest.fit(data_instances=train_instances)

    # predict
    predict_result = hetero_secure_boost_guest.predict(test_instances)

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

    def cal_accuracy(kvs):
        correct_num = 0
        for i, v in kvs:
            if v[0] == v[1]:
                correct_num += 1
            # else:
            # (i, v)
        print(correct_num)
        return [correct_num / len(kvs)]

    def cal_recall(kvs):
        correct_positive_num = 0
        positive_num = 0
        for i, v in kvs:
            if v[0] == v[1] and v[0] == 1:
                correct_positive_num += 1
            if v[0] == 1:
                positive_num += 1
        
        return [correct_positive_num / positive_num]

    # print(predict_result)

    statistic = predict_result.mapPartitions(cal_statistic).reduce(lambda a, b: a + b)

    # print('num is ', predict_result.count())
    print('(TP, TN, FP, FN) is {}'.format(statistic))

if __name__ == '__main__':
    heteto_sbt_guest()

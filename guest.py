# role：guest
# start parameters：5
#   - the address of csv file
#   - the number of hosts
#   - the proportion of data divided
#   - the type of the task, only support 'CLASSIFICATION'
#   - the port for federation

from ml.utils.logger import LOGGER, MyLoggerFactory
from computing.d_table import DTable
from ml.tree.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest
from i_o.utils import read_from_csv_with_lable
from ml.feature.instance import Instance
from federation.transfer_inst import TransferInstGuest
from ml.utils import consts
import random
import sys

my_logger = MyLoggerFactory.build("guest")

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

    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()
    if task_type == 'CLASSIFICATION':
        hetero_secure_boost_guest.model_param.task_type = consts.CLASSIFICATION
    # elif task_type == 'REGRESSION':
    #     hetero_secure_boost_guest.model_param.task_type = consts.REGRESSION
    else:
        raise ValueError('the value of param task_type wrong')

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

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
from ml.param.encrypt_param import EncryptParam
import random
import sys
import time
import argparse
import json

my_logger = MyLoggerFactory.build("guest")

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description='secureboost cmd')
    parser.add_argument("--dataset", type=str, help="pre define dataset", default="")
    parser.add_argument('--train_file', type=str, help='path for train file', default="")
    parser.add_argument('--test_file', type=str, help='path for test file', default="")
    parser.add_argument('--num_hosts', type=int, help='the number of hosts', default=1)
    parser.add_argument('--task_type', type=str, help='the proportion of data divided', default="CLASSIFICATION")
    parser.add_argument('--port', type=int, help='the port for federation', default=10086)
    parser.add_argument('--encrypt', type=str, help='select the add HE style: Paillier | Plaintext', default="Plaintext")
    args = parser.parse_args(sys.argv[1:])
    return args

def test_hetero_secure_boost_guest():

    # python guest.py data/weather/weather_train_guest.csv data/weather/weather_test_guest.csv 2 CLASSIFICATION 10086

    # python guest.py data/lr/lr_train_guest.csv data/lr/lr_test_guest.csv 2 CLASSIFICATION 10086

    # python guest.py data/asd/train_guest.csv data/asd/test_guest.csv 1 CLASSIFICATION 10086

    # python guest.py data/breast_hetero_mini/breast_hetero_mini_train_guest.csv data/breast_hetero_mini/breast_hetero_mini_test_guest.csv 1 CLASSIFICATION 10086

    # python guest.py data/credit2/credit2_train_guest.csv data/credit2/credit2_test_guest.csv 1 CLASSIFICATION 10086

    # python guest.py data/breast_hetero/breast_hetero_train_guest.csv data/breast_hetero/breast_hetero_test_guest.csv 1 CLASSIFICATION 10086

    # python guest.py data/vehicle_scale_hetero/vehicle_scale_hetero_train_guest.csv data/vehicle_scale_hetero/vehicle_scale_hetero_test_guest.csv 1 CLASSIFICATION 10086

    # 获取命令行参数
    args = getArgs()
    if len(args.dataset) > 0:
        if args.dataset not in DATASET_DICT:
            my_logger.error(f"dataset error! {args.dataset} haven't not set")
            exit(1)
        train_csv_address = DATASET_DICT[args.dataset]['train_file']
        test_csv_address = DATASET_DICT[args.dataset]['test_file']
    else:
        train_csv_address = args.train_file
        test_csv_address = args.test_file
    num_hosts = args.num_hosts
    task_type = args.task_type
    port = args.port
    encrypt_param_str = args.encrypt

    # 记录一些参数设置到日志
    my_logger.info('Here is the guest')

    my_logger.info(f'{"="*20} Init {"="*20}')
    # 实例化 guest 传输实体
    my_logger.info(f'--Port={port}, num_hosts={num_hosts}')
    transfer_inst = TransferInstGuest(port, num_hosts)

    # 实例化 hetero secure boost tree guest 实体
    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()

    # 设置 hetero secure boost tree guest 的参数
    hetero_secure_boost_guest.model_param.tree_param.max_depth=5


    my_logger.info('--task_type is {}'.format(task_type))
    if task_type == 'CLASSIFICATION':
        hetero_secure_boost_guest.model_param.task_type = consts.CLASSIFICATION
    # elif task_type == 'REGRESSION':
    #     hetero_secure_boost_guest.model_param.task_type = consts.REGRESSION
    else:
        raise ValueError('the value of param task_type wrong')

    my_logger.info(f'--encrypt={encrypt_param_str}')
    encrypt_param = EncryptParam(encrypt_param_str)
    if encrypt_param.check():
        hetero_secure_boost_guest.model_param.encrypt_param = encrypt_param

    # 使用设置的参数以及默认参数，对 hetero secure boost tree guest 进行初始化
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)

    # 给 hetero secure boost tree guest 分配一个传输实体
    hetero_secure_boost_guest.set_transfer_inst(transfer_inst)

    # 从训练集文件和测试集文件读取数据
    my_logger.info(f'--dataset log info')
    my_logger.info('\ttrain_file is {}'.format(train_csv_address))
    my_logger.info('\ttest_file is {}'.format(test_csv_address))
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


    # DEBUT_OUTPUT
    label_vec = dict()
    for inst in train_instances:
        label_vec[inst.inst_id] = inst.label
    for inst in test_instances:
        label_vec[inst.inst_id] = inst.label    
    
    # 使用上面得到的 Instance 的列表转化为 DTable 对象
    train_instances = DTable(False, train_instances)
    train_instances.schema['header'] = header1
    test_instances = DTable(False, test_instances)
    test_instances.schema['header'] = header2

    # 记录数据集相关信息到日志
    my_logger.info('length of train set is {}, schema is {}'.format(train_instances.count(), train_instances.schema))
    my_logger.info('length of test set is {}, schema is {}'.format(test_instances.count(), test_instances.schema))


    my_logger.info(f'{"="*20} Fit {"="*20}')
    # fit
    hetero_secure_boost_guest.fit(data_instances=train_instances)

    # predict
    predict_result = hetero_secure_boost_guest.predict(test_instances)

    def cal_accuracy(kvs):
        correct_num = 0
        for _, v in kvs:
            if v[0] == v[1]:
                correct_num += 1
        return [correct_num / len(kvs)]

    accuracy = predict_result.mapPartitions(cal_accuracy).reduce(lambda a, b: a + b)

    my_logger.info(f'accuracy is {accuracy}')

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

    # debug output
    tmp_predict_result = {}
    predict_result_col = list(predict_result.collect())
    for index in range(len(predict_result_col)):
        tmp_predict_result[predict_result_col[index][1][-1]] = predict_result_col[index][1][1]
    tmp_dict = {
        "dataset": {
            "train_file": train_csv_address,
            "test_file": test_csv_address,
        },
        "tree_params": {
            "depth": hetero_secure_boost_guest.model_param.tree_param.max_depth,
            'num_trees': hetero_secure_boost_guest.model_param.num_trees,
            'bin_nums': hetero_secure_boost_guest.model_param.bin_num,
        },
        "grad_hess": hetero_secure_boost_guest.get_tree_grad_hess(),
        "label": label_vec,
        "predict_nid": hetero_secure_boost_guest.get_tree_predict_result(),
        "predict_label": tmp_predict_result,
    }
    logging_time = time.strftime('%Y-%m-%d-%H_%M_%S')  
    with open(f"./new_log/guest_{logging_time}.json",'a') as wf:
        json.dump(tmp_dict, wf, indent=2)



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

from xgboost.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest
from i_o.utils import read_from_csv_with_lable, read_from_csv_with_no_lable
from computing.d_table import DTable
from ml.feature.instance import Instance
import random
import numpy as np
import functools



def test():

    hetero_secure_boost_guest = HeteroSecureBoostingTreeGuest()
    hetero_secure_boost_guest.model_param.subsample_feature_rate = 1
    hetero_secure_boost_guest._init_model(hetero_secure_boost_guest.model_param)

    

    header, ids, features, lables = read_from_csv_with_lable('data/vehicle_scale_hetero/vehicle_scale_hetero_guest.csv')
    header2, ids2, features2 = read_from_csv_with_no_lable('data/vehicle_scale_hetero/vehicle_scale_hetero_host.csv')

    # header, ids, features, lables = read_from_csv_with_lable('data/vehicle_scale_hetero/vehicle_scale_hetero_guest.csv')
    # header2, ids2, features2 = read_from_csv_with_no_lable('data/vehicle_scale_hetero/vehicle_scale_hetero_host.csv')
    header.extend(header2)
    


    instances = []
    for i, feature in enumerate(features):
        inst = Instance(inst_id=ids[i], features=np.hstack((feature, features2[i])), label=lables[i])
        # inst = Instance(inst_id=ids[i], features=feature, label=lables[i])
        instances.append(inst)
    
    train_instances, test_instances = data_split(instances, 0.8, True, 2)

    # ids = [a.inst_id for a in train_instances]

    # print(sorted(ids))

    # return
    

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
    # print('num is ', predict_result.count())

    # print(predict_result)

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

if __name__ == '__main__':
    test()
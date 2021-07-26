import xgboost as xgb
from xgboost import plot_importance
import pandas as pd
from sklearn.metrics import confusion_matrix
import random
import numpy as np

def xgboost_origin():
    # 算法参数
    params = {
        'booster':'gbtree',         # 确认完成
        'objective':'binary:logistic',  # 确认完成  更换数据需要更改
        # 'num_class':4,              # 确认完成  更换数据需要更改
        'gamma':1e-3,               # 确认完成
        'max_depth':5,              # 确认完成，单棵树最大深度5
        'lambda':0.1,               # 确认完成
        'subsample':1.0,            # 确认完成
        'colsample_bytree':0.8,     # 确认完成
        'min_child_weight':0,       # 确认完成
        # 'slient':1,                 # 确认完成
        'eta':0.3,                  # 确认完成，学习率0.3
        'seed':random.randint(1,1000),                # 确认完成
        'nthread':4,                # 确认完成
        'tree_method':'approx',     # 确认完成，近似策略
        'sketch_eps':1/32           # 确认完成，分32个箱子
    }

    df_train = pd.read_csv('data/lr/lr_train.csv')
    df_test = pd.read_csv('data/lr/lr_test.csv')

    df_train_x = df_train.iloc[:, 2:]
    df_train_y = df_train.iloc[:, 1:2]
    df_test_x = df_test.iloc[:, 2:]
    df_test_y = df_test.iloc[:, 1:2]

    d_train = xgb.DMatrix(df_train_x, label=df_train_y)
    d_test = xgb.DMatrix(df_test_x)

    model = xgb.train(params, d_train, 5)
    y_predict = model.predict(d_test)

    for i, v in enumerate(y_predict):
        # y_predict[i] = np.argmax(v, axis=1)
        if v > 1 - v:
            y_predict[i] = 1
        else:
            y_predict[i] = 0

    # y_predict = np.argmax(y_predict, axis=1)

    # print(y_predict)

    # from sklearn.metrics import accuracy_score   # 准确率
    # accuracy = accuracy_score(df_test_y,y_predict)
    # print("accuarcy: %.2f%%" % (accuracy*100.0))

    # print('accuracy is {}'.format(confusion_matrix(list(df_test_y), list(y_predict))))

    # print(df_train_x)
    # print(df_train_y)

    confusion = confusion_matrix(df_test_y,y_predict)
    print(confusion)



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



if __name__ == '__main__':

    xgboost_origin()
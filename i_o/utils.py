import numpy as np

def read_from_csv_with_lable(csv_address: str):
    """
    从带标签文件读取数据

    Parameters
    ----------
    csv_address: str
        文件路径
    
    Returns
    -------
    header:
        list,特征对应的表头
    ids:
        list，id的列表
    features:
        list，特征的列表，其中特征为numpy.ndarray类型
    lables:
        list,标签的列表

    Notes
    -----
    csv的第一列会被识别为id列，第二列会被识别为标签列
    """

    rows = []

    with open(csv_address) as file:
        for line in file:
            rows.append(line.rstrip("\n"))
    
    header = rows[0].split(',')

    # lable_pos = header.index('y')
    lable_pos = 1
    header.pop(lable_pos)

    # id_pos = header.index('id')
    id_pos = 0
    header.pop(id_pos)
    features = []
    lables = []
    ids = []

    for i in range(1,len(rows)):

        vals = rows[i].split(',')

        lables.append(vals.pop(lable_pos))
        ids.append(vals.pop(id_pos))
        vals = [float(val) for val in vals]
        features.append(np.array(vals))

    return header, ids, features, lables

def read_from_csv_with_no_lable(csv_address: str):
    """
    从不带标签文件读取数据

    Parameters
    ----------
    csv_address: str
        文件路径
    
    Returns
    -------
    header:
        list,特征对应的表头
    ids:
        list，id的列表
    features:
        list，特征的列表，其中特征为numpy.ndarray类型

    Notes
    -----
    csv的第一列会被识别为id列
    """

    rows = []

    with open(csv_address) as file:
        for line in file:
            rows.append(line.rstrip("\n"))
    
    header = rows[0].split(',')

    # id_pos = header.index('id')
    id_pos = 0
    header.pop(id_pos)

    features = []
    ids = []

    for i in range(1,len(rows)):

        vals = rows[i].split(',')

        ids.append(vals.pop(id_pos))
        vals = [float(val) for val in vals]
        features.append(np.array(vals))

    return header, ids, features

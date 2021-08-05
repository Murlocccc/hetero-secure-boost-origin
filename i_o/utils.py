import numpy as np

def read_from_csv_with_lable(csv_address: str):
    """
    read data from the csv file with lable
    Parameters
    ----------
    csv_address: str
        the address of the csv file
    
    Returns
    -------
    header:
        list of featrue_name
    ids:
        list of id
    features:
        list of features
    lables:
        list of lable
    Notes
    -----
    the first column of the csv file will be recognized as the id_col
    the second column of the csv file will be recognized as the lable_col
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

        # print(i)

        vals = rows[i].split(',')
        if '' in vals:
            vals.remove('')

        lables.append(float(vals.pop(lable_pos)))
        ids.append(int(vals.pop(id_pos)))
        vals = [float(val) for val in vals]
        features.append(np.array(vals))

    return header, ids, features, lables

def read_from_csv_with_no_lable(csv_address: str):
    """
    read data from the csv file with no lable
    Parameters
    ----------
    csv_address: str
        the address of the csv file
    
    Returns
    -------
    header:
        list of featrue_name
    ids:
        list of id
    features:
        list of features
    Notes
    -----
    the first column of the csv file will be recognized as the id_col
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

        ids.append(int(vals.pop(id_pos)))
        vals = [float(val) for val in vals]
        features.append(np.array(vals))

    return header, ids, features
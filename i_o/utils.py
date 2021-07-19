import numpy as np

def read_from_csv(csv_address:str):

    rows = []

    with open(csv_address) as file:
        for line in file:
            rows.append(line.rstrip("\n"))
    
    header = rows[0].split(',')

    lable_pos = header.index('y')
    header.pop(lable_pos)

    id_pos = header.index('id')
    header.pop(id_pos)

    features = []
    lables = []
    ids = []

    for i in range(1,len(rows)):

        vals = rows[i].split(',')

        lables.append(vals.pop(lable_pos))
        id = vals.pop(id_pos)
        ids.append(id)
        vals = [float(val) for val in vals]
        features.append(np.array(vals))

    return header, ids, features, lables

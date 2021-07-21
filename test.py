from computing.d_table import DTable
# from ml.tree.hetero_secureboosting_tree_guest import HeteroSecureBoostingTreeGuest
from i_o.utils import read_from_csv_with_lable

# 通过
def test_subtractByKey():
    t1 = DTable(False, [5,6,7])
    t2 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

    t3 = t1.subtractByKey(t2)

    print('t1 is ', t1)
    print('t2 is ', t2)
    print('t3 is ', t3)

# 通过
def test_union():
    t1 = DTable(False, [5,6,7])
    t2 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

    def func(a, b):
        return str(a) + str(b)
    
    t3 = t1.union(t2)
    t4 = t1.union(t2, func=func)

    print('t1 is ', t1)
    print('t2 is ', t2)
    print('t3 is ', t3)
    print('t4 is ', t4)

# 通过
def test_join():
    t1 = DTable(False, [5,6,7])
    t2 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

    def func(a, b):
        return str(a) + str(b)
    
    t3 = t1.join(t2)
    t4 = t1.join(t2, func=func)

    print('t1 is ', t1)
    print('t2 is ', t2)
    print('t3 is ', t3)
    print('t4 is ', t4)

# 通过
def test_filter():
    t1 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

    def key_is_int(k, v):
        return isinstance(k, int)
    
    def value_is_str(k, v):
        return isinstance(v, str)

    t2 = t1.filter(key_is_int)
    t3 = t1.filter(value_is_str)

    print('t1 is ', t1)
    print('t2 is ', t2)
    print('t3 is ', t3)

# 通过
def test_reduce():
    t1 = DTable(False, list(range(100)))

    reduce_ans = t1.reduce(lambda v1, v2: v1 + v2)

    print('reduce_ans is ', reduce_ans)

# 通过
def test_count():
    t1 = DTable(False, list(range(100)))

    count_ans = t1.count()

    print('count_ans is ', count_ans)

# 通过
def test_first():
    t1 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

    first_ans = t1.first()

    print('first is ', first_ans)

# 通过
def test_take():
    t1 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

    take_ans_1 = t1.take(0)
    take_ans_2 = t1.take(2)
    take_ans_3 = t1.take(8)

    print('t1 is ', t1)

    print('take_ans_1 is ', take_ans_1)
    print('take_ans_2 is ', take_ans_2)
    print('take_ans_3 is ', take_ans_3)

# 通过
def test_map():
    t1 = DTable(False, [5,6,7])

    def func(k, v):
        return k + v ** 2
    
    t2 = t1.map(func)

    print('t1 is ', t1)
    print('t2 is ', t2)

# 通过
def test_mapValues():
    t1 = DTable(False, [5,6,7])

    def func(v):
        return v ** 2
    
    t2 = t1.mapValues(func)

    print('t1 is ', t1)
    print('t2 is ', t2)

# 通过
def test_mapReducePartitions():
    t1 = DTable(True, [(1, 2), (2, 3), (3, 4), (4, 5)])

    def mapper(L: list):
        new_table = []
        for k, v in L:
            if k <= 2:
                k = 2
            else:
                k = 3
            new_table.append((k, v))
        return new_table
    
    def reducer(a, b):
        return a + b
    
    t2 = t1.mapReducePartitions(mapper, reducer)

    print('t1 is ', t1)
    print('t2 is ', t2)

# 通过
def test_read_from_csv():
    # data = read_from_csv('data/breast_hetero_mini/breast_hetero_mini_guest.csv')
    header, ids, features, lables = read_from_csv_with_lable('breast_hetero_mini_guest.csv')
    print(header)
    print()
    print(ids)
    print()
    print(features)
    print()
    print(lables)

# 
# def testHeteroSecureBoostTree():
#     heteroSecureBoostingTreeGuest = HeteroSecureBoostingTreeGuest()
#     data_insts = DTbale()


if __name__ == '__main__':
    # test_read_from_csv()
    l1 = [1,2,3]
    l2 = ['a','b','c']
    l3 = zip(l1,l2)
    l4 = list(l3)
    print('l3 is ', l3)
    print('l4 is ', l4)
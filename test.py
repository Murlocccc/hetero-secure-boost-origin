from computing.d_table import DTbale

# 通过
def test_subtractByKey():
    t1 = DTbale(False, [5,6,7])
    t2 = DTbale(True, [(0,3),(2,'asd'),('tt','pp')])

    t3 = t1.subtractByKey(t2)

    print('t1 is ', t1)
    print('t2 is ', t2)
    print('t3 is ', t3)

# 通过
def test_union():
    t1 = DTbale(False, [5,6,7])
    t2 = DTbale(True, [(0,3),(2,'asd'),('tt','pp')])

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
    t1 = DTbale(False, [5,6,7])
    t2 = DTbale(True, [(0,3),(2,'asd'),('tt','pp')])

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
    t1 = DTbale(True, [(0,3),(2,'asd'),('tt','pp')])

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
    t1 = DTbale(False, list(range(100)))

    reduce_ans = t1.reduce(lambda v1, v2: v1 + v2)

    print('reduce_ans is ', reduce_ans)

# 通过
def test_count():
    t1 = DTbale(False, list(range(100)))

    count_ans = t1.count()

    print('count_ans is ', count_ans)

# 通过
def test_first():
    t1 = DTbale(True, [(0,3),(2,'asd'),('tt','pp')])

    first_ans = t1.first()

    print('first is ', first_ans)

# 通过
def test_take():
    t1 = DTbale(True, [(0,3),(2,'asd'),('tt','pp')])

    take_ans_1 = t1.take(0)
    take_ans_2 = t1.take(2)
    take_ans_3 = t1.take(8)

    print('t1 is ', t1)

    print('take_ans_1 is ', take_ans_1)
    print('take_ans_2 is ', take_ans_2)
    print('take_ans_3 is ', take_ans_3)

# 通过
def test_map():
    t1 = DTbale(False, [5,6,7])

    def func(k, v):
        return k + v ** 2
    
    t2 = t1.map(func)

    print('t1 is ', t1)
    print('t2 is ', t2)

# 通过
def test_mapValues():
    t1 = DTbale(False, [5,6,7])

    def func(v):
        return v ** 2
    
    t2 = t1.mapValues(func)

    print('t1 is ', t1)
    print('t2 is ', t2)

if __name__ == '__main__':
    test_mapValues()
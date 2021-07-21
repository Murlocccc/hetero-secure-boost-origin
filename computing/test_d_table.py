from computing.d_table import DTable
import unittest


class TestDTable(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def tearDown(self) -> None:
        return super().tearDown()

    def test_subtractByKey(self):
        t1 = DTable(False, [5,6,7])
        t2 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

        t3 = t1.subtractByKey(t2)

        self.assertEqual(dict(t3.collect()), {1:6})

    def test_union(self):
        t1 = DTable(False, [5,6,7])
        t2 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

        def func(a, b):
            return str(a) + str(b)
    
        t3 = t1.union(t2)
        t4 = t1.union(t2, func=func)

        self.assertEqual(dict(t3.collect()), {0:5, 1:6, 2:7, 'tt':'pp'})
        self.assertEqual(dict(t4.collect()), {0:'53', 1:6, 2:'7asd', 'tt':'pp'})

    def test_join(self):
        t1 = DTable(False, [5,6,7])
        t2 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

        def func(a, b):
            return str(a) + str(b)

        t3 = t1.join(t2)
        t4 = t1.join(t2, func=func)

        self.assertEqual(dict(t3.collect()), {0:5, 2:7})
        self.assertEqual(dict(t4.collect()), {0:'53', 2:'7asd'})

    def test_filter(self):
        t1 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

        def key_is_int(k, v):
            return isinstance(k, int)
        
        def value_is_str(k, v):
            return isinstance(v, str)

        t2 = t1.filter(key_is_int)
        t3 = t1.filter(value_is_str)

        self.assertEqual(dict(t2.collect()), {0:3, 2:'asd'})
        self.assertEqual(dict(t3.collect()), {2:'asd', 'tt':'pp'})

    def test_reduce(self):
        t1 = DTable(False, list(range(100)))

        reduce_ans = t1.reduce(lambda v1, v2: v1 + v2)

        self.assertEqual(reduce_ans, 4950)

    def test_count(self):
        t1 = DTable(False, list(range(100)))

        count_ans = t1.count()

        self.assertEqual(count_ans, 100)

    def test_first(self):
        t1 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

        first_ans = t1.first()

        self.assertIn(first_ans, t1.get_values())

    def test_take(self):
        t1 = DTable(True, [(0,3),(2,'asd'),('tt','pp')])

        take_ans_1 = t1.take(0)
        take_ans_2 = t1.take(2)
        take_ans_3 = t1.take(8)

        self.assertEqual(len(take_ans_1), 0)
        self.assertEqual(len(take_ans_2), 2)
        self.assertEqual(len(take_ans_3), 3)
        self.assertTrue(set(take_ans_1).issubset(set(t1.get_kvs())))
        self.assertTrue(set(take_ans_2).issubset(set(t1.get_kvs())))
        self.assertTrue(set(take_ans_3).issubset(set(t1.get_kvs())))

    def test_map(self):
        t1 = DTable(False, [5,6,7])

        def func(k, v):
            return k + v ** 2
        
        t2 = t1.map(func)

        self.assertEqual(dict(t2.collect()), {0:25, 1:37, 2:51})

    def test_mapValues(self):
        t1 = DTable(False, [5,6,7])

        def func(v):
            return v ** 2
        
        t2 = t1.mapValues(func)

        self.assertEqual(dict(t2.collect()), {0:25, 1:36, 2:49})

    def test_mapPartitions(self):
        t1 = DTable(False, [1,2,3,4,5])

        def mapper(kviterator):
            return [sum(1 for _ in kviterator)]
        
        t2 = t1.mapPartitions(mapper)

        self.assertEqual(dict(t2.collect()), {0:5})


    def test_mapReducePartitions(self):
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

        self.assertEqual(dict(t2.collect()), {2:5, 3:9})


if __name__ == '__main__':
    unittest.main()

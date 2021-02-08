#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/6/14 23:31                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import math, sys, time
import pp

def example1():
    def IsPrime(n):
        """返回n是否是素数"""
        if not isinstance(n, int):
            raise TypeError("argument passed to is_prime is not of 'int' type")
        if n < 2:
            return False
        if n == 2:
            return True
        max = int(math.ceil(math.sqrt(n)))
        i = 2
        while i <= max:
            if n % i == 0:
                return False
            i += 1
        return True

    def SumPrimes(n):
        for i in range(15):
            sum([x for x in range(2, n) if IsPrime(x)])
        """计算从2-n之间的所有素数之和"""
        return sum([x for x in range(2, n) if IsPrime(x)])

    inputs = (100000, 100100, 100200, 100300, 100400, 100500, 100600, 100700)
    '''
    start_time = time.time()
    for input in inputs:
        print ( SumPrimes(input))
    print ('单线程执行，总耗时', time.time() - start_time, 's')
    '''
    # tuple of all parallel python servers to connect with
    ppservers = ()
    # ppservers = ("10.0.0.1",)
    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
        # Creates jobserver with ncpus workers
        job_server = pp.Server(ncpus, ppservers=ppservers)
    else:
        # Creates jobserver with automatically detected number of workers
        job_server = pp.Server(ppservers=ppservers)
    print("pp 可以用的工作核心线程数", job_server.get_ncpus(), "workers")
    start_time = time.time()
    jobs = [(input, job_server.submit(SumPrimes, (input,), (IsPrime,), ("math",))) for input in inputs]

    for input, job in jobs:
        (input, job())
        print("Sum of primes below", input, "is", job())
    print("多线程下执行耗时: ", time.time() - start_time, "s")

    print('================')
    job_server.print_stats()

def test_sklearn_parallel():
    # from sklearn.externals.joblib import Parallel, delayed
    # trees_pp = []
    # start_time = time.time()
    # for i in range(n_more_estimator):
    #     tree = MY_TreeClassifier(
    #         criterion=self.criterion,
    #         max_depth=self.max_depth,
    #         min_leaf_split=self.min_leaf_split,
    #         max_feature=self.max_feature,
    #         bootstrap=self.bootstrap,
    #         seed=self.seed,
    #         n_jobs=self.n_jobs
    #     )
    #     trees_pp.append(tree)
    # trees_pp = Parallel(n_jobs=16)(  # do not use backend="threading"
    #     delayed(_parallel_build_trees)(tree, X, y, i, len(trees_pp))
    #     for i, tree in enumerate(trees_pp))
    # pallal_time_1 = time.time() - start_time
    pass
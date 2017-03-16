# ==========================================================================
#   performance_comparison.py
#       Here we compare pulp and gurobi api performance.
# ==========================================================================
from pulp import *
from gurobipy import *
import numpy as np
import time

SIZE = 1000


def pulp():
    x = [LpVariable('x_%s' % i, 0, 1000) for i in range(SIZE)]
    ones = np.ones(SIZE)

    timer_start = time.time()
    sum = lpSum([x[i]*ones[i] for i in range(SIZE)])
    duration = time.time() - timer_start 

    print('PuLP Duration: ', duration)


def gurobipy():

    m = Model('Performance_Test')

    x = [m.addVar(vtype=GRB.CONTINUOUS, name='x_%s' % i) for i in range(SIZE)]
    ones = np.ones(SIZE)

    timer_start = time.time()
    sum = quicksum([x[i]*ones[i] for i in range(SIZE)])
    duration = time.time() - timer_start
    
    print('gurobipy Duration: ', duration)


if __name__ == '__main__':
    pulp()
    gurobipy()


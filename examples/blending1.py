import numpy as np
from gurobipy import Model, GRB

from grbtensor import *


#   This example is taken from the PuLP documentation
#   and is called "A Blending Problem"


def main():
    m = Model('blending1')

    # this is the only variable in the problem
    x = VarTensor([2], GRB.CONTINUOUS, m, name='x')

    # this vector is for the objective function
    A = Tensor([0.013, 0.008])

    # this vector is for the first equality constraint
    C1 = Tensor([1., 1.])
    # and the other side of the equation
    b1 = 100.

    # this matrix is for the first inequality constraint
    C2 = Tensor([[0.10, 0.20],
                 [0.08, 0.10]])
    # and the other side of that inequality
    b2 = Tensor([8., 6.])

    # this matrix is for the final inequality constraint
    C3 = Tensor([[0.001, 0.005],
                 [0.002, 0.005]])
    # and the other side of that inequality
    b3 = Tensor([2., 0.4])
    constr0 = zero.odot(x) == zero
    m.addConstr(

    # set objective
    obj = A['i']*x['i']
    m.setObjective(obj, GRB.MINIMIZE)

    # first constraint
    constraint1 = C1['i']*x['i'] == b1
    # if a constraint is a scalar, it must be added via the model by converting
    m.addConstr(constraint1)
    
    # the 'i' on the right hand side of the equation is just for clarity
    constraint2 = C2['ij']*x['j'] >= b2['i']
    constraint2.addConstr(m)

    # final constraint
    constraint3 = C3['ij']*x['j'] <= b3['i']
    constraint3.addConstr(m)
    
    # use primal simplex
    m.params.method = 0

    # optimize
    m.optimize()

    print(x.eval())


if __name__ == '__main__':
    main()


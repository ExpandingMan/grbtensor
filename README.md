***NOTE:*** A really good framework for doing this sort of stuff already exists in Julia and is called [JuMP.jl](https://github.com/JuliaOpt/JuMP.jl).  This has
been abandoned in favor of using that (which has no significance performance issues to speak of).


## grbtensor

This package is a wrapper for numpy arrays which facilitates a simple interface with
the gurobipy API for linear and quadratic (linearly constrained, or quadratically constrained)
optimization.  It also accomodates algebraic-style coding of tensor expressions.

In the following example, a constraint is added to a gurobi model: 

```python

m = gurobipy.Model()
x = grbtensor.VarTensor([2, 2], GRB.CONTINUOUS, m, name='x') 
a = grbtensor.Tensor([1, 0])
b = grbtensor.Tensor([0, 1])
constr = a['i']*x['ij']*b['j'] >= 5
m.addConstr(constr)

```

This constrains the gurobi variable $x_{01}$ to be greater than or equal to 5.
Einstein notation is used to clearly and easily specify tensor operations.

## gurobipy has no linear algebra functionality, and doesn't cooperate with anything that does

This is a huge problem, but right now there are no alternatives.  All types of inner products
had to be implemented with Python loops because they have to use the gurobipy "quicksum" function,
because the summation that is called from numpy runs in (at best) quadratic time and is
prohibitively slow.  "quicksum", meanwhile, is much much slower than Python or numpy sums for
anything other than gurobipy variables.  This means that one should avoid doing any operation
which results in a matrix.  

```python
# this is very slow for high dimensions because the result of the first
# operation is a matrix, and ultimately this involves matrix multiplication
expr = a['i']*b['j']*x['ij']
# this only involves operations which result in (at most) vectors and runs fast
expr = a['i']*x['ij']*b['j']
```

__TODO__:   Ultimately we want to set up to detect whether tensors contain numerical values or
gurobipy variables and use the appropriate computation.  Achieving this is extremely labor intensive,
so for now it isn't a priority.

## Design Principles

grbtensor adheres to the following design principles:

* __Performance__
... (We have had to make HUGE sacrifices here because of the slowness of 
... gurobipy.  Ultimately it would be nice to do some C++ coding, but there
... may never be time.)
... Creation and manipulation of tensors should be computationally efficient.
... All allocation should be done in bulk, grbtensor operations are all essentially
... wrappers for numpy operations.  Creating objectives and constraints for tensors
... with millions of elements should be fast, with all the heavy lifing happening
... under the hood in C or Fortran.  If complicated loops are ever needed, they should
... be implemented in Cython.

* __Pseudo-Algebraic Coding__
... Code should be readable and easy to formulate.  Inequalities and expressions
... should translate trivially from mathematical notation.
... grbtensor is designed to be used in an environment where one needs to formulate
... and implement hundreds of (possibly tensor-valued) constraints quickly.

* __Compliance with numpy (first) and pandas (second)__
... (Also made huge sacrifices here because of gurobipy limitations.)
... Tensor objects are just simple wrappers of numpy arrays.  Numpy arrays are
... performant and there is a huge number of functions available for working
... with them.  Since usually data is terribly fomratted, pandas is needed to
... get it into a useful form.  grbtensor should accomodate use of pandas where
... possible, but since numpy is far more appropriate mathematically and computationally,
... this usually involves taking slices of pandas dataframes as numpy arrays.


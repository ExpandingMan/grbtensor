import numpy as np
import numbers
import copy
import time
from gurobipy import Model, GRB, LinExpr
from gurobipy import Constr, TempConstr
from gurobipy import quicksum

# TODO:
#   > Do some basic sparse matrix handling.   
#   > Create more efficient shortcuts for purely numeric linear algebra using numpy.
#   > Currently most operations on VarTensors return Tensor.  What's best way to handle this?
#   - (doesn't currently cause any problems)
#   > Consider overloading assignment for tensor and using this in operations so that, for
#   - instance, indices are preserved
#   > Complete support for traces.

# ==========================================================================================
#   Tensor Products:
# ==========================================================================================
def tensorprod(t1, t2, axes, tensor_out=False):
    """
    Compute an arbitrary tensor product involving at most 2 contracted indices.
    Currently does not support traces.
    """
    assert len(axes) == 2, 'Axes must be specified for 2 tensors.'
    assert len(axes[0]) == len(axes[1]), 'Index mis-match.'
    # ensure that elements of axes are lists
    axes = [[ax] if not isinstance(ax, (list, tuple)) else ax for ax in axes]
    # convert to arry if Tensor
    if isinstance(t1, Tensor):
        t1 = t1.array
    if isinstance(t2, Tensor):
        t2 = t2.array
    # check if need to do outer instead
    if (len(axes[0]) == 0) & (len(axes[1]) == 0):
        return _outer(t1, t2)
    # make axes into a list of pairs to perform inner products on
    axes = np.swapaxes(np.array(axes), 0, 1).tolist()
    # perform the first inner product
    o = _tensorprod_on2(t1, t2, axes[0])
    # fix the shifted axes positions
    axes = _fix_axes_pos(axes)
    for ax in axes[1:]:
        # do traces on appropriate axes
        o = _trace(o, ax)
    if tensor_out:
        o = Tensor(o)
    return o


def _fix_axes_pos(axes):
    """
    Computes the new locations of indices after an inner product.
    :param axes: this should be passed after the numpy transpose is done
    """
    removed = axes[0]
    to_shift = axes[1:]
    for ax in to_shift:
        if ax[0] > removed[0]:
            ax[0] -= 1
        if ax[1] > removed[1]:
            ax[1] -= 1
    return axes


def _tensorprod_on2(t1, t2, axes):
    """
    Compute a tensor product involving at most two indices.
    Returns a numpy array.  Not to be called by user.
    :param axes: argument should be a list of length 2
    """
    # make the summation axes the first axes of each
    t1 = np.rollaxis(t1, axes[0], 0)
    t2 = np.rollaxis(t2, axes[1], 0)
    # create output tensor
    shape = list(t1.shape[1:]) + list(t2.shape[1:])
    o = np.empty(shape, dtype='O')
    # these slices are needed in the loop
    slice1 = [slice(t1.shape[0])]
    slice2 = [slice(t2.shape[0])]
    # create iterator object and enter loop
    it = np.nditer(o, op_flags=['readwrite'], flags=['multi_index', 'refs_ok'])
    while not it.finished:
        # it seems unavoidable to do this manipulation in the loop
        indices1 = slice1 + list(it.multi_index[:(len(t1.shape)-1)])
        indices2 = slice2 + list(it.multi_index[(len(t1.shape)-1):])
        it[0] = quicksum(t1[tuple(indices1)]*t2[tuple(indices2)])
        it.iternext()
    return o


# TODO: shouldn't create a new blank array
def _trace(t, axes):
    """
    Perform a trace across two axes.
    """
    # we only want the appropriate diagonal, numpy always makes the diagonal the LAST axis
    t = np.diagonal(t, axis1=axes[0], axis2=axes[1])
    # this is the output array
    o = np.empty(t.shape[:-1], dtype='O')
    # this slice is needed in the loop
    slice1 = [slice(t.shape[-1])]
    it = np.nditer(o, op_flags=['readwrite'], flags=['multi_index', 'refs_ok'])
    while not it.finished:
        indices = list(it.multi_index) + slice1
        it[0] = quicksum(t[tuple(indices)])
        it.iternext()
    return o 


# for some stupid reason this still runs unbelievably slow if using np.tensordot,
# even though there are no sums involved
def _outer(t1, t2):
    """
    Computes an outer product betwee numpy arrays.  The user should never call this,
    as "tensorprod" makes the call when appropriate.
    This is only subtly different than tensorprod, but I believe it's needed.
    """
    o = np.empty(list(t1.shape)+list(t2.shape), dtype='O')
    it = np.nditer(o, op_flags=['readwrite'], flags=['multi_index', 'refs_ok'])
    while not it.finished:
        indices1 = it.multi_index[:len(t1.shape)]
        indices2 = it.multi_index[len(t1.shape):]
        it[0] = t1[indices1]*t2[indices2]
        it.iternext()
    return o
    
# ===========================================================================================
 

# ============================================================================================
#   Tensor Class
# ============================================================================================
class Tensor:
    """
    Base class for tensor objects.
    """

    t_ = np.empty(())
    indices_ = None

    def __init__(self, array):
        """
        Initializer takes a numpy array to wrap.  If passed a list or tuple,
        this will be converted (trivially) to a numpy array.
        """
        if isinstance(array, (list, tuple)):
            array = np.array(array)
        self.t_ = array

    def __repr__(self):
        """
        Tensor objects print the same way as their underlying numpy arrays.
        """
        return repr(self.t_)

    # NOTE: in this case need to save indices
    #   don't think that's the case for any other operations
    def __neg__(self):
        """
        Negates the tensor (i.e. x -> -x for x in tensor).
        Conserves indices.
        """
        t = Tensor(-self.t_)
        t.indices_ = self.indices_
        return t

    def __add__(self, other):
        """
        Adds a numpy array or other Tensor.  Currently does not support broadcasting
        of scalars to maintain notational clarity.  This does not conserve indices,
        since we don't enforce index rules except when necessary.
        """
        if isinstance(other, Tensor):
            return Tensor(self.t_ + other.t_)
        elif isinstance(other, np.ndarray):
            return Tensor(self.t_ + other)

    def __sub__(self, other):
        """
        Subtracts a numpy array or other Tensor.  See __add__ for details.
        """
        if isinstance(other, Tensor):
            return Tensor(self.t_ - other.t_)
        elif isinstance(other, np.ndarray):
            return Tensor(self.t_ - other)

    def __mul__(self, other):
        """
        This supports all types of tensor products through indexing (also multiplication
        by scalars).  For example, an inner product between two vectors is x['i']*y['i'].
        CURRENTLY DOES NOT SUPPORT TRACES within a single tensor, i.e. you can't do a['ii'].
        If the result of a tensor product is a scalar, the object returned will be of the 
        same type as the elements of the underlying numpy array (i.e. the appropriate 
        scalar type).
        """
        if isinstance(other, numbers.Number):
            return Tensor(self.t_*other)
        elif isinstance(other, Tensor):
            if (self.indices_ is not None) & (other.indices_ is not None):
                return self._einstein(other)

    def __rmul__(self, other):
        """
        Multiplies with a scalar.  Tensor products are handled by the __mul__ method so
        that the rightmost Tensor object in a multiplication returns None.
        """
        if isinstance(other, numbers.Number):
            return Tensor(other*self.t_)
        elif isinstance(other, Tensor):
            return None

    # this supports slicing as part of the language
    def __getitem__(self, idx):
        """
        This works the same way as the numpy array __getitem__ method.
        If passed a string, this sets indices which specify how tensor
        products should be done.
        """
        if isinstance(idx, str):
            assert len(idx) == len(self.t_.shape), \
                'Invalid indices for tensor.'
            # self.indices_ = idx
            # NOTE: for now we return a shallow copy
            o = copy.copy(self)
            o.indices_ = idx
            return o
        else:
            return self.t_[idx]

    # ----overloading-relational-operators-----------
    def __le__(self, other):
        """
        A broadcast of the <= operation.  In the case of constants, it will return a tensor
        of booleans, in the case of gurobipy variables, it will return a tensor of gurobipy
        constraint objects.
        """
        assert self.shape == other.shape
        constr = np.empty(self.shape, dtype='O')
        it = np.nditer(self.t_, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            constr[it.multi_index] = self.t_[it.multi_index] <= other.t_[it.multi_index]
            it.iternext()
        return Tensor(constr)

    def __ge__(self, other):
        """
        A broadcast of the >= operation.  In the case of constants, it will return a tensor
        of booleans, in the case of gurobipy variables, it will return a tensor of gurobipy
        constraint objects.
        """
        assert self.shape == other.shape
        constr = np.empty(self.shape, dtype='O')
        it = np.nditer(self.t_, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            constr[it.multi_index] = self.t_[it.multi_index] >= other.t_[it.multi_index]
            it.iternext()
        return Tensor(constr)

    def __eq__(self, other):
        """
        A broadcast of the == operation.  In the case of constants, it will return a tensor
        of booleans, in the case of gurobipy variables, it will return a tensor of gurobipy
        constraint objects.
        """
        assert self.shape == other.shape
        constr = np.empty(self.shape, dtype='O')
        it = np.nditer(self.t_, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            constr[it.multi_index] = self.t_[it.multi_index] == other.t_[it.multi_index]
            it.iternext()
        return Tensor(constr)

    # TODO: doesn't work for traces
    def _einstein(self, other):
        """
        A private method which is called by __mul__ and performs the appropriate
        tensor product as specified by the indices of each tensor.
        """
        sum_indices = set(self.indices_).intersection(other.indices_)
        indices = list(self.indices_ + other.indices_)
        new_indices = [i for i in indices if i not in sum_indices]
        self_axes = []
        other_axes = []
        for i in sum_indices:
            indices.remove(i)
            indices.remove(i)
            self_axes.append(self.indices_.index(i))
            other_axes.append(other.indices_.index(i))
        # can't use numpy function because it's too slow for gurobipy objects
        # o = np.tensordot(self.t_, other.t_, axes=[self_axes, other_axes]) 
        o = tensorprod(self.t_, other.t_, axes=[self_axes, other_axes])
        # clear indices
        self.indices_ = None
        other.indices_ = None
        # if resulting object is rank-0, convert to scalar
        if o.shape == ():
            return np.asscalar(o)
        t = Tensor(o)
        t.indices_ = ''.join(new_indices)
        return t

    def odot(self, other):
        """
        Computes the Hadamard product (element-wise multiplication) with another tensor.
        This necessarily resets the indices.
        """
        return Tensor(self.t_*other.t_)

    def inverse(self):
        """
        Returns a tensor giving the element-wise multiplicative inverse of this one.
        The new tensor will inherit the indices of this one.
        """
        o = Tensor(1./self.t_)
        o.indices_ = self.indices_
        return o

    def broadcast_scalar(self, scalar):
        """
        Returns a Tensor of the same shape as this one, where all elements
        are equal to the passed scalar value.
        """
        return np.empty(self.t_.shape).fill(scalar)

    def _check_scalar(self):
        """
        Check if this tensor is rank-0, if it is, return a scalar.
        Otherwise, return self.
        """
        if self.t_.shape == ():
            return np.asscalar(self.t_)
        else:
            return self

    def tolist(self):
        """
        Calls the numpy array tolist method on the underlying array.
        """
        return self.t_.tolist()

    def get_array(self):
        """
        Gets the underlying numpy array.
        """
        return self.t_

    def get_shape(self):
        """
        Gets the shape of the underlying numpy array.
        """
        return self.t_.shape

    def getConstant(self):
        """
        Calls the LinExpr method 'getConstant' on every element of the tensor.
        This is apporopriate for converting tensors filled with gurobi LinExpr
        to numerical values.
        """
        o = np.empty(self.t_.shape)
        it = np.nditer(o, op_flags=['readwrite'], flags=['multi_index', 'refs_ok'])
        while not it.finished:
            assert isinstance(self.t_[it.multi_index], LinExpr), \
                'Can only call getConstant on LinExpr object.'
            it[0] = self.t_[it.multi_index].getConstant()
            it.iternext()
        o = Tensor(o)
        o.indices_ = self.indices_
        return o

    # TODO: do more sensible checking of whether tensor elements are constraints
    def addConstr(self, model, name=''):
        """
        If the tensor holds constraints, this will add them to the model.
        """
        firstobj = [0]*len(self.shape)
        firstobj = self.t_[firstobj].tolist()[0]
        # for now we are only checking the first object in the array!
        assert isinstance(firstobj, Constr) | isinstance(firstobj, TempConstr)
        it = np.nditer(self.t_, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            itname = name + ''.join(['_%s' % i for i in it.multi_index])
            model.addConstr(self.t_[it.multi_index], name=itname)
            it.iternext()

    # properties
    array = property(fget=get_array)
    shape = property(fget=get_shape)


class VarTensor(Tensor):
    """
    Tensor of Gurobi variables
    """

    model = None
    name = ''

    # bounds
    lb_ = 0.
    ub_ = GRB.INFINITY

    def __init__(self, shape, vtype, model, name, 
                 lb=0., ub=GRB.INFINITY, auto_update=True):
        """
        Creates a tensor of gurobipy variables.
        :param shape: the shape of the tensor
        :param vtype: the gurobi type of the tensor, for example GRB.BINARY
        :param model: the model to add the variables to
        :param name:  the gurobi name of the tensor
        :param lb: lower bound
        :param ub: upper bound
        :param auto_update: if true, update the model
        """
        self.model = model
        self.name = name
        self.lb_ = lb
        self.ub_ = ub
        self.t_ = np.empty(shape, dtype='O')
        it = np.nditer(self.t_, op_flags=['readwrite'], flags=['multi_index', 'refs_ok'])
        while not it.finished:
            idx = ['_%s' % k for k in it.multi_index]
            name_idx = name + ''.join(idx)
            it[0] = model.addVar(vtype=vtype, name=name_idx)
            it.iternext()
        if auto_update:
            model.update()

    def _alternate__init__(self, shape, vtype, model, name, auto_update=True):
        """
        This was a test to see if we could make the initializer faster.
        It seems that the reason nditer is so slow is because it needs to 
        generate the correct index names.
        We could try implementing the index generation in Cython, but
        the slowness of nditer suggests this would not help.
        We lose about a factor of 3 for a (1000, 1000) tensor.
        """
        self.model = model
        self.name = name
        n_elements = np.prod(shape)
        self.t_ = np.empty(n_elements, dtype='O')
        for i in range(n_elements):
            idx = self._generate_idx(i, shape)
            name_idx = name + ''.join(['_%s' % i for i in idx])
            self.t_[i] = model.addVar(vtype=vtype, name=name_idx, ub=self.ub_, lb=self.lb_)
        self.t_.reshape(shape)
        if auto_update:
            model.update()

    def _generate_idx(self, i, shape):
        """
        Generates the correct indices from the 'flat' index i.
        """
        if len(shape) == 2:
            return self._generate_idx_rank2(i, shape)
        idx = np.empty(len(shape), dtype=np.int32)
        idx[0] = i // np.prod(shape[:-1])
        idx[1:] = self._generate_idx(i % np.prod(shape[:-1]), shape=shape[1:])
        return idx

    def _generate_idx_rank2(self, i, shape):
        """
        Generates the correct indices from the 'flat' index i.
        This is called by the recursive algorithm _generate_idx.
        """
        idx = np.empty(len(shape), dtype=np.int32)
        idx[0] = i // shape[1]
        idx[1] = i % shape[0]
        return idx

    def eval(self):
        """
        Returns a new tensor with the values of the gurobi variables.
        This should be called after performing an optimization, but
        currently doesn't check the model status.
        """
        # TODO: this warning seems broken, investigate!
        # if not self.model.status != GRB.Status.OPTIMAL:
        #     print('WARNING: Attempting to evaluate non-optimized variables.')
        tmp = self.t_.ravel().tolist()
        shape = self.t_.shape
        tmp = self.model.getAttr('x', tmp)
        tmp = np.array(tmp).reshape(shape)
        return Tensor(tmp)

# ============================================================================================

# ============================================================================================
#   Testing
# ============================================================================================
def test():
    import time
    from gurobipy import LinExpr, quicksum
    
    size = 1000
    m = Model('TEST')

    # constants
    # A = Tensor(np.arange(size*size).reshape((size, size)))
    a = Tensor(np.arange(size))
    b = Tensor(np.ones((size)))
    # B = Tensor(np.ones((size, size)))

    # x = VarTensor([size, size], GRB.BINARY, m, 'x', auto_update=True)
    # x = VarTensor([size], GRB.BINARY, m, 'x', auto_update=True)

    print('Done declaring.')
        
    # computation
    start_time = time.time()

    # this is fucked
    # z = B['ij']*x['jk']
    z = B['ij']*B['ij']
   
    duration = time.time() - start_time

    print(duration)
    import ipdb; ipdb.set_trace()


def test_idx():
    m = Model()
    vt = VarTensor([2, 2], GRB.BINARY, m, name='vt')
    
    i = 11
    shape = (2, 2, 2, 2)
    idx = vt._generate_idx(i, shape)

    print(np.arange(np.prod(shape)).reshape(shape))
    print(i, idx)



if __name__ == '__main__':
    test()
    # performance_tests()


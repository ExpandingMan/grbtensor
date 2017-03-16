import numpy as np
import pandas as pd
from grbtensor.tensor import Tensor, VarTensor

#
#   This is a simple wrapper for pandas dataframes which makes it
#   easy for the user to return slices of it as Tensor objects.
#
#   It is intended for the user to create classes for cleaning or 
#   otherwise manipulate the dataframe which descend from this class.
#


# ============================================================================
#   TensorFrame
# ============================================================================
class TensorFrame:
    """
    Base class for wrapping pandas dataframes and easily producing Tensor 
    objects with them.  It is intended that the user creates a class which
    inherits from this one for manipulating the dataframe into the appropriate
    form.
    """
    
    df = None
    # this is for backing up full dataframe when setting restrictions
    df_ = None

    def __init__(self, df):
        """
        This initializer usually will not be called as it is expected for the
        user to create a class that inherits from this one.
        """
        self.df = df
        self.reset_backup()

    def __getitem__(self, arg):
        """
        When passing a string or list of strings, returns a Tensor formed
        from the specified columns and all rows.
        """
        # select one column
        if isinstance(arg, str):
            return Tensor(self.df[arg].values)
        # select multiple columns
        if isinstance(arg, list):
            if all(isinstance(a, str) for a in arg):
                return Tensor(self.df[arg].values)
        raise TypeError('Argument must be string or list of strings.')

    def get_shape(self):
        """
        Returns the shape of the underlying dataframe as a tuple.
        """
        return self.df.values.shape

    def get_category_vector(self, column, cat_name):
        """
        A common need is to get a binary vector which is one if a certain categorical
        column is in one category, and zero if it is in any other.
        This returns that result as a rank-1 tensor.
        """
        return Tensor((self.df[column] == cat_name).values.astype(np.int32))

    def reset_backup(self):
        """
        Sets the backup dataframe to be equal to the current dataframe.
        """
        self.df_ = self.df

    def reset_restrictions(self):
        """
        Resets all restrictions on the dataframe.
        """
        self.df = self.df_

    def restrict_to_subset(self, mask, reset=False):
        """
        Restricts the dataframe using a mask.  The unrestricted dataframe will be backed
        up in self.df_ so that restrictions can be lifted.
        :param reset: if true, reset restrictions before applying the new one
        """
        if reset:
            self.reset_restrictions()
        self.df = self.df[mask]


    
    # properties
    shape = property(fget=get_shape)
        
    


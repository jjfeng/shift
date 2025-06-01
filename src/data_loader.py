"""Loads data from a file or dataframe
Note that X denotes the conditional covariates which is named Z in the manuscript
"""

import copy
import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, X: np.ndarray, Y: np.ndarray, vdict: dict[int, int]=None, mdl_X: np.ndarray=None):
        self.X = X
        if mdl_X is None:
            self.mdl_X = X  # features for the ml model required to compute loss function even after removing features from X
        else:
            self.mdl_X = mdl_X
        self.Y = Y
        self.vdict = vdict
        print("DATALOADER NUM_P", list(self.vdict.values()), len(set(list(self.vdict.values()))), self.vdict)
       
    def _get_X(self):
        return self.X
   
    def _get_Y(self):
        return self.Y
    
    def subset_X(self, mask: np.ndarray):
        """Subset features of X
        """
        self.X = self.X[:, mask]
    
    @property
    def shape(self):
        return self.X.shape

    @property
    def num_p(self):
        """Number of groups
        """
        return len(set(list(self.vdict.values())))
    
    @property
    def num_n(self):
        """Number of samples
        """
        return self.X.shape[0]

    @property
    def total_p(self):
        """Number of features
        """
        return self.X.shape[1]
    
    def copy(self):
        return DataLoader(
            X=self.X.copy(),
            Y=self.Y.copy(),
            vdict=copy.deepcopy(self.vdict),
            mdl_X=self.mdl_X.copy()
        )
    

def train_test_loader_split(loader: DataLoader, test_size: float, random_state: int=None):
    """
    Splits the data into training and testing sets.
    """
    train_X, test_X, train_mdl_X, test_mdl_X, train_Y, test_Y = train_test_split(
        loader.X, loader.mdl_X, loader.Y, test_size=test_size, random_state=random_state
    )
    train_loader = DataLoader(
        X=train_X, Y=train_Y, vdict=loader.vdict, mdl_X=train_mdl_X
    )
    test_loader = DataLoader(
        X=test_X, Y=test_Y, vdict=loader.vdict, mdl_X=test_mdl_X
    )
    return train_loader, test_loader
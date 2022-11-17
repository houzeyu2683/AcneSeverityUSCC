
import numpy
from sklearn.model_selection import StratifiedKFold, KFold

class validation:

    def __init__(self, table=None, target=None, size=10, seed=0):

        self.table = table
        self.target = target
        self.size = size
        self.seed = seed 
        return

    def fold(self):

        numpy.random.seed(self.seed)
        if(self.target):

            validator  = StratifiedKFold(n_splits=self.size).split(self.table, self.table[self.target])
            pass

        else:

            validator  = KFold(n_splits=self.size).split(self.table)
            pass
        
        self.group = []
        for index, (a, b) in enumerate(validator):
            
            train = self.table.iloc[a]
            check  = self.table.iloc[b]
            self.group.append([train, check])
            pass

        return
    
    def get(self, index):

        if(index<self.size):

            self.train, self.check = self.group[index]
            return

        else:

            self.train = None
            self.check = None
            return

        pass
        
    pass


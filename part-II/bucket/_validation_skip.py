
import sklearn.model_selection
import pandas, numpy
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

class Validation:

    def __init__(self, data=None, seed=20221111):

        self.data  = data
        self.seed  = seed
        return

    def hold(self, percentage=0.2, stratification=None):
        
        group =  self.data[stratification].copy() if(stratification) else None
        split = sklearn.model_selection.train_test_split
        train, validation = split(
            self.data, 
            test_size=percentage, 
            stratify=group, 
            random_state=self.seed
        )
        pass

        output = (train, validation)
        return(output)

    def fold(self, size=4, stratification=None):

        if(self.style=='table'):

            if(stratification):

                index = {}
                function = sklearn.model_selection.StratifiedKFold
                engine = function(n_splits=size, shuffle=True, random_state=self.seed)
                iteration = engine.split(self.data, self.data[stratification])
                for block, (train, validation) in enumerate(iteration, 1):
                    
                    pair = [self.data.iloc[train,:], self.data.iloc[validation,:]]
                    index[block] = pair
                    continue

                pass
            
            else:

                index = {}
                function = sklearn.model_selection.KFold
                engine = function(n_splits=size, shuffle=True, random_state=self.seed)
                iteration = engine.split(self.data)
                for block, (train, validation) in enumerate(iteration, 1):
                    
                    pair = [self.data.iloc[train,:], self.data.iloc[validation,:]]
                    index[block] = pair
                    continue

                pass

            pass

        return

    pass

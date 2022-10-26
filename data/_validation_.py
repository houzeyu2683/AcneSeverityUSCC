
import sklearn.model_selection
import pandas, numpy
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

class validation:

    def __init__(self, data=None, style='table', seed=1):

        self.data  = data
        self.style = style
        self.seed  = seed
        return

    def hold(self, percentage=0.2, stratification=None):
        
        if(self.style=='table'):

            if(stratification): group = self.data[stratification].copy()
            function = sklearn.model_selection.train_test_split
            train, validation = function(
                self.data, 
                test_size=percentage, 
                stratify=stratification, 
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

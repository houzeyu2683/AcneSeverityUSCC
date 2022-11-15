
import pandas
import sklearn.model_selection
import torch

class table:

    def __init__(self, train=None, validation=None, test=None):
        
        self.train, self.validation, self.test = train, validation, test
        return

    def load(self):

        read = pandas.read_csv
        empty = pandas.DataFrame
        self.train = read(self.train, dtype="str", sep=",") if(self.train) else empty()
        self.validation = read(self.validation, dtype="str", sep=",") if(self.validation) else empty()
        self.test = read(self.test, dtype="str", sep=",") if(self.test) else empty()
        return

    def hold(self, size=0.2, stratification=None):

        
        assert self.validation.empty, 'self.validation already exist'
        if(stratification): stratification = self.train[stratification]
        train, validation = sklearn.model_selection.train_test_split(
            self.train, 
            test_size=size, 
            stratify=stratification, 
            random_state=1
        )
        self.train = train.reset_index(drop=True)
        self.validation = validation.reset_index(drop=True)
        self.method = 'hold'
        return

    def export(self, format='dataset'):

        if(format=='dataset'):

            class dataset: pass
            dataset.train = None if(self.train.empty) else unit(form=self.train)
            dataset.validation = None if(self.validation.empty) else unit(form=self.validation)
            dataset.test = None if(self.test.empty) else unit(form=self.test)
            return(dataset)

        pass

    pass

class unit(torch.utils.data.Dataset):

    def __init__(self, form=None):
        
        self.form = form
        return

    def __getitem__(self, index):

        item = self.form.loc[index]
        return(item)
    
    def __len__(self):

        length = len(self.form)
        return(length)

    pass

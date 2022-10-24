
import pandas
import sklearn.model_selection
import torch

class table:

    def __init__(self, train=None, test=None):
        
        if(train): self.train = pandas.read_csv(train, dtype="str", sep="\s+", header=None)
        if(test): self.test = pandas.read_csv(test, dtype="str", sep="\s+", header=None)
        return

    def hold(self, size=0.2, stratification=None):

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
            if(self.train is not None): dataset.train = collection(form=self.train)
            if(self.validation is not None): dataset.validation = collection(form=self.validation)
            if(self.test is not None): dataset.test = collection(form=self.test)
            return(dataset)
        
        pass

    pass

class collection(torch.utils.data.Dataset):

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

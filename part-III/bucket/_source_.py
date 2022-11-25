
import pandas
import sklearn.model_selection
# import pickle

createClass = lambda name: type(name, (type,), {})
# getPickle = lambda path: with open(path, 'rb') as paper: pickle.load(paper)
# getPickle = lambda path: pickle.load(open(path, 'rb').read())

'''Override for difference case.'''
class Source:

    def __init__(self, train=None, validation=None, test=None):
        
        self.train, self.validation, self.test = train, validation, test
        return

    def loadStructure(self):

        structure = createClass(name='structure')
        train = createClass(name='train')
        validation = createClass(name='validation')
        test = createClass(name='test')
        pass

        ##  load table.
        read = lambda path: pandas.read_csv(path)
        empty = pandas.DataFrame
        train.table = read(self.train) if(self.train) else empty()
        validation.table = read(self.validation) if(self.validation) else empty()
        test.table = read(self.test) if(self.test) else empty()
        pass

        ##  Update data size
        train.length = len(train.table)
        validation.length = len(validation.table)
        test.length = len(test.table)
        pass

        ##  Gather them together in a structure.
        structure.train = train
        structure.validation = validation
        structure.test = test
        self.structure = structure
        return

    # def splitValidation(self, percentage=0.2, stratification=None, seed=0):

    #     ##  Copy the train structure for split validation.
    #     assert not self.structure.train.table.empty, 'The [train.table] is empty.'
    #     assert self.structure.validation.table.empty, 'The [Validation.table] is not empty.'
    #     table = self.structure.train.table.copy()
    #     group = table[stratification].copy()
    #     pass

    #     split = sklearn.model_selection.train_test_split
    #     train, validation = split(
    #         table, 
    #         test_size=percentage, 
    #         stratify=group, 
    #         random_state=seed
    #     )
    #     index = createClass(name="index")
    #     index.train = train.index
    #     index.validation = validation.index
    #     pass

    #     ##  Use index to split validation if not table format.
    #     self.structure.train.table = table.iloc[index.train].reset_index(drop=True)
    #     self.structure.validation.table = table.iloc[index.validation].reset_index(drop=True)
    #     return

    pass


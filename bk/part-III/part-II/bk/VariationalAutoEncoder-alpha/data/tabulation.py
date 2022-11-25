
import pandas, numpy
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

class tabulation:

    def __init__(self, path=None):

        self.path = path
        pass
    
    def read(self):

        dataframe  = pandas.read_csv(self.path)
        self.data  = dataframe
        pass
    
    def distinct(self, what='data'):

        if(what=='data'):

            self.train = self.data[self.data['mode']=='train'].copy().reset_index(drop=True)
            self.exam  = self.data[self.data['mode']=='exam' ].copy().reset_index(drop=True)
            self.test  = self.data[self.data['mode']=='test' ].copy().reset_index(drop=True)
            pass

    def fold(self, size=None, target=None, seed=0):

        '''
        Input
            `size` : The size of k-fold.
            `target` : Classification target column.
            `seed` : Random seed.
        Return
            `self.grid` : A list consist of k-fold block, the element of list is a "pair" class.
        '''

        numpy.random.seed(seed)
        if(target):

            loader  = StratifiedKFold(n_splits=size).split(self.train, self.train[target])
            pass

        else:

            loader  = KFold(n_splits=size).split(self.train)
            pass
        
        self.grid = []
        for inside, outside in loader:
            
            inside   = self.train.iloc[inside]
            outside  = self.train.iloc[outside]
            self.grid.append([pair(inside, outside)])
            pass

        pass
    
    def hold(self, ratio=0.1, target=None, seed=0):
        
        '''
        Input
            `ratio` : The ratio of holdout.
            `target` : Classification target column.
            `seed` : Random seed.
        Return
            `self.grid` : A "pair" class.
        '''

        numpy.random.seed(seed)
        if(target):

            inside, outside = train_test_split(self.train, stratify=self.train[target], test_size=ratio)
            pass

        else:

            inside, outside = train_test_split(self.train, test_size=ratio)
            pass

        self.grid = pair(inside, outside)
        pass

    pass

class pair:

    def __init__(self, inside=None, outside=None):

        self.train = inside
        self.exam = outside
        pass
    
    pass

# class tabulation:

#     def __init__(self, path=None, target=None, split=0.1, seed=0):

#         self.path = path
#         self.target = target
#         self.split = split
#         self.seed = seed
#         return
    
#     def read(self):

#         form = pandas.read_csv(self.path)
#         data = form.loc[form['mode']!='test'].reset_index(drop=True)
#         test = form.loc[form['mode']=='test'].reset_index(drop=True)
#         pass
        
#         numpy.random.seed(self.seed)
#         if(self.target):

#             train, exam = train_test_split(data, stratify=data[self.target], test_size=self.split)
#             pass

#         else:

#             train, exam = train_test_split(data, test_size=self.split)
#             pass

#         train = train.reset_index(drop=True)
#         exam  = exam.reset_index(drop=True)
#         return(train, exam, test)

    # def split(dataframe, column, value):

    #     output = dataframe.loc[table[column]==value].copy()
    #     output = output.reset_index(drop=True)
    #     return(output)

    # pass








    # ##  Balance the data of table with target.
    # def balance(table, target, size):

    #     output = []
    #     for i in set(table[target]):

    #         selection = table[table[target]==i]
    #         pass
        
    #         if(len(selection)>size):

    #             selection = selection.sample(size)
    #             pass

    #         else:

    #             selection = selection.sample(size, replace=True)
    #             pass

    #         output = output + [selection]
    #         pass

    #     output = pandas.concat(output, axis=0)
    #     return(output)

    # ##
    # def unbalance(table, target, size):

    #     group = []
    #     for key, value in size.items():

    #         selection = table.loc[table[target]==key]
    #         pass

    #         if(len(selection)>value):
        
    #             group += [selection.sample(value)]
    #             pass
        
    #         else:
        
    #             group += [selection.sample(value, replace=True)]
    #             pass
        
    #     output = pandas.concat(group, axis=0)
    #     return(output)

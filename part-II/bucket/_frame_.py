
import core
import pickle
import pandas
import sklearn.model_selection
import yaml
import torch

class Bucket:

    def __init__(self, description='classification'):

        self.description = description
        return

    def loadEnvironment(self, path):

        self.environment = core.getEnvrionment(path)
        return

    '''Override in your case.'''
    def loadTable(self):

        table = core.createUnit(name='table')
        table.train = pandas.read_csv(self.environment['train']['table'])
        table.test  = pandas.read_csv(self.environment['test']['table'])
        pass

        self.table = table
        return

    '''Override in your case.'''
    def loadStructure(self):

        attribution = core.createUnit(name='attribution')
        attribution.train = core.getStructure(path=None)
        attribution.test = core.getStructure(path=None)
        pass

        self.attribution = attribution
        return

    '''Override in your case.'''
    def exportDataset(self):


        return

    pass
# index= 0 
# for k in x:

#     print(x[k][index])

'''Override in your case.'''
class Table(torch.utils.data.Dataset):

    def __init__(self, table):
            
        self.table = table
        return

    def __getitem__(self, index):

        item = {}
        for key in self.table:
            # image,label
            if(key=='image'): 
                
                if(item.get('index')): item['image'] += [self.table[key][index]]
                else: item['index'] = [self.table[key][index]]
                if(item.get('image')): item['image'] += [self.table[key][index]]
                else: item['image'] = [self.table[key][index]]
                pass

            if(key=='label'): 
                
                if(item.get('target')): item['label'] += [self.table[key][index]]
                else: item['label'] = [self.table[key][index]]
                pass

            # if(key=='feature'): value = self.dictionary[key][index,:]
            # if(key=='attribution'): value = self.dictionary[key][index,:]
            item[key] = value
            continue

        return(item)
        
    def __len__(self):

        length = len(self.dictionary['index'])
        return(length)

    pass    



class Table(torch.utils.data.Dataset):

    def __init__(self, table):
            
        self.table = table
        return

    def __getitem__(self, index):

        item = {}
        for key in self.dictionary:

            if(key=='index'): value = self.dictionary[key][index]
            if(key=='target'): value = self.dictionary[key][index]
            if(key=='feature'): value = self.dictionary[key][index,:]
            if(key=='attribution'): value = self.dictionary[key][index,:]
            item[key] = value
            continue

        return(item)
        
        def __len__(self):

            length = len(self.dictionary['index'])
            return(length)

        pass    


# def createDataset(dictionary):

#     class unit(torch.utils.data.Dataset):

#         def __init__(self, dictionary=dictionary):
            
#             self.dictionary = dictionary
#             return

#         def __getitem__(self, index):

#             item = {}
#             for key in self.dictionary:

#                 if(key=='index'): value = self.dictionary[key][index]
#                 if(key=='target'): value = self.dictionary[key][index]
#                 if(key=='feature'): value = self.dictionary[key][index,:]
#                 if(key=='attribution'): value = self.dictionary[key][index,:]
#                 item[key] = value
#                 continue

#             return(item)
        
#         def __len__(self):

#             length = len(self.dictionary['index'])
#             return(length)

#         pass    

#     dataset = unit(dictionary=dictionary) 
#     return(dataset)




# class Path:

#     def __init__(self, **path):

#         for key, value in path.items(): setattr(self, key, value)
#         return

#     pass



# class Bucket:

#     def __init__(self, **path):
        
#         self.path = Path(**path)
#         return

#     def loadDictionary(self):

#         if(self.train): 
            
#             with open(self.train, 'rb') as paper: self.train = pickle.load(paper)
#             pass
            
#         if(self.validation): 
            
#             with open(self.validation, 'rb') as paper: self.validation = pickle.load(paper)
#             pass

#         if(self.test): 
            
#             with open(self.test, 'rb') as paper: self.test = pickle.load(paper)
#             pass
        
#         return

#     '''Override your case'''
#     def loadTable(self, **path):

#         for 
#         read = pandas.read_csv
#         empty = pandas.DataFrame
#         self.train = read(self.train, dtype="str", sep=",") if(self.train) else empty()
#         self.validation = read(self.validation, dtype="str", sep=",") if(self.validation) else empty()
#         self.test = read(self.test, dtype="str", sep=",") if(self.test) else empty()
#         return

#     def splitValidation(self, percentage=0.2, stratification=None, seed=0):
        
#         assert not self.train.empty, 'train is empty'
#         assert self.validation.empty, 'validation is not empty'
#         group =  self.train[stratification].copy() if(stratification) else None
#         split = sklearn.model_selection.train_test_split
#         train, validation = split(
#             self.train, 
#             test_size=percentage, 
#             stratify=group, 
#             random_state=seed
#         )
#         pass

#         self.train = train.reset_index(drop=True)
#         self.validation = validation.reset_index(drop=True)
#         return

#     # def export(self, format='dataset'):

#     #     if(format=='dataset'):

#     #         class dataset: pass
#     #         dataset.train = None if(self.train.empty) else unit(form=self.train)
#     #         dataset.validation = None if(self.validation.empty) else unit(form=self.validation)
#     #         dataset.test = None if(self.test.empty) else unit(form=self.test)
#     #         return(dataset)

#     #     pass

#     pass

# # class unit(torch.utils.data.Dataset):

# #     def __init__(self, form=None):
        
# #         self.form = form
# #         return

# #     def __getitem__(self, index):

# #         item = self.form.loc[index]
# #         return(item)
    
# #     def __len__(self):

# #         length = len(self.form)
# #         return(length)

# #     pass


'''
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
'''

import functools
import pandas
import torch
import os
import PIL.Image
import torchvision.transforms
import yaml
import glob

def getEnvrionment(path='environment.yaml'):

    with open(path) as paper:
        
        environment = yaml.load(paper, yaml.loader.SafeLoader)
        pass

    return(environment)

def createPack(name='case'):

    assert name, 'name is None'
    class pack: pass
    pack.__qualname__ = name
    pack.__name__ = name
    return(pack)

def createDataset(dictionary):

    class unit(torch.utils.data.Dataset):

        def __init__(self, dictionary=dictionary):
            
            self.dictionary = dictionary
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

    dataset = unit(dictionary=dictionary) 
    return(dataset)

class Process:
    
    def __init__(self, item=None):
    
        self.item = item
        return

    def learnCase(self):
        
        environment = getEnvrionment(path='./environment.yaml')
        case = createPack(name='case')
        pass
        
        ##  Index process.
        case.index = self.item['index']
        pass

        ##  Feature process.
        case.feature = torch.tensor(self.item['feature']).type(torch.FloatTensor)
        pass

        ##  Attribution process.
        case.attribution = torch.tensor(self.item['attribution']).type(torch.FloatTensor)
        pass

        ##  Label process.
        label = environment['label']
        # print(self.item['target'])
        # print(label)
        case.target = torch.tensor(label.get(self.item['target'])).type(torch.LongTensor)
        pass

        return(case)

    def inferCase(self):

        environment = getEnvrionment(path='./environment.yaml')
        case = createPack(name='case')
        pass
        
        ##  Index process.
        case.index = self.item['index']
        pass

        ##  Feature process.
        case.feature = torch.tensor(self.item['feature']).type(torch.FloatTensor)
        pass

        ##  Attribution process.
        case.attribution = torch.tensor(self.item['attribution']).type(torch.FloatTensor)
        pass

        ##  Label process.
        label = environment['label']
        case.target = torch.tensor(label.get(self.item['target'])).type(torch.LongTensor)
        pass
    
        return(case)

def collectBatch(iteration=None, inference=False, device='cpu'):

    batch = createPack(name='batch')
    batch.iteration = iteration
    batch.inference = inference
    batch.size    = 0
    batch.index   = []
    batch.feature     = []
    batch.target      = []
    batch.attribution = []
    for item in iteration:
            
        process = Process(item=item)
        case = process.learnCase() if(not batch.inference) else process.inferCase()
        batch.index += [case.index]
        batch.feature += [case.feature.unsqueeze(0)]
        batch.target += [case.target.unsqueeze(0)]
        batch.attribution += [case.attribution.unsqueeze(0)]
        batch.size += 1
        continue

    batch.feature = torch.cat(batch.feature, axis=0).to(device)
    batch.target = torch.cat(batch.target, axis=0).to(device)
    batch.attribution = torch.cat(batch.attribution, axis=0).to(device)
    return(batch)

def createLoader(dataset=None, batch=32, inference=False, device='cuda'):

    # collection = functools.partial(collectBatch, inference=inference, device=device)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch, 
        shuffle=False if(inference) else True, 
        drop_last=False if(inference) else True, 
        collate_fn=functools.partial(collectBatch, inference=inference, device=device)
        )
    return(loader)

class Engine:

    def __init__(self, train=dict(), validation=dict(), test=dict()):

        self.train = train if(train!=dict()) else dict()
        self.validation = validation if(validation!=dict()) else dict()
        self.test = test if(test!=dict()) else dict()
        return

    def defineDataset(self):

        dataset = createPack(name='dataset')
        dataset.train = createDataset(dictionary=self.train) if(self.train!=dict()) else None
        dataset.validation = createDataset(dictionary=self.validation) if(self.validation!=dict()) else None
        dataset.test = createDataset(dictionary=self.test) if(self.test!=dict()) else None
        self.dataset = dataset
        return

    def defineLoader(self, batch=32, device='cuda', augmentation=True):

        loader = createPack(name='loader')
        pass

        if(augmentation):

            loader.train = createLoader(
                dataset=self.dataset.train, batch=batch, 
                inference=False, device=device
            ) if(self.dataset.train) else None
            pass

        else:

            loader.train = createLoader(
                dataset=self.dataset.train, batch=batch, 
                inference=True, device=device
            ) if(self.dataset.train) else None
            pass

        loader.validation = createLoader(
            dataset=self.dataset.validation, batch=batch, 
            inference=True, device=device
        ) if(self.dataset.validation) else None
        loader.test = createLoader(
            dataset=self.dataset.test, batch=batch, 
            inference=True, device=device
        ) if(self.dataset.test) else None
        self.loader = loader
        return

    def getSample(self):

        sample = next(iter(self.loader.train))
        return(sample)

    pass


# class loader:

#     def __init__(self, batch=32, device='cpu'):

#         self.batch  = batch
#         self.device = device
#         return
    
#     def define(self, train=None, validation=None, test=None):

#         if(train is not None):

#             self.train = torch.utils.data.DataLoader(
#                 dataset=train, batch_size=self.batch, 
#                 shuffle=True , drop_last=True, 
#                 collate_fn=functools.partial(collect, inference=False, device=self.device)
#             )
#             self.sample = self.check(self.train)
#             pass

#         if(validation is not None):

#             self.validation = torch.utils.data.DataLoader(
#                 dataset=validation, batch_size=self.batch, 
#                 shuffle=False , drop_last=False,
#                 collate_fn=functools.partial(collect, inference=True, device=self.device)
#             )
#             _ = self.check(self.validation)
#             pass

#         if(test is not None):

#             self.test = torch.utils.data.DataLoader(
#                 dataset=test, batch_size=self.batch, 
#                 shuffle=False , drop_last=False, 
#                 collate_fn=functools.partial(collect, inference=True, device=self.device)
#             )
#             _ = self.check(self.test)
#             pass

#         return

#     def check(self, loader):

#         try: batch = next(iter(loader)) 
#         except: print("error when process batch data")
#         return(batch)
    
#     pass
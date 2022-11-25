
import functools
import pandas
import torch
import os
import PIL.Image
import torchvision.transforms
import yaml
import glob

# def getEnvrionment(path='environment.yaml'):

#     with open(path) as paper:
        
#         environment = yaml.load(paper, yaml.loader.SafeLoader)
#         pass

#     return(environment)

# def createPack(name='case'):

#     assert name, 'name is None'
#     class pack: pass
#     pack.__qualname__ = name
#     pack.__name__ = name
#     return(pack)

class Set(torch.utils.data.Dataset):

    def __init__(self, structure):
        
        self.structure = structure
        return
    
    def __getitem__(self, index):

        item = self.structure.table.loc[index]
        return(item)
    
    def __len__(self):

        length = self.structure.length
        return(length)

    pass

createSet = lambda structure: Set(structure)

# def createDataset(table):

#     class unit(torch.utils.data.Dataset):

#         def __init__(self, table=table):
            
#             self.table = table
#             return

#         def __getitem__(self, index):

#             item = self.table.loc[index]
#             return(item)
        
#         def __len__(self):

#             length = len(self.table)
#             return(length)

#         pass    

#     dataset = unit(table=table) 
#     return(dataset)

# class Process:
    
#     def __init__(self, item=None):
    
#         self.item = item
#         return

#     def learnCase(self):
        
#         environment = getEnvrionment(path='./environment.yaml')
#         case = createPack(name='case')
#         pass
        
#         ##  Index process.
#         case.index = self.item['image']
#         pass

#         ##  Image process.
#         storage = environment['storage']
#         path = "".join(glob.glob(storage + self.item['image']))
#         image = PIL.Image.open(path).convert("RGB")
#         mu  = [0.46, 0.36, 0.29]
#         std = [0.27, 0.21, 0.18]
#         size = (240, 240)
#         position = (224, 224)
#         convert = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(size),
#             torchvision.transforms.RandomCrop(position),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mu, std),
#         ])
#         case.image = convert(image).type(torch.FloatTensor)
#         pass

#         ##  Label process.
#         label = environment['label']
#         case.target = torch.tensor(label.get(self.item['label'])).type(torch.LongTensor)

#         # ##  Class embedding process.
#         # embedding = self.item[environment['embedding']].astype("float")
#         # case.embedding = torch.tensor(embedding).type(torch.FloatTensor)
#         return(case)

#     def inferCase(self):

#         environment = getEnvrionment(path='./environment.yaml')
#         case = createPack(name='case')
#         pass
        
#         ##  Index process.
#         case.index = self.item['image']
#         pass

#         ##  Image process.
#         storage = environment['storage']
#         path = "".join(glob.glob(storage + self.item['image']))
#         image = PIL.Image.open(path).convert("RGB")
#         mu  = [0.46, 0.36, 0.29]
#         std = [0.27, 0.21, 0.18]
#         size = (240, 240)
#         position = (224, 224)
#         convert = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(size),
#             torchvision.transforms.CenterCrop(position),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mu, std),
#         ])
#         case.image = convert(image).type(torch.FloatTensor)
#         pass

#         ##  Label process.
#         label = environment['label']
#         case.target = torch.tensor(label.get(self.item["label"])).type(torch.LongTensor)

#         # ##  Class embedding process.
#         # embedding = self.item[environment['embedding']].astype("float")
#         # case.embedding = torch.tensor(embedding).type(torch.FloatTensor)
#         return(case)

# def collectBatch(iteration=None, inference=False, device='cpu'):

#     batch = createPack(name='batch')
#     batch.iteration = iteration
#     batch.inference = inference
#     batch.size    = 0
#     batch.index   = []
#     batch.image   = []
#     batch.target  = []
#     # batch.embedding = []
#     for item in iteration:
            
#         process = Process(item=item)
#         case = process.learnCase() if(not batch.inference) else process.inferCase()
#         batch.index += [case.index]
#         batch.image += [case.image.unsqueeze(0)]
#         batch.target += [case.target.unsqueeze(0)]
#         # batch.embedding += [case.embedding.unsqueeze(0)]
#         batch.size += 1
#         continue

#     batch.image = torch.cat(batch.image, axis=0).to(device)
#     batch.target = torch.cat(batch.target, axis=0).to(device)
#     # batch.embedding = torch.cat(batch.embedding, axis=0).to(device)
#     return(batch)

# def createLoader(dataset=None, batch=32, inference=False, device='cuda'):

#     # collection = functools.partial(collectBatch, inference=inference, device=device)
#     loader = torch.utils.data.DataLoader(
#         dataset=dataset, batch_size=batch, 
#         shuffle=False if(inference) else True, 
#         drop_last=False if(inference) else True, 
#         collate_fn=functools.partial(collectBatch, inference=inference, device=device)
#         )
#     return(loader)

# class Engine:

#     def __init__(self, train=pandas.DataFrame(), validation=pandas.DataFrame(), test=pandas.DataFrame()):

#         self.train = train if(not train.empty) else pandas.DataFrame()
#         self.validation = validation if(not validation.empty) else pandas.DataFrame()
#         self.test = test if(not test.empty) else pandas.DataFrame()
#         return

#     def defineDataset(self):

#         dataset = createPack(name='dataset')
#         dataset.train = createDataset(table=self.train) if(not self.train.empty) else None
#         dataset.validation = createDataset(table=self.validation) if(not self.validation.empty) else None
#         dataset.test = createDataset(table=self.test) if(not self.test.empty) else None
#         self.dataset = dataset
#         return

#     def defineLoader(self, batch=32, device='cuda', augmentation=True):

#         loader = createPack(name='loader')
#         pass

#         if(augmentation):

#             loader.train = createLoader(
#                 dataset=self.dataset.train, batch=batch, 
#                 inference=False, device=device
#             ) if(self.dataset.train) else None
#             pass

#         else:

#             loader.train = createLoader(
#                 dataset=self.dataset.train, batch=batch, 
#                 inference=True, device=device
#             ) if(self.dataset.train) else None
#             pass

#         loader.validation = createLoader(
#             dataset=self.dataset.validation, batch=batch, 
#             inference=True, device=device
#         ) if(self.dataset.validation) else None
#         loader.test = createLoader(
#             dataset=self.dataset.test, batch=batch, 
#             inference=True, device=device
#         ) if(self.dataset.test) else None
#         self.loader = loader
#         return

#     def getSample(self):

#         sample = next(iter(self.loader.train))
#         return(sample)

#     pass


# # class loader:

# #     def __init__(self, batch=32, device='cpu'):

# #         self.batch  = batch
# #         self.device = device
# #         return
    
# #     def define(self, train=None, validation=None, test=None):

# #         if(train is not None):

# #             self.train = torch.utils.data.DataLoader(
# #                 dataset=train, batch_size=self.batch, 
# #                 shuffle=True , drop_last=True, 
# #                 collate_fn=functools.partial(collect, inference=False, device=self.device)
# #             )
# #             self.sample = self.check(self.train)
# #             pass

# #         if(validation is not None):

# #             self.validation = torch.utils.data.DataLoader(
# #                 dataset=validation, batch_size=self.batch, 
# #                 shuffle=False , drop_last=False,
# #                 collate_fn=functools.partial(collect, inference=True, device=self.device)
# #             )
# #             _ = self.check(self.validation)
# #             pass

# #         if(test is not None):

# #             self.test = torch.utils.data.DataLoader(
# #                 dataset=test, batch_size=self.batch, 
# #                 shuffle=False , drop_last=False, 
# #                 collate_fn=functools.partial(collect, inference=True, device=self.device)
# #             )
# #             _ = self.check(self.test)
# #             pass

# #         return

# #     def check(self, loader):

# #         try: batch = next(iter(loader)) 
# #         except: print("error when process batch data")
# #         return(batch)
    
# #     pass
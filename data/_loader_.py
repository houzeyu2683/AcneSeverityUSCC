
import functools
import torch
import os
import PIL.Image
import torchvision.transforms
import yaml
import glob

with open('environment.yaml') as paper:
    
    environment = yaml.load(paper, yaml.loader.SafeLoader)
    pass

def create(name='case'):

    assert name, 'define name please'
    class prototype: pass
    prototype.__qualname__ = name
    prototype.__name__ = name
    return(prototype)

class process:
    
    def __init__(self, item=None):
    
        self.item = item
        return

    def learn(self):
        
        case = create(name='case')
        pass
        
        ##  Index process.
        case.index = self.item['image']
        pass

        ##  Image process.
        storage = environment['storage']
        path = "".join(glob.glob(storage + self.item['image']))
        image = PIL.Image.open(path).convert("RGB")
        mu  = [0.46, 0.36, 0.29]
        std = [0.27, 0.21, 0.18]
        size = (240, 240)
        position = (224, 224)
        convert = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.RandomCrop(position),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mu, std),
        ])
        case.image = convert(image).type(torch.FloatTensor)
        pass

        ##  Label process.
        label = environment['label']
        case.target = torch.tensor(label.get(self.item['vote'])).type(torch.LongTensor)
        return(case)

    def infer(self):

        case = create(name='case')
        pass
        
        ##  Index process.
        case.index = self.item['image']
        pass

        ##  Image process.
        storage = environment['storage']
        path = "".join(glob.glob(storage + self.item['image']))
        image = PIL.Image.open(path).convert("RGB")
        mu  = [0.46, 0.36, 0.29]
        std = [0.27, 0.21, 0.18]
        size = (240, 240)
        position = (224, 224)
        convert = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(position),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mu, std),
        ])
        case.image = convert(image).type(torch.FloatTensor)
        pass

        ##  Label process.
        label = environment['label']
        case.target = torch.tensor(label.get(self.item["vote"])).type(torch.LongTensor)
        return(case)

def collect(iteration=None, inference=False, device='cpu'):

    batch = create(name='batch')
    batch.iteration = iteration
    batch.inference = inference
    batch.size    = 0
    batch.index   = []
    batch.image   = []
    batch.target  = []
    for item in iteration:
            
        engine = process(item=item)
        case = engine.learn() if(not batch.inference) else engine.infer()
        batch.index += [case.index]
        batch.image += [case.image.unsqueeze(0)]
        batch.target += [case.target.unsqueeze(0)]
        batch.size += 1
        continue

    batch.image = torch.cat(batch.image, axis=0).to(device)
    batch.target = torch.cat(batch.target, axis=0).to(device)
    return(batch)

class loader:

    def __init__(self, batch=32, device='cpu'):

        self.batch  = batch
        self.device = device
        return
    
    def define(self, train=None, validation=None, test=None):

        if(train is not None):

            self.train = torch.utils.data.DataLoader(
                dataset=train, batch_size=self.batch, 
                shuffle=True , drop_last=True, 
                collate_fn=functools.partial(collect, inference=False, device=self.device)
            )
            pass
        
        if(validation is not None):

            self.validation = torch.utils.data.DataLoader(
                dataset=validation, batch_size=self.batch, 
                shuffle=False , drop_last=False,
                collate_fn=functools.partial(collect, inference=True, device=self.device)
            )
            pass

        if(test is not None):

            self.test = torch.utils.data.DataLoader(
                dataset=test, batch_size=self.batch, 
                shuffle=False , drop_last=False, 
                collate_fn=functools.partial(collect, inference=True, device=self.device)
            )
            pass

        return

    pass

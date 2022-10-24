
import functools
import torch
import os
import PIL.Image
import torchvision.transforms

label = {'0':0, '1':1, '2':2, '3':3}
storage = "/home/houzeyu2683/Desktop/Projects/Classification/AcneSeverity/resource/jpg"

class process:
    
    def __init__(self, item=None):
    
        self.item = item
        return

    def learn(self):
        
        image = PIL.Image.open(os.path.join(storage, self.item[0])).convert("RGB")
        mu  = [0.46, 0.36, 0.29]
        std = [0.27, 0.21, 0.18]
        size = (400, 400)
        position = (224, 224)
        convert = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.RandomCrop(position),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mu, std),
        ])
        self.image = convert(image).type(torch.FloatTensor)
        self.target = torch.tensor(label.get(self.item[1])).type(torch.LongTensor)
        return

    def infer(self):

        image = PIL.Image.open(os.path.join(storage, self.item[0])).convert("RGB")
        mu  = [0.46, 0.36, 0.29]
        std = [0.27, 0.21, 0.18]
        size = (400, 400)
        position = (224, 224)
        convert = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(position),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mu, std),
        ])
        self.image = convert(image).type(torch.FloatTensor)
        self.target = torch.tensor(label.get(self.item[1])).type(torch.LongTensor)
        return

def collect(iteration=None, inference=False, device='cpu'):

    class batch: 
        
        size = len(iteration)
        pass

    batch.iteration = iteration
    batch.inference = inference
    batch.image   = []
    batch.target  = []
    for item in iteration:
            
        engine = process(item=item)
        engine.learn() if(not batch.inference) else engine.infer()
        batch.image += [engine.image.unsqueeze(0)]
        batch.target += [engine.target.unsqueeze(0)]
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

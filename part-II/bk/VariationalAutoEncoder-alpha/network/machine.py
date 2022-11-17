
##  Packages.
import os, tqdm, torch, numpy, pickle, pandas
from torch.optim import lr_scheduler
from torchvision import utils

##  Class for machine learning process, case by case.
class machine:

    def __init__(self, model, optimizer=None, device='cuda', folder=None, checkpoint="0"):

        self.model      = model
        self.optimizer  = optimizer
        self.device     = device
        self.folder     = folder
        self.checkpoint = checkpoint
        pass

        ##  Create the folder for storage in default.
        os.makedirs(self.folder, exist_ok=True) if(self.folder) else None
        pass
        
        ##  Optimizer schedule in default.
        self.schedule = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9) if(self.optimizer) else None
        pass
    
    def bundle(self, batch):

        batch = batch.to(self.device)
        return(batch)

    def learn(self, loader):

        ##  Mode of learn.
        self.model.train()
        self.model = self.model.to(self.device)
        pass

        progress = tqdm.tqdm(loader, leave=False)
        record = None
        for batch in progress:

            ##  Handle batch.
            batch = self.bundle(batch)

            ##  Update weight.
            self.optimizer.zero_grad()
            value = self.model(batch)
            loss = self.model.cost(value)
            loss['total'].backward()
            # loss = loss['total']
            #loss = self.criterion.to(self.device)(score, target)
            # loss.backward()
            self.optimizer.step()
            # return(loss)
            score = {k: round(v.item(),3) for k, v in loss.items()}
            progress.set_description("{}".format(score))
            if(record==None):

                record = {k:[v] for k, v in score.items()}
                pass

            else:
                # return(record, score)
                for k in score:

                    record[k] += [score[k]]
                    pass

                pass

            pass
        
        record = pandas.DataFrame(record)
        record.to_csv(os.path.join(self.folder, "record-{}.csv".format(self.checkpoint)))
        pass

    def save(self, what='checkpoint'):

        ##  Save the checkpoint.
        if(what=='checkpoint'):

            path = os.path.join(self.folder, "model-"+self.checkpoint+".checkpoint")
            torch.save(self.model.state_dict(), path)
            print("save the weight of model to {}".format(path))
            pass
  
    def update(self, what='checkpoint'):

        if(what=='checkpoint'):
            
            try:
                
                self.checkpoint = str(int(self.checkpoint) + 1)
                print("update the checkpoint to {} for next iteration".format(self.checkpoint))
                pass

            except:

                print("the checkpoint is not integer, skip update checkpoint")
                pass

        if(what=='schedule'):

            self.schedule.step()
            rate = self.optimizer.param_groups[0]['lr']
            print("learning rate update to {}".format(rate))
            pass

    def load(self, what='weight', path=None):

        if(what=='weight'):
            
            try:

                self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                print("successful weight loading")
                pass
            
            except:

                print("fail weight loading")
                pass        
            
            pass

        return
    
    def evaluate(self, loader, title='train', number=1):
        
        link = {
            'origin':os.path.join(self.folder, "{}-origin-{}".format(title, self.checkpoint)),
            "reconstruction":os.path.join(self.folder, "{}-reconstruction-{}".format(title, self.checkpoint)),
            'generation':os.path.join(self.folder, "{}-generation-{}".format(title, self.checkpoint))
        }
        for index, batch in enumerate(loader):

            if(index<number):
                
                self.model.to(self.device).eval()
                with torch.no_grad():

                    batch = self.bundle(batch)
                    value = self.model(batch)
                    size = len(batch)
                    origin = utils.make_grid(value['image'])
                    reconstruction = utils.make_grid(value['reconstruction'])
                    generation = utils.make_grid(self.model.generate(size))
                    utils.save_image(origin, link['origin']+'-{}.jpg'.format(index), normalize=True)
                    utils.save_image(reconstruction, link['reconstruction']+'-{}.jpg'.format(index), normalize=True)
                    utils.save_image(generation, link['generation']+'-{}.jpg'.format(index), normalize=True)
                    pass

                pass
            
            else:

                return

            pass

        pass

    pass

    # def predict(self, loader):

    #     self.model.eval()
    #     self.model = self.model.to(self.device)
    #     pass

    #     likelihood = []
    #     for batch in tqdm.tqdm(loader, leave=False):

    #         with torch.no_grad():
            
    #             batch = self.bundle(batch)
    #             score, target = self.model(batch)
    #             pass
            
    #         likelihood += [score.cpu().numpy()]
    #         pass

    #     ##  Summarize record.
    #     likelihood  = numpy.concatenate(likelihood, axis=0)
    #     likelihood  = pandas.DataFrame(likelihood, columns=["c"+str(i) for i in range(10)])
    #     return(likelihood)

    # pass

        # event = {'train':train, "check":check}
        # for key in event:

        #     if(event[key]):

        #         record = {
        #             'loss':[]
        #         }
        #         for batch in tqdm.tqdm(event[key], leave=False):

        #             with torch.no_grad():

        #                 batch = self.bundle(batch)
        #                 score, target = self.model(batch)
        #                 loss  = self.criterion(score, target).cpu().detach().numpy().item()
        #                 pass

        #             record['loss']  += [loss]
        #             pass
                
        #         ##  Summarize record.
        #         record['loss']  = numpy.mean(record['loss'])
        #         pass

        #         ##  Insert evaluation to measurement.
        #         measurement[key] = record
        #         print("end of measure the {}".format(key))
        #         pass

        #     pass

        # self.measurement = measurement
        # pass

    # def predict(self, test):

    #     self.model = self.model.to(self.device)
    #     self.model.eval()
    #     pass

    #     likelihood = []
    #     # prediction = []
    #     for batch in tqdm.tqdm(test, leave=False):

    #         image, target = batch
    #         batch = image.to(self.device), target.to(self.device)
    #         score = self.model(batch).cpu().detach().numpy()
    #         likelihood += [score] 
    #         # prediction += score.argmax(dim=1)
    #         pass
        
    #     likelihood = pandas.DataFrame(numpy.concatenate(likelihood, axis=0), columns=["c"+ str(i) for i in range(10)])
    #     # prediction = numpy.array(prediction)
    #     return(likelihood)




# from itertools import chain
# x = [2]
# z= chain(*x)


# import torch
# import torch.nn as nn
# loss = nn.CrossEntropyLoss()

# x = torch.randn((2,6,3))
# y = torch.randint(0,3,(2,6))
# x.shape
# y.shape
# loss(x[0,:,:], y[0,:])

# x[0,:,:]
# output.shape
# target = target.to('cuda')
# torch.flatten(target).shape
# loss   = criterion.to('cuda')(output, torch.flatten(target))
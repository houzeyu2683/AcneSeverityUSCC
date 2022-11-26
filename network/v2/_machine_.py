
import os
import torch
import tqdm
import numpy
import pprint
import sys
import sklearn.metrics

createClass = lambda name: type(name, (), {})
runMultiplication = lambda x, y: sum([l*r for l, r in zip(x, y)])

class Feedback:

    def __init__(self, title=None):

        self.title = title
        return

    def convertDictionary(self):

        variable = vars(self)
        dictionary = dict(variable)
        return(dictionary)

    pass

class Machine:

    def __init__(self, model=None):

        self.model  = model
        return

    def loadWeight(self, path, device='cuda'):

        self.model.load_state_dict(torch.load(path, map_location=device))
        # self.model = torch.load(path, map_location=device)
        return

    def loadModel(self, path, device='cuda'):

        self.model = torch.load(path, map_location=device)
        return

    def defineOptimization(self, method='adam'):

        if(method=='adam'):

            self.gradient = torch.optim.Adam(
                self.model.parameters(), 
                lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
                amsgrad=False
            )
            pass

        if(method=='sgd'):

            self.gradient = torch.optim.SGD(
                self.model.parameters(), 
                lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False
            )
            pass

        self.schedule = torch.optim.lr_scheduler.StepLR(self.gradient, step_size=10, gamma=0.1)
        return

    def learnIteration(self, loader=None):
    
        Iteration = createClass(name='Iteration')
        iteration = Iteration()
        iteration.cost  = []
        iteration.size  = []
        pass

        self.model.train()
        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            self.gradient.zero_grad()
            cost = self.model.getCost(batch)
            cost.loss.backward()
            self.gradient.step()
            pass

            message = "loss : {:.2f}, divergence : {:.2f}, reconstruction : {:.2f}, projection : {:.2f}"
            description = message.format(cost.loss.item(), cost.divergence.item(), cost.reconstruction.item(), cost.projection.item())
            progress.set_description(description)
            pass

            iteration.cost  += [cost]
            iteration.size  += [batch.size]
            continue
        
        self.schedule.step()    
        pass
        
        feedback = Feedback(title='train')
        loss = [c.loss.item() for c in iteration.cost]
        divergence = [c.divergence.item() for c in iteration.cost]
        restruction = [c.reconstruction.item() for c in iteration.cost]
        projection = [c.projection.item() for c in iteration.cost]
        feedback.cost = {
            "loss": runMultiplication(loss, iteration.size) / sum(iteration.size),
            "divergence": runMultiplication(divergence, iteration.size) / sum(iteration.size),
            "reconstruction" : runMultiplication(restruction, iteration.size) / sum(iteration.size),
            "projection": runMultiplication(projection, iteration.size) / sum(iteration.size)
        }
        return(feedback)

    @torch.no_grad()
    def evaluateIteration(self, loader=None, title=None):

        Iteration = createClass(name='Iteration')
        iteration = Iteration()
        pass

        iteration.size        = []
        iteration.image       = []
        iteration.label       = []
        iteration.prediction  = []
        iteration.extraction  = []
        iteration.decoding    = []
        iteration.encoding    = []
        iteration.attribution = []
        iteration.cost        = []
        pass

        self.model.eval()
        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            encoding, decoding, _  = self.model.forwardProcedure(batch)
            cost  = self.model.getCost(batch)
            pass

            iteration.image       += [batch.image]
            iteration.size        += [batch.size]
            iteration.label       += [batch.label]
            iteration.prediction  += [batch.prediction]
            iteration.extraction  += [batch.extraction]
            iteration.decoding    += [decoding]
            iteration.attribution += [batch.attribution]
            iteration.encoding    += [encoding]
            iteration.cost        += [cost]
            continue
        
        iteration.image      = sum(iteration.image, [])
        iteration.label      = sum(iteration.label, [])
        iteration.prediction = sum(iteration.prediction, [])
        pass

        feedback = Feedback(title=title)
        feedback.size       = iteration.size
        feedback.image      = iteration.image
        feedback.label      = iteration.label
        feedback.prediction = iteration.prediction
        feedback.extraction  = torch.concat(iteration.extraction, dim=0).detach().cpu().numpy()
        feedback.decoding    = torch.concat(iteration.decoding, dim=0).detach().cpu().numpy()
        feedback.attribution = torch.concat(iteration.attribution, dim=0).detach().cpu().numpy()
        feedback.encoding    = torch.concat(iteration.encoding, dim=0).detach().cpu().numpy()
        pass

        loss = [c.loss.item() for c in iteration.cost]
        divergence = [c.divergence.item() for c in iteration.cost]
        restruction = [c.reconstruction.item() for c in iteration.cost]
        projection = [c.projection.item() for c in iteration.cost]
        feedback.cost = {
            "loss": runMultiplication(loss, iteration.size) / sum(iteration.size),
            "divergence": runMultiplication(divergence, iteration.size) / sum(iteration.size),
            "reconstruction" : runMultiplication(restruction, iteration.size) / sum(iteration.size),
            "projection": runMultiplication(projection, iteration.size) / sum(iteration.size)
        }
        return(feedback)

    def saveWeight(self, path):

        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model.state_dict(), path)

        return

    def saveModel(self, path):

        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model, path)

        return

    pass


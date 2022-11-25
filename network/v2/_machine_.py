
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

            message = "loss : {:.2f}, divergence : {:.2f}, reconstruction : {:.2f}, match : {:.2f}"
            description = message.format(cost.loss.item(), cost.divergence.item(), cost.reconstruction.item(), cost.match.item())
            progress.set_description(description)
            pass

            iteration.cost  += [cost]
            iteration.size  += [batch.size]
            continue
        
        self.schedule.step()    
        pass
        
        feedback = Feedback(title='train')
        feedback.loss           = runMultiplication([c.loss.item() for c in iteration.cost], iteration.size) / sum(iteration.size)
        feedback.divergence     = runMultiplication([c.divergence.item() for c in iteration.cost], iteration.size) / sum(iteration.size)
        feedback.reconstruction = runMultiplication([c.reconstruction.item() for c in iteration.cost], iteration.size) / sum(iteration.size)
        feedback.match          = runMultiplication([c.match.item() for c in iteration.cost], iteration.size) / sum(iteration.size)
        return(feedback)

    @torch.no_grad()
    def evaluateIteration(self, loader=None, title=None):

        Iteration = createClass(name='Iteration')
        iteration = Iteration()
        pass

        iteration.size        = []
        iteration.image       = []
        pass

        iteration.label       = []
        iteration.prediction  = []
        pass

        iteration.extraction  = []
        iteration.decoding    = []
        pass

        iteration.encoding    = []
        iteration.attribution = []
        pass

        iteration.cost        = []
        pass

        self.model.eval()
        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            encoding, decoding, _  = self.model.forwardProcedure(batch)
            cost  = self.model.getCost(batch)
            pass

            iteration.cost        += [cost]
            iteration.image       += [batch.image]
            iteration.size        += [batch.size]
            iteration.label       += [batch.label]
            iteration.prediction  += [batch.prediction]
            iteration.extraction  += [batch.extraction]
            iteration.decoding    += [decoding]
            iteration.attribution += [batch.attribution]
            iteration.encoding    += [encoding]
            continue
        
        iteration.image = sum(iteration.image, [])
        iteration.label = sum(iteration.label, [])
        iteration.prediction = sum(iteration.prediction, [])
        pass

        feedback = Feedback(title=title)
        feedback.cost = {
            "loss": runMultiplication([c.loss.item() for c in iteration.cost], iteration.size) / sum(iteration.size),
            "divergence": runMultiplication([c.divergence.item() for c in iteration.cost], iteration.size) / sum(iteration.size),
            "reconstruction" : runMultiplication([c.reconstruction.item() for c in iteration.cost], iteration.size) / sum(iteration.size),
            "match": runMultiplication([c.match.item() for c in iteration.cost], iteration.size) / sum(iteration.size)
        }
        feedback.size       = iteration.size
        feedback.image      = iteration.image
        feedback.label      = iteration.label
        feedback.prediction = iteration.prediction
        feedback.extraction  = torch.concat(iteration.extraction, dim=0).detach().cpu().numpy()
        feedback.decoding    = torch.concat(iteration.decoding, dim=0).detach().cpu().numpy()
        feedback.attribution = torch.concat(iteration.attribution, dim=0).detach().cpu().numpy()
        feedback.encoding    = torch.concat(iteration.encoding, dim=0).detach().cpu().numpy()
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


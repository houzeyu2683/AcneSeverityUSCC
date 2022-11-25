
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
        iteration.loss  = []
        iteration.size  = []
        iteration.divergence      = []
        iteration.reconstruction  = []
        pass

        self.model.train()
        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            self.gradient.zero_grad()
            cost = self.model.getCost(batch)
            cost.loss.backward()
            self.gradient.step()
            pass

            description = "loss : {:.2f}, divergence : {:.2f}, reconstruction : {:.2f}".format(cost.loss.item(), cost.divergence.item(), cost.reconstruction.item())
            progress.set_description(description)
            pass

            iteration.loss            += [cost.loss.item()]
            iteration.divergence      += [cost.divergence.item()]
            iteration.reconstruction  += [cost.reconstruction.item()]
            iteration.size  += [batch.size]
            continue
        
        self.schedule.step()    
        pass
        
        feedback = Feedback(title='train')
        feedback.loss           = runMultiplication(iteration.loss, iteration.size) / sum(iteration.size)
        feedback.divergence     = runMultiplication(iteration.divergence, iteration.size) / sum(iteration.size)
        feedback.reconstruction = runMultiplication(iteration.reconstruction, iteration.size) / sum(iteration.size)
        feedback.iteration      = iteration
        return(feedback)

    @torch.no_grad()
    def evaluateIteration(self, loader=None, title=None):

        Iteration = createClass(name='Iteration')
        iteration = Iteration()
        pass

        iteration.size             = []
        iteration.image            = []
        pass

        iteration.loss             = []
        iteration.divergence       = []
        iteration.reconstruction   = []
        pass

        iteration.extraction       = []
        iteration.decoding         = []
        pass

        iteration.prediction       = []
        iteration.label            = []
        pass

        iteration.encoding         = []
        iteration.attribution      = []
        pass

        self.model.eval()
        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            encoding  = self.model.encoder.getEncoding(batch)
            score     = self.model.getScore(batch)
            cost      = self.model.getCost(batch)
            pass

            iteration.image      += [batch.image]
            iteration.size       += [batch.size]
            iteration.target     += [batch.target.cpu().numpy()]
            iteration.score      += [score.cpu().numpy()]
            iteration.prediction += [score.cpu().numpy().argmax(1)]
            iteration.loss       += [cost.loss.item()]
            iteration.extraction += [extraction.cpu().numpy()]
            continue
        
        iteration.image = sum(iteration.image, [])
        pass

        feedback = Feedback(title=title)
        feedback.size       = iteration.size
        feedback.image      = iteration.image
        feedback.loss       = runMultiplication(iteration.loss, iteration.size) / sum(iteration.size)
        feedback.score      = numpy.concatenate(iteration.score, axis=0)
        feedback.prediction = numpy.concatenate(iteration.prediction, axis=-1)
        feedback.target     = numpy.concatenate(iteration.target, axis=-1)
        feedback.extraction = numpy.concatenate(iteration.extraction, axis=0)
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


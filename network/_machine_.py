
import os
import torch
import tqdm
import numpy
import sklearn.metrics

class machine:

    def __init__(self, model=None, device='cpu'):

        self.model  = model
        self.device = device
        return

    def optimization(self, method='adam'):

        self.model = self.model.to(self.device)
        pass

        if(method=='adam'):

            self.gradient = torch.optim.Adam(
                self.model.parameters(), 
                lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
                amsgrad=False, maximize=False
            )
            self.schedule = torch.optim.lr_scheduler.StepLR(self.gradient, step_size=10, gamma=0.1)
            return

        if(method=='sgd'):

            self.gradient = torch.optim.SGD(
                self.model.parameters(), 
                lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False
            )
            self.schedule = torch.optim.lr_scheduler.StepLR(self.gradient, step_size=10, gamma=0.1)
            return        

    def learn(self, train=None):

    
        assert train, 'train not found'
        class iteration:  pass
        # iteration.index = []
        iteration.loss  = []
        iteration.size  = []
        pass

        self.model = self.model.to(self.device)
        self.model.train()
        pass

        progress = tqdm.tqdm(train, leave=False)
        for batch in progress:

            self.gradient.zero_grad()
            batch = self.model.cost(batch)
            batch.loss.backward()
            self.gradient.step()
            description = "loss : {:.2f}".format(batch.loss.item())
            progress.set_description(description)
            pass

            # iteration.index += [*batch.index]
            iteration.loss  += [batch.loss.item()]
            iteration.size  += [batch.size]
            continue
        
        class feedback: pass
        feedback.loss = sum([l*s for l, s in zip(iteration.loss, iteration.size)]) / sum(iteration.size)
        self.schedule.step()    
        return(feedback)

    @torch.no_grad()
    def evaluate(self, **data):

        assert (len(data) == 1), "input loader one by one"
        item = data.items()
        name, loop = list(item)[0]
        pass

        class iteration: pass
        # iteration.index      = []
        iteration.loss       = []
        iteration.size       = []
        iteration.score      = []
        iteration.prediction = []
        iteration.target     = []
        pass

        self.model = self.model.to(self.device)
        self.model.eval()
        pass

        progress = tqdm.tqdm(loop, leave=False)
        for batch in progress:

            batch = self.model.cost(batch)
            # iteration.index      += [*batch.index]
            iteration.loss       += [batch.loss.item()]
            iteration.size       += [batch.size]
            iteration.score      += [batch.score.cpu().numpy()]
            iteration.prediction += [batch.score.cpu().numpy().argmax(1)]
            iteration.target     += [batch.target.cpu().numpy()]
            continue

        class feedback: pass
        feedback.loss       = sum([l*s for l, s in zip(iteration.loss, iteration.size)]) / sum(iteration.size)
        feedback.score      = numpy.concatenate(iteration.score, axis=0)
        feedback.prediction = numpy.concatenate(iteration.prediction, axis=-1)
        feedback.target     = numpy.concatenate(iteration.target, axis=-1)
        pass

        summary = dict()
        summary['{} loss'.format(name)]       = feedback.loss
        summary['{} score'.format(name)]      = feedback.score
        summary['{} prediction'.format(name)] = feedback.prediction
        summary['{} target'.format(name)]     = feedback.target
        summary['{} accuracy'.format(name)]        = sklearn.metrics.accuracy_score(feedback.target, feedback.prediction)
        summary['{} confusion table'.format(name)] = sklearn.metrics.confusion_matrix(feedback.target, feedback.prediction)
        summary['{} report'.format(name)]          = sklearn.metrics.classification_report(feedback.target, feedback.prediction)
        return(summary)

    def save(self, path):

        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model, path)
        return

    def write(self, text, path):

        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        with open(path, 'w') as paper: _ = paper.write(text) 
        return

    pass

import os
import torch
import tqdm
import numpy
import pprint
import sklearn.metrics

def create(name='case'):

    assert name, 'define name please'
    class prototype: pass
    prototype.__qualname__ = name
    prototype.__name__ = name
    return(prototype)

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
                amsgrad=False
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

    def learn(self, loader=None):
    
        assert loader, 'loader not found'
        iteration = create(name='iteration')
        iteration.loss  = []
        iteration.size  = []
        pass

        self.model = self.model.to(self.device)
        self.model.train()
        pass

        progress = tqdm.tqdm(loader, leave=False)
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
        
        feedback = create(name='feedback')
        feedback.loss = sum([l*s for l, s in zip(iteration.loss, iteration.size)]) / sum(iteration.size)    
        self.schedule.step()    
        return(feedback)

    @torch.no_grad()
    def evaluate(self, loader=None, title='data'):

        # assert (len(data) == 1), "input loader one by one"
        # item = data.items()
        # name, loop = list(item)[0]
        pass

        iteration = create(name='iteration')
        iteration.index      = []
        iteration.loss       = []
        iteration.size       = []
        iteration.score      = []
        iteration.prediction = []
        iteration.target     = []
        pass

        self.model = self.model.to(self.device)
        self.model.eval()
        pass

        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            batch = self.model.cost(batch)
            iteration.index      += [*batch.index]
            iteration.loss       += [batch.loss.item()]
            iteration.size       += [batch.size]
            iteration.score      += [batch.score.cpu().numpy()]
            iteration.prediction += [batch.score.cpu().numpy().argmax(1)]
            iteration.target     += [batch.target.cpu().numpy()]
            continue

        feedback = create(name='feedback')
        feedback.title      = title
        feedback.index      = iteration.index
        feedback.loss       = sum([l*s for l, s in zip(iteration.loss, iteration.size)]) / sum(iteration.size)
        feedback.score      = numpy.concatenate(iteration.score, axis=0)
        feedback.prediction = numpy.concatenate(iteration.prediction, axis=-1)
        feedback.target     = numpy.concatenate(iteration.target, axis=-1)
        pass

        # summary = dict()
        # summary['{} index'.format(title)]      = feedback.index
        # summary['{} loss'.format(title)]       = feedback.loss
        # summary['{} score'.format(title)]      = feedback.score
        # summary['{} prediction'.format(title)] = feedback.prediction
        # summary['{} target'.format(title)]     = feedback.target
        # summary['{} accuracy'.format(title)]        = sklearn.metrics.accuracy_score(feedback.target, feedback.prediction)
        # summary['{} confusion table'.format(title)] = sklearn.metrics.confusion_matrix(feedback.target, feedback.prediction)
        # summary['{} report'.format(title)]          = sklearn.metrics.classification_report(feedback.target, feedback.prediction)
        return(feedback)

    def save(self, path):

        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model, path)
        return

    def write(self, text, path):

        text = str(text)
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        with open(path, 'a') as paper: _ = paper.write(text) 
        return

    pass

# f = open("output.txt", "a")
# pprint.pprint("Hello stackoverflow!", file=f)
# pprint.pprint("I have a question.", file=f)
# f.close()

#     def write(text, path):

#         folder = os.path.dirname(path)
#         os.makedirs(folder, exist_ok=True)
#         with open(path, 'a') as paper: _ = paper.write(text) 
#         return


import os
import torch
import tqdm
import numpy
import pprint
import sys
import sklearn.metrics

def createPack(name='case'):

    assert name, 'define name please'
    class pack: pass
    pack.__qualname__ = name
    pack.__name__ = name
    return(pack)

class machine:

    def __init__(self, model=None):

        self.model  = model
        # self.device = device
        return

    def loadModel(self, path, device='cuda'):

        self.model = torch.load(path, map_location=device)
        return

    def defineOptimization(self, method='adam'):

        # self.model = self.model.to(self.device)
        pass

        if(method=='adam'):

            self.gradient = torch.optim.Adam(
                self.model.parameters(), 
                lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
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

    def learnIteration(self, loader=None):
    
        assert loader, 'loader not found'
        iteration = createPack(name='iteration')
        iteration.loss_total  = []
        iteration.loss_kl  = []
        iteration.loss_rec  = []
        iteration.size  = []
        pass

        # self.model = self.model.to(self.device)
        self.model.train()
        pass

        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            self.gradient.zero_grad()
            cost = self.model.getCost(batch)
            cost.loss['total'].backward()
            self.gradient.step()
            pass

            description = "total loss : {:.2f}, kl-divergence loss : {:.2f}, reconstruction loss : {:.2f}".format(cost.loss['total'].item(), cost.loss['kl-divergence'].item(), cost.loss['reconstruction'].item())
            progress.set_description(description)
            pass

            iteration.loss_total  += [cost.loss['total'].item()]
            iteration.loss_kl  += [cost.loss['kl-divergence'].item()]
            iteration.loss_rec  += [cost.loss['reconstruction'].item()]
            iteration.size  += [batch.size]
            continue
        
        feedback = createPack(name='feedback')
        feedback.loss_total = sum([l*s for l, s in zip(iteration.loss_total, iteration.size)]) / sum(iteration.size)
        feedback.loss_kl = sum([l*s for l, s in zip(iteration.loss_kl, iteration.size)]) / sum(iteration.size)    
        feedback.loss_rec = sum([l*s for l, s in zip(iteration.loss_rec, iteration.size)]) / sum(iteration.size)    
        self.schedule.step()    
        return(feedback)

    @torch.no_grad()
    def evaluateIteration(self, loader=None, title='validation'):

        iteration = createPack(name='iteration')
        iteration.index      = []
        iteration.loss       = []
        iteration.size       = []
        iteration.score      = []
        iteration.prediction = []
        iteration.target     = []
        pass

        # self.model = self.model.to(self.device)
        self.model.eval()
        pass

        progress = tqdm.tqdm(loader, leave=False)
        for batch in progress:

            cost = self.model.getCost(batch)
            iteration.index      += [*batch.index]
            iteration.loss       += [cost.loss.item()]
            iteration.size       += [batch.size]
            iteration.score      += [cost.score.cpu().numpy()]
            iteration.prediction += [cost.score.cpu().numpy().argmax(1)]
            iteration.target     += [batch.target.cpu().numpy()]
            continue

        feedback = createPack(name='feedback')
        feedback.title      = title
        feedback.index      = iteration.index
        feedback.loss       = sum([l*s for l, s in zip(iteration.loss, iteration.size)]) / sum(iteration.size)
        feedback.score      = numpy.concatenate(iteration.score, axis=0)
        feedback.prediction = numpy.concatenate(iteration.prediction, axis=-1)
        feedback.target     = numpy.concatenate(iteration.target, axis=-1)
        return(feedback)

    def saveModel(self, path):

        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        return

    def writeText(self, text, path):

        # text = str(text)
        # print(text)
        text = pprint.pformat(text)
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        with open(path, 'w') as paper: _ = paper.write(text) 
        return

    pass

# print(text['test']['confusion'])


# # print('This message will be displayed on the screen.')

# original_stdout = sys.stdout # Save a reference to the original standard output

# with open('filename.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print('This message will be written to a file.')
#     sys.stdout = original_stdout # Reset the standard output to its original value

# f = open("output.txt", "a")
# pprint.pprint("Hello stackoverflow!", file=f)
# pprint.pprint("I have a question.", file=f)
# f.close()

#     def write(text, path):

#         folder = os.path.dirname(path)
#         os.makedirs(folder, exist_ok=True)
#         with open(path, 'a') as paper: _ = paper.write(text) 
#         return

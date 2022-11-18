
import bucket
import network
import metric

environment = bucket.loadYaml(path='./environment.yaml')
pass

dataset = bucket.createClass('dataset')
dataset.train = bucket.Set(environment['train'])
dataset.validation = bucket.Set(environment['validation'])
dataset.test = bucket.Set(environment['test'])
pass

dataset.train.LoadData()
dataset.validation.LoadData()
dataset.test.LoadData()
pass

loader = bucket.createClass(name='loader')
loader.train = bucket.createLoader(dataset=dataset.train, batch=8, inference=False, device='cuda')
loader.validation = bucket.createLoader(dataset=dataset.validation, batch=1, inference=True, device='cuda')
loader.test = bucket.createLoader(dataset=dataset.test, batch=1, inference=True, device='cuda')
pass

sample = bucket.createClass(name='sample')
sample.train = bucket.getSample(loader.train)
sample.validation = bucket.getSample(loader.validation)
sample.test = bucket.getSample(loader.test)
pass

model = network.v1.Model(backbone='resnet', classification=2, device='cuda')
machine = network.v1.Machine(model=model)
machine.defineOptimization(method='adam')
pass

epoch = 20
loop = range(epoch)
history = environment['history']
for iteration in loop:

    ##  Learning process.
    checkpoint = "models/image-classifier/checkpoint/"
    _ = machine.learnIteration(loader=loader.train)
    machine.saveModel(path='{}/{}/weight.pt'.format(checkpoint, iteration))
    pass

    ##  Evaluate train data.
    title = 'train'
    feedback = machine.evaluateIteration(loader=loader.train, title=title)
    category = metric.Category(score=feedback.score, prediction=feedback.prediction, target=feedback.target)
    history[title]['accuracy']         += [category.getAccuracy()]
    history[title]['confusion table']  += [category.getConfusionTable()]
    history[title]['area under curve'] += [category.getAreaUnderCurve()]
    history[title]['loss'] += [feedback.loss]
    pass

    ##  Evaluate validation data.
    title = 'validation'
    feedback = machine.evaluateIteration(loader=loader.train, title=title)
    category = metric.Category(score=feedback.score, prediction=feedback.prediction, target=feedback.target)
    history[title]['accuracy']         += [category.getAccuracy()]
    history[title]['confusion table']  += [category.getConfusionTable()]
    history[title]['area under curve'] += [category.getAreaUnderCurve()]
    history[title]['loss'] += [feedback.loss]
    pass

    ##  Evaluate test data.
    title = 'test'
    feedback = machine.evaluateIteration(loader=loader.train, title=title)
    category = metric.Category(score=feedback.score, prediction=feedback.prediction, target=feedback.target)
    history[title]['accuracy']         += [category.getAccuracy()]
    history[title]['confusion table']  += [category.getConfusionTable()]
    history[title]['area under curve'] += [category.getAreaUnderCurve()]
    history[title]['loss'] += [feedback.loss]
    continue

##  Save the history.
bucket.writeText(text=history, path="{}/history.txt".format(checkpoint))
pass

##  Find and save the best checkpoint.
best  = environment['best']
track = history[best['focus']][best['metric']]
best['checkpoint'] = track.index(max(track))
start = "{}/{}/".format(checkpoint, best['checkpoint'])
end = "{}/{}/".format(checkpoint, 'best')
bucket.copyFolder(start=start, end=end)
pass

##  Save the extraction.
loader.train = bucket.createLoader(dataset=dataset.train, batch=1, inference=True, device='cuda')
loader.validation = bucket.createLoader(dataset=dataset.validation, batch=1, inference=True, device='cuda')
loader.test = bucket.createLoader(dataset=dataset.test, batch=1, inference=True, device='cuda')
machine = network.v1.Machine(model=None)
machine.loadModel(path="{}/weight.pt".format(end), device='cuda')
pass

title = 'train'
feedback = machine.evaluateIteration(loader=loader.train, title=title)
bucket.savePickle(feedback, path='resource/ACNE04/Extraction/{}.pkl'.format(title))
pass
title = 'validation'
feedback = machine.evaluateIteration(loader=loader.validation, title=title)
bucket.savePickle(feedback, path='resource/ACNE04/Extraction/{}.pkl'.format(title))
pass

title = 'test'
feedback = machine.evaluateIteration(loader=loader.test, title=title)
bucket.savePickle(feedback, path='resource/ACNE04/Extraction/{}.pkl'.format(title))
pass

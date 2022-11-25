
import bucket
import network
import metric

configuration = bucket.v2.loadYaml(path='configuration.yaml')
pass

Train      = bucket.v2.createClass('Train')
Validation = bucket.v2.createClass('Validation')
Test       = bucket.v2.createClass('Test')
train, validation, test = Train(), Validation(), Test()
pass

train.set      = bucket.v2.Set(configuration, 'train')
validation.set = bucket.v2.Set(configuration, 'validation')
test.set       = bucket.v2.Set(configuration, 'test')
pass

train.set.LoadData()
validation.set.LoadData()
test.set.LoadData()
pass

train.loader      = bucket.createLoader(set=train.set, batch=32, inference=False, device='cpu')
validation.loader = bucket.createLoader(set=validation.set, batch=16, inference=True, device='cpu')
test.loader       = bucket.createLoader(set=test.set, batch=16, inference=True, device='cpu')
pass

train.sample      = bucket.getSample(train.loader)
validation.sample = bucket.getSample(validation.loader)
test.sample       = bucket.getSample(test.loader)
pass

model = network.v2.Model(device='cuda')
machine = network.v1.Machine(model=model)
machine.defineOptimization(method='adam')
pass

checkpoint = bucket.loadYaml(path='checkpoint.yaml')
history    = checkpoint['history']
best       = checkpoint['best']
pass

epoch = 10
loop = range(epoch)
for iteration in loop:

    ##  Learning process.
    _ = machine.learnIteration(loader=train.loader)
    machine.saveModel(path='{}/{}/model.pt'.format(checkpoint['path'], iteration))
    machine.saveWeight(path='{}/{}/weight.pt'.format(checkpoint['path'], iteration))    
    pass

    ##  Evaluate train data.
    title = 'train'
    feedback = machine.evaluateIteration(loader=train.loader, title=title)
    category = metric.Category(score=feedback.score, prediction=feedback.prediction, target=feedback.target)
    history[title]['accuracy']         += [category.getAccuracy()]
    history[title]['confusion table']  += [category.getConfusionTable()]
    history[title]['area under curve'] += [category.getAreaUnderCurve()]
    history[title]['loss'] += [feedback.loss]
    pass

    ##  Evaluate validation data.
    title = 'validation'
    feedback = machine.evaluateIteration(loader=validation.loader, title=title)
    category = metric.Category(score=feedback.score, prediction=feedback.prediction, target=feedback.target)
    history[title]['accuracy']         += [category.getAccuracy()]
    history[title]['confusion table']  += [category.getConfusionTable()]
    history[title]['area under curve'] += [category.getAreaUnderCurve()]
    history[title]['loss'] += [feedback.loss]
    pass

    ##  Evaluate test data.
    title = 'test'
    feedback = machine.evaluateIteration(loader=test.loader, title=title)
    category = metric.Category(score=feedback.score, prediction=feedback.prediction, target=feedback.target)
    history[title]['accuracy']         += [category.getAccuracy()]
    history[title]['confusion table']  += [category.getConfusionTable()]
    history[title]['area under curve'] += [category.getAreaUnderCurve()]
    history[title]['loss'] += [feedback.loss]
    continue

##  Save the history.
bucket.saveYaml(content=history, path="{}/history.yaml".format(checkpoint['path']))
history = bucket.loadYaml(path="{}/history.yaml".format(checkpoint['path']))
pass

##  Find and save the best checkpoint.
track = history[best['focus']][best['metric']]
best['checkpoint'] = track.index(max(track))
source = "{}/{}/".format(checkpoint['path'], best['checkpoint'])
destination = "{}/{}/".format(checkpoint['path'], 'best')
bucket.copyFolder(source=source, destination=destination)
pass

##  Save the extraction.
train.loader = bucket.createLoader(set=train.set, batch=1, inference=True, device='cuda')
validation.loader = bucket.createLoader(set=validation.set, batch=1, inference=True, device='cuda')
test.loader = bucket.createLoader(set=test.set, batch=1, inference=True, device='cuda')
machine = network.v1.Machine(model=None)
machine.loadModel(path="{}/best/model.pt".format(checkpoint['path']), device='cuda')
pass

title = 'train'
feedback = machine.evaluateIteration(loader=train.loader, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/{}.pkl'.format(title))
pass

title = 'validation'
feedback = machine.evaluateIteration(loader=validation.loader, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/{}.pkl'.format(title))
pass

title = 'test'
feedback = machine.evaluateIteration(loader=test.loader, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/{}.pkl'.format(title))
pass

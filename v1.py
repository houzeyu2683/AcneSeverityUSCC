
import bucket
import network
import metric

configuration = bucket.v1.loadYaml(path='configuration.yaml')
pass

Train      = bucket.v1.createClass('Train')
Validation = bucket.v1.createClass('Validation')
Test       = bucket.v1.createClass('Test')
train, validation, test = Train(), Validation(), Test()
pass

train.set      = bucket.v1.Set(configuration, 'train')
validation.set = bucket.v1.Set(configuration, 'validation')
test.set       = bucket.v1.Set(configuration, 'test')
pass

train.set.LoadData()
validation.set.LoadData()
test.set.LoadData()
pass

train.loader      = bucket.v1.createLoader(set=train.set, batch=32, inference=False, device='cuda')
validation.loader = bucket.v1.createLoader(set=validation.set, batch=16, inference=True, device='cuda')
test.loader       = bucket.v1.createLoader(set=test.set, batch=16, inference=True, device='cuda')
pass

train.sample      = bucket.v1.getSample(train.loader)
validation.sample = bucket.v1.getSample(validation.loader)
test.sample       = bucket.v1.getSample(test.loader)
pass

model = network.v1.Model(backbone='resnet', classification=2, device='cuda')
machine = network.v1.Machine(model=model)
machine.defineOptimization(method='adam')
pass

checkpoint = bucket.v1.loadYaml(path='checkpoint.yaml')
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
bucket.v1.saveYaml(content=history, path="{}/history.yaml".format(checkpoint['path']))
history = bucket.v1.loadYaml(path="{}/history.yaml".format(checkpoint['path']))
pass

##  Find and save the best checkpoint.
track = history[best['focus']][best['metric']]
best['checkpoint'] = track.index(max(track))
source = "{}/{}/".format(checkpoint['path'], best['checkpoint'])
destination = "{}/{}/".format(checkpoint['path'], 'best')
bucket.v1.copyFolder(source=source, destination=destination)
pass

##  Save the extraction.
train.loader = bucket.v1.createLoader(set=train.set, batch=1, inference=True, device='cuda')
validation.loader = bucket.v1.createLoader(set=validation.set, batch=1, inference=True, device='cuda')
test.loader = bucket.v1.createLoader(set=test.set, batch=1, inference=True, device='cuda')
machine = network.v1.Machine(model=None)
machine.loadModel(path="{}/best/model.pt".format(checkpoint['path']), device='cuda')
pass

title = 'train'
feedback = machine.evaluateIteration(loader=train.loader, title=title)
bucket.v1.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/{}.pkl'.format(title))
pass

title = 'validation'
feedback = machine.evaluateIteration(loader=validation.loader, title=title)
bucket.v1.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/{}.pkl'.format(title))
pass

title = 'test'
feedback = machine.evaluateIteration(loader=test.loader, title=title)
bucket.v1.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/{}.pkl'.format(title))
pass

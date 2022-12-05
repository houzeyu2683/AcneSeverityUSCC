
import bucket
import network
import metric

configuration = bucket.v3.loadYaml(path='configuration.yaml')
configuration = configuration['v3']
device = configuration['device']
pass

Train      = bucket.v3.createClass('Train')
Validation = bucket.v3.createClass('Validation')
Test       = bucket.v3.createClass('Test')
train, validation, test = Train(), Validation(), Test()
pass

train.set      = bucket.v3.Set(configuration, 'train')
validation.set = bucket.v3.Set(configuration, 'validation')
test.set       = bucket.v3.Set(configuration, 'test')
pass

train.set.LoadData()
validation.set.LoadData()
test.set.LoadData()
pass

train.loader      = bucket.v3.createLoader(set=train.set, batch=32, inference=False, device=device)
validation.loader = bucket.v3.createLoader(set=validation.set, batch=16, inference=True, device=device)
test.loader       = bucket.v3.createLoader(set=test.set, batch=16, inference=True, device=device)
pass

train.sample      = bucket.v3.getSample(train.loader)
validation.sample = bucket.v3.getSample(validation.loader)
test.sample       = bucket.v3.getSample(test.loader)
pass

model = network.v3.Model(device=device)
machine = network.v3.Machine(model=model)
machine.defineOptimization(method='adam')
pass

checkpoint = bucket.v3.loadYaml(path='checkpoint.yaml')
checkpoint = checkpoint['v3']
history    = checkpoint['history']
best       = checkpoint['best']
pass

epoch = 5
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
    history[title]['cost']['loss']     += [feedback.cost['loss']]
    pass

    ##  Evaluate validation data.
    title = 'validation'
    feedback = machine.evaluateIteration(loader=validation.loader, title=title)
    category = metric.Category(score=feedback.score, prediction=feedback.prediction, target=feedback.target)
    history[title]['accuracy']         += [category.getAccuracy()]
    history[title]['confusion table']  += [category.getConfusionTable()]
    history[title]['area under curve'] += [category.getAreaUnderCurve()]
    history[title]['cost']['loss']     += [feedback.cost['loss']]
    pass

    ##  Evaluate test data.
    title = 'test'
    feedback = machine.evaluateIteration(loader=test.loader, title=title)
    category = metric.Category(score=feedback.score, prediction=feedback.prediction, target=feedback.target)
    history[title]['accuracy']         += [category.getAccuracy()]
    history[title]['confusion table']  += [category.getConfusionTable()]
    history[title]['area under curve'] += [category.getAreaUnderCurve()]
    history[title]['cost']['loss']     += [feedback.cost['loss']]
    continue

##  Find the best.
track = history[best['focus']][best['metric']]
best['checkpoint'] = track.index(max(track))
pass

##  Save the history.
bucket.v3.saveYaml(content=history, path="{}/history.yaml".format(checkpoint['path']))
history = bucket.v3.loadYaml(path="{}/history.yaml".format(checkpoint['path']))
pass

##  Save the best.
bucket.v3.saveYaml(content=best, path="{}/best.yaml".format(checkpoint['path']))
source = "{}/{}/".format(checkpoint['path'], best['checkpoint'])
destination = "{}/{}/".format(checkpoint['path'], 'best')
bucket.v3.copyFolder(source=source, destination=destination)
pass

'''
##  Reload the model and save the necessary thing.
train.loader = bucket.v3.createLoader(set=train.set, batch=1, inference=True, device='cpu')
validation.loader = bucket.v3.createLoader(set=validation.set, batch=1, inference=True, device='cuda')
test.loader = bucket.v3.createLoader(set=test.set, batch=1, inference=True, device='cpu')
machine = network.v3.Machine(model=None)
machine.loadModel(path="{}/best/model.pt".format(checkpoint['path']), device='cpu')
pass

title = 'train'
feedback = machine.evaluateIteration(loader=train.loader, title=title)
bucket.v1.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/V1/{}.pkl'.format(title))
pass

title = 'validation'
feedback = machine.evaluateIteration(loader=validation.loader, title=title)
bucket.v1.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/V1/{}.pkl'.format(title))
pass

title = 'test'
feedback = machine.evaluateIteration(loader=test.loader, title=title)
bucket.v1.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/V1/{}.pkl'.format(title))
pass
'''

import bucket
import network
import metric

configuration = bucket.v2.loadYaml(path='configuration.yaml')
configuration = configuration['v2']
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

train.loader      = bucket.createLoader(set=train.set, batch=32, inference=False, device='cuda')
validation.loader = bucket.createLoader(set=validation.set, batch=16, inference=True, device='cuda')
test.loader       = bucket.createLoader(set=test.set, batch=16, inference=True, device='cuda')
pass

train.sample      = bucket.getSample(train.loader)
validation.sample = bucket.getSample(validation.loader)
test.sample       = bucket.getSample(test.loader)
pass

model = network.v2.Model(device='cuda')
machine = network.v2.Machine(model=model)
machine.defineOptimization(method='adam')
pass

checkpoint = bucket.loadYaml(path='checkpoint.yaml')
checkpoint = checkpoint['v2']
history    = checkpoint['history']
best       = checkpoint['best']
pass

epoch = 20
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
    history[title]['cost']['loss']           += [feedback.cost['loss']]
    history[title]['cost']['divergence']     += [feedback.cost['divergence']]
    history[title]['cost']['reconstruction'] += [feedback.cost['reconstruction']]
    history[title]['cost']['match']          += [feedback.cost['match']]
    pass

    ##  Evaluate validation data.
    title = 'validation'
    feedback = machine.evaluateIteration(loader=validation.loader, title=title)
    history[title]['cost']['loss']           += [feedback.cost['loss']]
    history[title]['cost']['divergence']     += [feedback.cost['divergence']]
    history[title]['cost']['reconstruction'] += [feedback.cost['reconstruction']]
    history[title]['cost']['match']          += [feedback.cost['match']]
    pass

    ##  Evaluate test data.
    title = 'test'
    feedback = machine.evaluateIteration(loader=test.loader, title=title)
    history[title]['cost']['loss']           += [feedback.cost['loss']]
    history[title]['cost']['divergence']     += [feedback.cost['divergence']]
    history[title]['cost']['reconstruction'] += [feedback.cost['reconstruction']]
    history[title]['cost']['match']          += [feedback.cost['match']]
    continue

##  Find the best checkpoint.
track = history[best['focus']]['cost'][best['metric']]
best['checkpoint'] = track.index(min(track))
pass

##  Save the history.
bucket.v2.saveYaml(content=history, path="{}/history.yaml".format(checkpoint['path']))
history = bucket.loadYaml(path="{}/history.yaml".format(checkpoint['path']))
pass

##  Save the best.
bucket.v2.saveYaml(content=best, path="{}/best.yaml".format(checkpoint['path']))
source = "{}/{}/".format(checkpoint['path'], best['checkpoint'])
destination = "{}/{}/".format(checkpoint['path'], 'best')
bucket.v2.copyFolder(source=source, destination=destination)
pass

##  Reload the model and save the necessary thing.
train.loader = bucket.v2.createLoader(set=train.set, batch=1, inference=True, device='cuda')
validation.loader = bucket.v2.createLoader(set=validation.set, batch=1, inference=True, device='cuda')
test.loader = bucket.v2.createLoader(set=test.set, batch=1, inference=True, device='cuda')
machine = network.v2.Machine(model=None)
machine.loadModel(path="{}/best/model.pt".format(checkpoint['path']), device='cuda')
pass

title = 'train'
feedback = machine.evaluateIteration(loader=train.loader, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/V2/{}.pkl'.format(title))
pass

title = 'validation'
feedback = machine.evaluateIteration(loader=validation.loader, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/V2/{}.pkl'.format(title))
pass

title = 'test'
feedback = machine.evaluateIteration(loader=test.loader, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Feedback/V2/{}.pkl'.format(title))
pass


import bucket
import network
import metric

configuration = bucket.v2.loadYaml(path='configuration.yaml')
configuration = configuration['v2']
device = configuration['device']
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

train.loader      = bucket.v2.createLoader(set=train.set, batch=32, inference=False, device=device)
validation.loader = bucket.v2.createLoader(set=validation.set, batch=16, inference=True, device=device)
test.loader       = bucket.v2.createLoader(set=test.set, batch=16, inference=True, device=device)
pass

train.sample      = bucket.v2.getSample(train.loader)
validation.sample = bucket.v2.getSample(validation.loader)
test.sample       = bucket.v2.getSample(test.loader)
pass

model = network.v2.Model(device=device)
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

    loop = zip([train.loader, validation.loader, test.loader], ['train', 'validation', 'test'])
    for loader, title in loop:
 
        ##  Evaluate process.
        feedback = machine.evaluateIteration(loader=train.loader, title=title)
        cost = feedback.cost
        history[title]['cost']['iteration']['loss']           += cost['iteration']['loss']
        history[title]['cost']['iteration']['divergence']     += cost['iteration']['divergence']
        history[title]['cost']['iteration']['reconstruction'] += cost['iteration']['reconstruction']
        history[title]['cost']['iteration']['projection']     += cost['iteration']['projection']
        history[title]['cost']['epoch']['loss']            += [cost['epoch']['loss']]
        history[title]['cost']['epoch']['divergence']      += [cost['epoch']['divergence']]
        history[title]['cost']['epoch']['reconstruction']  += [cost['epoch']['reconstruction']]
        history[title]['cost']['epoch']['projection']      += [cost['epoch']['projection']]
        continue

    continue

##  Find the best checkpoint.
track = history[best['focus']]['cost']['epoch'][best['metric']]
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
train.loader = bucket.v2.createLoader(set=train.set, batch=1, inference=True, device=device)
validation.loader = bucket.v2.createLoader(set=validation.set, batch=1, inference=True, device=device)
test.loader = bucket.v2.createLoader(set=test.set, batch=1, inference=True, device=device)
machine = network.v2.Machine(model=None)
machine.loadModel(path="{}/best/model.pt".format(checkpoint['path']), device=device)
pass

title = 'train'
feedback = machine.evaluateIteration(loader=train.loader, title=title)
bucket.savePickle(feedback.information, path='resource/ACNE04/Feedback/V2/{}.pkl'.format(title))
pass

title = 'validation'
feedback = machine.evaluateIteration(loader=validation.loader, title=title)
bucket.savePickle(feedback.information, path='resource/ACNE04/Feedback/V2/{}.pkl'.format(title))
pass

title = 'test'
feedback = machine.evaluateIteration(loader=test.loader, title=title)
bucket.savePickle(feedback.information, path='resource/ACNE04/Feedback/V2/{}.pkl'.format(title))
pass

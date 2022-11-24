
import bucket
import network
import metric

environment = bucket.loadYaml(path='./environment.yaml')
pass

Dataset = bucket.createClass('Train')
dataset = Dataset()
dataset.train      = bucket.Set(environment, 'train')
dataset.validation = bucket.Set(environment, 'validation')
dataset.test       = bucket.Set(environment, 'test')
pass

dataset.train.LoadData()
dataset.validation.LoadData()
dataset.test.LoadData()
pass

dataset.train. = bucket.loadPickle(path=data['train']['extraction'])

Loader = bucket.createClass(name='Loader')
loader = Loader()
loader.train      = bucket.createLoader(dataset=dataset.train, batch=32, inference=False, device='cuda')
loader.validation = bucket.createLoader(dataset=dataset.validation, batch=16, inference=True, device='cuda')
loader.test       = bucket.createLoader(dataset=dataset.test, batch=16, inference=True, device='cuda')
pass

Sample = bucket.createClass(name='sample')
sample = Sample()
sample.train      = bucket.getSample(loader.train)
sample.validation = bucket.getSample(loader.validation)
sample.test       = bucket.getSample(loader.test)
pass

model = network.v1.Model(backbone='resnet', classification=2, device='cuda')
machine = network.v1.Machine(model=model)
machine.defineOptimization(method='adam')
pass

checkpoint = environment['checkpoint']
history    = checkpoint['history']
best       = checkpoint['best']
pass

epoch = 5
loop = range(epoch)
for iteration in loop:

    ##  Learning process.
    _ = machine.learnIteration(loader=loader.train)
    machine.saveModel(path='{}/{}/model.pt'.format(checkpoint['path'], iteration))
    machine.saveWeight(path='{}/{}/weight.pt'.format(checkpoint['path'], iteration))    
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
loader.train = bucket.createLoader(dataset=dataset.train, batch=1, inference=True, device='cuda')
loader.validation = bucket.createLoader(dataset=dataset.validation, batch=1, inference=True, device='cuda')
loader.test = bucket.createLoader(dataset=dataset.test, batch=1, inference=True, device='cuda')
machine = network.v1.Machine(model=None)
machine.loadModel(path="{}/best/model.pt".format(checkpoint['path']), device='cuda')
pass

title = 'train'
feedback = machine.evaluateIteration(loader=loader.train, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Extraction/{}.pkl'.format(title))
pass

title = 'validation'
feedback = machine.evaluateIteration(loader=loader.validation, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Extraction/{}.pkl'.format(title))
pass

title = 'test'
feedback = machine.evaluateIteration(loader=loader.test, title=title)
bucket.savePickle(feedback.convertDictionary(), path='resource/ACNE04/Extraction/{}.pkl'.format(title))
pass

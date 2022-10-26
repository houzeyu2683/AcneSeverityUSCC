
import data
import network

train = "resource/v2/csv/train.csv"
validation = None
test = "resource/v2/csv/test.csv"

table = data.table(train=train, validation=validation, test=test)
table.load()
table.hold(size=0.2, stratification='vote')

dataset = table.export(format='dataset')
loader = data.loader(batch=8, device='cuda')
loader.define(train=dataset.train, validation=dataset.validation, test=dataset.test)

model = network.model(backbone='mobilenet', classification=2)
machine = network.machine(model=model, device='cuda')
machine.optimization(method='adam')

epoch = 100
loop = range(epoch)
version = "0930(mobilenet)"
summary   = {'train':[], 'validation':[], "test":[]}
iteration = {'train':[], 'validation':[], "test":[]}
report    = {'train':[], 'validation':[], "test":[]}
for step in loop:

    _ = machine.learn(train=loader.train)
    node = 5
    if(step%node==node-1):

        s = machine.evaluate(train=loader.train)
        summary['train']   += [s]
        report['train']    += [s['train report']]
        s = machine.evaluate(validation=loader.validation)
        summary['validation']   += [s]
        report['validation']    += [s['validation report']]
        s = machine.evaluate(test=loader.test)
        summary['test']   += [s]
        report['test']    += [s['test report']]
        machine.save('./log-{}/{}-checkpoint/model.pt'.format(version, step))
        machine.write(text=report['train'][-1], path='./log-{}/{}-checkpoint/train report.txt'.format(version, step))
        machine.write(text=report['validation'][-1], path='./log-{}/{}-checkpoint/validation report.txt'.format(version, step))
        machine.write(text=report['test'][-1], path='./log-{}/{}-checkpoint/test report.txt'.format(version, step))
        pass

    continue

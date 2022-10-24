
import data
import network

train = "~/Desktop/Projects/Classification/AcneSeverity/resource/txt/tr.txt"
test = "~/Desktop/Projects/Classification/AcneSeverity/resource/txt/te.txt"

table = data.table(train=train, test=test)
table.hold(size=0.1, stratification=None)
dataset = table.export(format='dataset')

loader = data.loader(batch=8, device='cuda')
loader.define(train=dataset.train, validation=dataset.validation, test=dataset.test)

model = network.model(backbone='mobilenet')
machine = network.machine(model=model, device='cuda')
# batch = machine.model.cost(next(iter(loader.train)))
machine.optimization(method='adam')

epoch = 20
loop = range(epoch)
date = "0930(mobilenet)"
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
        machine.save('./log-{}/{}-checkpoint/model.pt'.format(date, step))
        machine.write(text=report['train'][-1], path='./log-{}/{}-checkpoint/train report.txt'.format(date, step))
        machine.write(text=report['validation'][-1], path='./log-{}/{}-checkpoint/validation report.txt'.format(date, step))
        machine.write(text=report['test'][-1], path='./log-{}/{}-checkpoint/test report.txt'.format(date, step))
        pass

    continue

# for i in range(200):
#     print(i%10==9)
#     continue


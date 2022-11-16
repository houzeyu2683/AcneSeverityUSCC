
import data
import network
import metric

sheet = data.Sheet(
    train='./resource/ACNE04/Attribution/train-feature.pkl', 
    validation='./resource/ACNE04/Attribution/validation-feature.pkl', 
    test="./resource/ACNE04/Attribution/test-feature.pkl"
)
sheet.loadDictionary()

engine = data.Engine(train=sheet.train, validation=sheet.validation, test=sheet.test)
engine.defineDataset()
engine.defineLoader(batch=32, device='cuda')
loader = engine.loader
batch = engine.getSample()

model = network.v2.Model(device='cuda')
# value = model.forward(batch)
# model.cost(value)

machine = network.v2.machine(model=model)
machine.defineOptimization(method='adam')

epoch = 20
loop = range(epoch)
# history = {
#     'test':{'accuracy':[], 'report':[], 'auc':[], 'confusion':[]}, 
#     'validation':{'accuracy':[], 'report':[], 'auc':[], 'confusion':[]}
# }
history = []
for iteration in loop:

    feedback = machine.learnIteration(loader=engine.loader.train)
    history += [dict(feedback.__dict__)]
    machine.writeText(history, './output/history.txt')
    machine.saveModel(path='./output/checkpoint-{}/acne-vae.pt'.format(iteration))
    # pass

    # feedback = machine.evaluateIteration(loader=engine.loader.validation, title='validation')
    # classification = metric.Classification(
    #     score=feedback.score, 
    #     prediction=feedback.prediction, 
    #     target=feedback.target
    # )
    # history[feedback.title]['accuracy'] += [classification.getAccuracy()]
    # history[feedback.title]['report'] += [classification.getReport()]
    # history[feedback.title]['auc'] += [classification.getAreaUnderCurve()]
    # history[feedback.title]['confusion'] += [classification.getConfusionTable()]
    # pass

    # feedback = machine.evaluateIteration(loader=engine.loader.test, title='test')
    # classification = metric.Classification(
    #     score=feedback.score, 
    #     prediction=feedback.prediction, 
    #     target=feedback.target
    # )
    # history[feedback.title]['accuracy'] += [classification.getAccuracy()]
    # history[feedback.title]['report'] += [classification.getReport()]
    # history[feedback.title]['auc'] += [classification.getAreaUnderCurve()]
    # history[feedback.title]['confusion'] += [classification.getConfusionTable()]
    # pass

    # machine.writeText(history, './output/history.txt')
    continue



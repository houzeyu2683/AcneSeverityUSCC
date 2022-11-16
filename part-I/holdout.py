
import data
import network
import metric

sheet = data.Sheet(
    train='./resource/ACNE04/Classification/NNEW_trainval_0.csv', 
    validation=None, 
    test="./resource/ACNE04/Classification/NNEW_test_0.csv"
)
sheet.loadTable()
sheet.splitValidation(percentage=0.2, stratification='label')

engine = data.Engine(train=sheet.train, validation=sheet.validation, test=sheet.test)
engine.defineDataset()
engine.defineLoader(batch=32, device='cuda')
loader = engine.loader
# batch = engine.getSample()

model = network.v1.Model(backbone='resnet', classification=2, device='cuda')
machine = network.v1.machine(model=model)
machine.defineOptimization(method='adam')

epoch = 20
loop = range(epoch)
history = {
    'test':{'accuracy':[], 'report':[], 'auc':[], 'confusion':[]}, 
    'validation':{'accuracy':[], 'report':[], 'auc':[], 'confusion':[]}
}
for iteration in loop:

    _ = machine.learnIteration(loader=engine.loader.train)
    machine.saveModel(path='./output/checkpoint-{}/acne-classifier.pt'.format(iteration))
    pass

    feedback = machine.evaluateIteration(loader=engine.loader.validation, title='validation')
    classification = metric.Classification(
        score=feedback.score, 
        prediction=feedback.prediction, 
        target=feedback.target
    )
    history[feedback.title]['accuracy'] += [classification.getAccuracy()]
    history[feedback.title]['report'] += [classification.getReport()]
    history[feedback.title]['auc'] += [classification.getAreaUnderCurve()]
    history[feedback.title]['confusion'] += [classification.getConfusionTable()]
    pass

    feedback = machine.evaluateIteration(loader=engine.loader.test, title='test')
    classification = metric.Classification(
        score=feedback.score, 
        prediction=feedback.prediction, 
        target=feedback.target
    )
    history[feedback.title]['accuracy'] += [classification.getAccuracy()]
    history[feedback.title]['report'] += [classification.getReport()]
    history[feedback.title]['auc'] += [classification.getAreaUnderCurve()]
    history[feedback.title]['confusion'] += [classification.getConfusionTable()]
    pass

    machine.writeText(history, './output/history.txt')
    continue



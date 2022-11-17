
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
history = {
    'train':{'loss_total':[], 'loss_kl':[], 'loss_rec':[]}
}
# history = []
for iteration in loop:

    train_feedback = machine.learnIteration(loader=engine.loader.train)
    history['train']['loss_total'] += [train_feedback.loss_total]
    history['train']['loss_kl']    += [train_feedback.loss_kl]
    history['train']['loss_rec']   += [train_feedback.loss_rec]
    machine.writeText(history, './output/history.txt')
    machine.saveModel(path='./output/checkpoint-{}/acne-vae-weight.pt'.format(iteration))
    continue


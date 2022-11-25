
import data
import network

sheet = data.Sheet(
    train='./resource/ACNE04/Classification/NNEW_trainval_0.csv', 
    validation=None, 
    test="./resource/ACNE04/Classification/NNEW_test_0.csv"
)
sheet.loadTable()
sheet.splitValidation(percentage=0.2, stratification='label')

engine = data.Engine(train=sheet.train, validation=sheet.validation, test=sheet.test)
engine.defineDataset()
engine.defineLoader(batch=1, device='cuda')
loader = engine.loader

model = network.v1.Model(backbone='resnet', classification=2, device='cuda')
machine = network.v1.machine(model=model)
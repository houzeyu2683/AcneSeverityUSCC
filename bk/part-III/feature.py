
import data
import network
import metric

import pandas
import pickle
import numpy
import tqdm

def runExtraction(loader, model, embedding):

    extraction = {}
    extraction['index'] = []
    extraction['feature'] = []
    extraction['attribution'] = []
    extraction['target'] = []
    
    for batch in tqdm.tqdm(loader):

        index = batch.index
        target = batch.target.detach().numpy().flatten().tolist()
        feature = model.getExtraction(batch).detach().numpy()
        attribution = embedding.iloc[target, :].values
        pass

        extraction['index'] += index
        extraction['target'] += target
        extraction["feature"] += [feature]
        extraction['attribution'] += [attribution]
        continue
    
    extraction['feature'] = numpy.concatenate(extraction['feature'], axis=0)
    extraction['attribution'] = numpy.concatenate(extraction['attribution'], axis=0)
    return(extraction)

def saveExtraction(extraction, path=None):

    with open(path, 'wb') as paper:

        pickle.dump(extraction, paper)
        pass

    return

sheet = data.Sheet(
    train='./resource/ACNE04/Classification/NNEW_trainval_0.csv', 
    validation=None, 
    test="./resource/ACNE04/Classification/NNEW_test_0.csv"
)
sheet.loadTable()
sheet.splitValidation(percentage=0.2, stratification='label')

engine = data.Engine(train=sheet.train, validation=sheet.validation, test=sheet.test)
engine.defineDataset()
engine.defineLoader(batch=1, device='cpu', augmentation=False)
loader = engine.loader

machine = network.v1.machine(model=None)
machine.loadModel(path='./output/resnet/checkpoint-19/acne-classifier.pt', device='cpu')
model = machine.model

embedding = pandas.read_csv('./resource/ACNE04/Attribution/embedding_17-10-13-636644.csv')

extraction = runExtraction(loader=loader.train, model=model, embedding=embedding)
saveExtraction(extraction=extraction, path='output/train-feature.pkl')

extraction = runExtraction(loader=loader.validation, model=model, embedding=embedding)
saveExtraction(extraction=extraction, path='output/validation-feature.pkl')

extraction = runExtraction(loader=loader.test, model=model, embedding=embedding)
saveExtraction(extraction=extraction, path='output/test-feature.pkl')


import pandas
import os
import pickle
import torch
import yaml

def getEnvrionment(path='environment.yaml'):

    with open(path) as paper:
        
        environment = yaml.load(paper, yaml.loader.SafeLoader)
        pass

    return(environment)

def getModel(path=None, device='cpu'):

    model = torch.load(path, map_location=device)
    return(model)

def getExtractiveLayer(path='./output/resnet/checkpoint-19/acne-classifier.pt', backbone='resnet', device='cpu'):

    model = getModel(path=path, device=device)
    pass

    if(backbone=='resnet'):

        layer = model.layer['0']
        pass

    return(layer)

def getExtractiveFeature(model=None, )

def saveExtraction(extraction=None, path='./output/***.pkl'):

    ##  儲存 2048 feature
    return

if(__name__=='__main__'):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='./environment.yaml', help='environment file', type=str)
    parser.add_argument('--output', default='./output/', help='output folder', type=str)
    parser.add_argument('--env', default='./environment.yaml', help='environment file', type=str)
    getEnvrionment(path='environment.yaml')


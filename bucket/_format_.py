
import yaml
import json
import pickle
import os
import pprint
import shutil

def loadYaml(path='environment.yaml'):

    with open(path) as paper:
        
        output = yaml.load(paper, yaml.loader.SafeLoader)
        pass

    return(output)

def loadJson(path):

    with open(path) as paper:
        
        output = json.loads(paper)
        pass

    return(output)

def loadPickle(path):

    with open(path, 'rb') as paper:

        output = pickle.load(paper)
        pass

    return(output)

def savePickle(content, path):

    if(not os.path.exists(path)): os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as paper:

        output = pickle.dump(content, paper)
        pass

    return(output)

def writeText(content, path):

    text = pprint.pformat(content)
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w') as paper: _ = paper.write(text) 
    return

def copyFolder(start, end):

    exist = os.path.exists(end)
    if(exist): shutil.rmtree(end)
    shutil.copytree(start, end)
    return

def saveYaml(content, path):

    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    with open(path, 'w') as paper: _ = yaml.dump(content, paper, default_flow_style=None)
    return


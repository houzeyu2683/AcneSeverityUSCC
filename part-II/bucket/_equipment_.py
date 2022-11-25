
import yaml
import pickle

def getThing(path):

    with open(path, 'rb') as paper:

        thing = pickle.load(paper)
        pass

    return(thing)

def createThing(name='thing'):

    thing = type(name, (type,), {})
    return(thing)

def getEnvironment(path):

    with open(path, 'r') as paper:

        environment = yaml.load(paper, yaml.SafeLoader)
        pass

    return(environment)

class Structure:

    def __init__(self, name=None):
        
        self.name = name
        return

    def insertSomething(self, **something):

        return

    def insertDictionary(self, dictionary=None):

        return

    pass    

class Data:

    def __init__(self):

        return        
    
    def createSet(self):

        return

    def createLoader(self):

        return

    pass




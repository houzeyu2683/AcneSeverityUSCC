
import pickle

def getStructure(path):

    with open(path, 'rb') as paper:

        structure = pickle.load(paper)
        pass

    return(structure)

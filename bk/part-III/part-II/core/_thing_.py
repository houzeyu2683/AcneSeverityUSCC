
import pickle

def getThing(path):

    with open(path, 'rb') as paper:

        thing = pickle.load(paper)
        pass

    return(thing)

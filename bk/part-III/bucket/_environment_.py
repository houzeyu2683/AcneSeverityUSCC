
import yaml

def loadEnvironment(path):

    with open(path, 'r') as paper:

        environment = yaml.load(paper, yaml.SafeLoader)
        pass

    return(environment)

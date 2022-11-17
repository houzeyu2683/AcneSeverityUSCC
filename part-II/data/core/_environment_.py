
import yaml

def getEnvrionment(path='environment.yaml'):

    with open(path) as paper:
        
        environment = yaml.load(paper, yaml.loader.SafeLoader)
        pass

    return(environment)
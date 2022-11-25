

##
##  Packages.
import torch, pickle


# ##
# ##
# path='SOURCE/PICKLE/VOCABULARY.pickle'
# with open(path, 'rb') as paper:

#     vocabulary = pickle.load(paper)
#     pass


##
##
class criterion:

    ##  Cross entropy loss.
    def cel():
        
        loss = torch.nn.CrossEntropyLoss()
        return(loss)




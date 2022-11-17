
import os
import pickle
from torch.utils.data import DataLoader

class loader:

    def __init__(self, train=None, exam=None, test=None, batch=32):

        if(train):
            
            self.train = DataLoader(train, batch_size=batch, shuffle=True , drop_last=True)
            pass

        if(exam):
            
            self.exam = DataLoader(exam, batch_size=64, shuffle=False , drop_last=False)
            pass

        if(test):

            self.test  = DataLoader(test , batch_size=64, shuffle=False, drop_last=False)
            pass

        pass
    
    def save(self, what='batch', folder='./'):

        if(what=='batch'):

            name = 'BATCH.pickle'
            path = os.path.join(folder, name)
            batch = []
            batch += [next(iter(self.train))] if(self.train) else [None]
            batch += [next(iter(self.train))] if(self.exam ) else [None]
            batch += [next(iter(self.train))] if(self.test ) else [None]            
            batch = tuple(batch)
            with open(path, 'wb') as paper:

                pickle.dump(batch, paper, protocol=pickle.HIGHEST_PROTOCOL)
                pass

            pass

        pass
    
    pass



# a = {'hello': 'world'}

    # def sample(self, collection):

    #     batch = {
    #         'text':[],
    #         "author":[],
    #         "target":[]
    #     }
    #     for _, (text, author, target) in enumerate(collection):

    #         batch['text'] += [torch.tensor(text, dtype=torch.long)]
    #         batch['author'] += [torch.tensor(author, dtype=torch.long)]
    #         batch['target'] += [torch.tensor(target, dtype=torch.long)]
    #         pass

    #     batch['text'] = torch.nn.utils.rnn.pad_sequence(batch['text'], padding_value=self.vocabulary['<pad>'])
    #     batch['author'] = torch.nn.utils.rnn.pad_sequence(batch['author'], padding_value=self.vocabulary['<pad>'])
    #     batch['target'] = torch.tensor(batch['target'])
    #     output = list(batch.values())
    #     return(output)
            # target = torch.tensor(target, dtype=torch.long)            
            # batch['target'] += [target]
            # pass

            # index = [self.vocabulary[i] for i in token]
            # # index = torch.tensor(index, dtype=torch.long)
            # batch['index'] += index

            # point = len(index)
            # batch['point'] += [point]
            # pass

        # batch['index']   = torch.nn.utils.rnn.pad_sequence(batch['index'], padding_value=vocabulary['<pad>'])
        # batch['target']  = torch.tensor(batch['target'])
    #     batch['index'] = torch.tensor(batch['index'], dtype=torch.long)
    #     batch['target'] = torch.tensor(batch['target'], dtype=torch.long)
    #     batch['point'] = torch.tensor(batch['point'][:-1]).cumsum(dim=0)
    #     return(batch['index'], batch['target'], batch['point'])

    # pass


# index = [vocabulary['<bos>']] + [vocabulary[i] for i in token] + [vocabulary['<eos>']]
    # def load(self, what='vocabulary', path=None):

    #     if(what=='vocabulary'):

    #         with open(path, 'rb') as paper:

    #             vocabulary = pickle.load(paper)
    #             pass

    #         self.vocabulary = vocabulary
    #         pass

    #     pass

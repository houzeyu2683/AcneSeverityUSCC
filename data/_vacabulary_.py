
import itertools
import collections

class vocabulary:

    def __init__(self, text=["the first content", "the second content"], default=["<PAD>", "<CLS>", "<UNK>", "<SOS>","<EOS>"]):

        self.text = text
        self.default = default
        pass
    
    def tokenize(self, content='the content'):

        token = content.split(" ")
        return(token)

    def build(self):
        
        token = map(self.tokenize, self.text)
        token = itertools.chain(*token)
        token = list(token)
        count = collections.Counter(token)
        index = self.default
        index = index + [c for c in count]
        index = {t:i for i, t in enumerate(index)}
        self.count  = count
        self.index  = index
        self.word   = {i:t for t, i in self.index.items()}
        self.length = len(count) + len(self.default)
        return

    def encode(self, content="the first content"):

        index = []
        index += [self.index['<CLS>'], self.index['<SOS>']]
        for t in self.tokenize(content):

            if(t in self.index.keys()): index += [self.index[t]]
            if(t not in self.index.keys()): index += [self.index['<UNK>']]
            continue
        
        index += [self.index['<EOS>']]
        return(index)

    def decode(self, index=[3,5,6,7,4]):
        
        token = [self.word[i] for i in index]
        return(token)

    pass

# import torchtext
# def tokenize(self, content='the content'):

#     engine = torchtext.data.utils.get_tokenizer('basic_english')
#     token = engine(content)
#     return(token)
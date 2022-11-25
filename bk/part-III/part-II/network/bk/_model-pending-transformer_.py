
import torch
import math

class mask:

    length = 17
    batch = 6
    tensor = torch.randint(0, 100, (length, batch))
    pass

    def padding(tensor=tensor, value=0):

        matrix = (tensor==value).transpose(0,1)
        if(tensor.is_cuda): matrix = matrix.cuda()
        return(matrix)

    def recursion(tensor=tensor):

        shape = (len(tensor), )*2
        matrix = torch.triu(torch.full(shape, float('-inf')), diagonal=1)
        if(tensor.is_cuda): matrix = matrix.cuda()
        return(matrix)

    pass

class position(torch.nn.Module):

    def __init__(self):

        super().__init__()
        length = 50000
        embedding = 256
        location = torch.arange(length).unsqueeze(1)
        term = torch.exp(torch.arange(0, embedding, 2) * (-math.log(10000.0) / embedding))
        code = torch.zeros(length, 1, embedding)
        code[:, 0, 0::2] = torch.sin(location * term)
        code[:, 0, 1::2] = torch.cos(location * term)
        self.code = code
        return

    def forward(self, tensor):

        tensor = tensor + self.code[:tensor.size(0)]
        return(tensor)

    pass

class model(torch.nn.Module):

    def __init__(self, vocabulary=None):

        super().__init__()
        self.vocabulary = vocabulary
        layer = {
            "0":torch.nn.Embedding(num_embeddings=150000, embedding_dim=256),
            '1':position(),
            "2":torch.nn.TransformerEncoder(
                encoder_layer=torch.nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=2048, dropout=0.1),
                num_layers=1,
                norm=torch.nn.LayerNorm(normalized_shape=256)
            ),
            '3':torch.nn.Sequential(
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(128),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(64),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(32),
                torch.nn.Linear(32, 4),
                torch.nn.Softmax(1),
            )
        }
        self.layer = torch.nn.ModuleDict(layer)
        return

    def forward(self, batch):

        layer = self.layer
        vocabulary = self.vocabulary
        block = mask.padding(batch.x, vocabulary.index['<PAD>'])
        pass

        x = layer['0'](batch.x)
        x = layer['1'](x)
        x = layer['2'](src = x, mask = None, src_key_padding_mask = block)
        s = layer['3'](x[0,:,:])
        pass

        score = s
        return(score)

    def cost(self, batch):

        class loss:
    
            function = torch.nn.CrossEntropyLoss()
            pass
        
        batch.s = self.forward(batch)
        loss.score = loss.function(batch.s, batch.y)
        return(loss, batch)

    pass

# class TextClassificationModel(torch.nn.Module):

#     def __init__(self, vocabulary):
#         super(TextClassificationModel, self).__init__()
#         self.embedding = torch.nn.EmbeddingBag(150000, 64, sparse=True)
#         self.fc = torch.nn.Linear(64, 4)
#         # self.init_weights()

#     def init_weights(self):
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc.weight.data.uniform_(-initrange, initrange)
#         self.fc.bias.data.zero_()

#     def forward(self, text, offsets):
#         embedded = self.embedding(text, offsets)
#         return self.fc(embedded)


# embedding_sum = torch.nn.EmbeddingBag(10, 3, mode='sum', padding_idx=2)
# input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9], dtype=torch.long)
# offsets = torch.tensor([0,4], dtype=torch.long)
# embedding_sum(input, offsets)
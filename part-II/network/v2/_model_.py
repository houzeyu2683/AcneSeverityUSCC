
import torch
import torchvision
import torch
from torch import nn
from torch.nn import functional

def createPack(name='case'):

    assert name, 'define name please'
    class pack: pass
    pack.__qualname__ = name
    pack.__name__ = name
    return(pack)

class encoder(nn.Module):

    def __init__(self):

        super(encoder, self).__init__()
        layer = {
            'to code' : nn.Sequential(
                nn.Linear(2048, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(),
                nn.Linear(256, 27),
            ),
            'to mu': nn.Linear(27, 128),
            'to log(sigma^2)' : nn.Linear(27, 128)
        }
        self.layer = nn.ModuleDict(layer)
        pass

    def forward(self, batch):

        x = batch.feature
        l = self.layer
        pass

        code = l['to code'](x)
        mu = l['to mu'](code)
        std = l['to log(sigma^2)'](code)
        pass
        # mu = self.fc_mu(result)
        # log_var = self.fc_var(result)
        output = code, mu, std
        return(output)

    pass

##
class decoder(nn.Module):

    def __init__(self):

        super(decoder, self).__init__()
        layer = {
            "to feature" : nn.Linear(27, 256),
            "to image" : nn.Sequential(
                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(),
                nn.Linear(512, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 2048)
            )
        }
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, batch):

        x = batch.encode_feature
        l = self.layer
        feature = l['to feature'](x)
        decode_feature = l['to image'](feature)
        return(decode_feature)

##
class Model(nn.Module):

    def __init__(self, device='cpu'):

        super(Model, self).__init__()
        self.device = device
        self.encoder = encoder()
        self.decoder = decoder()
        return

    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    def reparameterize(self, value):

        std = torch.exp(0.5 * value['log(sigma^2)'])
        eps = torch.randn_like(std)
        z = eps.to(self.device) * std.to(self.device) + value['mu'].to(self.device)
        return(z)

    def forward(self, batch):

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        # mu, log_var = self.encode(input)
        # value = {
        #     "image":None,
        #     "mu":None,
        #     "log(sigma^2)":None,
        #     'reconstruction':None            
        # }
        value = {}
        value['image'] = batch.feature
        value['encode_feature'], value['mu'], value['log(sigma^2)'] = self.encoder(batch)
        batch.encode_feature = value['encode_feature']
        # mu, log_var = self.encoder_layer(x)
        z = self.reparameterize(value)
        value['reconstruction'] = self.decoder(batch)
        # return  [self.decode(z), input, mu, log_var]
        return(value)

    def cost(self, value):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        loss = {
            "kl-divergence":None,
            "reconstruction":None,
            "total":None
        }
        weight = {"kl-divergence":0.001}
        loss['reconstruction'] = functional.mse_loss(value['reconstruction'], value['image'])
        divergence = - 0.5 * torch.sum(1 + value['log(sigma^2)'] - value['mu'] ** 2 - value['log(sigma^2)'].exp(), dim = 1) 
        loss['kl-divergence'] = torch.mean(divergence, dim = 0)
        loss['total'] = loss['reconstruction'] + weight['kl-divergence'] * loss['kl-divergence']
        return(loss)

    def getCost(self, batch):

        cost = createPack(name='cost')
        value = self.forward(batch)
        cost.loss = self.cost(value)
        return(cost)
    # def generate(self, number):

    #     device = "cuda" if next(self.decoder.parameters()).is_cuda else "cpu"
    #     z = torch.randn(number, 128).to(device)
    #     samples = self.decoder(z)
    #     return samples
    #     # return self.forward(input=x)[0]

    pass






'''
class Model(torch.nn.Module):

    def __init__(self, device='cuda'):

        super().__init__()
        if(backbone=='densenet'):

            net = [i for i in torchvision.models.densenet121(weights="DenseNet121_Weights.DEFAULT").children()][:-1]
            # net = [i for i in torchvision.models.densenet121(pretrained=True).children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(torch.nn.AvgPool2d((7,7))),
                '2':torch.nn.Sequential(
                    torch.nn.Linear(1024, classification),
                    # torch.nn.LogSoftmax(dim=1)
                )
            }
            pass

        if(backbone=='resnet'):

            net = [i for i in torchvision.models.resnet152(weights='ResNet152_Weights.IMAGENET1K_V1').children()][:-1]
            # net = [i for i in torchvision.models.resnet152(pretrained=True).children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(
                    torch.nn.Linear(2048, classification)
                )
            }
            pass

        # if(backbone=='efficientnet'):

        #     # net = [i for i in torchvision.models.efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1").children()][:-1]
        #     net = [i for i in torchvision.models.efficientnet_b0(pretrained=True).children()][:-1]
        #     layer = {
        #         "0":torch.nn.Sequential(*net),
        #         '1':torch.nn.Sequential(
        #             torch.nn.Linear(1280, classification)
        #         )
        #     }
        #     pass

        # if(backbone=='mobilenet'):

        #     net = torchvision.models.MobileNetV2(num_classes=classification)
        #     layer = {
        #         "0":net
        #     }
        #     pass

        self.device = device
        self.classification = classification
        self.backbone = backbone
        self.layer = torch.nn.ModuleDict(layer).to(device)
        return

    def forward(self, batch):

        l = self.layer
        x = batch.image
        # n = batch.size
        pass

        if(self.backbone=='densenet'):

            c0 = x
            c1 = l['0'](c0)
            # c2 = l['1'](c1).squeeze()
            c2 = l['1'](c1).flatten(1, -1)
            c3 = l['2'](c2)
            s = c3
            pass

        if(self.backbone=='resnet'):

            c0 = x
            # c1 = l['0'](c0).squeeze()
            c1 = l['0'](c0).flatten(1, -1)
            c2 = l['1'](c1)
            s = c2
            pass

        # if(self.backbone=='shufflenet'):

        #     c0 = x
        #     c1 = l['0'](c0)
        #     c2 = l['1'](c1).squeeze()
        #     c3 = l['2'](c2)
        #     s = c3
        #     pass

        # if(self.backbone=='efficientnet'):

        #     c0 = x
        #     c1 = l['0'](c0).squeeze()
        #     c2 = l['1'](c1)
        #     s = c2
        #     pass


        # if(self.backbone=='mobilenet'):

        #     c0 = x
        #     c1 = l['0'](c0)
        #     s = c1
        #     pass

        # score = s.unsqueeze(dim=0) if(n==1) else s
        score = s
        return(score)

    def getExtraction(self, batch):

        l = self.layer
        x = batch.image
        pass

        if(self.backbone=='densenet'):

            c0 = x
            c1 = l['0'](c0)
            # c2 = l['1'](c1).squeeze()
            c2 = l['1'](c1).flatten(1, -1)
            c3 = l['2'](c2)
            f = c2
            s = c3
            pass

        if(self.backbone=='resnet'):

            c0 = x
            # c1 = l['0'](c0).squeeze()
            c1 = l['0'](c0).flatten(1, -1)
            c2 = l['1'](c1)
            f = c1
            s = c2
            pass
        
        _ = s
        return(f)

    def getScore(self, batch):

        # 0    1.043859
        # 1    0.833664
        # 2    2.080128
        # 3    2.424322
        score = self.forward(batch)
        return(score)

    def getCost(self, batch):
        # 0    1.043859
        # 1    0.833664
        # 2    2.080128
        # 3    2.424322
        # weight = torch.tensor([1.04, 0.83, 2.08, 2.42]).cuda()
        target = batch.target
        weight = None
        criteria  = torch.nn.CrossEntropyLoss(weight)
        score = self.getScore(batch)
        loss = criteria(score, target)
        pass
    
        cost = createPack(name='cost')
        cost.loss = loss
        cost.score = score
        return(cost)

    pass

# import pandas
# import yaml 

# with open('environment.yaml') as paper:
    
#     environment = yaml.load(paper, yaml.loader.SafeLoader)
#     pass

# class_embedding = pandas.read_csv(environment['embedding'])

# class model_v2(torch.nn.Module):

#     def __init__(self, backbone='densenet', classification=2):

#         super().__init__()
#         if(backbone=='densenet'):

#             net = [i for i in torchvision.models.densenet121(weights="DenseNet121_Weights.DEFAULT").children()][:-1]
#             layer = {
#                 "0":torch.nn.Sequential(*net),
#                 '1':torch.nn.Sequential(torch.nn.AvgPool2d((7,7))),
#                 '2':torch.nn.Sequential(
#                     torch.nn.Linear(1024, classification)
#                 )
#             }
#             pass

#         if(backbone=='resnet'):
    
#             # net = [i for i in torchvision.models.resnet101(weights="ResNet101_Weights.IMAGENET1K_V1").children()][:-1]
#             net = [i for i in torchvision.models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1").children()][:-1]
#             layer = {
#                 "0":torch.nn.Sequential(*net),
#                 '1':torch.nn.Sequential(
#                     torch.nn.Linear(2048, classification)
#                 )
#             }
#             pass

#         if(backbone=='efficientnet'):

#             net = [i for i in torchvision.models.efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1").children()][:-1]
#             layer = {
#                 "0":torch.nn.Sequential(*net),
#                 '1':torch.nn.Sequential(
#                     torch.nn.Linear(1280, classification)
#                 )
#             }
#             pass

#         if(backbone=='mobilenet'):

#             net = torchvision.models.MobileNetV2(num_classes=classification)
#             layer = {
#                 "0":net
#             }

#         self.classification = classification
#         self.backbone = backbone
#         self.layer = torch.nn.ModuleDict(layer)
#         return

#     def forward(self, batch):

#         l = self.layer
#         x = batch.image
#         pass

#         if(self.backbone=='densenet'):

#             c0 = x
#             c1 = l['0'](c0)
#             c2 = l['1'](c1).squeeze()
#             c3 = l['2'](c2)
#             p = c3
#             pass

#         if(self.backbone=='resnet'):

#             c0 = x
#             c1 = l['0'](c0).squeeze()
#             c2 = l['1'](c1)
#             p = c2
#             pass

#         if(self.backbone=='shufflenet'):

#             c0 = x
#             c1 = l['0'](c0)
#             c2 = l['1'](c1).squeeze()
#             c3 = l['2'](c2)
#             p = c3
#             pass

#         if(self.backbone=='efficientnet'):

#             c0 = x
#             c1 = l['0'](c0).squeeze()
#             c2 = l['1'](c1)
#             p = c2
#             pass


#         if(self.backbone=='mobilenet'):

#             c0 = x
#             c1 = l['0'](c0)
#             p = c1
#             pass

#         batch.score = p
#         return(batch)

#     def cost(self, batch):
#         # 0    1.043859
#         # 1    0.833664
#         # 2    2.080128
#         # 3    2.424322
#         # weight = torch.tensor([1.04, 0.83, 2.08, 2.42]).cuda()
#         weight = None
#         loss  = torch.nn.CrossEntropyLoss(weight)
#         batch = self.forward(batch)
#         pass

#         l = loss(batch.score, batch.target)
#         pass

#         batch.loss = l
#         return(batch)

#     pass

'''
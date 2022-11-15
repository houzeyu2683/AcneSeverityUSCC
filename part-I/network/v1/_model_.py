
import torch
import torchvision


def createPack(name='case'):

    assert name, 'define name please'
    class pack: pass
    pack.__qualname__ = name
    pack.__name__ = name
    return(pack)

class Model(torch.nn.Module):

    def __init__(self, backbone='densenet', classification=2, device='cuda'):

        super().__init__()
        if(backbone=='densenet'):

            # net = [i for i in torchvision.models.densenet121(weights="DenseNet121_Weights.DEFAULT").children()][:-1]
            net = [i for i in torchvision.models.densenet121(pretrained=True).children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(torch.nn.AvgPool2d((7,7))),
                '2':torch.nn.Sequential(
                    torch.nn.Linear(1024, classification)
                )
            }
            pass

        if(backbone=='resnet'):
    
            net = [i for i in torchvision.models.resnet152(pretrained=True).children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(
                    torch.nn.Linear(2048, classification)
                )
            }
            pass

        if(backbone=='efficientnet'):

            # net = [i for i in torchvision.models.efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1").children()][:-1]
            net = [i for i in torchvision.models.efficientnet_b0(pretrained=True).children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(
                    torch.nn.Linear(1280, classification)
                )
            }
            pass

        if(backbone=='mobilenet'):

            net = torchvision.models.MobileNetV2(num_classes=classification)
            layer = {
                "0":net
            }

        self.device = device
        self.classification = classification
        self.backbone = backbone
        self.layer = torch.nn.ModuleDict(layer).to(device)
        return

    def forward(self, batch):

        l = self.layer
        x = batch.image
        n = batch.size
        pass

        if(self.backbone=='densenet'):

            c0 = x
            c1 = l['0'](c0)
            c2 = l['1'](c1).squeeze()
            c3 = l['2'](c2)
            s = c3
            pass

        if(self.backbone=='resnet'):

            c0 = x
            c1 = l['0'](c0).squeeze()
            c2 = l['1'](c1)
            s = c2
            pass

        if(self.backbone=='shufflenet'):

            c0 = x
            c1 = l['0'](c0)
            c2 = l['1'](c1).squeeze()
            c3 = l['2'](c2)
            s = c3
            pass

        if(self.backbone=='efficientnet'):

            c0 = x
            c1 = l['0'](c0).squeeze()
            c2 = l['1'](c1)
            s = c2
            pass


        if(self.backbone=='mobilenet'):

            c0 = x
            c1 = l['0'](c0)
            s = c1
            pass

        score = s.unsqueeze(dim=0) if(n==1) else s
        return(score)

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



import torch
import torchvision

class model(torch.nn.Module):

    def __init__(self, backbone='densenet', classification=2):

        super().__init__()
        if(backbone=='densenet'):

            net = [i for i in torchvision.models.densenet121(weights="DenseNet121_Weights.DEFAULT").children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(torch.nn.AvgPool2d((7,7))),
                '2':torch.nn.Sequential(
                    torch.nn.Linear(1024, classification)
                )
            }
            pass

        if(backbone=='resnet'):
    
            # net = [i for i in torchvision.models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1").children()][:-1]
            net = [i for i in torchvision.models.resnet152(pretrained=True).children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                "1":torch.nn.Sequential(
                    torch.nn.Linear(2048, 1024), torch.nn.ReLU(),
                    torch.nn.Linear(1024,  512), torch.nn.ReLU(),
                    torch.nn.Linear( 512,   27), torch.nn.ReLU()
                ),
                "2":torch.nn.Sequential(
                    torch.nn.Linear(  27,  512), torch.nn.ReLU(),
                    torch.nn.Linear( 512, 1024), torch.nn.ReLU(),
                    torch.nn.Linear(1024, 2048), torch.nn.Sigmoid()
                ),
                '3':torch.nn.Sequential(
                    torch.nn.Linear(2048+27, classification)
                )
            }
            pass

        if(backbone=='efficientnet'):

            net = [i for i in torchvision.models.efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1").children()][:-1]
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

        self.classification = classification
        self.backbone = backbone
        self.layer = torch.nn.ModuleDict(layer)
        return

    def forward(self, batch):

        l = self.layer
        # m  = batch.inference
        x1 = batch.image
        # x2 = batch.embedding
        pass

        if(self.backbone=='densenet'):

            c0 = x
            c1 = l['0'](c0)
            c2 = l['1'](c1).squeeze()
            c3 = l['2'](c2)
            p = c3
            pass

        if(self.backbone=='resnet'):

            c0 = x1
            c1 = l['0'](c0).squeeze()
            c2 = l['1'](c1)
            c3 = l['2'](c2)
            c4 = l['3'](torch.cat([c1, c2], dim=1))
            p = [c1,c2,c3,c4]
            pass

        if(self.backbone=='shufflenet'):

            c0 = x
            c1 = l['0'](c0)
            c2 = l['1'](c1).squeeze()
            c3 = l['2'](c2)
            p = c3
            pass

        if(self.backbone=='efficientnet'):

            c0 = x
            c1 = l['0'](c0).squeeze()
            c2 = l['1'](c1)
            p = c2
            pass


        if(self.backbone=='mobilenet'):

            c0 = x
            c1 = l['0'](c0)
            p = c1
            pass

        score = p
        return(score)

    def cost(self, batch):
        # 0    1.043859
        # 1    0.833664
        # 2    2.080128
        # 3    2.424322
        # weight = torch.tensor([1.04, 0.83, 2.08, 2.42]).cuda()
        weight = None
        loss_1  = torch.nn.CrossEntropyLoss(weight)
        loss_2  = torch.nn.MSELoss()
        loss_3  = torch.nn.MSELoss()
        score = self.forward(batch)
        l_1 = loss_1(score[3], batch.target)
        l_2 = loss_2(score[0], score[2])
        l_3 = loss_3(score[1], batch.embedding)
        # print(l_1, l_2, l_3)
        loss = 1 * l_1 + 1 * l_2 + 1 * l_3
        pass

        batch.score = score
        batch.loss = loss
        pass

        return(batch)

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


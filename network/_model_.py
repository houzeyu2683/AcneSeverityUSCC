
import torch
import torchvision

class model(torch.nn.Module):

    def __init__(self, backbone='densenet'):

        super().__init__()
        if(backbone=='densenet'):

            net = [i for i in torchvision.models.densenet121(weights="DenseNet121_Weights.DEFAULT").children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(torch.nn.AvgPool2d((7,7))),
                '2':torch.nn.Sequential(
                    torch.nn.Linear(1024, 2),
                    torch.nn.Softmax(dim=1)
                    # torch.nn.LogSoftmax(dim=1)
                )
            }
            pass

        if(backbone=='resnet'):
    
            # net = [i for i in torchvision.models.resnet101(weights="ResNet101_Weights.IMAGENET1K_V1").children()][:-1]
            net = [i for i in torchvision.models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1").children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(
                    torch.nn.Linear(2048, 2),
                    # torch.nn.Softmax(dim=1)
                    torch.nn.LogSoftmax(dim=1)
                )
            }
            pass

        if(backbone=='shufflenet'):

            net = [i for i in torchvision.models.shufflenet_v2_x0_5(weights="ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1").children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                '1':torch.nn.Sequential(torch.nn.AvgPool2d((7,7))),
                '2':torch.nn.Sequential(
                    torch.nn.Linear(1024, 4),
                    # torch.nn.Softmax(dim=1)
                    # torch.nn.LogSoftmax(dim=1)
                )
            }
            pass

        if(backbone=='efficientnet'):

            net = [i for i in torchvision.models.efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1").children()][:-1]
            layer = {
                "0":torch.nn.Sequential(*net),
                # '1':torch.nn.Sequential(torch.nn.AvgPool2d((7,7))),
                '1':torch.nn.Sequential(
                    torch.nn.Linear(1280, 2),
                    # torch.nn.Softmax(dim=1)
                    # torch.nn.LogSoftmax(dim=1)
                )
            }
            pass

        if(backbone=='mobilenet'):

            net = torchvision.models.MobileNetV2(num_classes=2)
            layer = {
                "0":net
            }

        self.backbone = backbone
        self.layer = torch.nn.ModuleDict(layer)
        return

    def forward(self, batch):

        l = self.layer
        x = batch.image
        pass

        if(self.backbone=='densenet'):

            c0 = x
            c1 = l['0'](c0)
            c2 = l['1'](c1).squeeze()
            c3 = l['2'](c2)
            p = c3
            pass

        if(self.backbone=='resnet'):

            c0 = x
            c1 = l['0'](c0).squeeze()
            c2 = l['1'](c1)
            p = c2
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

        batch.score = p
        return(batch)

    def cost(self, batch):
        # 0    1.043859
        # 1    0.833664
        # 2    2.080128
        # 3    2.424322
        # weight = torch.tensor([1.04, 0.83, 2.08, 2.42]).cuda()
        weight = None
        loss  = torch.nn.CrossEntropyLoss(weight)
        batch = self.forward(batch)
        pass

        l = loss(batch.score, batch.target)
        pass

        batch.loss = l
        return(batch)

    pass

# net = [i for i in torchvision.models.(weights="ResNet152_Weights.IMAGENET1K_V1").children()][:-1]
# x = torch.randn((4,3,224,224))
# torch.nn.Sequential(*net)(x).shape




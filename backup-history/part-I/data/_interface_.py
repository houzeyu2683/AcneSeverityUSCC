
import PIL.Image
import torchvision
import torch

def createPack(name='case'):

    assert name, 'name is None'
    class pack: pass
    pack.__qualname__ = name
    pack.__name__ = name
    return(pack)

class Interface:

    def __init__(self, path):
        
        self.path = path
        return

    def createCase(self):

        case = createPack(name='case')
        case.size = 1
        pass
        
        image = PIL.Image.open(self.path).convert("RGB")
        mu  = [0.46, 0.36, 0.29]
        std = [0.27, 0.21, 0.18]
        size = (240, 240)
        position = (224, 224)
        convert = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.CenterCrop(position),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mu, std),
        ])
        case.image = convert(image).unsqueeze(0).type(torch.FloatTensor)
        return(case)

    pass
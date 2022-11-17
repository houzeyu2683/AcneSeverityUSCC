
##
##
import torch
import PIL.Image, PIL.ImageStat
from torchvision import transforms as kit

##
##
class dataset(torch.utils.data.Dataset):

    def __init__(self, table):

        self.table = table
        pass

    def __len__(self):

        length = len(self.table)
        return(length)

    def __getitem__(self, index):

        item  = self.table.iloc[index, :]
        picture = process(item).image()
        case = picture
        return(case)

    def get(self, index):

        return self.__getitem__(index)

    pass

##
##
class process:

    def __init__(self, item):

        self.item = item
        pass

    def image(self):

        picture  = PIL.Image.open(self.item['link']).convert("RGB")
        size     = (64, 64)
        if(self.item['mode']=='train'):

            blueprint = [
                kit.RandomHorizontalFlip(), 
                kit.CenterCrop(148), 
                kit.Resize(size), 
                kit.ToTensor(),
                kit.Lambda(lambda x: 2 * x - 1)
            ]
            convert = kit.Compose(blueprint)
            picture = convert(picture).type(torch.float)
            return(picture)
            
        else:

            blueprint = [
                kit.CenterCrop(148), 
                kit.Resize(size), 
                kit.ToTensor(),
                kit.Lambda(lambda x: 2 * x - 1)
            ]
            convert = kit.Compose(blueprint)
            picture = convert(picture).type(torch.float)
            return(picture)

        pass

    pass

    # def imagev2(self):

    #     picture  = PIL.Image.open(self.item['link']).convert("RGB")
    #     size     = (64, 64)
    #     SetRange = kit.Lambda(lambda X: 2 * X - 1.)
    #     # SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))
    #     convert = kit.Compose([kit.RandomHorizontalFlip(), kit.CenterCrop(148), kit.Resize(size), kit.ToTensor(), SetRange])
    #     picture = convert(picture).type(torch.float)
    #     return(picture)
        # if self.params['dataset'] == 'celeba':
        #     transform = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.params['img_size']),
        #                                     transforms.ToTensor(),
        #                                     SetRange])
        # else:
        #     raise ValueError('Undefined dataset type')
        # return transform
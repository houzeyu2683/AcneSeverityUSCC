

import PIL.Image, PIL.ImageStat, torch
from torchvision import transforms as kit

class process:

    def image(item):

        mu       = (0.5,0.5,0.5)
        sigma    = (0.5,0.5,0.5)
        size     = (64, 64)


        if(item['mode']=='train'):

            pipeline = [
                kit.RandomRotation((-60, 60)),
                kit.Resize(size),
                kit.ToTensor(),
                kit.Normalize(mean = mu, std = sigma)
            ]
            transform = kit.Compose(pipeline)
            pass

            image = PIL.Image.open(item['image']).convert("RGB")
            image = transform(image).type(torch.float)
            return(image)

        if(item['mode']=='test'):

            pipeline = [
                kit.Resize(size),
                kit.ToTensor(),
                kit.Normalize(mean = mu, std = sigma)
            ]
            transform = kit.Compose(pipeline)
            pass

            image = PIL.Image.open(item['image']).convert("RGB")
            image = transform(image).type(torch.float)
            return(image)

        return
    
    pass

    def target(item):

        target = torch.tensor(int(item['target']), dtype=torch.long)
        return(target)


# class target:

#     def learn(self, item):


#     def guide(self, item):

#         target = self.learn(item)
#         return(target)
    
#     pass

# class image:

#     def learn(self, item):

#         mu       = (0.5,0.5,0.5)
#         sigma    = (0.5,0.5,0.5)
#         size     = (64, 64)
#         pipeline = [
#             kit.RandomRotation((-60, 60)),
#             kit.Resize(size),
#             kit.ToTensor(),
#             kit.Normalize(mean = mu, std = sigma)
#         ]
#         action = kit.Compose(pipeline)
#         pass

#         image   = PIL.Image.open(item).convert("RGB")
#         image = action(image).type(torch.float)
#         return(image)

#     def guide(self, item):

#         mu      = (0.5,0.5,0.5)
#         sigma   = (0.5,0.5,0.5)
#         size    = (64, 64)
#         pipeline = [
#             kit.Resize(size),
#             kit.ToTensor(),
#             kit.Normalize(mean = mu, std = sigma)
#         ]
#         action = kit.Compose(pipeline)
#         pass

#         image   = PIL.Image.open(item).convert("RGB")
#         image = action(image).type(torch.float)
#         return(image)

#     pass



import torch
from torch import nn
from torch.nn import functional
# x = torch.randn((16, 3, 64, 64))
# a(x).shape
class encoder(nn.Module):

    def __init__(self):

        super(encoder, self).__init__()
        a = nn.Sequential(
            nn.Conv2d(3, 32, (3,3), (2,2), 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (3,3), (2,2), 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, (3,3), (2,2), 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, (3,3), (2,2), 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        b = nn.Linear(256*16, 128)
        c = nn.Linear(256*16, 128)
        self.layer = nn.ModuleDict({
            "x->a":a,
            "a->b":b,
            "a->c":c            
        })
        pass

    def forward(self, x):
        
        a = self.layer['x->a'](x).squeeze()
        b = self.layer['a->b'](a)
        c = self.layer['a->c'](a)
        value = {
            'mu':b,
            'log(sigma)':c,
        }
        return(value)
    
    pass

class decoder(nn.Module):

    def __init__(self):

        super(decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (3,3), (2,2), 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, (3,3), (2,2), 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, (3,3), (2,2), 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, (3,3), (2,2), 1, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, (3,3), (2,2), 1, 1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 3, (3,3), (2,2), 1, 1),
            nn.Tanh()
        )
        pass

    def forward(self, z):

        z = z.unsqueeze(2).unsqueeze(3)
        value = {
            "reconstruction":self.layer(z)
        }
        return(value)
    
    pass

class model(nn.Module):

    def __init__(self):
        
        super(model, self).__init__()
        self.encode = encoder()
        self.decode = decoder()
        pass

    def reparameterize(self, value):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param eta(log_var): (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * value['log(sigma)'])
        eps = torch.randn_like(std)
        z   = eps * std + value['mu'] 
        return(z)

    def forward(self, batch):

        x = batch
        value = {}
        value.update({'image':x})
        value.update(self.encode(x))
        z = self.reparameterize(value)
        value.update(self.decode(z))
        return(value)

    def cost(self, value):

        loss = {
            'MSE':None,
            'KL-divergence':None,
        }
        # Compute reconstruction loss (alpha) and kl divergence (beta)
        # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
        # reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        loss['MSE'] = functional.mse_loss(value['reconstruction'], value['image'], reduction='sum')
        # reconst_loss = functional.mse_loss(pixel, x, reduction='sum')
        # kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # kl_div = - torch.sum(1 + eta - mu.pow(2) - eta.exp())
        loss['KL-divergence'] = - 0.5 * torch.sum(1 + value['log(sigma)'] - value['mu'].pow(2) - value['log(sigma)'].exp())
        # Backprop and optimize
        loss['total'] = sum(loss.values())
        return(loss['total'])

    def generate(self, number):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(number, 128).to('cpu')
        value = self.decode.to('cpu')(z)
        return(value['reconstruction'])

    pass

# import torch
# x = torch.randn((12,3,64,64))
# z = torch.randn((12,128))
# vae = VAE()
# vae.encode(x)
# vae.decode(z)
# vae.forward(x)



# import torch
# z = torch.randn((12, 128))
# z = z.unsqueeze(2).unsqueeze(3)
# z = nn.ConvTranspose2d(128, 64, (3,3), (2,2), 1, 1)(z)
# z = nn.BatchNorm2d(64)(z)
# z = nn.ConvTranspose2d(64, 32, (3,3), (2,2), 1, 1)(z)
# z = nn.BatchNorm2d(32)(z)
# z = nn.ConvTranspose2d(32, 16, (3,3), (2,2), 1, 1)(z)
# z = nn.BatchNorm2d(16)(z)
# z = nn.ConvTranspose2d(16, 8, (3,3), (2,2), 1, 1)(z)
# z = nn.BatchNorm2d(8)(z)
# z = nn.ConvTranspose2d(8, 4, (3,3), (2,2), 1, 1)(z)
# z = nn.BatchNorm2d(4)(z)
# z = nn.ConvTranspose2d(4, 3, (3,3), (2,2), 1, 1)(z)

# z.shape


'''
class VanillaVAE(nn.Module):

    def __init__(self):
        
        super(VanillaVAE, self).__init__()
        self.latent_dim = 128

        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = 3

         = nn.Sequential(
            nn.Conv2d(3, 64, (3,3), (2,2), 1),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, (3,3), (2,2), 1),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(128, 256, (3,3), (2,2), 1),
            nn.MaxPool2d((2,2))
        )
        nn.ModuleDict()
        batch = torch.randn((16, 3, 64, 64))
        batch = nn.Conv2d(3, 64, (3,3), (2,2), 1)(batch)
        batch = nn.MaxPool2d((2,2))(batch)
        batch = nn.Conv2d(64, 128, (3,3), (2,2), 1)(batch)
        batch = nn.MaxPool2d((2,2))(batch)
        batch = nn.Conv2d(128, 256, (3,3), (2,2), 1)(batch)
        batch = nn.MaxPool2d((2,2))(batch)
        batch = batch.squeeze()
        mu    = nn.Linear(256, 128)(batch)
        alpha = nn.Linear(256, 128)(batch)
        # # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim,
        #                       kernel_size= 3, stride= 2, padding  = 1),
        #             nn.BatchNorm2d(h_dim),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, self.latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self, result, M_N):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = result[0]  #
        input = result[1]   #
        mu = result[2]      #
        log_var = result[3] # forward result

        kld_weight = M_N # Account for the minibatch samples from the dataset, # batch / # test dataset =====> 必要？
        recons_loss = nn.MSELoss(reduction='sum')(recons, input)  ##  輸入跟解碼的loss
        # print(recons_loss)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)  ##  letant code 跟 N(0,1) 的差距 => 越小越好

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self, num_samples, current_device='cpu'):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 128)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        跟 forward 很像 但就是給你 output 而已
        """

        return self.forward(x)[0]

    pass

'''

'''
模型參考來源 https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py 。
'''

# import torch
# from torch import nn

# batch = torch.randn((16, 3, 64, 64))
# batch = nn.Conv2d(3, 64, (3,3), (2,2), 1)(batch)
# batch = nn.MaxPool2d((2,2))(batch)
# batch = nn.Conv2d(64, 128, (3,3), (2,2), 1)(batch)
# batch = nn.MaxPool2d((2,2))(batch)
# batch = nn.Conv2d(128, 256, (3,3), (2,2), 1)(batch)
# batch = nn.MaxPool2d((2,2))(batch)
# batch = batch.squeeze()
# mu    = nn.Linear(256, 128)(batch)
# alpha = nn.Linear(256, 128)(batch)


# batch.shape


# model = VanillaVAE()
# x = torch.randn((12, 3, 64, 64))
# results = model.forward(x)
# model.encode(x)[0].shape
# model.encode(x)[1].shape

# z = model.reparameterize(model.encode(x)[0], model.encode(x)[1])
# model.decode(z)

# num_val_imgs = 24
# model.loss_function(*results, M_N = 12 / num_val_imgs)

# model.sample(20)

# x = torch.randn(12, 3, 64, 64)

# in_channels = 3
# modules = []
# hidden_dims = [32, 64, 128, 256, 512]
    
# #
# # Build Encoder
# for h_dim in hidden_dims:
#     modules.append(
#         nn.Sequential(
#             nn.Conv2d(in_channels, out_channels=h_dim,
#                       kernel_size= 3, stride= 2, padding  = 1),
#             nn.BatchNorm2d(h_dim),
#             nn.LeakyReLU())
#     )
#     in_channels = h_dim
# #
# encoder = nn.Sequential(*modules)
# latent_dim = 128
# fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
# fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
# '''
# `encoder(x).shape`
# '''

# # Build Decoder
# modules = []
# #
# decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
# #
# hidden_dims.reverse()
# #
# for i in range(len(hidden_dims) - 1):
#     modules.append(
#         nn.Sequential(
#             nn.ConvTranspose2d(hidden_dims[i],
#                                hidden_dims[i + 1],
#                                kernel_size=3,
#                                stride = 2,
#                                padding=1,
#                                output_padding=1),
#             nn.BatchNorm2d(hidden_dims[i + 1]),
#             nn.LeakyReLU())
#     )
# #
# decoder = nn.Sequential(*modules)


# final_layer = nn.Sequential(
#                     nn.ConvTranspose2d(hidden_dims[-1],
#                                            hidden_dims[-1],
#                                        kernel_size=3,
#                                        stride=2,
#                                        padding=1,
#                                        output_padding=1),
#                     nn.BatchNorm2d(hidden_dims[-1]),
#                     nn.LeakyReLU(),
#                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
#                               kernel_size= 3, padding= 1),
#                     nn.Tanh())





# class model(nn.Module):

#     def __init__(self):

#         super(model, self).__init__()
#         pass
        
#         self.layer = nn.Sequential( 
#             *list(torchvision.models.resnet18(True).children())[:-1],
#             nn.Flatten(1,-1),
#             nn.Linear(in_features=512, out_features=10),
#             nn.Softmax(dim=1)
#         )
#         return

#     def forward(self, batch):

#         image, target = batch
#         score = self.layer(image)
#         return(score, target)

#     pass

# import torchvision
# from torch import nn
# layer = nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1])
# import torch
# x = torch.randn(size=(1,3,64,64))
# layer(x).shape
# batch[0]
# image['residual'](batch[0])
# import torch
# from torch import nn

# index = torch.randint(low=0, high=100, size=(13, 4))
# target = torch.randint(low=0, high=100, size=(3, 4))
# embed_layer = nn.Embedding(num_embeddings=100, embedding_dim=6)
# embed = embed_layer(index)
# embed.shape

# embed_target = embed_layer(target)


# encoder_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=6, nhead=1), num_layers=1)
# encode = encoder_layer(embed)


# decoder_layer = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=6, nhead=1), num_layers=1)
# decode = decoder_layer(embed_target, encode)
# decode.shape

# ##
# ##
# import torch, torchvision, pickle
# import torch.nn as nn


# ##
# ##
# path='SOURCE/PICKLE/VOCABULARY.pickle'
# with open(path, 'rb') as paper:

#     vocabulary = pickle.load(paper)
#     pass


# ##
# ##
# class mask:

#     def encode(text):

#         if(text.is_cuda):

#             device = "cuda"
#             pass

#         else:

#             device = 'cpu'
#             pass

#         length = text.shape[0]
#         mask = torch.zeros((length, length), device=device).type(torch.bool)
#         return mask
    
#     def decode(text):

#         if(text.is_cuda):

#             device = "cuda"
#             pass

#         else:

#             device = 'cpu'
#             pass

#         length = text.shape[0]
#         mask = (torch.triu(torch.ones((length, length), device=device)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask

#     def pad(text):

#         mask = (text == vocabulary['<pad>']).transpose(0, 1)      
#         return mask


# ##
# ##
# class model(torch.nn.Module):

#     def __init__(self):
        
#         super(model, self).__init__()
#         pass

#         self.size = {
#             "vocabulary" : len(vocabulary.itos),
#             "embedding" : 256
#         }
#         pass

#         embedding = nn.ModuleDict({
#             "01" : nn.Embedding(self.size['vocabulary'], self.size['embedding'])
#         })

#         encoding = nn.ModuleDict({
#             "02" : nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.size['embedding'], nhead=2), num_layers=4)
#         })

#         sequence = nn.ModuleDict({
#             "03" : nn.GRU(self.size['embedding'], self.size['embedding'], 1),
#             "04" : nn.GRU(self.size['embedding'], self.size['embedding'], 1)
#         })

#         classification = nn.ModuleDict({
#             "05" : nn.Sequential(nn.Linear(self.size['embedding'], 3))
#         })

#         layer = {
#             "embedding": embedding,
#             "encoding" : encoding,
#             "sequence" : sequence,
#             "classification" : classification
#         }
#         self.layer = nn.ModuleDict(layer)
#         pass
    
#     def forward(self, batch):
        
#         ##
#         index, _ = batch

#         ##
#         cell = {}
#         cell['01'] = self.layer['embedding']['01'](index)
#         cell['02'] = self.layer['encoding']['02'](
#             cell['01'],
#             mask.encode(index),
#             mask.pad(index)
#         )
#         cell['03'], memory = self.layer['sequence']['03'](cell['02'])
#         cell['04'], memory = self.layer['sequence']['04'](cell['03'], memory)

#         cell['05'] = self.layer['classification']['05'](cell['04'][-1,:,:])
#         return cell['05']        

        # # cell['01'], _ = self.layer['encoding']['02'](cell['01'].transpose(0,1).unsqueeze(dim=2))
        # # cell['02'] = self.layer['image']['02'](cell['01']).squeeze()
        # # index = torch.as_tensor(cell['02'] * self.size['vocabulary'], dtype=torch.long)
        # # cell['03'] = self.layer['token']['03'](cell['00'])
        # # length = (cell['03'] * (512-3)).int().flatten().tolist()

        # # ##
        # # for column, row in enumerate(length):

        # #     index[0, column] = self.vocabulary['<bos>']
        # #     index[row, column] = self.vocabulary['<eos>']
        # #     index[row+1:, column] = self.vocabulary['<pad>']
        # #     pass 
        
        # ##
        # # cell['04'] = self.layer['token']['04'](
        # #     self.layer['token']['embedding'](index), 
        # #     mask.encode(index), 
        # #     mask.pad(index)            
        # # )

        # ##
        # # text = dictionary.convert(text, vocabulary=self.vocabulary)
        # cell['05'] = self.layer['token']['05'](
        #     self.layer['token']['embedding'](token),
        #     cell['04'],
        #     mask.decode(token),
        #     None,
        #     mask.pad(token, vocabulary=self.vocabulary),
        #     None
        # )
        # cell['06'] = self.layer['token']['06'](cell['05'])
        # return(cell['06'])

    # def convert(self, image, size=128):

    #     ##
    #     if(image.is_cuda):

    #         device = 'cuda'
    #         pass

    #     else:
     
    #         device = 'cpu'
    #         pass
        
    #     ##
    #     cell = {}
    #     cell['00'] = self.layer['image']['00'](image).squeeze()
    #     cell['01'], _ = self.layer['image']['01'](cell['00'].transpose(0,1).unsqueeze(dim=2))
    #     cell['02'] = self.layer['image']['02'](cell['01']).squeeze()
    #     index = torch.as_tensor(cell['02'] * self.size['vocabulary'], dtype=torch.long)
    #     cell['03'] = self.layer['token']['03'](cell['00'])
    #     length = (cell['03'] * (512-3)).int().flatten().tolist()

    #     ##
    #     for column, row in enumerate(length):

    #         index[0, column] = self.vocabulary['<bos>']
    #         index[row, column] = self.vocabulary['<eos>']
    #         index[row+1:, column] = self.vocabulary['<pad>']
    #         pass 
        
    #     ##
    #     cell['04'] = self.layer['token']['04'](
    #         self.layer['token']['embedding'](index), 
    #         mask.encode(index), 
    #         mask.pad(index, vocabulary=self.vocabulary)            
    #     )

    #     batch = len(image)
    #     sequence = torch.ones(1, batch).fill_(self.vocabulary['<bos>']).type(torch.long).to(device)

    #     for _ in range(size):

    #         code = self.layer['token']['05'](
    #             self.layer['token']['embedding'](sequence), 
    #             cell['04'], 
    #             mask.decode(sequence), 
    #             None, 
    #             None
    #         )
    #         probability = self.layer['token']['06'](code.transpose(0, 1)[:, -1])
    #         _, prediction = torch.max(probability, dim = 1)
    #         sequence = torch.cat([sequence, prediction.unsqueeze(dim=0)], dim=0)
    #         pass

    #     output = []
    #     for i in range(batch):

    #         character = "".join([self.vocabulary.itos[token] for token in sequence[:,i]])
    #         character = "InChI=1S/" + character
    #         character = character.replace("<bos>", "").replace("<eos>", "").replace('<pad>', "")
    #         output += [character]
    #         pass

    #     return output

    #     output = []
    #     for item in range(batch):
            
    #         memory = midden['encoder memory'][:,item:item+1,:]

    #     # print("midden['encoder memory']")
    #     # print(midden['encoder memory'].shape)
    #         ##  Generate sequence.
    #         sequence = torch.ones(1, 1).fill_(vocabulary['<bos>']).type(torch.long).to(device)
    #         for i in range(length):

    #             midden['decoder output'] = self.layer['text decoder'](
    #                 self.layer['text to embedding'](sequence), 
    #                 memory, 
    #                 mask.decode(sequence), 
    #                 None, 
    #                 None
    #             )
    #             print("midden['decoder output'] ")
    #             print(midden['decoder output'].shape)
    #             probability = self.layer['text to vacabulary'](midden['decoder output'].transpose(0, 1)[:, -1])
    #             _, prediction = torch.max(probability, dim = 1)
    #             index = prediction.item()
    #             sequence = torch.cat([sequence, torch.ones(1, 1).type_as(midden['image to index 03']).fill_(index)], dim=0)
    #             pass

    #             if index == vocabulary['<eos>']:
                    
    #                 break
            
    #         character = "InChI=1S/" + "".join([vocabulary.itos[tok] for tok in sequence]).replace("<bos>", "").replace("<eos>", "")
    #         output += [character]
    #         pass

    #     return output




# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     src = src.to(device)
#     src_mask = src_mask.to(device)

#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
#     for i in range(max_len-1):
#         memory = memory.to(device)
#         memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)

#         tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
#         print('tgt_mask----')
#         print(tgt_mask)
#         print(tgt_mask.shape)


#         out = model.decode(ys, memory, tgt_mask)
#         out = out.transpose(0, 1)
#         print("output===")
#         print(out.shape)
#         print(out)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim = 1)
#         next_word = next_word.item()

#         ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=0)
#         if next_word == EOS_IDX:
#           break
#     return ys

# def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
#     model.eval()
#     tokens = [BOS_IDX] + [src_vocab.stoi[tok] for tok in src_tokenizer(src)]+ [EOS_IDX]
#     num_tokens = len(tokens)
#     src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
#     return " ".join([tgt_vocab.itos[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")

# # batch = torch.randn((8, 3, 224, 224)), torch.randint(0, 141, (10, 8))
# # image, text = batch
# # m = model()
# # x = m(batch)

# # x.shape



# z
# nn.Linear(24, 3)(y[-1,:,:])

# vocabulary = data.process.vocabulary.load(path='SOURCE/PICKLE/VOCABULARY.pickle')

# image = torch.randn((8,3,224,224))

# L01 = nn.Sequential(*list(torchvision.models.resnet18(True).children())[:-1])
# L02 = nn.Sequential(nn.GRU(1, 141, 1))
# L03 = nn.Softmax(dim=2)
# L04 = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())

# M01 = L01(image).squeeze()
# M02, _ = L02(M01.transpose(0,1).unsqueeze(dim=2))
# M03 = L03(M02).argmax(dim=2)

# M04 = (L04(M01) * 512).int().flatten().tolist() # seq length
# for column, row in enumerate(M04):
#     M03[0, column] = vocabulary['<bos>']
#     M03[row, column] = vocabulary['<eos>']
#     M03[row+1:, column] = vocabulary['<pad>']
#     pass 

# L05 = nn.Embedding(141, 256)
# M05 = L05(M03)
# M05.shape

# L06 = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=2), num_layers=4)
# M06 = L06(M05, mask.encode(M03), mask.pad(M03, vocabulary=vocabulary))



# def convert(text):

#     output = []
    
#     for item in text:
#         item = [vocabulary['<bos>']] + [vocabulary[i] for i in item] + [vocabulary['<eos>']]
#         item = torch.tensor(item, dtype=torch.long)
#         output += [item]
#     output = torch.nn.utils.rnn.pad_sequence(output, padding_value=vocabulary['<pad>'])
#     return(output)

# text = [["H", "2", "o"], ["H", "2", "o"], ["C", "20"], ["C", "20"], ["C", "20"], ["C", "20"], ["C", "20"], ["H", "2", "o"]]
# L07 = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=256, nhead=2), num_layers=4)
# M07 = L07(L05(convert(text)), M06, mask.decode(convert(text)), None, mask.pad(convert(text), vocabulary=vocabulary))

# L08 =  nn.Sequential(nn.Linear(256, 141), nn.Softmax(dim=2))
# M08 =  L08(M07)
# M08


        # ##  Decoder, encode to index of text.
        # midden['decoder output'] = self.layer['text decoder'](
        #     self.layer['text to embedding'](text), 
        #     midden['encoder memory'], 
        #     mask.decode(text), 
        #     None, 
        #     mask.pad(text), 
        #     None
        # )
        # output = self.layer['text to vacabulary'](midden['decoder output'])
        # # print("self.generator(outs)-----")
        # # print(self.generator(outs).shape)
        # return output


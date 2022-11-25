

##
import torch
from torch import nn
from torch.nn import functional


##
class encoder(nn.Module):

    def __init__(self):

        super(encoder, self).__init__()
        layer = {
            'to code' : nn.Sequential(
                nn.Conv2d(3, out_channels=32, kernel_size= 3, stride= 2, padding  = 1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, out_channels=64, kernel_size= 3, stride= 2, padding  = 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.Conv2d(64, out_channels=128, kernel_size= 3, stride= 2, padding  = 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, out_channels=256, kernel_size= 3, stride= 2, padding  = 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.Conv2d(256, out_channels=512, kernel_size= 3, stride= 2, padding  = 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Flatten()
            ),
            'to mu': nn.Linear(512 * 4, 128),
            'to log(sigma^2)' : nn.Linear(512 * 4, 128)
        }
        self.layer = nn.ModuleDict(layer)
        pass

    def forward(self, x):

        code = self.layer['to code'](x)
        # result = torch.flatten(self.fc_encoder(x), start_dim=1)
        output = self.layer['to mu'](code), self.layer['to log(sigma^2)'](code)
        # mu = self.fc_mu(result)
        # log_var = self.fc_var(result)
        return(output)


##
class decoder(nn.Module):

    def __init__(self):

        super(decoder, self).__init__()
        layer = {
            "to feature" : nn.Linear(128, 512 * 4),
            "to image" : nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, out_channels= 3, kernel_size= 3, padding= 1),
                nn.Tanh()
            )
        }
        self.layer = nn.ModuleDict(layer)
        return

    def forward(self, z):

        feature = self.layer['to feature'](z).view(-1, 512, 2, 2)
        image = self.layer['to image'](feature)
        return(image)


##
class model(nn.Module):

    def __init__(self):

        super(model, self).__init__()
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
        z = eps * std + value['mu']
        return(z)

    def forward(self, x):

        # mu, log_var = self.encode(input)
        value = {
            "image":None,
            "mu":None,
            "log(sigma^2)":None,
            'reconstruction':None            
        }
        value['image'] = x
        value['mu'], value['log(sigma^2)'] = self.encoder(x)
        # mu, log_var = self.encoder_layer(x)
        z = self.reparameterize(value)
        value['reconstruction'] = self.decoder(z)
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

        #loss['kl-divergence'] = -1 * loss['kl-divergence']
        return(loss)
        # kld_weight = 0.0008 # Account for the minibatch samples from the dataset
        # recons_loss = nn.functional.mse_loss(recons, input)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        # loss = recons_loss + kld_weight * kld_loss
        # return {'total': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def generate(self, number):

        device = "cuda" if next(self.decoder.parameters()).is_cuda else "cpu"
        z = torch.randn(number, 128).to(device)
        samples = self.decoder(z)
        return samples
        # return self.forward(input=x)[0]
    pass

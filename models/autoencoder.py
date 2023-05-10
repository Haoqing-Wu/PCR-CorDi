import torch
from torch.nn import Module

from models.ddpm import *
from models.pointnet import *
from utils.common import *
from models.unet import *


class Autoencoder(Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointNetEncoder(args.latent_dim)
        self.decoder = PointNetDecoder(args.latent_dim)

    def get_loss(self, pcd):
        code_mu, code_sigma = self.encoder(pcd)
        code = reparameterize_gaussian(mean=code_mu, logvar=code_sigma)  # (B, F)
        recon = self.decoder(code)
        loss = chamfer(pcd, recon)
        return loss, recon
    
    def forward(self, pcd):
        code_mu, code_sigma = self.encoder(pcd)    
        return code_mu, code_sigma

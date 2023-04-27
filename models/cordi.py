import torch
from torch.nn import Module

from models.ddpm import *
from models.pointnet import *
from utils.common import *


class Cordi(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_src = PointNetEncoder(args.latent_dim)
        self.encoder_tgt = PointNetEncoder(args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(
                point_dim=6,
                context_dim=args.latent_dim * 2,
                residual=args.residual
                ),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
                )
        )
        
    def get_loss(self, corr, src, tgt,  writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        z_src_mu, z_src_sigma = self.encoder_src(src)
        z_src = reparameterize_gaussian(mean=z_src_mu, logvar=z_src_sigma)  # (B, F)
        z_tgt_mu, z_tgt_sigma = self.encoder_tgt(tgt)
        z_tgt = reparameterize_gaussian(mean=z_tgt_mu, logvar=z_tgt_sigma)  # (B, F)
        loss = self.diffusion.get_loss(corr, z_src, z_tgt)

        return loss
    
    def sample(self, x_T, src, tgt, flexibility):
        z_src_mu, z_src_sigma = self.encoder_src(src)
        z_src = reparameterize_gaussian(mean=z_src_mu, logvar=z_src_sigma)  # (B, F)
        z_tgt_mu, z_tgt_sigma = self.encoder_tgt(tgt)
        z_tgt = reparameterize_gaussian(mean=z_tgt_mu, logvar=z_tgt_sigma)  # (B, F)
        samples = self.diffusion.sample(x_T, z_src, z_tgt, flexibility=flexibility)
        return samples
import torch
from torch.nn import Module

from models.ddpm import *
from models.pointnet import *
from models.flow import *
from utils.common import *



class Cordi(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_src = PointNetEncoder(args.latent_dim)
        self.encoder_tgt = PointNetEncoder(args.latent_dim)
        self.encoder = PointNetEncoder(args.latent_dim)
        self.flow = build_latent_flow(args)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(
                point_dim=6,
                context_dim=args.latent_dim * 1,
                residual=args.residual
                ),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
                )
        )
        
    def get_loss(self, vector, src, tgt,  writer=None, it=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        z_src_mu, z_src_sigma = self.encoder_src(src)
        z_src = reparameterize_gaussian(mean=z_src_mu, logvar=z_src_sigma)  # (B, F)
        z_tgt_mu, z_tgt_sigma = self.encoder_tgt(tgt)
        z_tgt = reparameterize_gaussian(mean=z_tgt_mu, logvar=z_tgt_sigma)  # (B, F)

        batch_size, _, _ = vector.size()
        z_mu, z_sigma = self.encoder(vector)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        # H[Q(z|X)]
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )

        # P(z), Prior probability, parameterized by the flow: z -> w.
        w, delta_log_pw = self.flow(z, torch.zeros([batch_size, 1]).to(z), reverse=False)
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(dim=1, keepdim=True)   # (B, 1)
        log_pz = log_pw - delta_log_pw.view(batch_size, 1)  # (B, 1)
    

        neg_elbo = self.diffusion.get_loss(vector, z, z_src, z_tgt)
        # Loss
        kl_weight = 0.001
        loss_entropy = -entropy.mean()
        loss_prior = -log_pz.mean()
        loss_recons = neg_elbo
        loss = kl_weight*(loss_entropy + loss_prior) + neg_elbo

        return loss
    
    def sample(self, w, src, tgt, flexibility, truncate_std=None):
        z_src_mu, z_src_sigma = self.encoder_src(src)
        z_src = reparameterize_gaussian(mean=z_src_mu, logvar=z_src_sigma)  # (B, F)
        z_tgt_mu, z_tgt_sigma = self.encoder_tgt(tgt)
        z_tgt = reparameterize_gaussian(mean=z_tgt_mu, logvar=z_tgt_sigma)  # (B, F)

        batch_size, _ = w.size()
        if truncate_std is not None:
            w = truncated_normal_(w, mean=0, std=1, trunc_std=truncate_std)
        # Reverse: z <- w.
        z = self.flow(w, reverse=True).view(batch_size, -1)
        samples = self.diffusion.sample(z, z_src, z_tgt, flexibility=flexibility)
        return samples
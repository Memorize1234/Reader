import functools
import torch
from torch import nn
import torch.nn.functional as F
from utils import global_vars as gl
from .perceptual import LPIPS


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()

def adopt_weight(weight, global_step, threshold=0, warm_up_end=0, value=0.):
    if global_step < threshold:
        weight = value
    elif global_step < warm_up_end:
        weight = weight * (global_step-threshold) / (warm_up_end - threshold)
    return weight

def hinge_d_loss(logits_real, logits_fake, sample_weight):
    logits_real = F.relu(1. - logits_real).flatten(1).mean(-1)
    logits_fake = F.relu(1. + logits_fake).flatten(1).mean(-1)
    if sample_weight is not None:
        sample_weight = sample_weight * len(sample_weight) / sample_weight.sum()
        logits_fake = logits_fake * sample_weight
    loss_real = torch.mean(logits_real)
    loss_fake = torch.mean(logits_fake)
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class GANLoss(nn.Module):
    def __init__(self, discriminator, disc_start, warm_up_end, pretrained=None, freeze=False, 
                 normalize=False, disc_loss='hinge', use_adaptive_weight=True, weight=1.0, factor=1.0):
        super().__init__()
        self.discriminator = NLayerDiscriminator(**discriminator)
        self.discriminator.apply(NLayerDiscriminator.weights_init)
        if pretrained is not None:
            self.discriminator.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=True)
            print(f'loaded pretrained GAN discriminator from {pretrained}')
        steps_per_epoch = gl.get('steps_per_epoch')
        self.discriminator_iter_start = disc_start * steps_per_epoch
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        self.use_adaptive_weight = use_adaptive_weight
        self.freeze = freeze
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
        self.weight = weight
        self.factor = factor

        self.warm_up_end = warm_up_end * steps_per_epoch
        
        self.normalize = normalize
        self.register_buffer('norm_mean', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer('norm_std', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.weight
        return d_weight    
    
    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor, update_g, global_step, sample_weight=None, last_layer=None, rec_loss=None):
        """
        inputs: original image batch tensor with shape (bs, 3, H, W)
        reconstructions: Tensor with shape (bs, 3, H, W)
        rec_loss: sum of L1/L2 loss and LPIPS loss
        last_layer: weight of the last layer of the decoder
        """
        if self.normalize:
            # convert to VQGAN normalization
            reconstructions = ((reconstructions * self.norm_std + self.norm_mean) * 2.0 - 1.0)
            if not update_g:
                inputs = ((inputs * self.norm_std + self.norm_mean) * 2.0 - 1.0)
        
        if update_g:
            logits_fake = self.discriminator(reconstructions.contiguous()).flatten(1).mean(-1)
            if sample_weight is not None:
                sample_weight = sample_weight * len(sample_weight) / sample_weight.sum()
                logits_fake = logits_fake * sample_weight
            g_loss = -torch.mean(logits_fake)
            if self.use_adaptive_weight:
                assert rec_loss is not None and last_layer is not None
                rec_loss = torch.mean(rec_loss)
                weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer)
            else:
                weight = self.weight
            factor = adopt_weight(self.factor, global_step, threshold=self.discriminator_iter_start, warm_up_end=self.warm_up_end)
            g_loss = weight * factor * g_loss
            return g_loss, weight
        else:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            factor = adopt_weight(self.factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = factor * self.disc_loss(logits_real, logits_fake, sample_weight)
            return d_loss, logits_real.mean().item(), logits_fake.mean().item()
            

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
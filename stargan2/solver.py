import os
import csv
import time
import datetime
from os.path import join as ospj

import numpy as np
from numpy.random import randint
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.utils as vutils

from stargan2.model import build_model

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(
            module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

class Solver(nn.Module):
    """Initialize networks.

    Args:
        X: Real images normalized to [-1, 1]
        y: Labels
        n_domains: Number of domains
        latent_dim: Dimensions of the latent vector z
        style_dim: Style code dimension
        lambda_reg: Weight for R1 regularization
        lambda_cyc: Weight for cyclic consistency loss
        lambda_sty: Weight for style reconstruction loss
        lambda_ds: Weight for diversity sensitive loss
        weight_decay: Weight decay for optimizer
    """

    def __init__(self, working_dir, img_size, n_domains=2, style_dim=64, latent_dim=64,
                 lambda_sty=1, lambda_ds=1, lambda_reg=1,
                 lambda_cyc=1, weight_decay=1e-4, cuda=True):
        super().__init__()

        self.working_dir = working_dir

        self.n_domains = n_domains
        self.style_dim = style_dim
        self.latent_dim = latent_dim
        self.lambda_sty = lambda_sty
        self.lambda_ds = lambda_ds
        # Number of iterations to optimize diversity sensitive loss
        self.ds_iter = 100000
        self.lambda_cyc = lambda_cyc
        self.lambda_reg = lambda_reg

        if not os.path.exists('results'):
            os.mkdir('results')
        working_dir = ospj('results', working_dir)
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        self.sample_dir = ospj(working_dir, 'samples')
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

        self.working_dir = working_dir

        self.d_real_losses = []
        self.d_reg_losses = []
        self.d_fake_latent_losses = []
        self.d_fake_ref_losses = []
        self.g_latent_adv_losses = []
        self.g_latent_sty_losses = []
        self.g_latent_ds_losses = []
        self.g_latent_cyc_losses = []
        self.g_ref_adv_losses = []
        self.g_ref_sty_losses = []
        self.g_ref_ds_losses = []
        self.g_ref_cyc_losses = []

        lr = 1e-4
        # mapping_network learning rate
        f_lf = 1e-6
        beta1, beta2 = .0, .99

        self.device = torch.device(
            'cuda' if cuda and torch.cuda.is_available() else 'cpu')

        # witout high-pass and mask
        # https://github.com/clovaai/stargan-v2/issues/6#issuecomment-620907653
        # https://github.com/clovaai/stargan-v2/issues/70#issuecomment-709668736
        (self.G, self.D,
         self.mapping_network,
         self.style_encoder) = build_model(img_size, style_dim, latent_dim, n_domains)

        nets = (self.G, self.D,
                self.mapping_network,
                self.style_encoder)

        # copy all nets to device
        [net.to(self.device) for net in nets]

        self.G_opt = Adam(params=self.G.parameters(), lr=lr,
                          betas=[beta1, beta2], weight_decay=weight_decay)
        self.D_opt = Adam(params=self.D.parameters(), lr=lr,
                          betas=[beta1, beta2], weight_decay=weight_decay)
        self.mapping_network_opt = Adam(params=self.mapping_network.parameters(), lr=f_lf,
                                        betas=[beta1, beta2], weight_decay=weight_decay)
        self.style_encoder_opt = Adam(params=self.style_encoder.parameters(), lr=lr,
                                      betas=[beta1, beta2], weight_decay=weight_decay)

        [net.apply(he_init) for net in nets]

    def train(self, epochs, loader, loader_ref, val_dataset=None):
        # nets_ema = self.nets_ema
        device = self.device

        # remember the initial value of ds weight
        initial_lambda_ds = self.lambda_ds

        cum_y = 0
        cum_y_trg = 0
        cum_n = 0

        for epoch in range(epochs):
            start_time = time.time()

            d_real_losses = []
            d_reg_losses = []
            d_fake_latent_losses = []
            d_fake_ref_losses = []

            g_latent_adv_losses = []
            g_latent_sty_losses = []
            g_latent_ds_losses = []
            g_latent_cyc_losses = []

            g_ref_adv_losses = []
            g_ref_sty_losses = []
            g_ref_ds_losses = []
            g_ref_cyc_losses = []

            # data_iter = iter(loader)
            # ref_iter = iter(loader_ref)

            for i, data in enumerate(zip(loader, loader_ref)):
                print(f'Epoch {epoch+1:3d}: {(100*(i+1)/total_iter):6.2f}%', end='\r', flush=True)

                # unpack real data
                x_real, y_org = data[0]
                # unpack reference data
                x_ref1, x_ref2, y_trg = data[1]

                batch = len(x_real)

                x_real = x_real.to(device)
                x_real.requires_grad_()
                y_org = y_org.to(device)

                x_ref1 = x_ref1.to(device)
                x_ref2 = x_ref2.to(device)
                y_trg = y_trg.to(device)

                # generate latent vectors
                z_trg1 = torch.randn(batch, self.latent_dim).to(device)
                z_trg2 = torch.randn(batch, self.latent_dim).to(device)

                # *** train the discriminator ***

                # generate fakes with no_grad
                # since we don't want to update G, style_encoder and mapping_network
                with torch.no_grad():
                    # using styles from noise
                    x_fake_noise = self.G(
                        x_real, self.mapping_network(z_trg1, y_trg))
                    # using styles from reference images
                    x_fake_ref = self.G(
                        x_real, self.style_encoder(x_ref1, y_trg))

                d_loss, d_loss_stats = self.d_loss(
                    x_real, y_org, y_trg, x_fake_noise)

                loss_real, loss_fake, loss_reg = d_loss_stats
                d_real_losses.append(loss_real)
                d_fake_latent_losses.append(loss_fake)
                d_reg_losses.append(loss_reg)

                self.D_opt.zero_grad()
                d_loss.backward()
                self.D_opt.step()

                d_loss, d_loss_stats = self.d_loss(
                    x_real, y_org, y_trg, x_fake_ref)

                loss_real, loss_fake, loss_reg = d_loss_stats
                d_real_losses.append(loss_real)
                d_fake_ref_losses.append(loss_fake)
                d_reg_losses.append(loss_reg)

                self.D_opt.zero_grad()
                d_loss.backward()
                self.D_opt.step()

                # *** train the generator ***

                # generate styles from noise
                s_trg_z1 = self.mapping_network(z_trg1, y_trg)
                s_trg_z2 = self.mapping_network(z_trg2, y_trg)

                g_loss, g_loss_stats = self.g_loss(
                    x_real, y_org, y_trg, s_trg_z1, s_trg_z2)

                loss_adv, loss_sty, loss_ds, loss_cyc = g_loss_stats
                g_latent_adv_losses.append(loss_adv)
                g_latent_sty_losses.append(loss_sty)
                g_latent_ds_losses.append(loss_ds)
                g_latent_cyc_losses.append(loss_cyc)

                self.G_opt.zero_grad()
                self.mapping_network_opt.zero_grad()
                self.style_encoder_opt.zero_grad()
                g_loss.backward()
                self.G_opt.step()
                self.mapping_network_opt.step()
                self.style_encoder_opt.step()

                # generate styles from reference images
                s_trg_ref1 = self.style_encoder(x_ref1, y_trg)
                s_trg_ref2 = self.style_encoder(x_ref2, y_trg)

                g_loss, g_loss_stats = self.g_loss(
                    x_real, y_org, y_trg, s_trg_ref1, s_trg_ref2)

                loss_adv, loss_sty, loss_ds, loss_cyc = g_loss_stats
                g_ref_adv_losses.append(loss_adv)
                g_ref_sty_losses.append(loss_sty)
                g_ref_ds_losses.append(loss_ds)
                g_ref_cyc_losses.append(loss_cyc)

                self.G_opt.zero_grad()
                g_loss.backward()
                self.G_opt.step()

                # compute moving average of network parameters
                # https://arxiv.org/abs/1806.04498
                # https://github.com/clovaai/stargan-v2/issues/62
                # self.moving_average(self.G, nets_ema.generator, beta=0.999)
                # self.moving_average(self.mapping_network,
                #                nets_ema.mapping_network, beta=0.999)
                # self.moving_average(self.style_encoder,
                #                nets_ema.style_encoder, beta=0.999)

                # decay weight for diversity sensitive loss
                if self.lambda_ds > 0:
                    self.lambda_ds -= (initial_lambda_ds / self.ds_iter)
            print()

            self.d_real_losses.append(sum(d_real_losses)/len(d_real_losses))
            self.d_reg_losses.append(sum(d_reg_losses)/len(d_reg_losses))
            self.d_fake_latent_losses.append(
                sum(d_fake_latent_losses)/len(d_fake_latent_losses))
            self.d_fake_ref_losses.append(
                sum(d_fake_ref_losses)/len(d_fake_ref_losses))
            self.g_latent_adv_losses.append(
                sum(g_latent_adv_losses)/len(g_latent_adv_losses))
            self.g_latent_sty_losses.append(
                sum(g_latent_sty_losses)/len(g_latent_sty_losses))
            self.g_latent_ds_losses.append(
                sum(g_latent_ds_losses)/len(g_latent_ds_losses))
            self.g_latent_cyc_losses.append(
                sum(g_latent_cyc_losses)/len(g_latent_cyc_losses))
            self.g_ref_adv_losses.append(
                sum(g_ref_adv_losses)/len(g_ref_adv_losses))
            self.g_ref_sty_losses.append(
                sum(g_ref_sty_losses)/len(g_ref_sty_losses))
            self.g_ref_ds_losses.append(
                sum(g_ref_ds_losses)/len(g_ref_ds_losses))
            self.g_ref_cyc_losses.append(
                sum(g_ref_cyc_losses)/len(g_ref_cyc_losses))

            self.print_log(epoch+1, start_time)
            # generate images for debugging
            if val_dataset:
                self.save_images(val_dataset, epoch+1, 2)

            # save model checkpoints
            # if (i+1) % args.save_every == 0:
            #     self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            # if (i+1) % args.eval_every == 0:
            #     calculate_metrics(nets_ema, args, i+1, mode='latent')
            #     calculate_metrics(nets_ema, args, i+1, mode='reference')

    def d_loss(self, x_real, y_org, y_trg, x_fake):
        # with real images
        x_real.requires_grad_()
        out = self.D(x_real, y_org)
        loss_real = self.adv_loss(out, 1)
        loss_reg = self.r1_reg(out, x_real)

        # with fake images
        out = self.D(x_fake, y_trg)
        loss_fake = self.adv_loss(out, 0)

        loss = loss_real + loss_fake + self.lambda_reg * loss_reg
        return loss, (loss_real.item(), loss_fake.item(), loss_reg.item())

    def g_loss(self, x_real, y_org, y_trg, s_trg, s_trg2):
        # adversarial loss
        x_fake = self.G(x_real, s_trg)
        out = self.D(x_fake, y_trg)
        loss_adv = self.adv_loss(out, 1)

        # style reconstruction loss
        s_pred = self.style_encoder(x_fake, y_trg)
        loss_sty = torch.mean(torch.abs(s_pred - s_trg))

        # diversity sensitive loss
        # TODO: detach?
        x_fake2 = self.G(x_real, s_trg2).detach()
        loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

        # cycle-consistency loss
        s_org = self.style_encoder(x_real, y_org)
        x_rec = self.G(x_fake, s_org)
        loss_cyc = torch.mean(torch.abs(x_rec - x_real))

        loss = loss_adv + self.lambda_sty * loss_sty \
            - self.lambda_ds * loss_ds + self.lambda_cyc * loss_cyc
        return loss, (loss_adv.item(),
                      loss_sty.item(),
                      loss_ds.item(),
                      loss_cyc.item())

    def adv_loss(self, logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target).to(self.device)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    def r1_reg(self, d_out, x_in):
        # zero-centered gradient penalty for real images
        batch = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = 0.5 * grad_dout2.view(batch, -1).sum(1).mean(0)
        return reg

    def print_log(self, epoch, start_time):
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
        print(f"\nEpoch {epoch}: {elapsed}\n"
              f"D       real: {self.d_real_losses[-1]}\n"
              f"         reg: {self.d_reg_losses[-1]}\n"
              f"      fake z: {self.d_fake_latent_losses[-1]}\n"
              f"    fake ref: {self.d_fake_ref_losses[-1]}\n"
              f"G latent adv: {self.g_latent_adv_losses[-1]}\n"
              f"         sty: {self.g_latent_sty_losses[-1]}\n"
              f"         cyc: {self.g_latent_cyc_losses[-1]}\n"
              f"          ds: {self.g_latent_ds_losses[-1]}\n"
              f"G    ref adv: {self.g_ref_adv_losses[-1]}\n"
              f"         sty: {self.g_ref_sty_losses[-1]}\n"
              f"         cyc: {self.g_ref_cyc_losses[-1]}\n"
              f"          ds: {self.g_ref_ds_losses[-1]}\n"
              )

    @torch.no_grad()
    def save_images(self, val_dataset, epoch, n=8):
        device = self.device

        # sample from validation dataset
        x_src = []
        y_src = []
        for i in random.sample(range(len(val_dataset)), n):
            x_src.append(val_dataset[i][0])
            y_src.append(val_dataset[i][1])

        x_src = torch.stack(x_src).to(device)
        y_src = torch.tensor(y_src).to(device)

        x_ref = []
        y_ref = []
        for i in random.sample(range(len(val_dataset)), n):
            x_ref.append(val_dataset[i][0])
            y_ref.append(val_dataset[i][1])

        x_ref = torch.stack(x_ref).to(device)
        y_ref = torch.tensor(y_ref).to(device)

        # translate and reconstruct (reference-guided)
        reconst = self.translate_and_reconstruct(x_src, y_src, x_ref, y_ref)
        filename = ospj(self.sample_dir, f'{epoch:03d}_cycle_consistency.jpg')
        vutils.save_image(reconst.cpu(), filename, nrow=n*2)

        # latent-guided image synthesis
        y_trg_list = [torch.tensor(y).repeat(n).to(device)
                      for y in range(self.n_domains)]
        z_trg_list = torch.randn(
            n, 1, self.latent_dim).repeat(1, n, 1).to(device)
        for psi in [0.5, 0.7, 1.0]:
            lat = self.translate_using_latent(
                x_src, y_trg_list, z_trg_list, psi)
            filename = ospj(self.sample_dir, f'{epoch:03d}_latent_psi_{psi:.1f}.jpg')
            vutils.save_image(lat.cpu(), filename, nrow=n)

        # reference-guided image synthesis
        ref = self.translate_using_reference(x_src, x_ref, y_ref)
        filename = ospj(self.sample_dir, f'{epoch:03d}_reference.jpg')
        vutils.save_image(ref.cpu(), filename, nrow=n+1)

    @torch.no_grad()
    def translate_and_reconstruct(self, x_src, y_src, x_ref, y_ref):
        N, C, H, W = x_src.size()
        s_ref = self.style_encoder(x_ref, y_ref)
        x_fake = self.G(x_src, s_ref)
        s_src = self.style_encoder(x_src, y_src)
        x_rec = self.G(x_fake, s_src)
        x_concat = [x_src, x_ref, x_fake, x_rec]
        x_concat = torch.cat(x_concat, dim=0)

        return denormalize(x_concat)
    
    @torch.no_grad()
    def translate_using_latent(self, x_src, y_trg_list, z_trg_list, psi):
        N, C, H, W = x_src.size()
        latent_dim = z_trg_list[0].size(1)
        x_concat = [x_src]

        for i, y_trg in enumerate(y_trg_list):
            z_many = torch.randn(10000, latent_dim).to(x_src.device)
            y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
            s_many = self.mapping_network(z_many, y_many)
            s_avg = torch.mean(s_many, dim=0, keepdim=True)
            s_avg = s_avg.repeat(N, 1)

            for z_trg in z_trg_list:
                s_trg = self.mapping_network(z_trg, y_trg)
                s_trg = torch.lerp(s_avg, s_trg, psi)
                x_fake = self.G(x_src, s_trg)
                x_concat += [x_fake]

        return denormalize(torch.cat(x_concat, dim=0))        

    @torch.no_grad()
    def translate_using_reference(self, x_src, x_ref, y_ref):
        N, C, H, W = x_src.size()

        s_ref = self.style_encoder(x_ref, y_ref)
        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
        x_concat = []
        for i, s_ref in enumerate(s_ref_list):
            x_fake = self.G(x_src, s_ref)
            x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
            x_concat += [x_fake_with_ref]

        return denormalize(torch.cat(x_concat, dim=0))

    def save_model(self):
        torch.save(self.G.state_dict(), ospj(self.working_dir, 'G.pth'))
        torch.save(self.D.state_dict(), ospj(self.working_dir, 'D.pth'))
        torch.save(self.F.state_dict(), ospj(self.working_dir, 'F.pth'))
        torch.save(self.E.state_dict(), ospj(self.working_dir, 'E.pth'))

    # @torch.no_grad()
    # def sample(self, epoch, n_samples=16):
    #     os.makedirs(self.result_dir, exist_ok=True)

    #     # src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
    #     # ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

    #     utils.translate_using_reference(
    #         nets_ema, args, src.x, ref.x, ref.y, fname)

    #     N, C, H, W = x_src.size()
    #     s_ref = nets.style_encoder(x_ref, y_ref)
    #     x_fake = nets.generator(x_src, s_ref)
    #     s_src = nets.style_encoder(x_src, y_src)
    #     x_rec = nets.generator(x_fake, s_src)
    #     x_concat = [x_src, x_ref, x_fake, x_rec]
    #     x_concat = torch.cat(x_concat, dim=0)
    #     save_image(x_concat, N, filename)
    #     del x_concat

    #     out = (x + 1) / 2
    #     return out.clamp_(0, 1)

    #     x = denormalize(x)
    #     vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

    # def moving_average(self, model, model_test, beta=0.999):
    #     for param, param_test in zip(model.parameters(), model_test.parameters()):
    #         param_test.data = torch.lerp(param.data, param_test.data, beta)

    # @torch.no_grad()
    # def evaluate(self):
    #     args = self.args
    #     nets_ema = self.nets_ema
    #     resume_iter = args.resume_iter
    #     self._load_checkpoint(args.resume_iter)
    #     calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
    #     calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')

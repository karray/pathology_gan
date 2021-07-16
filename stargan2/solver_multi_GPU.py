import os
import csv
import time
import datetime
from os.path import join as ospj

import numpy as np
from numpy.random import randint
import random
import wandb
import requests

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.utils as vutils

from stargan2.model import build_model

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

SEND = 'https://api.telegram.org/bot'+os.environ['TG']+'/'
def send(text):
    return requests.post(SEND+'sendMessage', json={'chat_id': 80968060, 'text': text}).json()['result']['message_id']

def update_msg(text, msg_id):
    return requests.post(SEND+'editMessageText', json={'chat_id': 80968060, 'text': text, 'message_id': msg_id})

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

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

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
        cuda: List of 4 stings representing devise names
    """

    def __init__(self, name, run_name, img_size, n_domains=2, style_dim=64, latent_dim=64,
                 lambda_sty=1, lambda_ds=1, lambda_reg=1,
                 lambda_cyc=1, weight_decay=1e-4, pretrained_path=None):
        super().__init__()

        self.name = name
        self.run_name = run_name

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
        working_dir = ospj('results', f'{name}_{run_name}')
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
        self.acc = []

        lr = 1e-4
        # mapping network F learning rate
        f_lf = 1e-6
        beta1, beta2 = .0, .99

        (self.G, self.D, self.F, self.E) = build_model(
            img_size, style_dim, latent_dim, n_domains)

        self.G_device = torch.device('cuda:0')
        self.D_device = torch.device('cuda:1')
        self.F_device = torch.device('cuda:2')
        self.E_device = torch.device('cuda:3')

        if pretrained_path:
            self.load_models(pretrained_path)
        else:
            [net.apply(he_init) for net in (self.G, self.D, self.F, self.E)]

        self.G.to(self.G_device)
        self.D.to(self.D_device)
        self.F.to(self.F_device)
        self.E.to(self.E_device)

        self.G_opt = Adam(params=self.G.parameters(), lr=lr,
                          betas=[beta1, beta2], weight_decay=weight_decay)
        self.D_opt = Adam(params=self.D.parameters(), lr=lr,
                          betas=[beta1, beta2], weight_decay=weight_decay)
        self.F_opt = Adam(params=self.F.parameters(), lr=f_lf,
                          betas=[beta1, beta2], weight_decay=weight_decay)
        self.E_opt = Adam(params=self.E.parameters(), lr=lr,
                          betas=[beta1, beta2], weight_decay=weight_decay)

    def train(self, total_iter, loader, loader_ref, clf=None, val=None):
        if clf:
            clf.to(self.F_device)
            clf.eval()

        run = wandb.init(project=self.name)
        wandb.run.name = self.run_name
        wandb.run.save()

        # nets_ema = self.nets_ema
        # device = self.device

        # remember the initial value of ds weight
        initial_lambda_ds = self.lambda_ds

        G_device = self.G_device
        D_device = self.D_device
        F_device = self.F_device
        E_device = self.E_device

        data_iter = sample_data(loader)
        ref_iter = sample_data(loader_ref)

        best_acc = 0

        msg_id = send(self.name+': 0')
        for i in range(1, total_iter+1):
            # print(f'{(100*i/total_iter):6.2f}%', end='\r', flush=True)

            data = next(data_iter)
            data_ref = next(ref_iter)

            # unpack real data
            x_real, y_org = data
            # unpack reference data
            x_ref1, x_ref2, y_trg = data_ref

            batch = len(x_real)

            x_real_D = x_real.to(D_device)
            x_real_G = x_real.to(G_device)
            x_real_D.requires_grad_()
            x_real_G.requires_grad_()
            y_org_D = y_org.to(D_device)

            x_ref1 = x_ref1.to(E_device)
            x_ref2 = x_ref2.to(E_device)

            y_trg_D = y_trg.to(D_device)
            y_trg_F = y_trg.to(F_device)
            y_trg_E = y_trg.to(E_device)

            # generate latent vectors
            z_trg1 = torch.randn(batch, self.latent_dim).to(F_device)
            z_trg2 = torch.randn(batch, self.latent_dim).to(F_device)

            # *** train the discriminator ***

            # generate fakes with no_grad
            # since we don't want to update G, style_encoder and mapping_network
            with torch.no_grad():
                # using styles from noise
                s_latent = self.F(z_trg1, y_trg_F).to(G_device)
                x_fake_noise = self.G(x_real_G, s_latent).to(D_device)
                # using styles from reference images
                s_ref = self.E(x_ref1.to(E_device), y_trg_E).to(G_device)
                x_fake_ref = self.G(x_real_G, s_ref).to(D_device)

            d_loss, d_loss_stats = self.d_loss(
                x_real_D, y_org_D, y_trg_D, x_fake_noise)

            loss_real1, loss_fake, loss_reg1 = d_loss_stats
            self.d_fake_latent_losses.append(loss_fake)

            self.D_opt.zero_grad()
            d_loss.backward()
            self.D_opt.step()

            d_loss, d_loss_stats = self.d_loss(
                x_real_D, y_org_D, y_trg_D, x_fake_ref)

            loss_real2, loss_fake, loss_reg2 = d_loss_stats
            self.d_fake_ref_losses.append(loss_fake)

            self.d_real_losses.append((loss_real1+loss_real2)/2)
            self.d_reg_losses.append((loss_reg1+loss_reg2)/2)

            self.D_opt.zero_grad()
            d_loss.backward()
            self.D_opt.step()

            # *** train the generator ***

            # generate styles from noise
            s_trg_z1 = self.F(z_trg1, y_trg_F).to(G_device)
            s_trg_z2 = self.F(z_trg2, y_trg_F).to(G_device)

            g_loss, g_loss_stats = self.g_loss(
                x_real, y_org, y_trg, s_trg_z1, s_trg_z2)

            loss_adv, loss_sty, loss_ds, loss_cyc = g_loss_stats
            self.g_latent_adv_losses.append(loss_adv)
            self.g_latent_sty_losses.append(loss_sty)
            self.g_latent_ds_losses.append(loss_ds)
            self.g_latent_cyc_losses.append(loss_cyc)

            self.G_opt.zero_grad()
            self.F_opt.zero_grad()
            self.E_opt.zero_grad()
            g_loss.backward()
            self.G_opt.step()
            self.F_opt.step()
            self.E_opt.step()

            # generate styles from reference images
            s_trg_ref1 = self.E(x_ref1, y_trg_F).to(G_device)
            s_trg_ref2 = self.E(x_ref2, y_trg_F).to(G_device)

            g_loss, g_loss_stats = self.g_loss(
                x_real, y_org, y_trg, s_trg_ref1, s_trg_ref2)

            loss_adv, loss_sty, loss_ds, loss_cyc = g_loss_stats
            self.g_ref_adv_losses.append(loss_adv)
            self.g_ref_sty_losses.append(loss_sty)
            self.g_ref_ds_losses.append(loss_ds)
            self.g_ref_cyc_losses.append(loss_cyc)

            self.G_opt.zero_grad()
            g_loss.backward()
            self.G_opt.step()

            # compute moving average of network parameters
            # https://arxiv.org/abs/1806.04498
            # https://github.com/clovaai/stargan-v2/issues/62
            # self.moving_average(self.G, nets_ema.generator, beta=0.999)
            # self.moving_average(self.F,
            #                nets_ema.mapping_network, beta=0.999)
            # self.moving_average(self.E,
            #                nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if self.lambda_ds > 0:
                self.lambda_ds -= (initial_lambda_ds / self.ds_iter)
            
            if i%100==0:
                if clf:
                    acc = self.eval_G(clf, val)
                    self.acc.append(acc)

                update_msg(self.name+': '+str(i/total_iter), msg_id)
                self.print_log(i)

                if clf and acc > best_acc:
                    best_acc = acc
                    wandb.run.summary["best_acc"] = acc
                    self.save_model('best_')

            # generate images for debugging
            if i%1000==0:
                self.save_model()
                self.save_images(val, i, 4)

            # save model checkpoints
            # if (i+1) % args.save_every == 0:
            #     self._save_checkpoint(step=i+1)

            # compute FID and LPIPS if necessary
            # if (i+1) % args.eval_every == 0:
            #     calculate_metrics(nets_ema, args, i+1, mode='latent')
            #     calculate_metrics(nets_ema, args, i+1, mode='reference')
        self.save_model()
        self.save_stats()

        send(self.name+' done')
        run.finish()

    @torch.no_grad()
    def eval_G(self, clf, validation_loader):
        self.G.eval()
        self.E.eval()
        acc = .0
        for i, data in enumerate(validation_loader):
            X = data[0].to(self.E_device)
            y_s = data[2].to(self.F_device)
            s = self.E(X, y_s).to(self.G_device)

            X_g = self.G(X.to(self.G_device), s).to(self.F_device)
            predicted = torch.round(clf(0.5 * (X_g + 1.0)))
            
            y = data[1].to(self.F_device)
            acc+=(predicted == y).sum()/float(predicted.shape[0])     
    #             acc_g+=(predicted_g == y).sum()/float(predicted_g.shape[0])     
        self.G.train()
        self.E.train()
        return (acc/(i+1)).detach().item()


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
        x_real_G = x_real.to(self.G_device)
        x_fake = self.G(x_real_G, s_trg)
        out = self.D(x_fake.to(self.D_device), y_trg.to(
            self.D_device)).to(self.G_device)
        loss_adv = self.adv_loss(out, 1)

        # style reconstruction loss
        s_pred = self.E(x_fake.to(self.E_device), y_trg.to(
            self.E_device)).to(self.G_device)
        loss_sty = torch.mean(torch.abs(s_pred - s_trg))

        # diversity sensitive loss
        # TODO: detach?
        x_fake2 = self.G(x_real_G, s_trg2).detach()
        loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

        # cycle-consistency loss
        s_org = self.E(x_real.to(self.E_device), y_org.to(
            self.E_device)).to(self.G_device)
        x_rec = self.G(x_fake, s_org)
        loss_cyc = torch.mean(torch.abs(x_rec - x_real_G))

        loss = (loss_adv + self.lambda_sty * loss_sty
            - self.lambda_ds * loss_ds + self.lambda_cyc * loss_cyc)
        return loss, (loss_adv.item(),
                      loss_sty.item(),
                      loss_ds.item(),
                      loss_cyc.item())

    def adv_loss(self, logits, target):
        assert target in [1, 0]
        targets = torch.full_like(logits, fill_value=target).to(logits.device)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        # loss = F.mse_loss(logits, targets)
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

    def print_log(self, i):
        stats = {
              "d_real_losses": self.d_real_losses[-1],
              "d_reg_losses": self.d_reg_losses[-1],
              "d_fake_latent_losses": self.d_fake_latent_losses[-1],
              "d_fake_ref_losses": self.d_fake_ref_losses[-1],
              "g_latent_adv_losses": self.g_latent_adv_losses[-1],
              "g_latent_sty_losses": self.g_latent_sty_losses[-1],
              "g_latent_cyc_losses": self.g_latent_cyc_losses[-1],
              "g_latent_ds_losses": self.g_latent_ds_losses[-1],
              "g_ref_adv_losses": self.g_ref_adv_losses[-1],
              "g_ref_sty_losses": self.g_ref_sty_losses[-1],
              "g_ref_cyc_losses": self.g_ref_cyc_losses[-1],
              "g_ref_ds_losses": self.g_ref_ds_losses[-1]
        }
        if len(self.acc)>0:
            stats["acc"] = self.acc[-1]
        wandb.log(stats)
        # elapsed = time.time() - start_time
        # elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
        # print(f"Iter {i}: {elapsed}\n"
        #       f"D       real: {self.d_real_losses[-1]}\n"
        #       f"         reg: {self.d_reg_losses[-1]}\n"
        #       f"      fake z: {self.d_fake_latent_losses[-1]}\n"
        #       f"    fake ref: {self.d_fake_ref_losses[-1]}\n"
        #       f"G latent adv: {self.g_latent_adv_losses[-1]}\n"
        #       f"         sty: {self.g_latent_sty_losses[-1]}\n"
        #       f"         cyc: {self.g_latent_cyc_losses[-1]}\n"
        #       f"          ds: {self.g_latent_ds_losses[-1]}\n"
        #       f"G    ref adv: {self.g_ref_adv_losses[-1]}\n"
        #       f"         sty: {self.g_ref_sty_losses[-1]}\n"
        #       f"         cyc: {self.g_ref_cyc_losses[-1]}\n"
        #       f"          ds: {self.g_ref_ds_losses[-1]}\n"
        #       )

    @torch.no_grad()
    def save_images(self, val_dataset, epoch, n=4):
        domains, _ = val_dataset._find_classes(val_dataset.root)
        # sample from validation dataset
        x_src = []
        y_src = []
        for i in random.sample(range(len(val_dataset)), n):
            x_src.append(val_dataset[i][0])
            y_src.append(val_dataset[i][1])

        x_src = torch.stack(x_src)
        y_src = torch.tensor(y_src)

        x_ref = []
        y_ref_E = []
        for i in random.sample(range(len(val_dataset)), n):
            x_ref.append(val_dataset[i][0])
            y_ref_E.append(val_dataset[i][1])

        x_ref = torch.stack(x_ref)
        y_ref_E = torch.tensor(y_ref_E).to(self.E_device)

        # translate and reconstruct (reference-guided)
        filename = ospj(self.sample_dir, f'{epoch:03d}_cycle_consistency.jpg')
        reconst = self.translate_and_reconstruct(x_src, y_src, x_ref, y_ref_E, filename)
        # vutils.save_image(reconst.cpu(), filename, nrow=n*2)

        # latent-guided image synthesis
        y_trg_list = [torch.tensor(y).repeat(n).to(self.F_device)
                      for y in range(self.n_domains)]
        z_trg_list = torch.randn(
            1, 1, self.latent_dim).repeat(1, n, 1).to(self.F_device)
        for psi in [0.5, 0.7, 1.0]:
            filename = ospj(self.sample_dir,
                            f'{epoch:03d}_latent_psi_{psi:.1f}.jpg')
            lat = self.translate_using_latent(
                x_src, y_trg_list, z_trg_list, psi, filename, domains)
            # vutils.save_image(lat.cpu(), filename, nrow=n)

        # reference-guided image synthesis
        filename = ospj(self.sample_dir, f'{epoch:03d}_reference.jpg')
        ref = self.translate_using_reference(
            x_src.to(self.G_device), x_ref, y_ref_E, filename)

        return reconst, lat, ref

    @torch.no_grad()
    def translate_and_reconstruct(self, x_src, y_src, x_ref, y_ref_E, filename):
        N, C, H, W = x_src.size()
        x_src_G = x_src.to(self.G_device)
        s_ref = self.E(x_ref.to(self.E_device), y_ref_E)
        x_fake = self.G(x_src_G, s_ref.to(self.G_device))
        s_src = self.E(x_src.to(self.E_device), y_src.to(self.E_device))
        x_rec = self.G(x_fake, s_src.to(self.G_device))
        x_concat = [x_src_G, x_ref.to(self.G_device), x_fake, x_rec]
        x_concat = torch.cat(x_concat, dim=0)

        rec =  denormalize(x_concat).cpu()
        img = vutils.make_grid(rec, padding=0, nrow=N).numpy()
        self.save_plot(img, [50, 160, 305,395], ['real', 'real ref', 'fake', 'reconstructed'], filename)
        
        return rec

    @torch.no_grad()
    def translate_using_latent(self, x_src, y_trg_list, z_trg_list, psi, filename, domains):
        x_src = x_src.to(self.G_device)
        N, C, H, W = x_src.size()
        latent_dim = z_trg_list[0].size(1)
        x_concat = [x_src]

        for i, y_trg in enumerate(y_trg_list):
            z_many = torch.randn(10000, latent_dim).to(self.F_device)
            y_many = torch.LongTensor(10000).to(self.F_device).fill_(y_trg[0])
            s_many = self.F(z_many, y_many)
            s_avg = torch.mean(s_many, dim=0, keepdim=True)
            s_avg = s_avg.repeat(N, 1)

            for z_trg in z_trg_list:
                s_trg = self.F(z_trg, y_trg)
                s_trg = torch.lerp(s_avg, s_trg, psi).to(self.G_device)
                x_fake = self.G(x_src, s_trg)
                x_concat += [x_fake]

        lat = denormalize(torch.cat(x_concat, dim=0)).cpu()
        img = vutils.make_grid(lat, padding=0, nrow=N).numpy()
        labels = ['input', ] + domains
        
        self.save_plot(img, [40, 185, 300, 420, 560, 690][:len(labels)], labels, filename)

        return lat

    @torch.no_grad()
    def translate_using_reference(self, x_src_G, x_ref, y_ref_E, filename):
        N, C, H, W = x_src_G.size()
        x_ref_G = x_ref.to(self.G_device)
        wb = torch.ones(1, C, H, W).to(x_src_G.device)
        x_src_with_wb = torch.cat([wb, x_src_G], dim=0)

        s_ref = self.E(x_ref.to(self.E_device), y_ref_E).to(self.G_device)
        s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
        x_concat = [x_src_with_wb]
        for i, s_ref in enumerate(s_ref_list):
            x_fake = self.G(x_src_G, s_ref)
            x_fake_with_ref = torch.cat([x_ref_G[i:i+1], x_fake], dim=0)
            x_concat += [x_fake_with_ref]

        ref = denormalize(torch.cat(x_concat, dim=0)).cpu()
        vutils.save_image(ref, filename, padding=0, nrow=N+1)

        return ref

    def save_plot(self, img, ticks, labels, filename):
        fig = plt.figure()
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)
        plt.box(False)
        plt.xticks([], [])
        plt.yticks(ticks, labels)
        plt.yticks(rotation=90)
        plt.tick_params(length=0, labelsize=7)
        plt.savefig(filename, format='jpg', bbox_inches = 'tight', pad_inches = 0)
        plt.close()


    def load_models(self, root):
        self.G.load_state_dict(torch.load(ospj(root, 'G.pth'), map_location='cuda:0'))
        self.D.load_state_dict(torch.load(ospj(root, 'D.pth'), map_location='cuda:1'))
        self.F.load_state_dict(torch.load(ospj(root, 'F.pth'), map_location='cuda:2'))
        self.E.load_state_dict(torch.load(ospj(root, 'E.pth'), map_location='cuda:3'))

    def save_model(self, prefix=''):
        torch.save(self.G.state_dict(), ospj(self.working_dir, prefix+'G.pth'))
        torch.save(self.D.state_dict(), ospj(self.working_dir, prefix+'D.pth'))
        torch.save(self.F.state_dict(), ospj(self.working_dir, prefix+'F.pth'))
        torch.save(self.E.state_dict(), ospj(self.working_dir, prefix+'E.pth'))
        wandb.save(ospj(self.working_dir, prefix+'G.pth'))
        wandb.save(ospj(self.working_dir, prefix+'D.pth'))
        wandb.save(ospj(self.working_dir, prefix+'F.pth'))
        wandb.save(ospj(self.working_dir, prefix+'E.pth'))


    def save_stats(self):
        np.save(ospj(self.working_dir, 'd_real'), self.d_real_losses)
        np.save(ospj(self.working_dir, 'd_reg'), self.d_reg_losses)
        np.save(ospj(self.working_dir, 'd_fake_latent'), self.d_fake_latent_losses)
        np.save(ospj(self.working_dir, 'd_fake_ref'), self.d_fake_ref_losses)
        np.save(ospj(self.working_dir, 'g_latent_adv'), self.g_latent_adv_losses)
        np.save(ospj(self.working_dir, 'g_latent_sty'), self.g_latent_sty_losses)
        np.save(ospj(self.working_dir, 'g_latent_ds'), self.g_latent_ds_losses)
        np.save(ospj(self.working_dir, 'g_latent_cyc'), self.g_latent_cyc_losses)
        np.save(ospj(self.working_dir, 'g_ref_adv'), self.g_ref_adv_losses)
        np.save(ospj(self.working_dir, 'g_ref_sty'), self.g_ref_sty_losses)
        np.save(ospj(self.working_dir, 'g_ref_ds'), self.g_ref_ds_losses)
        np.save(ospj(self.working_dir, 'g_ref_cyc'), self.g_ref_cyc_losses)
        np.save(ospj(self.working_dir, 'acc'), self.acc)

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

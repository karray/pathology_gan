import math
import random
import os
from os.path import join as ospj

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
# from tqdm import tqdm

# from munch import Munch

import wandb
import requests

from stylegan2.model import Generator, Discriminator
# from dataset import MultiResolutionDataset
# from distributed import (
#     get_rank,
#     synchronize,
#     reduce_loss_dict,
#     reduce_sum,
#     get_world_size,
# )
# from non_leaking import augment, AdaptiveAugment


# def data_sampler(dataset, shuffle, distributed):
#     if distributed:
#         return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

#     if shuffle:
#         return data.RandomSampler(dataset)

#     else:
#         return data.SequentialSampler(dataset)

SEND = 'https://api.telegram.org/bot'+os.environ['TG']+'/'
def send(text):
    return requests.post(SEND+'sendMessage', json={'chat_id': 80968060, 'text': text}).json()['result']['message_id']

def update_msg(text, msg_id):
    return requests.post(SEND+'editMessageText', json={'chat_id': 80968060, 'text': text, 'message_id': msg_id})

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# def accumulate(model1, model2, decay=0.999):
#     par1 = dict(model1.named_parameters())
#     par2 = dict(model2.named_parameters())

#     for k in par1.keys():
#         par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3])

    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * \
        (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


class Solver:
    def __init__(self, name, run_name, img_size):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.name = name
        self.run_name = run_name
        self.working_dir = ospj('results', f'{name}_{run_name}')
        self.samples_dir = ospj(self.working_dir, 'samples')
        os.makedirs(self.samples_dir, exist_ok=True)

        self.path = ''  # "path to the lmdb dataset"
        self.iter = 800000  # "total training iterations"
        self.size = img_size  # "image sizes for the model"
        self.r1 = 10  # "weight of the r1 regularization"
        self.path_regularize = 2  # "weight of the path length regularization",
        self.path_batch_shrink = 2
        self.d_reg_every = 16  # "interval of the applying r1 regularization",
        self.g_reg_every = 4  # "interval of the applying path length regularization",
        self.mixing = 0.9  # "probability of latent code mixing"
        self.lr = 0.002  # "learning rate")
        self.channel_multiplier = 2
        self.local_rank = 0  # "local rank for distributed training"
        self.ada_target = 0.6  # "target augmentation probability for adaptive augmentation",
        self.ada_length = 500 * 1000
        self.ada_every = 256  # "probability update interval of the adaptive augmentation",



        self.latent = 512
        self.n_mlp = 8

        self.start_iter = 0

        self.G = Generator(
            self.size, self.latent, self.n_mlp, channel_multiplier=self.channel_multiplier
        ).to(self.device)
        self.D = Discriminator(
            self.size, channel_multiplier=self.channel_multiplier
        ).to(self.device)

        g_reg_ratio = self.g_reg_every / (self.g_reg_every + 1)
        d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.G.parameters(),
            lr=self.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        self.d_optim = optim.Adam(
            self.D.parameters(),
            lr=self.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )


    def train(self, loader, batch):
        run = wandb.init(project=self.name)
        wandb.run.name = self.run_name
        wandb.run.save()

        device = self.device
        loader = sample_data(loader)

        # if clf:
        #     clf.to(device)

        mean_path_length = 0

        d_loss_val = 0
        r1_loss = torch.tensor(0.0, device=device)
        g_loss_val = 0
        path_loss = torch.tensor(0.0, device=device)
        path_lengths = torch.tensor(0.0, device=device)
        mean_path_length_avg = 0
        loss_dict = {}

        best_acc = 0

        msg_id = send(self.name+': 0')
        for i in range(1, self.iter+1):
            real_img = next(loader)[0].to(device)

            requires_grad(self.G, False)
            requires_grad(self.D, True)

            noise = mixing_noise(batch, self.latent, self.mixing, device)
            fake_img, _ = self.G(noise)

            real_img_aug = real_img

            fake_pred = self.D(fake_img)
            real_pred = self.D(real_img_aug)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            loss_dict["d"] = d_loss
            loss_dict["real_score"] = real_pred.mean()
            loss_dict["fake_score"] = fake_pred.mean()

            self.D.zero_grad()
            d_loss.backward()
            self.d_optim.step()

            d_regularize = i % self.d_reg_every == 0

            if d_regularize:
                real_img.requires_grad = True
                real_pred = self.D(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)

                self.D.zero_grad()
                (self.r1 / 2 * r1_loss * self.d_reg_every +
                 0 * real_pred[0]).backward()

                self.d_optim.step()

            loss_dict["r1"] = r1_loss

            requires_grad(self.G, True)
            requires_grad(self.D, False)

            noise = mixing_noise(batch, self.latent, self.mixing, device)
            fake_img, _ = self.G(noise)

            fake_pred = self.D(fake_img)
            g_loss = g_nonsaturating_loss(fake_pred)

            loss_dict["g"] = g_loss

            self.G.zero_grad()
            g_loss.backward()
            self.g_optim.step()

            g_regularize = i % self.g_reg_every == 0

            if g_regularize:
                path_batch_size = max(1, batch // self.path_batch_shrink)
                noise = mixing_noise(
                    path_batch_size, self.latent, self.mixing, device)
                fake_img, latents = self.G(noise, return_latents=True)

                path_loss, mean_path_length, path_lengths = g_path_regularize(
                    fake_img, latents, mean_path_length
                )

                self.G.zero_grad()
                weighted_path_loss = self.path_regularize * self.g_reg_every * path_loss

                if self.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                weighted_path_loss.backward()

                self.g_optim.step()

                mean_path_length_avg = mean_path_length.item()

            loss_dict["path"] = path_loss
            loss_dict["path_length"] = path_lengths.mean()


            if i % 1000 == 0:
                loss_reduced = loss_dict

                d_loss_val = loss_reduced["d"].mean().item()
                g_loss_val = loss_reduced["g"].mean().item()
                r1_val = loss_reduced["r1"].mean().item()
                path_loss_val = loss_reduced["path"].mean().item()
                real_score_val = loss_reduced["real_score"].mean().item()
                fake_score_val = loss_reduced["fake_score"].mean().item()
                path_length_val = loss_reduced["path_length"].mean().item()

                # acc = self.eval_G(clf, valloader)

                update_msg(self.name+': '+str(i/self.iter), msg_id)

                # if acc > best_acc:
                #     best_acc = acc
                #     wandb.run.summary["best_acc"] = acc
                #     self.save_model('best_')

                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 10000 == 0:
                with torch.no_grad():
                    # g_ema.eval()
                    # sample, _ = g_ema([sample_z])
                    # self.G.eval()
                    # sample, _ = self.G([sample_z])
                    utils.save_image(
                        fake_img,
                        ospj(self.samples_dir, f'{i:6d}.png'),
                        nrow=int(batch**0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
        self.save_model('final_')

    @torch.no_grad()
    def eval_G(self, clf, validation_loader):
        self.G.eval()
        acc = .0
        for i, data in enumerate(validation_loader):
            X = data[0].to(self.device)
            y = data[1].to(self.device)

            noise = mixing_noise(len(y), self.latent, self.mixing, self.device)
            X_g, _ = self.G(noise)
            predicted = torch.round(clf(0.5 * (X_g + 1.0)))

            acc += (predicted == y).sum()/float(predicted.shape[0])
    #             acc_g+=(predicted_g == y).sum()/float(predicted_g.shape[0])
        self.G.train()
        return (acc/(i+1)).detach().item()

    def load_models(self, root):
        self.G.load_state_dict(torch.load(ospj(root, 'G.pth')))
        self.D.load_state_dict(torch.load(ospj(root, 'D.pth')))

    def save_model(self, prefix=''):
        torch.save(self.G.state_dict(), ospj(self.working_dir, prefix+'G.pth'))
        torch.save(self.D.state_dict(), ospj(self.working_dir, prefix+'D.pth'))
        wandb.save(ospj(self.working_dir, prefix+'G.pth'))
        wandb.save(ospj(self.working_dir, prefix+'D.pth'))

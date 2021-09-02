import requests
import os
from os.path import join as ospj

import torch

from torchvision.utils import save_image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.backends import cudnn
cudnn.benchmark = True

import wandb

from StyleGAN2_img2img.model import *
from StyleGAN2_img2img.loss import discriminator_hinge_loss, generator_hinge_loss

SEND = 'https://api.telegram.org/bot'+os.environ['TG']+'/'

def send(text):
    return requests.post(SEND+'sendMessage', json={'chat_id': 80968060, 'text': text}).json()['result']['message_id']

def update_msg(text, msg_id):
    return requests.post(SEND+'editMessageText', json={'chat_id': 80968060, 'text': text, 'message_id': msg_id})

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

class Solver():
    """
    ## Configurations
    """

    def __init__(self, name: str, run_name: str, image_size: int, data_path: str,
                 batch_size: int = 32,
                 d_latent: int = 32,
                 mapping_network_layers: int = 8,
                 learning_rate: float = 1e-3,
                 mapping_network_learning_rate: float = 1e-5,
                 gradient_accumulate_steps: int = 1,
                 style_mixing_prob: float = 0.9,
                 device_name: str = 'cuda'):

        self.transform = transforms.Compose([
            # Resize the image
            transforms.Resize(image_size),
            # Convert to PyTorch tensor
            transforms.ToTensor(),
        ])
        data_set = ImageFolder(data_path, self.transform)
        self.loader = sample_data(DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True))

        self.name = name
        self.run_name = run_name

        if not os.path.exists('results'):
            os.mkdir('results')
        working_dir = ospj('results', f'{name}_{run_name}')
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

        self.sample_dir = ospj(working_dir, 'samples')
        if not os.path.exists(self.sample_dir):
            os.mkdir(self.sample_dir)

        self.working_dir = working_dir

        self.device: torch.device = torch.device(
            device_name if torch.cuda.is_available() else 'cpu')

        # [Gradient Penalty Regularization Loss](index.html#gradient_penalty)
        self.gradient_penalty = GradientPenalty()
        # Gradient penalty coefficient $\gamma$
        self.gradient_penalty_coefficient: float = 10.

        # Batch size
        self.batch_size = batch_size
        # Dimensionality of $z$ and $w$
        self.d_latent: int = d_latent
        # Height/width of the image
        self.image_size: int = image_size
        # Number of layers in the mapping network
        self.mapping_network_layers: int = mapping_network_layers
        # Generator & Discriminator learning rate
        self.learning_rate: float = learning_rate
        # Mapping network learning rate ($100 \times$ lower than the others)
        self.mapping_network_learning_rate: float = mapping_network_learning_rate
        # Number of steps to accumulate gradients on. Use this to increase the effective batch size.
        self.gradient_accumulate_steps: int = gradient_accumulate_steps
        # Probability of mixing styles
        self.style_mixing_prob: float = style_mixing_prob

        # ### Lazy regularization
        # Instead of calculating the regularization losses, the paper proposes lazy regularization
        # where the regularization terms are calculated once in a while.
        # This improves the training efficiency a lot.

        # The interval at which to compute gradient penalty
        self.lazy_gradient_penalty_interval: int = 4
        # Path length penalty calculation interval
        self.lazy_path_penalty_interval: int = 32
        # Skip calculating path length penalty during the initial phase of training
        self.lazy_path_penalty_after: int = 5_000

        # How often to log generated images
        self.log_generated_interval: int = 500
        self.log_interval: int = 100
        # How often to save model checkpoints
        self.save_checkpoint_interval: int = 2_000

        # $\log_2$ of image resolution
        log_resolution = int(math.log2(self.image_size))

        self.E_content = ContentEncoder(log_resolution).to(self.device)
        self.E_style = StyleEncoder(log_resolution, d_latent).to(self.device)

        # Create discriminator and generator
        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(
            log_resolution, self.d_latent).to(self.device)
        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = self.generator.n_blocks
        # Create mapping network
        self.mapping_network = MappingNetwork(
            self.d_latent, self.mapping_network_layers).to(self.device)
        # Create path length penalty loss
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        # Discriminator and generator losses
        self.discriminator_loss = discriminator_hinge_loss
        self.generator_loss = generator_hinge_loss

        # $\beta_1$ and $\beta_2$ for Adam optimizer
        self.adam_betas: Tuple[float, float] = (0.0, 0.99)

        # Create optimizers
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate, betas=self.adam_betas
        )
        self.E_style_optimizer = torch.optim.Adam(
            self.E_style.parameters(),
            lr=self.mapping_network_learning_rate, betas=self.adam_betas
        )
        self.E_content_optimizer = torch.optim.Adam(
            self.E_content.parameters(),
            lr=self.mapping_network_learning_rate, betas=self.adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate, betas=self.adam_betas
        )

    # def get_content(self, encodings):
    #     """
    #     ### Generate content
    #     This generates content for each [generator block](index.html#generator_block)
    #     """
    #     # List to store content
    #     noise = []
    #     # Noise resolution starts from $4$
    #     resolution = 4

    #     # Generate noise for each generator block
    #     for i in range(self.n_gen_blocks):
    #         # The first block has only one $3 \times 3$ convolution
    #         if i == 0:
    #             n1 = None
    #         # Generate noise to add after the first convolution layer
    #         else:
    #             n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
    #         # Generate noise to add after the second convolution layer
    #         n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

    #         # Add noise tensors to the list
    #         noise.append((n1, n2))

    #         # Next block has $2 \times$ resolution
    #         resolution *= 2

    #     # Return noise tensors
    #     return noise

    def generate_images(self, style_embeddings, content_embeddings):
        """
        ### Generate images
        This generate images using the generator
        """

        # Get $w$ and expand it for the generator blocks [n_blocks, batch, latent_dim]
        w = self.mapping_network(style_embeddings)[
            None, :, :].expand(self.n_gen_blocks, -1, -1)

        images = self.generator(w, content_embeddings)

        return images, w

    def step(self, idx: int):
        """
        ### Training Step
        """

        #### Train the discriminator ###

        self.discriminator_optimizer.zero_grad()

        disc_loss_real_accum = 0
        disc_loss_fake_accum = 0

        # Accumulate gradients for `gradient_accumulate_steps`
        for i in range(self.gradient_accumulate_steps):
            real_images = next(self.loader)[0].to(self.device)

            with torch.no_grad():
                style_embeddings = self.E_style(real_images)
                content_embeddings = self.E_content(real_images)
                # flip the order of style embeddings
                generated_images, _ = self.generate_images(
                    torch.flip(style_embeddings, [0]), content_embeddings)

            fake_output = self.discriminator(generated_images)

            # We need to calculate gradients w.r.t. real images for gradient penalty
            if idx % self.lazy_gradient_penalty_interval == 0:
                real_images.requires_grad_()
            # Discriminator classification for real images
            real_output = self.discriminator(real_images)

            # Get discriminator loss
            real_loss, fake_loss = self.discriminator_loss(
                real_output, fake_output)
            disc_loss = real_loss + fake_loss

            disc_loss_real_accum+= real_loss.detach().cpu().item()
            disc_loss_fake_accum+= fake_loss.detach().cpu().item()

            # Add gradient penalty
            if idx % self.lazy_gradient_penalty_interval == 0:
                # Calculate and log gradient penalty
                gp = self.gradient_penalty(real_images, real_output)
                # tracker.add('loss.gp', gp)
                # Multiply by coefficient and add gradient penalty
                disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * \
                    gp * self.lazy_gradient_penalty_interval

            # Compute gradients
            disc_loss.backward()

            # Log discriminator loss
            # tracker.add('loss.discriminator', disc_loss)


        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(), max_norm=1.0)
        
        # Take optimizer step
        self.discriminator_optimizer.step()

        #### Train the generator ####

        # Reset gradients
        self.E_style_optimizer.zero_grad()
        self.E_content_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        gen_loss_accum = 0
        style_loss_accum = 0
        content_loss_accum = 0
        cycle_loss_accum = 0

        # Accumulate gradients for `gradient_accumulate_steps`
        for i in range(self.gradient_accumulate_steps):
            real_images = next(self.loader)[0].to(self.device)

            style_embeddings = self.E_style(real_images)
            content_embeddings = self.E_content(real_images)
            # flip the order of style
            generated_images, w = self.generate_images(
                torch.flip(style_embeddings, [0]), content_embeddings)
            # Discriminator classification for generated images
            fake_output = self.discriminator(generated_images)

            # Get generator loss
            gen_loss = self.generator_loss(fake_output)
            gen_loss_accum+= gen_loss.detach().cpu().item()

            # style and content reconstruction loss
            style_rec = torch.flip(self.E_style(generated_images), [0])
            contnet_rec = self.E_content(generated_images)
            style_loss = torch.mean(torch.abs(style_rec - style_embeddings))
            content_loss = torch.mean(torch.abs(contnet_rec - content_embeddings))
            style_loss_accum+= style_loss.detach().cpu().item()
            content_loss_accum+= content_loss.detach().cpu().item()

            # cycle-consistency loss
            reconstructed_images, _ = self.generate_images(style_rec, contnet_rec)
            loss_cyc = torch.mean(torch.abs(reconstructed_images - real_images))
            cycle_loss_accum+= loss_cyc.detach().cpu().item()

            gen_loss+= style_loss + content_loss + loss_cyc

            # Add path length penalty
            if idx > self.lazy_path_penalty_after and idx % self.lazy_path_penalty_interval == 0:
                # Calculate path length penalty
                plp = self.path_length_penalty(w, generated_images)
                # Ignore if `nan`
                if not torch.isnan(plp):
                    gen_loss = gen_loss + plp
                    wandb.log({"path_penalty": plp.detach().cpu().item()})

            # Calculate gradients
            gen_loss.backward()

        if idx % self.log_interval == 0:
            wandb.log(
                {
                    "disc_loss_real": disc_loss_real_accum,
                    "disc_loss_fake": disc_loss_fake_accum,
                    "disc_loss": (disc_loss_real_accum+disc_loss_fake_accum)/2,
                    "gen_loss": gen_loss_accum,
                    "style_loss": style_loss_accum,
                    "content_loss": content_loss_accum,
                    "cycle_loss": cycle_loss_accum,
                }
            )

        if idx % self.save_checkpoint_interval == 0:
            torch.save(self.discriminator.state_dict(), ospj(self.working_dir, 'discriminator.pth'))
            torch.save(self.generator.state_dict(), ospj(self.working_dir, 'generator.pth'))
            torch.save(self.E_content.state_dict(), ospj(self.working_dir, 'E_content.pth'))
            torch.save(self.E_style.state_dict(), ospj(self.working_dir, 'E_style.pth'))

            wandb.save(ospj(self.working_dir, 'discriminator.pth'))
            wandb.save(ospj(self.working_dir, 'generator.pth'))
            wandb.save(ospj(self.working_dir, 'E_content.pth'))
            wandb.save(ospj(self.working_dir, 'E_style.pth'))


        # Clip gradients for stabilization
        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            self.mapping_network.parameters(), max_norm=1.0)

        # Take optimizer step
        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()  
        self.E_content_optimizer.step()
        self.E_style_optimizer.step()

        # Log generated images
        if idx % self.log_generated_interval == 0:
            save_image(torch.cat([ torch.clamp(real_images[:6], 0, 1), generated_images[:6]], dim=0), ospj(self.sample_dir, f'{idx:06d}.jpg'), padding=0, nrow=6)
        # Save model checkpoints
        # if idx % self.save_checkpoint_interval == 0:
        #     experiment.save_checkpoint()


    def train(self, training_steps):
        """
        ## Train model
        """
        run = wandb.init(project=self.name)
        wandb.run.name = self.run_name
        wandb.run.save()
        name = self.name+'_'+self.run_name
        msg_id = send(name+': 0')
        # Loop for `training_steps`
        for i in range(1, training_steps+1):
            # Take a training step
            self.step(i)
            if i%100==0:
                update_msg(f'{name}: {i/training_steps:1.4f}', msg_id)
        send(name+' done')
        wandb.finish() 

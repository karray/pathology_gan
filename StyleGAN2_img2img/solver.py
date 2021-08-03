
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import wandb
from os.path import join as ospj


class Solver:
    """
    ## Configurations
    """

    def __init__(self, name: str, run_name: str, image_size: int,
                batch_size: int=32,
                d_latent: int=32,
                mapping_network_layers: int=8,
                learning_rate: float=1e-3,
                mapping_network_learning_rate: float = 1e-5,
                gradient_accumulate_steps: int = 1,
                style_mixing_prob: float = 0.9
                device: str='cuda'):

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

        device: torch.device =  torch.device(
                device if cuda and torch.cuda.is_available() else 'cpu')

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
        # How often to save model checkpoints
        self.save_checkpoint_interval: int = 2_000

        # $\log_2$ of image resolution
        log_resolution = int(math.log2(self.image_size))

        # Create discriminator and generator
        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        # Get number of generator blocks for creating style and noise inputs
        self.n_gen_blocks = self.generator.n_blocks
        # Create mapping network
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)
        # Create path length penalty loss
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        # Discriminator and generator losses
        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

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
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate, betas=self.adam_betas
        )

    def get_noise(self, batch_size: int):
        """
        ### Generate noise
        This generates noise for each [generator block](index.html#generator_block)
        """
        # List to store noise
        noise = []
        # Noise resolution starts from $4$
        resolution = 4

        # Generate noise for each generator block
        for i in range(self.n_gen_blocks):
            # The first block has only one $3 \times 3$ convolution
            if i == 0:
                n1 = None
            # Generate noise to add after the first convolution layer
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)
            # Generate noise to add after the second convolution layer
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=self.device)

            # Add noise tensors to the list
            noise.append((n1, n2))

            # Next block has $2 \times$ resolution
            resolution *= 2

        # Return noise tensors
        return noise

    def generate_images(self, batch_size: int):
        """
        ### Generate images
        This generate images using the generator
        """

        # Get $w$ and expand it for the generator blocks
        w = self.mapping_network(encodings)[None, :, :].expand(self.n_gen_blocks, -1, -1)
        # Get noise
        noise = self.get_noise(batch_size)

        # Generate images
        images = self.generator(w, noise)

        # Return images and $w$
        return images, w

    def step(self, idx: int):
        """
        ### Training Step
        """

        # Train the discriminator
        with monit.section('Discriminator'):
            # Reset gradients
            self.discriminator_optimizer.zero_grad()

            # Accumulate gradients for `gradient_accumulate_steps`
            for i in range(self.gradient_accumulate_steps):
                # Update `mode`. Set whether to log activation
                with self.mode.update(is_log_activations=(idx + 1) % self.log_generated_interval == 0):
                    # Sample images from generator
                    generated_images, _ = self.generate_images(self.batch_size)
                    # Discriminator classification for generated images
                    fake_output = self.discriminator(generated_images.detach())

                    # Get real images from the data loader
                    real_images = next(self.loader).to(self.device)
                    # We need to calculate gradients w.r.t. real images for gradient penalty
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        real_images.requires_grad_()
                    # Discriminator classification for real images
                    real_output = self.discriminator(real_images)

                    # Get discriminator loss
                    real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
                    disc_loss = real_loss + fake_loss

                    # Add gradient penalty
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        # Calculate and log gradient penalty
                        gp = self.gradient_penalty(real_images, real_output)
                        tracker.add('loss.gp', gp)
                        # Multiply by coefficient and add gradient penalty
                        disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

                    # Compute gradients
                    disc_loss.backward()

                    # Log discriminator loss
                    tracker.add('loss.discriminator', disc_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                # Log discriminator model parameters occasionally
                tracker.add('discriminator', self.discriminator)

            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            # Take optimizer step
            self.discriminator_optimizer.step()

        # Train the generator
        with monit.section('Generator'):
            # Reset gradients
            self.generator_optimizer.zero_grad()
            self.mapping_network_optimizer.zero_grad()

            # Accumulate gradients for `gradient_accumulate_steps`
            for i in range(self.gradient_accumulate_steps):
                # Sample images from generator
                generated_images, w = self.generate_images(self.batch_size)
                # Discriminator classification for generated images
                fake_output = self.discriminator(generated_images)

                # Get generator loss
                gen_loss = self.generator_loss(fake_output)

                # Add path length penalty
                if idx > self.lazy_path_penalty_after and (idx + 1) % self.lazy_path_penalty_interval == 0:
                    # Calculate path length penalty
                    plp = self.path_length_penalty(w, generated_images)
                    # Ignore if `nan`
                    if not torch.isnan(plp):
                        tracker.add('loss.plp', plp)
                        gen_loss = gen_loss + plp

                # Calculate gradients
                gen_loss.backward()

                # Log generator loss
                tracker.add('loss.generator', gen_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                # Log discriminator model parameters occasionally
                tracker.add('generator', self.generator)
                tracker.add('mapping_network', self.mapping_network)

            # Clip gradients for stabilization
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

            # Take optimizer step
            self.generator_optimizer.step()
            self.mapping_network_optimizer.step()

        # Log generated images
        if (idx + 1) % self.log_generated_interval == 0:
            tracker.add('generated', torch.cat([generated_images[:6], real_images[:3]], dim=0))
        # Save model checkpoints
        if (idx + 1) % self.save_checkpoint_interval == 0:
            experiment.save_checkpoint()

        # Flush tracker
        tracker.save()

    def train(self, training_steps):
        """
        ## Train model
        """

        # Loop for `training_steps`
        for i in monit.loop(training_steps):
            # Take a training step
            self.step(i)
            #
            if (i + 1) % self.log_generated_interval == 0:
                tracker.new_line()


def main():
    """
    ### Train StyleGAN2
    """

    # Create an experiment
    experiment.create(name='stylegan2')
    # Create configurations object
    configs = Configs()

    # Set configurations and override some
    experiment.configs(configs, {
        'device.cuda_device': 0,
        'image_size': 64,
        'log_generated_interval': 200
    })

    # Initialize
    configs.init()
    # Set models for saving and loading
    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)

    # Start the experiment
    with experiment.start():
        # Run the training loop
        configs.train()

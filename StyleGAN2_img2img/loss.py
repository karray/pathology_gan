import torch.utils.data
from torch.nn import functional as F

def discriminator_hinge_loss(f_real: torch.Tensor, f_fake: torch.Tensor):
    r"""
    ## Discriminator Loss
    We want to find $w$ to maximize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big(x^{(i)} \big) +
     \frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    * `f_real` is $f_w(x)$
    * `f_fake` is $f_w(g_\theta(z))$

    This returns the a tuple with losses for $f_w(x)$ and $f_w(g_\theta(z))$,
    which are later added.
    They are kept separate for logging.
    """

    # We use ReLUs to clip the loss to keep $f \in [-1, +1]$ range.
    return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


def generator_hinge_loss(f_fake: torch.Tensor):
    r"""
    ## Generator Loss
    We want to find $\theta$ to minimize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$
    The first component is independent of $\theta$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$

    * `f_fake` is $f_w(g_\theta(z))$
    """
    return -f_fake.mean()
from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb
from torch.distributions import Normal
import torch.nn.functional as F
import math

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


Sample = namedtuple("Sample", "trajectories values chains")


@torch.no_grad()
def default_sample_fn(model, x, cond, t, cond_key):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        n_timesteps=1000,
        loss_type="l1",
        clip_denoised=False,
        predict_epsilon=True,
        action_weight=1.0,
        loss_discount=1.0,
        loss_weights=None,
        hidden_dim=256,
        cond_key=[0],
        node_height=1,
        eta_weight=1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.cond_key = cond_key
        self.model = model
        self.node_height = node_height
        self.eta_weight = eta_weight

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        """
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        weights_dict    : dict
            { i: c } multiplies dimension i of observation loss by c
        """
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None:
            weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum("h,t->ht", discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, : self.action_dim] = action_weight
        return loss_weights

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = torch.cat(
            [
                self.posterior_log_variance_clipped[1:2],
                self.posterior_log_variance_clipped[1:],
            ]
        )
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, return_x_recon=False):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_x_recon:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        cond,
        verbose=False,
        return_chain=False,
        sample_fn=default_sample_fn,
        **sample_kwargs
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim, self.cond_key)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, self.cond_key, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim, self.cond_key)

            progress.update(
                {"t": i, "vmin": values.min().item(), "vmax": values.max().item()}
            )
            if return_chain:
                chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain:
            chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        """
        conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, **sample_kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def mean_flat(self, tensor):
        """
        Take the mean over all non-batch dimensions.
        """
        return tensor.mean(dim=list(range(1, len(tensor.shape))))

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def p_losses(self, x_start, cond, t):

        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim, self.cond_key)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim, self.cond_key)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        prob = self.cal_prob(x_start, cond, t)
        entropy = self.cal_entropy(prob)

        if self.node_height == 1:
            loss -= (0.1 * entropy)
        elif self.node_height > 1:
            loss += (0.1 * self.eta_weight * entropy)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()        
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

    def cal_entropy(self, prob, eps=1e-8):
        prob = prob.clamp(min=eps, max=1.0 - eps)
        log_prob = torch.log(prob)
        entropy = -torch.sum(prob * log_prob, dim=-1)
        return torch.mean(entropy)

    def cal_prob(self, x, cond, t=None, eps=1e-8):
        log_prob = self.raw_log_prob(x, cond, t)  # [batch_size, element_number]

        min_log_prob = log_prob.min(dim=1, keepdim=True).values
        scaling_exponent = torch.ceil((-min_log_prob / 10).clamp(min=0))
        weights = 10 ** scaling_exponent
        log_prob = log_prob / weights

        prob = torch.exp(log_prob)
        prob_dist = prob / (prob.sum(dim=-1, keepdim=True) + eps)

        return prob_dist


    def raw_log_prob(self, x, cond, t=None):
        device = self.betas.device
        batch_size, element_num, dim = x.shape

        if t is None:
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=device).long()

        t_expanded = t.unsqueeze(1).expand(-1, element_num).reshape(-1)

        x_reshaped = x.view(-1, 1, dim)

        # Optimize cond_flat
        if isinstance(cond, dict):
            cond_flat = {
                k: v.repeat_interleave(element_num, dim=0) if isinstance(v, torch.Tensor) else v
                for k, v in cond.items()
            }
        else:
            cond_flat = cond

        noise = torch.randn_like(x_reshaped)
        x_noisy = self.q_sample(x_start=x_reshaped, t=t_expanded, noise=noise)

        x_noisy = apply_conditioning(x_noisy, cond_flat, self.action_dim, self.cond_key)

        model_out = self.model(x_noisy, cond_flat, t_expanded).reshape(-1, dim)
        x_target = x.view(-1, dim)

        if self.clip_denoised:
            model_out = model_out.clamp(-1.0, 1.0)

        posterior_variance = extract(self.posterior_variance, t_expanded, x_target.shape)
        posterior_variance = posterior_variance.clamp(min=1e-6)

        log_prob = -0.5 * (
            (x_target - model_out) ** 2 / posterior_variance +
            torch.log(2 * torch.tensor(math.pi) * posterior_variance)
        ).sum(-1)

        return log_prob.view(batch_size, element_num)

class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):

        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim, self.cond_key)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x_start, cond, t):

        return self.model(x_start, cond, t)

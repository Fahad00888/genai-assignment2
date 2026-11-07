import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

def gather_timestep_coeff(schedule, timesteps, x_shape):
    """
    In diffusion, we keep 1D schedules (e.g., $\sqrt{\bar\alpha_t}$) indexed by
    timestep. This helper picks the coefficient for each item in the batch and
    reshapes it so it can multiply an image/feature tensor of shape `x_shape`.

    To use this function, consider the example
        alpha_t * x
    To compute this in code, we would write
        gather_timestep_coeff(alpha, t, x.shape) * x

    Args:
        schedule: 1D tensor containing values for each timestep.
        timesteps: 1D tensor containing a batch of time indices.
        x_shape: The reference shape.
    Returns:
        The timestep coefficients, reshaped for broadcasting.
    """
    b, *_ = timesteps.shape
    out = schedule.gather(-1, timesteps)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Passes the input timesteps through the cosine schedule for the diffusion process
    Args:
        timesteps: Number of timesteps int. 
        s: The strength of the schedule.
    Returns:
        1D tensor of length timesteps with computed alphas indexed from t = 0 to t = T-1.
    """
    # Define alpha_bar for t in [0, T], then compute alpha_t for t = 0..T-1.
    steps = timesteps
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)


# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        # ---- Noise schedule (Eq. 14), indexed t = 0 .. T-1 ----
        alpha = cosine_schedule(self.num_timesteps)              # alpha_t
        alpha_bar = torch.cumprod(alpha, dim=0)                   # \bar{alpha}_t = prod_{i<=t} alpha_i
        # \bar{alpha}_{t-1}: shift right by one and put 1 at t = 0 (no noise before the first step)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])

        # Buffers move with .to(device) and are saved in state_dict,
        # but they are NOT trainable parameters.
        def keep(name, value):
            self.register_buffer(name, value.float())

        keep("alpha", alpha)
        keep("alpha_bar", alpha_bar)
        keep("alpha_bar_prev", alpha_bar_prev)

        # Forward process, Eq. 13: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps
        keep("signal_scale", torch.sqrt(alpha_bar))
        keep("noise_scale", torch.sqrt(1.0 - alpha_bar))

        # Posterior q(x_{t-1} | x_t, x_0), Eq. 18 / Alg. 2 lines 7-8
        #   mean = coef_xt * x_t + coef_x0 * x0_hat
        #   var  = (1 - abar_{t-1}) / (1 - abar_t) * (1 - alpha_t)
        keep("coef_xt", torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
        keep("coef_x0", torch.sqrt(alpha_bar_prev) * (1.0 - alpha) / (1.0 - alpha_bar))
        keep("posterior_std", torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * (1.0 - alpha)))

    def _at(self, name, t, x):
        """Shortcut: pick schedule `name` at timesteps t, shaped to broadcast with x."""
        return gather_timestep_coeff(getattr(self, name), t, x.shape)
    
    ##############################################
    # FORWARD DIFFUSION    
    ##############################################

    def q_sample(self, x_0, t, noise):
        """
        Applies alpha interpolation between x_0 and noise to simulate sampling
        x_t from the noise distribution.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to sample at.
            noise: The noise tensor to sample from.
        Returns:
            The sampled images.
        """
        # Eq. 13: jump straight from x_0 to x_t in one step (closed form, O(1) in t)
        x_t = self._at("signal_scale", t, x_0) * x_0 + self._at("noise_scale", t, x_0) * noise
        return x_t


    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial images.
            t: 1D tensor containing a batch of time indices to compute the loss at.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        # Alg. 1 lines 5-6: noise the image, ask the U-Net which noise was added,
        # and penalise the L1 distance between true and predicted noise.
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        loss = F.l1_loss(predicted_noise, noise)
        return loss
    
    def forward(self, x_0, noise):
        """
        Acts as a wrapper for p_losses.
        Args:
            x_0: The initial images.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # Alg. 1 line 3: one random timestep per image, t ~ Uniform{0, ..., T-1}
        t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
        return self.p_losses(x_0, t, noise)

    ############################################################
    # BACKWARD DIFFUSION
    ############################################################

    def noise_like(self, shape, device):
        """
        Generates noise with the same shape as the input.
        Args:
            shape: The shape of the noise.
            device: The device on which to create the noise.
        Returns:
            The generated noise.
        """
        noise = lambda: torch.randn(shape, device=device)
        return noise()

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Computes the (t_index)th sample from the (t_index + 1)th sample using
        the reverse diffusion process.
        Args:
            x: The sampled image at timestep t_index + 1.
            t: 1D tensor of the index of the time step.
            t_index: Scalar of the index of the time step.
        Returns:
            The sampled image at timestep t_index.
        """
        ## TODO: Implement the p_sample function #######
        # sample x_{t-1} from the gaussian distribution wrt. posterior mean and posterior variance
        # Hint: use gather_timestep_coeff function to get the coefficients at time t
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Alg. 2 line 4: predicted noise
        eps = self.model(x, t)
        # Alg. 2 lines 5-6: estimate the clean image (Eq. 19) and keep it in [-1, 1]
        x0_hat = (x - self._at("noise_scale", t, x) * eps) / self._at("signal_scale", t, x)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        # Alg. 2 line 7: posterior mean
        mean = self._at("coef_xt", t, x) * x + self._at("coef_x0", t, x) * x0_hat

        if t_index == 0:
            # last step: z = 0, return the mean without adding noise
            return mean
        else:
            # Alg. 2 lines 3, 8-9: add scaled Gaussian noise
            z = self.noise_like(x.shape, x.device)
            return mean + self._at("posterior_std", t, x) * z

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Passes noise through the entire reverse diffusion process to generate
        final image samples.
        Args:
            img: The initial noise that is randomly sampled from the noise distribution.
        Returns:
            The sampled images.
        """
        b = img.shape[0]
        #### TODO: Implement the p_sample_loop function ####
        # 1. loop through the time steps from the last to the first
        # 2. inside the loop, sample x_{t-1} from the reverse diffusion process
        # 3. clamp and unnormalize the generated image to valid pixel range
        # Hint: to get time index, you can use torch.full()
        for step in range(self.num_timesteps - 1, -1, -1):          # T-1, ..., 0
            t = torch.full((b,), step, device=img.device, dtype=torch.long)
            img = self.p_sample(img, t, step)
        # back from [-1, 1] model space to [0, 1] pixel space
        img = unnormalize_to_zero_to_one(torch.clamp(img, -1.0, 1.0))
        return img

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Wrapper function for p_sample_loop.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        #### TODO: Implement the sample function ####
        # Hint: use self.noise_like function to generate noise. DO NOT USE torch.randn
        # Alg. 2 line 1: start from pure Gaussian noise x_{T-1} ~ N(0, I)
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = self.noise_like(shape, self.alpha.device)
        return self.p_sample_loop(img)
        # ####################################################

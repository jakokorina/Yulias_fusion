"""
Original code: https://github.com/ashawkey/stable-dreamfusion
"""

from diffusers import DDPMScheduler, DDIMScheduler, KandinskyV22Pipeline, KandinskyV22CombinedPipeline, DiffusionPipeline
from transformers import logging

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import custom_bwd, custom_fwd


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        (gt_grad,) = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class KandinskyDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device

        print(f"[INFO] loading kandisnky diffusion...")

        model_key = "kandinsky-community/kandinsky-2-2-decoder"

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = KandinskyV22Pipeline.from_pretrained(
            model_key, torch_dtype=self.precision_t
        )

        self.encoder = pipe.movq.encoder#.to(device)
        self.quant_conv = pipe.movq.quant_conv#.to(device)
        self.quantize = pipe.movq.quantize#.to(device)
        
        self.unet = pipe.unet.to(device)

        self.encoder.eval()
        self.quant_conv.eval()
        self.quantize.eval()
        self.unet.eval()
        
        self.scheduler = DDPMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.precision_t
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f"[INFO] loaded kandisnky diffusion!")

    @torch.no_grad()
    def compute_text_emb(self, prompt, negative_prompt=""):
        # prompt: [str]
        pipe_prior = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16)
        pipe_prior.to(self.device)
        
        image_embeds, negative_image_embeds = pipe_prior(prompt, guidance_scale=1.0).to_tuple()
        image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0).to(
            dtype=self.unet.dtype, device=self.device
        )
        del pipe_prior
        return image_embeds

    def sds(
        self,
        text_emb,
        rgb,
        guidance_scale=100,
        grad_scale=1,
    ):
        """Score distillation sampling"""
        self.encoder.to(self.device)
        self.quant_conv.to(self.device)
        self.quantize.to(self.device)

        self.unet.to(self.device)

        num_rays, _ = rgb.shape
        h = w = int(num_rays ** (1 / 2))
        rgb = rearrange(rgb, "(h w) c -> 1 c h w", h=h, w=w)

        rgb = F.interpolate(rgb, (768, 768), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(rgb)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            
            image_embeds = text_emb
            added_cond_kwargs = {"image_embeds": image_embeds}
            
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

        # perform guidance (high scale from paper!)
        
        noise_pred, variance_pred = noise_pred.split(latents.shape[1], dim=1)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        _, variance_pred_text = variance_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        noise_pred = torch.cat([noise_pred, variance_pred_text], dim=1)

        noise_pred, _ = noise_pred.split(latents.shape[1], dim=1)

        # w(t), sigma_t^2
        w = 1 - self.alphas[t]
        grad = grad_scale * w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        h = self.encoder(imgs)
        h = self.quant_conv(h)
        latents, _, _ = self.quantize(h)
        return latents

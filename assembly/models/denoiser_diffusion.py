"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

import torch
import torch.nn.functional as F
from diffusers.schedulers import DDPMScheduler
from scipy.spatial.transform import Rotation as R

from .denoiser_base import DenoiserBase


class DenoiserDiffusion(DenoiserBase):
    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        val_noise_scheduler: DDPMScheduler,
        **kwargs,
    ):
        super().__init__(
            noise_scheduler=noise_scheduler,
            val_noise_scheduler=val_noise_scheduler,
            **kwargs,
        )

        self.noise_scheduler = noise_scheduler
        self.val_noise_scheduler = val_noise_scheduler

    def forward(self, data_dict: dict):
        B, P = data_dict["points_per_part"].shape
        part_valids = data_dict["points_per_part"] != 0

        gt_trans = data_dict["translations"][part_valids]  # (valid_P, 3)
        gt_rots = data_dict["quaternions"][part_valids]  # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)  # (valid_P, 7)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.get("num_train_timesteps"),
            (B,),
            device=self.device,
        ).long()  # (B,)
        timesteps = timesteps.repeat(P, 1).T  # (B, P)
        timesteps = timesteps[part_valids]  # (valid_P,)

        noise = torch.randn(gt_trans_and_rots.shape, device=self.device)
        noise_rots = (
            torch.tensor(R.random(gt_rots.size(0)).as_quat()).float().to(self.device)
        )[..., [3, 0, 1, 2]]
        noise[..., 3:] = noise_rots

        noisy_trans_and_rots = self.noise_scheduler.add_noise(
            gt_trans_and_rots, noise, timesteps
        )  # (valid_P, 7)
        noisy_trans_and_rots[data_dict["ref_part"][part_valids]] = gt_trans_and_rots[
            data_dict["ref_part"][part_valids]
        ]  # (valid_P, 7)

        # Extract features with adjacency model
        latent = self._extract_features(data_dict)

        denoiser_out = self.denoiser(
            x=noisy_trans_and_rots,
            timesteps=timesteps,
            latent=latent,
            part_valids=part_valids,
            scale=data_dict["scale"][part_valids],
            ref_part=data_dict["ref_part"][part_valids],
        )
        model_pred = denoiser_out["pred"]

        output_dict = {
            "model_pred": model_pred,
            "gt_noise": noise,
        }

        return output_dict

    def _loss(self, data_dict: dict, output_dict: dict):
        model_pred = output_dict["model_pred"]  # (valid_P, 7)
        gt_noise = output_dict["gt_noise"]  # (valid_P, 7)

        part_valids = data_dict["points_per_part"] != 0  # (B, P)
        ref_part_mask = ~data_dict["ref_part"][part_valids]  # (valid_P,)

        mse_loss = F.mse_loss(
            model_pred[ref_part_mask],
            gt_noise[ref_part_mask],
        )

        return {
            "mse_loss": mse_loss,
        }, set(["mse_loss"])

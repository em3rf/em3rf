"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

import torch
import torch.nn.functional as F

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R

from .denoiser_base import DenoiserBase
from .modules.scheduler import SE3FlowMatchEulerDiscreteScheduler
from .modules.loss_func import NoOverlapLoss


class DenoiserFlowMatching(DenoiserBase):
    def __init__(
        self,
        noise_scheduler: SE3FlowMatchEulerDiscreteScheduler,
        val_noise_scheduler: SE3FlowMatchEulerDiscreteScheduler,
        use_no_overlap_loss: bool = False,         # <--- NEW
        no_overlap_weight: float = 0.1,            # <--- NEW
        no_overlap_grid_size: int = 32,            # <--- NEW
        **kwargs,
    ):
        super().__init__(
            noise_scheduler=noise_scheduler,
            val_noise_scheduler=val_noise_scheduler,
            **kwargs,
        )

        self.noise_scheduler = noise_scheduler
        self.val_noise_scheduler = val_noise_scheduler

        # <--- NEW BLOCK --->
        self.use_no_overlap_loss = use_no_overlap_loss
        self.no_overlap_weight = no_overlap_weight
        if self.use_no_overlap_loss:
            print(f"✅ No-Overlap Loss is ENABLED! (Weight: {self.no_overlap_weight}, Grid Size: {no_overlap_grid_size})")
            self.no_overlap_criterion = NoOverlapLoss(grid_size=no_overlap_grid_size)
        else:
            print("❌ No-Overlap Loss is DISABLED.")
        # <----------------->

    def get_sigmas(self, timesteps: torch.Tensor, ndim: int, dtype: torch.dtype):
        sigmas = self.noise_scheduler.sigmas
        schedule_timesteps = self.noise_scheduler.timesteps.to(device=timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(self, data_dict: dict):
        B, P = data_dict["points_per_part"].shape
        part_valids = data_dict["points_per_part"] != 0

        gt_trans = data_dict["translations"][part_valids]  # (valid_P, 3)
        gt_rots = data_dict["quaternions"][part_valids]  # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)  # (valid_P, 7)

        # Sample random timestep for each object i.e. B dimension
        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=B,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler.config.get("num_train_timesteps")).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(
            device=gt_trans_and_rots.device
        )  # (B,)
        timesteps = timesteps.repeat(P, 1).T  # (B, P)
        timesteps = timesteps[part_valids]  # (valid_P,)

        # Add noise according to flow matching
        sigmas = self.get_sigmas(
            timesteps, ndim=gt_trans_and_rots.ndim, dtype=gt_trans_and_rots.dtype
        ).to(
            self.device
        )  # (valid_P, 7)
        noise = torch.randn(gt_trans_and_rots.shape, device=self.device)
        noise_rots = (
            torch.tensor(R.random(gt_rots.size(0)).as_quat()).float().to(self.device)
        )[..., [3, 0, 1, 2]]
        noise[..., 3:] = noise_rots

        noisy_trans_and_rots, gt_vec_field = self.noise_scheduler.scale_noise(
            sample=gt_trans_and_rots,
            timestep=timesteps,
            noise=noise,
        )

        gt_vec_field[data_dict["ref_part"][part_valids]] = 0.0
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

        weighting = compute_loss_weighting_for_sd3("none", sigmas)

        # precondition
        model_pred_trans = (
            model_pred[..., :3] * (-sigmas) + noisy_trans_and_rots[..., :3]
        )
        model_pred_rots = transforms.matrix_to_quaternion(
            transforms.axis_angle_to_matrix(-sigmas * model_pred[..., 3:])
            @ transforms.quaternion_to_matrix(noisy_trans_and_rots[..., 3:])
        )

        output_dict = {
            "model_pred": model_pred,
            "model_pred_trans": model_pred_trans,
            "model_pred_rots": model_pred_rots,
            "target": gt_vec_field,
            "gt_trans": gt_trans,
            "gt_rots": gt_rots,
            "weighting": weighting,
        }

        return output_dict

    def _loss(self, data_dict: dict, output_dict: dict):
        model_pred = output_dict["model_pred"]  # (valid_P, 6)
        model_pred_trans = output_dict["model_pred_trans"]  # (valid_P, 3)
        model_pred_rots = output_dict["model_pred_rots"]  # (valid_P, 4)
        target = output_dict["target"]  # (valid_P, 6)
        gt_trans = output_dict["gt_trans"]  # (valid_P, 3)
        gt_rots = output_dict["gt_rots"]  # (valid_P, 4)
        weighting = output_dict["weighting"]  # (valid_P,)

        vec_mse_loss = F.mse_loss(model_pred, target, reduction="none")
        vec_mse_loss = (vec_mse_loss * weighting).mean()

        trans_mse_loss = F.mse_loss(model_pred_trans, gt_trans, reduction="none")
        trans_mse_loss = (trans_mse_loss * weighting).mean()

        rot_mse_loss = F.mse_loss(model_pred_rots, gt_rots, reduction="none")
        rot_mse_loss = (rot_mse_loss * weighting).mean()

        cos_theta = torch.sum(model_pred_rots * gt_rots, dim=-1)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        rot_rmse = torch.acos(cos_theta)
        rot_rmse = torch.rad2deg(rot_rmse)
        rot_rmse = torch.sqrt(rot_rmse.pow(2).mean())

        # return {
        #     "vec_mse_loss": vec_mse_loss,
        #     "trans_mse_loss": trans_mse_loss,
        #     "rot_mse_loss": rot_mse_loss,
        #     "rot_rmse": rot_rmse,
        # }, set(["vec_mse_loss"])
        # Define the base loss dictionary
        loss_dict = {
            "vec_mse_loss": vec_mse_loss,
            "trans_mse_loss": trans_mse_loss,
            "rot_mse_loss": rot_mse_loss,
            "rot_rmse": rot_rmse,
        }
        counted_losses = set(["vec_mse_loss"])

        # <--- NEW BLOCK: Compute No-Overlap Loss --->
        if self.use_no_overlap_loss:
            pts = data_dict["pointclouds"]
            points_per_part = data_dict["points_per_part"]
            part_valids = points_per_part != 0
            B, P = points_per_part.shape
            valid_P = part_valids.sum().item()
            
            batch_indices = torch.arange(B, device=self.device).unsqueeze(1).repeat(1, P)[part_valids]
            
            # Handle both Uniform (ndim=4) and Weighted (ndim=3) point cloud sampling
            if pts.ndim == 4:
                N = pts.shape[2]
                expanded_part_scale = data_dict["scale"].unsqueeze(-1).expand(-1, -1, N, -1)
                pts_scaled = (pts * expanded_part_scale)[part_valids] # (valid_P, N, 3)
                
                rots_exp = model_pred_rots.unsqueeze(1).expand(-1, N, -1)
                trans_exp = model_pred_trans.unsqueeze(1).expand(-1, N, -1)
                transformed_points = transforms.quaternion_apply(rots_exp, pts_scaled) + trans_exp
                
                transformed_points = transformed_points.view(-1, 3)
                point_to_part_idx = torch.arange(valid_P, device=self.device).unsqueeze(1).expand(-1, N).reshape(-1)
            else:
                C = pts.shape[-1]
                scale = data_dict["scale"][part_valids]
                scale = scale.repeat_interleave(points_per_part[part_valids], dim=0)
                pts_scaled = pts.view(-1, C) * scale # (N_total, 3)
                
                part_indices = torch.arange(valid_P, device=self.device)
                point_to_part_idx = part_indices.repeat_interleave(points_per_part[part_valids], dim=0)
                
                point_rots = model_pred_rots[point_to_part_idx]
                point_trans = model_pred_trans[point_to_part_idx]
                transformed_points = transforms.quaternion_apply(point_rots, pts_scaled) + point_trans

            no_overlap_val = self.no_overlap_criterion(
                transformed_points=transformed_points,
                point_to_part_idx=point_to_part_idx,
                valid_P=valid_P,
                batch_indices=batch_indices
            )
            
            loss_dict["no_overlap_loss"] = no_overlap_val
            loss_dict["weighted_no_overlap_loss"] = no_overlap_val * self.no_overlap_weight
            counted_losses.add("weighted_no_overlap_loss")
        # <----------------------------------------->

        return loss_dict, counted_losses

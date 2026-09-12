import copy
import json
from typing import List
from functools import cache
from PIL import Image

import hydra
import gradio as gr
import torch
import lightning as L
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R
import torch.amp

from assembly.data.inference import MeshInferenceDataset
from assembly.models.denoiser.modules.evaluation.evaluator import (
    calc_part_acc_weighted,
    calc_shape_cd_weighted,
    rot_metrics,
    trans_metrics,
)

gr.set_static_paths(["gallery"])

if not OmegaConf.has_resolver("getIndex"):
    OmegaConf.register_new_resolver("getIndex", lambda lst, idx: lst[idx])

with hydra.initialize(version_base=None, config_path="./configs", job_name="app"):
    cfg = hydra.compose(config_name="app")

global_model: L.LightningModule = hydra.utils.instantiate(cfg.get("model"))
fabric: L.Fabric = hydra.utils.instantiate(cfg.get("fabric"))
global_model = fabric.setup_module(global_model)
ckpt = fabric.load(cfg.get("ckpt_path"))
# replace key: "adjacency_model." with "feature_extractor."
if "adjacency_model" in ckpt["state_dict"]:
    ckpt["state_dict"] = {
        k.replace("adjacency_model.", "feature_extractor."): v
        for k, v in ckpt["state_dict"].items()
    }
    torch.save(ckpt, cfg.get("ckpt_path"))
global_model.load_state_dict(state_dict=ckpt["state_dict"])
global_model.eval()


def inference(
    meshes: List[str],
    meshes_paths: List[str],
    mesh_type: str = "obj",
    settings: dict = None,
):
    assert len(meshes_paths) > 0 or len(meshes) > 0, "Please upload at least one mesh."
    # Extract settings
    seed = settings.get("seed", 1116)
    num_points_to_sample = settings.get("samplePoints", 5000)
    num_inference_steps = settings.get("steps", 20)
    max_iters = settings.get("maxIterations", 6)
    one_step_init = settings.get("oneStepInit", False)
    sample_strategy = settings.get("sampleStrategy", "uniform")
    lora_ckpt = settings.get("loraCheckpoint", None)

    # Copy the model when using LoRA
    model = copy.deepcopy(global_model)
    if lora_ckpt is not None:
        model.enable_lora(lora_ckpt)

    torch_rng = torch.Generator("cuda").manual_seed(seed)
    numpy_rng = np.random.default_rng(seed)
    fabric.seed_everything(seed)

    dataset = MeshInferenceDataset(
        name="inference",
        meshes_paths=meshes_paths if len(meshes_paths) > 0 else meshes,
        pad_to_parts=len(meshes_paths) if len(meshes_paths) > 0 else len(meshes),
        num_points_to_sample=num_points_to_sample,
        mesh_type=mesh_type,
        sample_method="weighted",
        seed=seed,
        sample_strategy=sample_strategy,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    dataloader = fabric.setup_dataloaders(dataloader)

    scheduler = copy.deepcopy(model.val_noise_scheduler)
    for data_dict in dataloader:
        points_per_part = data_dict["points_per_part"]
        part_valids = points_per_part != 0
        part_scale = data_dict["scale"][part_valids]  # (valid_P, 1)
        ref_part = data_dict["ref_part"][part_valids]  # (valid_P,)
        pointclouds = data_dict["pointclouds_gt"]  # (B, P, N, 3) or (B, N_sum, 3)
        points_per_valid_part = points_per_part[part_valids].tolist()
        splited_pointclouds = []

        if pointclouds.ndim == 4:
            pointclouds = pointclouds[part_valids]
            splited_pointclouds = torch.split_with_sizes(
                pointclouds.view(-1, 3), points_per_valid_part
            )
            pointclouds = pointclouds.tolist()
        else:
            pointclouds = pointclouds.view(-1, 3)
            splited_pointclouds = torch.split_with_sizes(
                pointclouds, points_per_valid_part
            )
            pointclouds = [pc.tolist() for pc in splited_pointclouds]

        yield {"type": "mesh_scale", "data": data_dict["mesh_scale"].item()}

        yield {
            "type": "input",
            "data": {
                "pointclouds": pointclouds,
                "initial_translation": data_dict["translations"][part_valids].tolist(),
                "initial_rotation": data_dict["quaternions"][part_valids].tolist(),
            },
        }

        gt_trans = data_dict["translations"][part_valids]  # (valid_P, 3)
        gt_rots = data_dict["quaternions"][part_valids]  # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)  # (valid_P, 7)

        noisy_trans_and_rots = torch.empty_like(gt_trans_and_rots).normal_(
            generator=torch_rng
        )  # (valid_P, 7)
        noise_rots = (
            torch.tensor(R.random(gt_rots.size(0), random_state=numpy_rng).as_quat())
            .float()
            .to(noisy_trans_and_rots.device)
        )[..., [3, 0, 1, 2]]
        noisy_trans_and_rots[..., 3:] = noise_rots

        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        yield {
            "type": "transformation",
            "data": {
                "translation": noisy_trans_and_rots[:, :3].tolist(),
                "rotation": noisy_trans_and_rots[:, 3:].tolist(),
                "step": -1,
                "iter": -1,
            },
        }

        with fabric.autocast(), torch.no_grad():
            feature = model.feature_extractor(data_dict)
            latent = feature["point"]
            latent["batch"] = latent["batch"].clone()
            coarse_seg_pred = feature["coarse_seg_pred"]
            yield {
                "type": "fracture_segmentation",
                "data": [
                    pcd[seg].tolist()
                    for seg, pcd in zip(
                        torch.split_with_sizes(
                            coarse_seg_pred > 0.5,
                            points_per_valid_part,
                        ),
                        splited_pointclouds,
                    )
                ],
            }

        if one_step_init:
            scheduler.set_timesteps(1)
            t = scheduler.timesteps[0]
            timesteps = (
                t.reshape(-1)
                .repeat(len(noisy_trans_and_rots))
                .to(noisy_trans_and_rots.device)
            )
            with fabric.autocast(), torch.no_grad():
                model_pred = model.denoiser(
                    x=noisy_trans_and_rots,
                    timesteps=timesteps,
                    latent=latent,
                    part_valids=part_valids,
                    scale=part_scale,
                    ref_part=ref_part,
                )
                model_pred = model_pred["pred"]

                noisy_trans_and_rots = scheduler.step(
                    model_pred, t, noisy_trans_and_rots
                ).prev_sample
                noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part].to(
                    dtype=noisy_trans_and_rots.dtype
                )

            yield {
                "type": "transformation",
                "data": {
                    "translation": noisy_trans_and_rots[:, :3].tolist(),
                    "rotation": noisy_trans_and_rots[:, 3:].tolist(),
                    "step": 0,
                    "iter": -1,
                },
            }

        for iter in range(max_iters):
            scheduler.set_timesteps(num_inference_steps)
            for step, t in enumerate(scheduler.timesteps):
                timesteps = (
                    t.reshape(-1)
                    .repeat(len(noisy_trans_and_rots))
                    .to(noisy_trans_and_rots.device)
                )

                with fabric.autocast(), torch.no_grad():
                    model_pred = model.denoiser(
                        x=noisy_trans_and_rots,
                        timesteps=timesteps,
                        latent=latent,
                        part_valids=part_valids,
                        scale=part_scale,
                        ref_part=ref_part,
                    )
                    model_pred = model_pred["pred"]

                    noisy_trans_and_rots = scheduler.step(
                        model_pred, t, noisy_trans_and_rots
                    ).prev_sample
                    noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part].to(
                        dtype=noisy_trans_and_rots.dtype
                    )

                    yield {
                        "type": "transformation",
                        "data": {
                            "translation": noisy_trans_and_rots[:, :3].tolist(),
                            "rotation": noisy_trans_and_rots[:, 3:].tolist(),
                            "step": step + 1,
                            "iter": iter + 1,
                        },
                    }

            if iter + 1 == max_iters:
                break

        B, P = points_per_part.shape
        pred_trans = noisy_trans_and_rots[..., :3].detach()
        pred_rots = noisy_trans_and_rots[..., 3:].detach()

        # Recover SE3 back to padded mode
        pred_trans_padded = torch.zeros(
            (B, P, 3), device=pred_trans.device, dtype=pred_trans.dtype
        )
        pred_rots_padded = torch.zeros(
            (B, P, 4), device=pred_rots.device, dtype=pred_rots.dtype
        )
        gt_trans_padded = torch.zeros(
            (B, P, 3), device=gt_trans.device, dtype=pred_trans.dtype
        )
        gt_rots_padded = torch.zeros(
            (B, P, 4), device=gt_rots.device, dtype=pred_rots.dtype
        )
        pred_trans_padded[part_valids] = pred_trans
        pred_rots_padded[part_valids] = pred_rots
        gt_trans_padded[part_valids] = gt_trans.to(dtype=gt_trans_padded.dtype)
        gt_rots_padded[part_valids] = gt_rots.to(dtype=gt_rots_padded.dtype)

        # Two scenarios: (B, P, N, 3) or (B, N_sum, 3)
        # First one is for uniform sampling, second one is for weighted sampling
        # We have to calculate shape_cd and part_acc differently
        pts = data_dict["pointclouds"]
        B, N_sum, C = pts.shape
        scale = data_dict["scale"][part_valids]
        scale = scale.repeat_interleave(points_per_part[part_valids], dim=0)
        pts = (pts.view(-1, C) * scale).view(B, N_sum, C)

        # Calculate Part Acc
        acc = calc_part_acc_weighted(
            pts,
            gt_trans=gt_trans,
            gt_rots=gt_rots,
            pred_trans=pred_trans,
            pred_rots=pred_rots,
            points_per_part=points_per_part,
            part_valids=part_valids,
            part_valids_wo_redundancy=part_valids,
        )[0]

        # Calculate Shape Chamfer Distance
        shape_cd = calc_shape_cd_weighted(
            pts,
            gt_trans=gt_trans,
            gt_rots=gt_rots,
            pred_trans=pred_trans,
            pred_rots=pred_rots,
            points_per_part=points_per_part,
            part_valids=part_valids,
            part_valids_wo_redundancy=part_valids,
        )[0]

        rmse_r = rot_metrics(pred_rots_padded, gt_rots_padded, part_valids, "rmse")[0]
        rmse_t = trans_metrics(pred_trans_padded, gt_trans_padded, part_valids, "rmse")[
            0
        ]

        yield {
            "type": "metrics",
            "data": {
                "Part Accuracy": acc.item(),
                "Shape CD (10<sup>-3</sup>)": shape_cd.item() * 1e3,
                "RMSE(R) (Degrees)": rmse_r.item(),
                "RMSE(T) (10<sup>-2</sup>)": rmse_t.item() * 1e2,
            },
        }

    yield {"type": "end", "data": None}
    # Release memory
    torch.cuda.empty_cache()


def get_gallery():
    try:
        gallery_data = json.load(open("gallery/gallery.json", "r"))
    except Exception as e:
        print(e)
        gallery_data = []
    return gallery_data


with gr.Blocks(title="3D Assembly") as demo:
    meshes = gr.Files(
        label="Upload fragments to assemble",
        file_count="multiple",
    )
    meshes_paths = gr.JSON()
    settings = gr.JSON()
    mesh_type = gr.Textbox()
    output = gr.JSON()
    inference_btn = gr.Button()
    inference_btn.click(
        fn=inference,
        inputs=[
            meshes,
            meshes_paths,
            mesh_type,
            settings,
        ],
        outputs=output,
        concurrency_limit=10,
    )


if __name__ == "__main__":
    demo.launch(share=False)

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchmetrics
import torch_scatter

from .loss import dice_loss
from assembly.backbones.vn_wrapper import VNBackboneWrapper


class FracSeg(pl.LightningModule):
    def __init__(
        self,
        pc_feat_dim: int,
        encoder: nn.Module,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        seg_warmup_epochs: int = 10,
        grid_size: float = 0.02,
        **kwargs,
    ):
        super().__init__()
        self.pc_feat_dim = pc_feat_dim
        self.encoder = encoder
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.seg_warmup_epochs = seg_warmup_epochs
        self.grid_size = grid_size

        self.batch_norm = nn.BatchNorm1d(self.pc_feat_dim)
        
        self.coarse_segmenter = nn.Sequential(
            nn.Linear(self.pc_feat_dim, 16, 1),  # (N_sum, 16)
            nn.ReLU(inplace=True),
            nn.Linear(16, 1, 1),  # (N_sum, 1)
            nn.Flatten(0, 1),  # (N_sum,)
        )

    def criteria(self, input_dict, output_dict):
        # loss
        coarse_seg_loss = dice_loss(
            output_dict["coarse_seg_pred"],
            output_dict["coarse_seg_gt"].float(),
        )

        loss = coarse_seg_loss

        # metrics
        coarse_seg_acc = torchmetrics.functional.accuracy(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_recall = torchmetrics.functional.recall(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_precision = torchmetrics.functional.precision(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )
        coarse_seg_f1 = torchmetrics.functional.f1_score(
            output_dict["coarse_seg_pred_binary"],
            output_dict["coarse_seg_gt"],
            task="binary",
        )

        return loss, {
            "coarse_seg_loss": coarse_seg_loss,
            "coarse_seg_acc": coarse_seg_acc,
            "coarse_seg_recall": coarse_seg_recall,
            "coarse_seg_precision": coarse_seg_precision,
            "coarse_seg_f1": coarse_seg_f1,
        }

    def training_step(self, batch, batch_idx):   #CHANGED SIGNATURE by adding batch_idx

        # --- ADD THIS PRINT STATEMENT ---
        print(f"\n[DEBUG] Processing Batch Index: {batch_idx} | Total Weight Updates (Global Step): {self.global_step}")
        # ------------------------------
        
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            {f"train/{k}": v for k, v in metrics.items()},
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log(
            "test/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch):
        out_dict = self.forward(batch)
        loss, metrics = self.criteria(batch, out_dict)
        self.log(
            "val/loss",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def forward(self, batch):
        out_dict = dict()
        
        pointclouds: torch.Tensor = batch["pointclouds"]  # (B, N, 3)
        normals: torch.Tensor = batch["pointclouds_normals"]  # (B, N, 3)
        points_per_part: torch.Tensor = batch["points_per_part"]  # (B, P)
        valid_pcs = points_per_part != 0

        # pylint: disable=invalid-name
        B, N, C = pointclouds.shape
        _, P = points_per_part.shape

        points_per_part_offset = torch.cumsum(points_per_part, dim=-1)  # (B, P)

        with torch.no_grad():
            valid_graph = valid_pcs.unsqueeze(2) & valid_pcs.unsqueeze(1)
            valid_graph = (
                valid_graph
                & ~torch.eye(valid_graph.shape[1], device=valid_graph.device).bool()
            )
            out_dict["valid_graph"] = valid_graph
            # Flip the diagonal of the connectivity matrix to 1
            out_dict["graph_gt"] = batch["graph"]

        part_pcds = pointclouds.view(-1, C)
        part_normals = normals.view(-1, C)
        points_offset = torch.cumsum(points_per_part[valid_pcs], dim=-1)  # (X,)

        # Extract color if it exists in the batch
        part_colors = None
        if "pointclouds_colors_gt" in batch:
            part_colors = batch["pointclouds_colors_gt"].view(-1, 3) # (N_sum, 3)

        # breakpoint()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
            # Pass the color into the encoder dictionary
            super_point, point = self.encoder(
                {
                    "coord": part_pcds,
                    "offset": points_offset,
                    "feat": torch.cat([part_pcds, part_normals], dim=-1),
                    "color": part_colors, # Added color here
                    "grid_size": torch.tensor(self.grid_size).to(part_pcds.device),
                }
            )

        
            points_features = point["feat"]  # (X * N, pc_feat_dim)
            # breakpoint()
            points_features = self.batch_norm(points_features)
            # breakpoint()
            out_dict["point"] = point
            out_dict["point"]["normal"] = part_normals
            out_dict["super_point"] = super_point
            assert points_features.isnan().sum() == 0, "points_features has nan"

        coarse_seg_pred = self.coarse_segmenter(points_features)  # (B, N)
        coarse_seg_pred = torch.sigmoid(coarse_seg_pred)
        coarse_seg_pred_binary = coarse_seg_pred > 0.5
        out_dict["coarse_seg_pred"] = coarse_seg_pred
        out_dict["coarse_seg_pred_binary"] = coarse_seg_pred_binary

        with torch.no_grad():
            if "fracture_surface_gt" in batch:
                out_dict["coarse_seg_gt"] = batch["fracture_surface_gt"].view(-1)

            if self.training and out_dict["coarse_seg_gt"] is not None:
                out_dict["coarse_seg"] = out_dict["coarse_seg_gt"]
            else:
                out_dict["coarse_seg"] = out_dict["coarse_seg_pred_binary"]

            fracture_surface_points_per_part = torch_scatter.segment_csr(
                src=out_dict["coarse_seg"].view(B, N).float(),
                indptr=F.pad(points_per_part_offset, (1, 0)),
                reduce="sum",
            ).view(B, P)
            out_dict["fracture_surface_points_per_part"] = (
                fracture_surface_points_per_part
            )

        return out_dict

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(),
        )

        if self.lr_scheduler is None:
            return {
                "optimizer": optimizer,
            }

        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
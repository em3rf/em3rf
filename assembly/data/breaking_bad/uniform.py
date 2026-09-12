from itertools import combinations
from typing import List

import trimesh
import numpy as np

from .base import BreakingBadBase
from ..transform import recenter_pc, rotate_pc, shuffle_pc, rotate_whole_part


class BreakingBadUniform(BreakingBadBase):
    """
    The Breaking Bad dataset with uniformly sampled points.
    Each piece is sampled with same number of points.
    """

    def sample_points(
        self,
        meshes: List[trimesh.Trimesh],
        shared_faces: List[np.ndarray],
    ):
        sampled_pcds = [
            trimesh.sample.sample_surface(
                mesh=mesh,
                count=self.num_points_to_sample,
            )
            for mesh in meshes
        ]
        pointclouds_gt = [pcd[0] for pcd in sampled_pcds]
        pointclouds_normals_gt = [
            mesh.face_normals[pcd[1]] for mesh, pcd in zip(meshes, sampled_pcds)
        ]
        fracture_surface = [
            (mask != -1)[pcd[1]] for mask, pcd in zip(shared_faces, sampled_pcds)
        ]
        return pointclouds_gt, pointclouds_normals_gt, fracture_surface

    def transform(self, data):
        num_parts = data["num_parts"]
        pointclouds_gt = data["pointclouds_gt"]
        pointclouds_normals_gt = data["pointclouds_normals_gt"]
        fracture_surface_gt = data["fracture_surface_gt"]
        graph = data["graph"]

        pointclouds_gt = np.stack(pointclouds_gt)  # (P, N, 3)
        pointclouds_normals_gt = np.stack(pointclouds_normals_gt)  # (P, N, 3)

        # Init Pose
        pointclouds_gt, pointclouds_normals_gt, init_rot = rotate_whole_part(
            pointclouds_gt,
            pointclouds_normals_gt,
        )

        pointclouds, pointclouds_normals, quaternions, translations = [], [], [], []
        for part_idx in range(num_parts):
            pointcloud, translation = recenter_pc(pointclouds_gt[part_idx])
            pointcloud, pointcloud_normals, quaternion = rotate_pc(
                pointcloud, pointclouds_normals_gt[part_idx]
            )
            pointcloud, pointcloud_normals, order = shuffle_pc(
                pointcloud, pointcloud_normals
            )
            # Shuffle gt as well
            pointclouds_gt[part_idx] = pointclouds_gt[part_idx][order]
            pointclouds_normals_gt[part_idx] = pointclouds_normals_gt[part_idx][order]
            fracture_surface_gt[part_idx] = fracture_surface_gt[part_idx][order]

            pointclouds.append(pointcloud)
            pointclouds_normals.append(pointcloud_normals)
            quaternions.append(quaternion)
            translations.append(translation)

        pointclouds = self._pad_data(np.stack(pointclouds, axis=0))  # (P, N, 3)
        pointclouds_normals = self._pad_data(
            np.stack(pointclouds_normals, axis=0)
        )  # (P, N, 3)
        quaternions = self._pad_data(np.stack(quaternions, axis=0))  # (P, 4)
        translations = self._pad_data(np.stack(translations, axis=0))  # (P, 3)
        pointclouds_gt = self._pad_data(np.stack(pointclouds_gt, axis=0))  # (P, N, 3)
        pointclouds_normals_gt = self._pad_data(
            np.stack(pointclouds_normals_gt, axis=0)
        )
        fracture_surface_gt = self._pad_data(np.stack(fracture_surface_gt, axis=0))

        # Normalize transformed point clouds
        scale = np.max(np.abs(pointclouds), axis=(1, 2), keepdims=True)
        scale[scale == 0] = 1
        pointclouds /= scale

        # Ref-part
        ref_part = np.zeros((self.max_parts), dtype=np.float32)
        ref_part_idx = np.argmax(scale[: (num_parts - self.num_redundancy)])
        ref_part[ref_part_idx] = 1
        ref_part = ref_part.astype(bool)

        return {
            "name": data["name"],
            "num_parts": num_parts,
            "pointclouds": pointclouds,
            "pointclouds_gt": pointclouds_gt,
            "pointclouds_normals": pointclouds_normals,
            "pointclouds_normals_gt": pointclouds_normals_gt,
            "fracture_surface_gt": fracture_surface_gt.astype(np.int8),
            "quaternions": quaternions,
            "translations": translations,
            "points_per_part": self._pad_data(
                np.array([self.num_points_to_sample] * num_parts)
            ).astype(np.int64),
            "graph": graph,
            "scale": scale.squeeze(-1),
            "ref_part": ref_part,
            "init_rot": init_rot,
            "removal": self.num_removal,
            "redundancy": self.num_redundancy,
            "removal_pieces": data["removal_pieces"],
            "redundant_pieces": data["redundant_pieces"],
        }
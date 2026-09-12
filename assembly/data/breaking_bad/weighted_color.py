from typing import List

import trimesh
import numpy as np

from .base_color import BreakingBadBaseColor
from ..transform import recenter_pc, rotate_pc, shuffle_pc, rotate_whole_part


class BreakingBadWeightedColor(BreakingBadBaseColor):
    """
    The Breaking Bad dataset with points sampled based on the area of each part.
    """

    def sample_points(
        self,
        meshes: List[trimesh.Trimesh],
        shared_faces: List[np.ndarray],
    ):
        assert self.max_parts * self.min_points_per_part <= self.num_points_to_sample, (
            f"Total number of points ({self.num_points_to_sample}) is less than the minimum number of points "
            f"({self.max_parts * self.min_points_per_part}) required for the dataset."
        )
        
        areas = [mesh.area for mesh in meshes]
        total_area = sum(areas)
        points_per_part = [
            self.min_points_per_part
            + int(
                (self.num_points_to_sample - self.min_points_per_part * len(meshes))
                * area
                / total_area
            )
            for area in areas
        ]
        anchor_idx = np.argmax(points_per_part)
        points_per_part[anchor_idx] += self.num_points_to_sample - sum(points_per_part)
        points_per_part[
            np.argmin(points_per_part)
        ] += self.num_points_to_sample - np.sum(points_per_part)
        
        
        sampled_pcds = []
        if self.mesh_sample_strategy == "poisson":
            for i in range(len(meshes)):
                pcd, idx = trimesh.sample.sample_surface_even(meshes[i], count=points_per_part[i])
                if len(pcd) < points_per_part[i]:
                    concat_pcd, concat_idx = trimesh.sample.sample_surface(
                        meshes[i],
                        count=points_per_part[i] - len(pcd),
                    )
                    pcd = np.concatenate([pcd, concat_pcd], axis=0)
                    idx = np.concatenate([idx, concat_idx], axis=0)
                # sampled_pcds.append((pcd, idx))
                
                # --- New Part: Sample Colors ADEELA STARTS ---
                if hasattr(meshes[i].visual, "vertex_colors") and len(meshes[i].visual.vertex_colors) > 0:
                    vertex_colors = meshes[i].visual.vertex_colors[:, :3]  # RGB, ignore alpha if present
        
                    # Get vertices of the faces where samples were drawn
                    faces = meshes[i].faces[idx]
                    tri_vertices = meshes[i].vertices[faces]
                    tri_colors = vertex_colors[faces]
        
                    # Compute barycentric coordinates for each sampled point
                    # (Trimesh provides a helper for this)
                    bary = trimesh.triangles.points_to_barycentric(tri_vertices, pcd)
        
                    # Interpolate colors
                    sampled_colors = np.einsum("ij,ijk->ik", bary, tri_colors)
                    sampled_colors = (
                        (sampled_colors * 255).astype(np.uint8)
                        if sampled_colors.max() <= 1.0
                        else np.clip(sampled_colors, 0, 255).astype(np.uint8)
                    )

                    
                    # Store both point cloud + color
                    sampled_pcds.append((pcd, idx, sampled_colors))
                    
                    '''
                    # # to visualize sampled_pcd with sampled_colors for verification
                    colors_uint8 = np.clip(sampled_colors, 0, 255).astype(np.uint8)
                    colored_mesh = trimesh.Trimesh( vertices=pcd, faces=None, vertex_colors=colors_uint8)
                    colored_mesh.export('sampled_colored_pcd_test.obj')
                    # breakpoint()
                    '''
                else:
                    # If no vertex colors, store just the points
                    sampled_pcds.append((pcd, idx))
                
                
                
        else:
            sampled_pcds = [
                trimesh.sample.sample_surface(
                    mesh=meshes[i],
                    count=points_per_part[i],
                    sample_color=True,      # https://trimesh.org/trimesh.sample.html
                )
                for i in range(len(meshes))
            ]
            
            sampled_pcds = [
                (
                    pts,
                    face_idx,
                    np.clip((colors[:, :3] * (255 if colors.max() <= 1.0 else 1)), 0, 255).astype(np.uint8)
                )
                for (pts, face_idx, colors) in sampled_pcds
            ]
            
            # --- New Part: Sample Colors ADEELA ENDS ---
            
        pointclouds_gt = [pcd[0] for pcd in sampled_pcds]
        pointclouds_colors_gt = [pcd[2] for pcd in sampled_pcds]    #ADEELA
        # breakpoint()
        pointclouds_normals_gt = [meshes[i].face_normals[pcd[1]] for i, pcd in enumerate(sampled_pcds)]
        fracture_surface = [
            (mask != -1)[pcd[1]] for mask, pcd in zip(shared_faces, sampled_pcds)
        ]
        return pointclouds_gt, pointclouds_normals_gt, fracture_surface, pointclouds_colors_gt  #ADEELA

    def transform(self, data):
        
        num_parts = data["num_parts"]
        pointclouds_gt = data["pointclouds_gt"]
        pointclouds_colors_gt = data["pointclouds_colors_gt"]       #ADEELA
        pointclouds_normals_gt = data["pointclouds_normals_gt"]
        fracture_surface_gt = data["fracture_surface_gt"]
        graph = data["graph"]

        points_per_part = np.array([len(pc) for pc in pointclouds_gt])  # (valid_P,)
        offset = np.concatenate([[0], np.cumsum(points_per_part)])  # (valid_P+1,)

        pointclouds_gt = np.concatenate(pointclouds_gt)  # (N, 3)                    5000x3
        pointclouds_colors_gt = np.concatenate(pointclouds_colors_gt)  # (N, 3)      5000x3    #ADEELA
        pointclouds_normals_gt = np.concatenate(pointclouds_normals_gt)  # (N, 3)    5000x3
        fracture_surface_gt = np.concatenate(fracture_surface_gt)  # (N,)            5000

        # Init Pose
        pointclouds_gt, pointclouds_normals_gt, init_rot = rotate_pc(
            pointclouds_gt,
            pointclouds_normals_gt,
        )
        
        # breakpoint()
        pointclouds, pointclouds_normals, quaternions, translations = [], [], [], []
        scale = []
        for part_idx in range(num_parts):
            start = offset[part_idx]
            end = offset[part_idx + 1]

            pointcloud, translation = recenter_pc(pointclouds_gt[start:end])
            pointcloud, pointcloud_normals, quaternion = rotate_pc(
                pointcloud, pointclouds_normals_gt[start:end]
            )
            pointcloud, pointcloud_normals, order = shuffle_pc(
                pointcloud, pointcloud_normals
            )

            # Shuffle gt as well
            # np.where(order == 0)[0][0]   to find out where 0th index gone after shuffling
            pointclouds_gt[start:end] = pointclouds_gt[start:end][order]
            pointclouds_colors_gt[start:end] = pointclouds_colors_gt[start:end][order]    #ADEELA
            pointclouds_normals_gt[start:end] = pointclouds_normals_gt[start:end][order]
            fracture_surface_gt[start:end] = fracture_surface_gt[start:end][order]

            # Rescale
            current_scale = np.max(np.abs(pointcloud))
            scale.append(current_scale)
            pointcloud /= current_scale

            pointclouds.append(pointcloud)
            pointclouds_normals.append(pointcloud_normals)
            quaternions.append(quaternion)
            translations.append(translation)
            
        # breakpoint()
        # Concatenate
        pointclouds = np.concatenate(pointclouds).astype(np.float32)  # [N, 3]
        pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)
        quaternions = np.stack(quaternions).astype(np.float32)  # [P, 4]
        translations = np.stack(translations).astype(np.float32)  # [P, 3]
        scale = np.array(scale).astype(np.float32)

        # Pad data
        points_per_part = self._pad_data(points_per_part).astype(np.int64)
        quaternions = self._pad_data(quaternions)
        translations = self._pad_data(translations)
        scale = self._pad_data(scale)

        # Ref-part
        ref_part = np.zeros((self.max_parts), dtype=np.float32)
        ref_part_idx = np.argmax(points_per_part[: (num_parts - self.num_redundancy)])
        if self.random_anchor:
            # only points > 5% of the total points can be the ref_part
            can_be_anchor = (
                points_per_part[:num_parts] > self.num_points_to_sample * 0.05
            )
            # sample a ref part
            ref_part_idx = np.random.choice(
                np.where(can_be_anchor)[0], 1, replace=False
            )[0]

        ref_part[ref_part_idx] = 1
        ref_part = ref_part.astype(bool)

        if self.multi_ref and num_parts > 2 and np.random.rand() > 1 / num_parts:
            can_be_ref = points_per_part[:num_parts] > self.num_points_to_sample * 0.05
            can_be_ref[ref_part_idx] = False
            can_be_ref_num = np.sum(can_be_ref)
            if can_be_ref_num > 0:
                # random select more ref parts
                num_more_ref = np.random.randint(
                    1, min(can_be_ref_num + 1, num_parts - 1)
                )
                more_ref_part_idx = np.random.choice(
                    np.where(can_be_ref)[0], num_more_ref, replace=False
                )
                ref_part[more_ref_part_idx] = True

        return {
            "index": data["index"],
            "name": data["name"],
            "num_parts": num_parts,
            "pointclouds": pointclouds,
            "pointclouds_gt": pointclouds_gt.astype(np.float32),
            "pointclouds_colors_gt": (pointclouds_colors_gt.astype(np.float32) / 127.5) - 1.0, #"pointclouds_colors_gt": pointclouds_colors_gt.astype(np.float32),      #ADEELA
            "pointclouds_normals": pointclouds_normals.astype(np.float32),
            "pointclouds_normals_gt": pointclouds_normals_gt,
            "fracture_surface_gt": fracture_surface_gt.astype(np.int8),
            "quaternions": quaternions,
            "translations": translations,
            "points_per_part": points_per_part.astype(np.int64),
            "graph": graph,
            "scale": scale[:, np.newaxis],
            "ref_part": ref_part,
            "init_rot": init_rot,
            "removal": self.num_removal,
            "redundancy": self.num_redundancy,
            "removal_pieces": data["removal_pieces"],
            "redundant_pieces": data["redundant_pieces"],
            "pieces": data["pieces"],
            "mesh_scale": data["mesh_scale"],
        }

if __name__ == "__main__":
    dataset = BreakingBadWeightedColor(
        split="train",
        data_root="/home/duan/shl/JZY/breaking_bad_vol.hdf5",
        category="everyday",
        num_points_to_sample=5000,
    )
    dataset.visualize(0)

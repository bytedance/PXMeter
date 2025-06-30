# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def align_src_to_tar(
    src_pose: np.ndarray,
    tar_pose: np.ndarray,
    atom_mask: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    allow_reflection: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find optimal transformation, rotation (and reflection) of two poses using NumPy.

    Args:
        src_pose (np.ndarray): [N, 3] the pose to perform transformation on
        tar_pose (np.ndarray): [N, 3] the target pose to align src_pose to
        atom_mask (np.ndarray): [N] a mask for atoms
        weight (np.ndarray): [N] a weight vector to be applied
        allow_reflection (bool): whether to allow reflection when finding optimal alignment

    Returns:
        rot: optimal rotation matrix
        translate: optimal translation vector
    """
    if atom_mask is not None:
        atom_mask = atom_mask.astype(float)
        src_pose = src_pose * atom_mask[..., None]
        tar_pose = tar_pose * atom_mask[..., None]
    else:
        atom_mask = np.ones(src_pose.shape[:-1], dtype=float)

    if weight is None:
        weight = atom_mask
    else:
        weight = weight.astype(float)
        weight = weight * atom_mask

    weighted_n_atoms = np.sum(weight, axis=-1, keepdims=True)[..., None]
    src_pose_centroid = (
        np.sum(src_pose * weight[..., None], axis=-2, keepdims=True) / weighted_n_atoms
    )
    src_pose_centered = src_pose - src_pose_centroid
    tar_pose_centroid = (
        np.sum(tar_pose * weight[..., None], axis=-2, keepdims=True) / weighted_n_atoms
    )
    tar_pose_centered = tar_pose - tar_pose_centroid
    H_mat = (src_pose_centered * weight[..., None]).swapaxes(-2, -1) @ (
        tar_pose_centered * atom_mask[..., None]
    )

    u, s, vh = np.linalg.svd(H_mat)
    u = u.swapaxes(-1, -2)
    vh = vh.T

    if not allow_reflection:
        det = np.linalg.det(vh @ u)
        diagonal = np.stack([np.ones_like(det), np.ones_like(det), det], axis=-1)
        rot = np.diagflat(diagonal) @ u
        rot = vh @ rot
    else:
        rot = vh @ u
    translate = tar_pose_centroid - src_pose_centroid @ rot.swapaxes(-1, -2)
    return rot, translate


def apply_transform(pose: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """
    Apply a rotation and translation to a pose.

    Arguments:
        pose: [..., N, 3] the pose to perform transformation on
        rot: optimal rotation matrix
        trans: optimal translation vector

    Returns:
        transformed_pose: [..., N, 3] the transformed pose
    """
    return pose @ rot.swapaxes(-1, -2) + trans


def rmsd(
    pose1: np.ndarray,
    pose2: np.ndarray,
    mask: np.ndarray = None,
    eps: float = 0.0,
    reduce: bool = True,
):
    """
    Compute RMSD between two poses, with the same shape.

    Arguments:
        pred_pose, true_pose: [..., N, 3], two poses with the same shape.
        mask: [..., N], mask to indicate which elements to compute.
        eps: Add a tolerance to avoid floating number issues.
        reduce: Decide the return shape of RMSD.

    Returns:
        rmsd_value: If reduce is True, return the mean of RMSD over batches;
                    else return an array containing each RMSD separately.
    """
    assert pose1.shape == pose2.shape  # [..., N, 3]

    if mask is None:
        mask = np.ones(pose2.shape[:-1], dtype=float)
    else:
        mask = mask.astype(float)

    # Compute squared error.
    err2 = (np.square(pose1 - pose2).sum(axis=-1) * mask).sum(axis=-1) / (
        mask.sum(axis=-1) + eps
    )

    # Calculate RMSD with added epsilon tolerance.
    rmsd_value = np.sqrt(err2 + eps)

    # Option to reduce RMSD to a mean value.
    if reduce:
        rmsd_value = rmsd_value.mean()

    return rmsd_value


def partially_aligned_rmsd(
    src_pose: np.ndarray,
    tar_pose: np.ndarray,
    align_mask: np.ndarray | None = None,
    rmsd_mask: np.ndarray | None = None,
    weight: np.ndarray | None = None,
    eps: float = 0.0,
    reduce: bool = True,
    allow_reflection: bool = False,
):
    """
    RMSD when aligning parts of the complex coordinate,
    does NOT take permutation symmetricity into consideration

    Args:
        src_pose (np.ndarray): [N, 3] Source pose.
        tar_pose (np.ndarray): [N, 3] Target pose.
        align_mask (np.ndarray): [N] A mask representing which coordinates to align.
        rmsd_mask (np.ndarray): [N] A mask representing which coordinates to compute RMSD.
        weight (np.ndarray): [N] A weight tensor assining weights in alignment for each atom.
        eps (float): Add a tolerance to avoid floating number issue in sqrt.
        reduce: Decide the return shape of RMSD.
        allow_reflection (bool): Whether to allow reflection when finding optimal alignment

    Returns:
        aligned_part_rmsd : the RMSD of part being align_masked
        rmsd_value: the RMSD  of part being rmsd_mask
        rot (np.ndarray): optimal rotation matrix
        translate (np.ndarray): optimal translation vector
    """
    rot, translate = align_src_to_tar(
        src_pose,
        tar_pose,
        atom_mask=align_mask,
        weight=weight,
        allow_reflection=allow_reflection,
    )
    transformed_src_pose = apply_transform(src_pose, rot, translate)
    rmsd_value = rmsd(
        transformed_src_pose, tar_pose, mask=rmsd_mask, eps=eps, reduce=reduce
    )
    aligned_part_rmsd = rmsd(
        transformed_src_pose, tar_pose, mask=align_mask, eps=eps, reduce=reduce
    )
    return aligned_part_rmsd, rmsd_value, rot, translate

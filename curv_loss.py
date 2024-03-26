import torch
from pytorch3d.ops import cot_laplacian
from typing import Union
from abc import ABC, abstractmethod
from pytorch3d.structures import Meshes,Pointclouds

from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

# from Bongratz et al. 2022

def curv_from_cotcurv_laplacian(verts_packed, faces_packed):
    """ Construct the cotangent curvature Laplacian as done in
    pytorch3d.loss.mesh_laplacian_smoothing and use it for approximation of the
    mean curvature at each vertex. See also
    - Nealen et al. "Laplacian Mesh Optimization", 2006
    """
    # No backprop through the computation of the Laplacian (taken as a
    # constant), similar to pytorch3d.loss.mesh_laplacian_smoothing
    with torch.no_grad():
        L, inv_areas = cot_laplacian(verts_packed, faces_packed)
        L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1,1)
        norm_w = 0.25 * inv_areas

    return torch.norm(
        (L.mm(verts_packed) - L_sum * verts_packed) * norm_w,
        dim=1
    )

def point_weigths_from_curvature(
    curvatures: torch.Tensor,
    points: torch.Tensor,
    max_weight: Union[float, int, torch.Tensor],
    padded_coordinates=(0.0, 0.0, 0.0)
):
    """ Calculate Chamfer weights from curvatures such that they are in
    [1, max_weight]. In addition, the weight of padded points is set to zero."""

    if not isinstance(max_weight, torch.Tensor):
        max_weight = torch.tensor(max_weight).float()

    # Weights in [1, max_weight]
    weights = torch.minimum(1 + curvatures, max_weight.to(curvatures.device))

    # Set weights of padded vertices to 0
    padded_coordinates = torch.Tensor(padded_coordinates).to(points.device)
    weights[torch.isclose(points, padded_coordinates).all(dim=2)] = 0.0

    return weights

# modification of pytorch3d.loss.chamfer_distance source code implementation to include the points weights
def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def chamfer_distance_pw(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    batch_weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    point_weights=None,
    oriented_cosine_similarity=False,
    norm: int = 2,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        batch_weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        point_weights: Optional FloatTensor of shape (N,P2,) giving weights for
        points in y.
        oriented_cosine: If set to True and x_normals and y_normals are not
        None, the cosine similarity considers the orientation of normals.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if batch_weights is not None:
        if batch_weights.size(0) != N:
            raise ValueError("batch_weights must be of shape (N,).")
        if not (batch_weights >= 0).all():
            raise ValueError("batch_weights cannot be negative.")
        if batch_weights.sum() == 0.0:
            batch_weights = batch_weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * batch_weights).sum() * 0.0,
                    (x.sum((1, 2)) * batch_weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * batch_weights) * 0.0, (x.sum((1, 2)) * batch_weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if point_weights is not None:
        x_weights = knn_gather(
            point_weights.view(N, P2, 1), x_nn.idx, y_lengths
        ).view(N, P1)
        y_weights = point_weights.view(N, P2)
        cham_x *= x_weights
        cham_y *= y_weights

    if batch_weights is not None:
        cham_x *= batch_weights.view(N, 1)
        cham_y *= batch_weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        if oriented_cosine_similarity:
            cham_norm_x = 1 - F.cosine_similarity(
                x_normals, x_normals_near, dim=2, eps=1e-6
            )
            cham_norm_y = 1 - F.cosine_similarity(
                y_normals, y_normals_near, dim=2, eps=1e-6
            )
        else:
            cham_norm_x = 1 - torch.abs(
                F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
            )
            cham_norm_y = 1 - torch.abs(
                F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
            )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if batch_weights is not None:
            cham_norm_x *= batch_weights.view(N, 1)
            cham_norm_y *= batch_weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        if point_weights is not None:
            # Normalize by weight sum
            cham_x /= x_weights.sum(dim=1)
            cham_y /= y_weights.sum(dim=1)
        else:
            cham_x /= x_lengths
            cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = batch_weights.sum() if batch_weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals



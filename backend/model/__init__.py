"""Faz 4-5: Gaussian model + deformation field + training."""
from .gaussian_model import GaussianModel
from .deformation import DeformationField
from .covariance import build_covariance_3d, project_gaussians, quat_to_rotmat

__all__ = [
    "GaussianModel",
    "DeformationField",
    "build_covariance_3d",
    "project_gaussians",
    "quat_to_rotmat",
]

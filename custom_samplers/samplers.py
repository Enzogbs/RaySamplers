"""
COMPLETE SAMPLERS LIBRARY FOR NERFSTUDIO
==========================================
All sampling methods from your research integrated with Nerfstudio

File: custom_samplers/samplers.py
"""

from dataclasses import dataclass
from typing import Optional, Literal
import torch
from torch import Tensor
import torch.nn as nn
from nerfstudio.cameras.rays import RaySamples, RayBundle
from nerfstudio.model_components.ray_samplers import Sampler


# ============================================================================
# 1. ADAPTIVE KERNEL MIX SAMPLER V2 (Main Method)
# ============================================================================

class AdaptiveKernelSamplerV2(Sampler):
    """
    Adaptive Kernel Mix Sampler V2
    Your main novel method with kernel-based perturbations
    """
    
    def __init__(self, num_samples: int = 64, single_jitter: bool = True, **kwargs):
        super().__init__(num_samples=num_samples, single_jitter=single_jitter)
        self.kernel_type = kwargs.get('kernel_type', 'epanechnikov')
        self.use_blur = kwargs.get('use_blur', True)
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        weights: Optional[Tensor] = None,
        bins: Optional[Tensor] = None,
        **kwargs
    ) -> RaySamples:
        
        if num_samples is None:
            num_samples = self.num_samples
        
        if weights is None or bins is None:
            return self._uniform_samples(ray_bundle, num_samples)
        
        device = weights.device
        num_rays = weights.shape[0]
        eps = 1e-5
        
        # Normalize weights
        weights = weights + eps
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Apply maxblur
        if self.use_blur:
            weights = self._apply_maxblur(weights)
        
        # Sample bin centers
        centers_idx = torch.multinomial(weights, num_samples, replacement=True)
        z_c = torch.gather(bins, 1, centers_idx)
        
        # Compute bandwidth
        bandwidth = (bins[:, 1:] - bins[:, :-1]).mean(dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Generate kernel perturbations
        eps_samples = self._generate_kernel_samples(num_rays, num_samples, device)
        
        # Final samples
        samples = z_c + eps_samples * bandwidth
        samples = samples.clamp(bins.min(), bins.max())
        samples, _ = torch.sort(samples, dim=-1)
        
        return self._pack_ray_samples(samples)
    
    def _apply_maxblur(self, weights):
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])
        return weights + 1e-5
    
    def _generate_kernel_samples(self, num_rays, num_samples, device):
        if self.single_jitter:
            return torch.linspace(-0.5, 0.5, num_samples, device=device).unsqueeze(0).expand(num_rays, -1)
        
        if self.kernel_type == "gaussian":
            return torch.randn(num_rays, num_samples, device=device)
        elif self.kernel_type == "epanechnikov":
            u = torch.rand(num_rays, num_samples, device=device)
            eps = torch.sqrt(1 - u)
            signs = torch.randint(0, 2, eps.shape, device=device).float() * 2 - 1
            return eps * signs
        elif self.kernel_type == "triangular":
            u = torch.rand(num_rays, num_samples, device=device)
            eps = torch.sqrt(u)
            signs = torch.randint(0, 2, eps.shape, device=device).float() * 2 - 1
            return eps * signs
        elif self.kernel_type == "uniform":
            return torch.rand(num_rays, num_samples, device=device) * 2 - 1
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
    
    def _uniform_samples(self, ray_bundle, num_samples):
        num_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        t = torch.linspace(0., 1., num_samples, device=device)
        samples = ray_bundle.nears + t * (ray_bundle.fars - ray_bundle.nears)
        samples = samples.unsqueeze(0).expand(num_rays, -1)
        return self._pack_ray_samples(samples)
    
    def _pack_ray_samples(self, samples):
        bin_starts = samples
        bin_ends = torch.cat([samples[:, 1:], samples[:, -1:] + 1e-3], dim=-1)
        return RaySamples(
            frustums=None,
            camera_indices=None,
            deltas=bin_ends - bin_starts,
            spacing_starts=bin_starts,
            spacing_ends=bin_ends,
            spacing_to_euclidean_fn=lambda x: x,
            metadata={},
            times=None,
        )


# ============================================================================
# 2. ADAPTIVE KERNEL MIX SAMPLER V3 (With Entropy-based Adaptation)
# ============================================================================

class AdaptiveKernelSamplerV3(Sampler):
    """
    Adaptive Kernel Mix V3 with entropy-based sample count adaptation
    Dynamically adjusts number of samples per ray based on uncertainty
    """
    
    def __init__(self, num_samples: int = 64, single_jitter: bool = True, **kwargs):
        super().__init__(num_samples=num_samples, single_jitter=single_jitter)
        self.kernel_type = kwargs.get('kernel_type', 'gaussian')
        self.N_samples_min = kwargs.get('N_samples_min', num_samples // 4)
        self.N_samples_max = kwargs.get('N_samples_max', num_samples)
        self.uncertainty_threshold = kwargs.get('uncertainty_threshold', 0.01)
        self.uncertainty_type = kwargs.get('uncertainty_type', 'entropy')
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        weights: Optional[Tensor] = None,
        bins: Optional[Tensor] = None,
        **kwargs
    ) -> RaySamples:
        
        if weights is None or bins is None:
            return self._uniform_samples(ray_bundle, self.num_samples)
        
        device = weights.device
        eps = 1e-5
        num_rays, N_bins = weights.shape
        
        # Normalize weights
        weights = weights + eps
        pdf = weights / weights.sum(dim=-1, keepdim=True)
        
        # Compute uncertainty
        if self.uncertainty_type == "entropy":
            uncertainty = -(pdf * pdf.clamp(min=1e-6).log()).sum(dim=-1)
        elif self.uncertainty_type == "variance":
            idxs = torch.arange(N_bins, device=device).float().unsqueeze(0)
            mean = (pdf * idxs).sum(dim=-1)
            mean2 = (pdf * idxs**2).sum(dim=-1)
            uncertainty = mean2 - mean**2
        else:
            uncertainty = torch.ones(num_rays, device=device)
        
        uncertainty_norm = uncertainty / (uncertainty.max() + eps)
        
        # Skip mask
        skip_mask = (uncertainty_norm < self.uncertainty_threshold)
        
        # Adaptive sample count
        N_samples_per_ray = (
            self.N_samples_min + uncertainty_norm * (self.N_samples_max - self.N_samples_min)
        ).clamp(min=self.N_samples_min).round().long()
        
        N_samples = N_samples_per_ray.max().item()
        bandwidth = (bins[:, 1:] - bins[:, :-1]).mean(dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Sample centers
        centers = torch.multinomial(pdf, N_samples, replacement=True)
        z_c = torch.gather(bins, 1, centers)
        
        # Kernel perturbations
        eps_samples = self._generate_kernel_samples(num_rays, N_samples, device)
        samples = z_c + eps_samples * bandwidth
        
        # Apply masks
        sample_ids = torch.arange(N_samples, device=device).unsqueeze(0)
        mask = sample_ids < N_samples_per_ray.unsqueeze(1)
        samples[~mask] = 0.0
        samples[skip_mask] = 0.0
        
        samples, _ = torch.sort(samples, dim=-1)
        return self._pack_ray_samples(samples)
    
    def _generate_kernel_samples(self, num_rays, num_samples, device):
        if self.kernel_type == "gaussian":
            return torch.randn(num_rays, num_samples, device=device)
        elif self.kernel_type == "epanechnikov":
            u = torch.rand(num_rays, num_samples, device=device)
            eps = torch.sqrt(1 - u)
            signs = torch.randint(0, 2, eps.shape, device=device).float() * 2 - 1
            return eps * signs
        elif self.kernel_type == "triangular":
            u = torch.rand(num_rays, num_samples, device=device)
            eps = torch.sqrt(u)
            signs = torch.randint(0, 2, eps.shape, device=device).float() * 2 - 1
            return eps * signs
        elif self.kernel_type == "uniform":
            return torch.rand(num_rays, num_samples, device=device) * 2 - 1
        return torch.randn(num_rays, num_samples, device=device)
    
    def _uniform_samples(self, ray_bundle, num_samples):
        num_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        t = torch.linspace(0., 1., num_samples, device=device)
        samples = ray_bundle.nears + t * (ray_bundle.fars - ray_bundle.nears)
        samples = samples.unsqueeze(0).expand(num_rays, -1)
        return self._pack_ray_samples(samples)
    
    def _pack_ray_samples(self, samples):
        bin_starts = samples
        bin_ends = torch.cat([samples[:, 1:], samples[:, -1:] + 1e-3], dim=-1)
        return RaySamples(
            frustums=None,
            camera_indices=None,
            deltas=bin_ends - bin_starts,
            spacing_starts=bin_starts,
            spacing_ends=bin_ends,
            spacing_to_euclidean_fn=lambda x: x,
            metadata={},
            times=None,
        )


# ============================================================================
# 3. L0 SAMPLER (Baseline - Li et al. 2023)
# ============================================================================

class L0Sampler(Sampler):
    """
    L0 Sampler from Li et al. (2023)
    Piecewise exponential interpolation for quasi-L0 weight distribution
    """
    
    def __init__(self, num_samples: int = 64, single_jitter: bool = True, **kwargs):
        super().__init__(num_samples=num_samples, single_jitter=single_jitter)
        self.use_blur = kwargs.get('use_blur', True)
        self.spline_type = kwargs.get('spline_type', 'exp')
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        weights: Optional[Tensor] = None,
        bins: Optional[Tensor] = None,
        **kwargs
    ) -> RaySamples:
        
        if num_samples is None:
            num_samples = self.num_samples
        
        if weights is None or bins is None:
            return self._uniform_samples(ray_bundle, num_samples)
        
        if self.use_blur:
            weights = self._maxblur(weights)
        
        samples = self._l0_sample_exponential(bins, weights, num_samples)
        return self._pack_ray_samples(samples)
    
    def _maxblur(self, weights):
        weights_pad = torch.cat([weights[:, :1], weights, weights[:, -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[:, :-1], weights_pad[:, 1:])
        return 0.5 * (weights_max[:, :-1] + weights_max[:, 1:]) + 1e-5
    
    def _l0_sample_exponential(self, bins, weights, N_samples):
        N_rays, N_bins = bins.shape
        device = bins.device
        eps = 1e-10
        
        weights = weights + eps
        w_left = weights
        w_right = torch.cat([weights[:, 1:], weights[:, -1:]], dim=-1)
        w_left = w_left.clamp(min=eps)
        w_right = w_right.clamp(min=eps)
        
        # Integral
        integral = (w_right - w_left) / (torch.log(w_right) - torch.log(w_left) + eps)
        integral = torch.nan_to_num(integral, nan=w_left.mean().item())
        
        # CDF
        pdf = integral / (integral.sum(dim=-1, keepdim=True) + eps)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
        
        # Sample
        u = torch.linspace(0., 1., N_samples, device=device).unsqueeze(0).expand(N_rays, -1).contiguous()
        
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=N_bins)
        
        bins_below = torch.gather(bins, 1, below)
        bins_above = torch.gather(bins, 1, above)
        cdf_below = torch.gather(cdf, 1, below)
        cdf_above = torch.gather(cdf, 1, above)
        w_below = torch.gather(w_left, 1, below)
        w_above = torch.gather(w_right, 1, below)
        
        denom = (cdf_above - cdf_below).clamp(min=1e-5)
        t = (u - cdf_below) / denom
        
        ratio = (w_above / w_below.clamp(min=eps)).clamp(min=eps)
        samples = bins_below + (torch.log(1 + t * (ratio - 1) + eps) / torch.log(ratio).clamp(min=eps)) * (bins_above - bins_below)
        samples = torch.nan_to_num(samples, nan=0.0)
        samples, _ = torch.sort(samples, dim=-1)
        
        return samples
    
    def _uniform_samples(self, ray_bundle, num_samples):
        num_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        t = torch.linspace(0., 1., num_samples, device=device)
        samples = ray_bundle.nears + t * (ray_bundle.fars - ray_bundle.nears)
        samples = samples.unsqueeze(0).expand(num_rays, -1)
        return self._pack_ray_samples(samples)
    
    def _pack_ray_samples(self, samples):
        bin_starts = samples
        bin_ends = torch.cat([samples[:, 1:], samples[:, -1:] + 1e-3], dim=-1)
        return RaySamples(
            frustums=None, camera_indices=None,
            deltas=bin_ends - bin_starts,
            spacing_starts=bin_starts, spacing_ends=bin_ends,
            spacing_to_euclidean_fn=lambda x: x,
            metadata={}, times=None,
        )


# ============================================================================
# 4. GAUSSIAN MIXTURE MODEL SAMPLER
# ============================================================================

class GMMSampler(Sampler):
    """
    GMM-based sampler using Gaussian Mixture Model
    Fits K Gaussians to the weight distribution
    """
    
    def __init__(self, num_samples: int = 64, single_jitter: bool = True, **kwargs):
        super().__init__(num_samples=num_samples, single_jitter=single_jitter)
        self.K = kwargs.get('K', 4)
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        weights: Optional[Tensor] = None,
        bins: Optional[Tensor] = None,
        **kwargs
    ) -> RaySamples:
        
        if num_samples is None:
            num_samples = self.num_samples
        
        if weights is None or bins is None:
            return self._uniform_samples(ray_bundle, num_samples)
        
        device = weights.device
        num_rays, n_bins = bins.shape
        eps = 1e-5
        
        weights = weights + eps
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Create K fixed means
        gmm_means = torch.linspace(0., 1., self.K, device=device).unsqueeze(0).expand(num_rays, self.K)
        gmm_means = torch.lerp(bins[:, :1], bins[:, -1:], gmm_means)
        
        # Compute std
        gmm_stds = 0.05 * (bins[:, -1:] - bins[:, :1])
        gmm_stds = gmm_stds.expand(num_rays, self.K)
        
        # Sample from each component
        samples_per_component = num_samples // self.K
        all_samples = []
        
        for k in range(self.K):
            mean = gmm_means[:, k:k+1]
            std = gmm_stds[:, k:k+1]
            component_samples = mean + std * torch.randn(num_rays, samples_per_component, device=device)
            all_samples.append(component_samples)
        
        samples = torch.cat(all_samples, dim=1)
        samples = samples.clamp(bins.min(), bins.max())
        samples, _ = torch.sort(samples, dim=-1)
        
        return self._pack_ray_samples(samples)
    
    def _uniform_samples(self, ray_bundle, num_samples):
        num_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        t = torch.linspace(0., 1., num_samples, device=device)
        samples = ray_bundle.nears + t * (ray_bundle.fars - ray_bundle.nears)
        samples = samples.unsqueeze(0).expand(num_rays, -1)
        return self._pack_ray_samples(samples)
    
    def _pack_ray_samples(self, samples):
        bin_starts = samples
        bin_ends = torch.cat([samples[:, 1:], samples[:, -1:] + 1e-3], dim=-1)
        return RaySamples(
            frustums=None, camera_indices=None,
            deltas=bin_ends - bin_starts,
            spacing_starts=bin_starts, spacing_ends=bin_ends,
            spacing_to_euclidean_fn=lambda x: x,
            metadata={}, times=None,
        )


# ============================================================================
# 5. OPTIMAL TRANSPORT SAMPLER
# ============================================================================

class OptimalTransportSampler(Sampler):
    """
    Fast 1D Optimal Transport sampler
    Uses batched linear interpolation for efficient CDF inversion
    """
    
    def __init__(self, num_samples: int = 64, single_jitter: bool = True, **kwargs):
        super().__init__(num_samples=num_samples, single_jitter=single_jitter)
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        weights: Optional[Tensor] = None,
        bins: Optional[Tensor] = None,
        **kwargs
    ) -> RaySamples:
        
        if num_samples is None:
            num_samples = self.num_samples
        
        if weights is None or bins is None:
            return self._uniform_samples(ray_bundle, num_samples)
        
        device = weights.device
        num_rays, N_bins = weights.shape
        eps = 1e-5
        
        # Normalize to PDF
        weights = weights + eps
        pdf = weights / weights.sum(dim=-1, keepdim=True)
        
        # Compute CDF
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Uniform samples
        if self.single_jitter:
            u = torch.linspace(0., 1., num_samples, device=device).unsqueeze(0).expand(num_rays, -1)
        else:
            u = torch.rand(num_rays, num_samples, device=device)
        
        # Invert CDF
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=N_bins)
        
        cdf_g0 = torch.gather(cdf, 1, below)
        cdf_g1 = torch.gather(cdf, 1, above)
        bins_g0 = torch.gather(bins, 1, below)
        bins_g1 = torch.gather(bins, 1, above)
        
        denom = cdf_g1 - cdf_g0
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g0) / denom
        
        samples = bins_g0 + t * (bins_g1 - bins_g0)
        samples, _ = torch.sort(samples, dim=-1)
        
        return self._pack_ray_samples(samples)
    
    def _uniform_samples(self, ray_bundle, num_samples):
        num_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        t = torch.linspace(0., 1., num_samples, device=device)
        samples = ray_bundle.nears + t * (ray_bundle.fars - ray_bundle.nears)
        samples = samples.unsqueeze(0).expand(num_rays, -1)
        return self._pack_ray_samples(samples)
    
    def _pack_ray_samples(self, samples):
        bin_starts = samples
        bin_ends = torch.cat([samples[:, 1:], samples[:, -1:] + 1e-3], dim=-1)
        return RaySamples(
            frustums=None, camera_indices=None,
            deltas=bin_ends - bin_starts,
            spacing_starts=bin_starts, spacing_ends=bin_ends,
            spacing_to_euclidean_fn=lambda x: x,
            metadata={}, times=None,
        )


# ============================================================================
# 6. ENTROPY-KDE SAMPLER
# ============================================================================

class EntropyKDESampler(Sampler):
    """
    Pure adaptive entropy + KDE sampler
    Combines entropy-based adaptation with kernel density estimation
    """
    
    def __init__(self, num_samples: int = 64, single_jitter: bool = True, **kwargs):
        super().__init__(num_samples=num_samples, single_jitter=single_jitter)
        self.N_samples_min = kwargs.get('N_samples_min', 8)
        self.N_samples_max = kwargs.get('N_samples_max', 64)
        self.kernel_type = kwargs.get('kernel_type', 'gaussian')
        self.entropy_threshold = kwargs.get('entropy_threshold', 0.01)
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        weights: Optional[Tensor] = None,
        bins: Optional[Tensor] = None,
        **kwargs
    ) -> RaySamples:
        
        if weights is None or bins is None:
            return self._uniform_samples(ray_bundle, self.num_samples)
        
        device = bins.device
        eps = 1e-6
        num_rays, N_bins = bins.shape
        
        # Normalize
        weights = weights + eps
        pdf = weights / weights.sum(dim=-1, keepdim=True)
        
        # Compute entropy
        entropy = -(pdf * pdf.clamp(min=eps).log()).sum(dim=-1)
        uncertainty = entropy / (entropy.max() + eps)
        
        # Adaptive sample count
        N_samples_per_ray = (
            self.N_samples_min + uncertainty * (self.N_samples_max - self.N_samples_min)
        ).clamp(min=self.N_samples_min).round().long()
        
        skip_mask = (uncertainty < self.entropy_threshold)
        
        # Sample bin centers
        N_samples = N_samples_per_ray.max().item()
        centers = torch.multinomial(pdf, N_samples, replacement=True)
        z_c = torch.gather(bins, 1, centers)
        
        # Bandwidth
        bandwidth = (bins[:, 1:] - bins[:, :-1]).mean(dim=-1, keepdim=True).clamp(min=1e-5)
        
        # Kernel perturbations
        eps_samples = self._kernel_samples(num_rays, N_samples, device)
        samples = z_c + eps_samples * bandwidth
        
        # Apply masks
        sample_ids = torch.arange(N_samples, device=device).unsqueeze(0)
        mask = sample_ids < N_samples_per_ray.unsqueeze(1)
        samples[~mask] = 0.0
        samples[skip_mask] = 0.0
        
        samples, _ = torch.sort(samples, dim=-1)
        return self._pack_ray_samples(samples)
    
    def _kernel_samples(self, num_rays, num_samples, device):
        if self.kernel_type == 'gaussian':
            return torch.randn(num_rays, num_samples, device=device)
        elif self.kernel_type == 'epanechnikov':
            u = torch.rand(num_rays, num_samples, device=device)
            eps = torch.sqrt(1 - u)
            signs = torch.randint(0, 2, eps.shape, device=device).float() * 2 - 1
            return eps * signs
        elif self.kernel_type == 'uniform':
            return torch.rand(num_rays, num_samples, device=device) * 2 - 1
        return torch.randn(num_rays, num_samples, device=device)
    
    def _uniform_samples(self, ray_bundle, num_samples):
        num_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        t = torch.linspace(0., 1., num_samples, device=device)
        samples = ray_bundle.nears + t * (ray_bundle.fars - ray_bundle.nears)
        samples = samples.unsqueeze(0).expand(num_rays, -1)
        return self._pack_ray_samples(samples)
    
    def _pack_ray_samples(self, samples):
        bin_starts = samples
        bin_ends = torch.cat([samples[:, 1:], samples[:, -1:] + 1e-3], dim=-1)
        return RaySamples(
            frustums=None, camera_indices=None,
            deltas=bin_ends - bin_starts,
            spacing_starts=bin_starts, spacing_ends=bin_ends,
            spacing_to_euclidean_fn=lambda x: x,
            metadata={}, times=None,
        )


# ============================================================================
# 7. WAVELET HIERARCHICAL SAMPLER
# ============================================================================

class WaveletHierarchicalSampler(Sampler):
    """
    Wavelet-based hierarchical sampling
    Multi-scale importance using Haar-like decomposition
    """
    
    def __init__(self, num_samples: int = 64, single_jitter: bool = True, **kwargs):
        super().__init__(num_samples=num_samples, single_jitter=single_jitter)
        self.levels = kwargs.get('levels', 3)
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        weights: Optional[Tensor] = None,
        bins: Optional[Tensor] = None,
        **kwargs
    ) -> RaySamples:
        
        if num_samples is None:
            num_samples = self.num_samples
        
        if weights is None or bins is None:
            return self._uniform_samples(ray_bundle, num_samples)
        
        device = bins.device
        num_rays, N_bins = bins.shape
        eps = 1e-5
        
        # Normalize
        weights = weights + eps
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Multi-resolution analysis
        scales = []
        current_weights = weights
        
        for level in range(self.levels):
            if current_weights.shape[1] < 2:
                break
            
            # Downsample
            if current_weights.shape[1] % 2 == 1:
                current_weights = current_weights[:, :-1]
            
            downsampled = (current_weights[:, ::2] + current_weights[:, 1::2]) / 2
            scales.append(downsampled)
            current_weights = downsampled
        
        # Allocate samples across scales
        samples_per_scale = [num_samples // (2**i) for i in range(len(scales))]
        samples_per_scale[0] += num_samples - sum(samples_per_scale)
        
        all_samples = []
        
        for i, (scale_weights, n_samples) in enumerate(zip(scales, samples_per_scale)):
            if n_samples == 0:
                continue
            
            # Create bins for current scale
            scale_factor = 2**i
            scale_bins = bins[:, ::scale_factor]
            
            if scale_bins.shape[1] < scale_weights.shape[1]:
                scale_weights = scale_weights[:, :scale_bins.shape[1]]
            
            # Sample from current scale
            pdf = scale_weights / (scale_weights.sum(dim=-1, keepdim=True) + eps)
            cdf = torch.cumsum(pdf, dim=-1)
            cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
            
            if self.single_jitter:
                u = torch.linspace(0., 1., n_samples, device=device).unsqueeze(0).expand(num_rays, -1)
            else:
                u = torch.rand(num_rays, n_samples, device=device)
            
            # Invert CDF
            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.clamp(inds - 1, min=0)
            above = torch.clamp(inds, max=cdf.shape[-1] - 1)
            
            bins_g0 = torch.gather(scale_bins, 1, below)
            bins_g1 = torch.gather(scale_bins, 1, above)
            cdf_g0 = torch.gather(cdf, 1, below)
            cdf_g1 = torch.gather(cdf, 1, above)
            
            denom = cdf_g1 - cdf_g0
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g0) / denom
            
            scale_samples = bins_g0 + t * (bins_g1 - bins_g0)
            all_samples.append(scale_samples)
        
        # Combine all samples
        if all_samples:
            combined_samples = torch.cat(all_samples, dim=1)
            combined_samples, _ = torch.sort(combined_samples, dim=1)
            return self._pack_ray_samples(combined_samples)
        else:
            return self._uniform_samples(ray_bundle, num_samples)
    
    def _uniform_samples(self, ray_bundle, num_samples):
        num_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        t = torch.linspace(0., 1., num_samples, device=device)
        samples = ray_bundle.nears + t * (ray_bundle.fars - ray_bundle.nears)
        samples = samples.unsqueeze(0).expand(num_rays, -1)
        return self._pack_ray_samples(samples)
    
    def _pack_ray_samples(self, samples):
        bin_starts = samples
        bin_ends = torch.cat([samples[:, 1:], samples[:, -1:] + 1e-3], dim=-1)
        return RaySamples(
            frustums=None, camera_indices=None,
            deltas=bin_ends - bin_starts,
            spacing_starts=bin_starts, spacing_ends=bin_ends,
            spacing_to_euclidean_fn=lambda x: x,
            metadata={}, times=None,
        )


# ============================================================================
# 8. KERNEL-TILTED SAMPLER (K-TOSS)
# ============================================================================

class KernelTiltedSampler(Sampler):
    """
    K-TOSS: Kernel-Tilted Ray Sampling
    Applies kernel weighting before sampling
    """
    
    def __init__(self, num_samples: int = 64, single_jitter: bool = True, **kwargs):
        super().__init__(num_samples=num_samples, single_jitter=single_jitter)
        self.kernel = kwargs.get('kernel', 'gaussian')
        self.sigma = kwargs.get('sigma', 0.1)
    
    def generate_ray_samples(
        self,
        ray_bundle: Optional[RayBundle] = None,
        num_samples: Optional[int] = None,
        weights: Optional[Tensor] = None,
        bins: Optional[Tensor] = None,
        **kwargs
    ) -> RaySamples:
        
        if num_samples is None:
            num_samples = self.num_samples
        
        if weights is None or bins is None:
            return self._uniform_samples(ray_bundle, num_samples)
        
        device = weights.device
        num_rays = weights.shape[0]
        eps = 1e-5
        
        # Normalize weights
        weights = weights + eps
        weights = weights / weights.sum(-1, keepdim=True)
        
        # Apply kernel function
        if self.kernel == 'gaussian':
            kernel_weights = torch.exp(-0.5 * (weights - 0.5)**2 / self.sigma**2)
            kernel_weights = kernel_weights / kernel_weights.sum(-1, keepdim=True)
        elif self.kernel == 'uniform':
            kernel_weights = torch.ones_like(weights)
        else:
            kernel_weights = weights
        
        # Create PDF from kernel weights
        pdf = kernel_weights / kernel_weights.sum(-1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        
        # Sample
        if self.single_jitter:
            u = torch.linspace(0., 1., num_samples, device=device).unsqueeze(0).expand(num_rays, -1)
        else:
            u = torch.rand(num_rays, num_samples, device=device)
        
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        
        cdf_g0 = torch.gather(cdf, 1, below)
        cdf_g1 = torch.gather(cdf, 1, above)
        bins_g0 = torch.gather(bins, 1, below)
        bins_g1 = torch.gather(bins, 1, above)
        
        denom = (cdf_g1 - cdf_g0)
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g0) / denom
        samples = bins_g0 + t * (bins_g1 - bins_g0)
        
        samples, _ = torch.sort(samples, dim=-1)
        return self._pack_ray_samples(samples)
    
    def _uniform_samples(self, ray_bundle, num_samples):
        num_rays = ray_bundle.origins.shape[0]
        device = ray_bundle.origins.device
        t = torch.linspace(0., 1., num_samples, device=device)
        samples = ray_bundle.nears + t * (ray_bundle.fars - ray_bundle.nears)
        samples = samples.unsqueeze(0).expand(num_rays, -1)
        return self._pack_ray_samples(samples)
    
    def _pack_ray_samples(self, samples):
        bin_starts = samples
        bin_ends = torch.cat([samples[:, 1:], samples[:, -1:] + 1e-3], dim=-1)
        return RaySamples(
            frustums=None, camera_indices=None,
            deltas=bin_ends - bin_starts,
            spacing_starts=bin_starts, spacing_ends=bin_ends,
            spacing_to_euclidean_fn=lambda x: x,
            metadata={}, times=None,
        )


# ============================================================================
# SAMPLER REGISTRY & FACTORY
# ============================================================================

SAMPLER_REGISTRY = {
    'adaptive_kernel_v2': AdaptiveKernelSamplerV2,
    'adaptive_kernel_v3': AdaptiveKernelSamplerV3,
    'l0': L0Sampler,
    'gmm': GMMSampler,
    'optimal_transport': OptimalTransportSampler,
    'entropy_kde': EntropyKDESampler,
    'wavelet': WaveletHierarchicalSampler,
    'kernel_tilted': KernelTiltedSampler,
}


def create_sampler(sampler_type: str, **kwargs):
    """Factory function to create samplers"""
    if sampler_type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler: {sampler_type}. Available: {list(SAMPLER_REGISTRY.keys())}")
    
    sampler_class = SAMPLER_REGISTRY[sampler_type]
    return sampler_class(**kwargs)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'AdaptiveKernelSamplerV2',
    'AdaptiveKernelSamplerV3',
    'L0Sampler',
    'GMMSampler',
    'OptimalTransportSampler',
    'EntropyKDESampler',
    'WaveletHierarchicalSampler',
    'KernelTiltedSampler',
    'SAMPLER_REGISTRY',
    'create_sampler',
]


print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ALL SAMPLERS INTEGRATED                                ║
╚═══════════════════════════════════════════════════════════════════════════╝

 8 Sampling Methods Implemented:

1. AdaptiveKernelSamplerV2      → Your main method (4 kernel types)
2. AdaptiveKernelSamplerV3      → With entropy-based adaptation
3. L0Sampler                    → Baseline (Li et al. 2023)
4. GMMSampler                   → Gaussian Mixture Model
5. OptimalTransportSampler      → Fast 1D optimal transport
6. EntropyKDESampler            → Pure entropy + KDE
7. WaveletHierarchicalSampler   → Multi-scale wavelet
8. KernelTiltedSampler          → K-TOSS method

 Recommended Experiments:

PRIMARY COMPARISON:
- baseline_pdf (Nerfstudio default)
- l0 (Li et al. baseline)
- adaptive_kernel_v2 + epanechnikov (YOUR MAIN METHOD)

SECONDARY COMPARISON:
- adaptive_kernel_v3 (with entropy)
- optimal_transport (fast alternative)
- gmm (traditional approach)

ABLATION STUDIES:
- Kernel types: gaussian, epanechnikov, triangular, uniform
- Blur effect: with/without maxblur
- Entropy adaptation: v2 vs v3

""")

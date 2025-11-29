#!/usr/bin/env python3
"""
Extract divergence fields from CoTracker3 point tracks
Input: CoTracker3 tracks [B, N, T, 2] 
Output: Divergence maps [B, T-1, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


class TCAMDivergenceExtractor(nn.Module):
    def __init__(self, grid_size=(64, 64), temporal_window=3, smoothing_sigma=1.5):
        """
        Args:
            grid_size: Resolution for divergence computation
            temporal_window: Frames for temporal smoothing
            smoothing_sigma: Gaussian smoothing parameter
        """
        super().__init__()
        self.grid_size = grid_size
        self.temporal_window = temporal_window
        self.smoothing_sigma = smoothing_sigma
        
        # Register grid coordinates as buffers
        grid_h, grid_w = grid_size
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, grid_h), 
            torch.linspace(0, 1, grid_w),
            indexing='ij'
        )
        self.register_buffer('grid_coords', torch.stack([x_grid, y_grid], dim=-1))
        
    def forward(self, tracks, visibility, image_size=(480, 640)):
        """
        Args:
            tracks: [B, N, T, 2] point tracks from CoTracker3
            visibility: [B, N, T] visibility masks
        Returns:
            divergence_maps: [B, T-1, H, W] 
            confidence_maps: [B, T-1, H, W]
        """
        B, N, T, _ = tracks.shape
        H, W = image_size
        grid_h, grid_w = self.grid_size
        
        device = tracks.device
        dtype = tracks.dtype
        
        # Initialize output tensors
        divergence_maps = torch.zeros(B, T-1, grid_h, grid_w, device=device, dtype=dtype)
        confidence_maps = torch.zeros(B, T-1, grid_h, grid_w, device=device, dtype=dtype)
        
        for b in range(B):
            for t in range(T-1):
                # Get valid tracks for consecutive frames
                valid_t = visibility[b, :, t] > 0.5
                valid_t1 = visibility[b, :, t+1] > 0.5
                valid_both = valid_t & valid_t1
                
                if valid_both.sum() < 4:  # Need minimum points for divergence
                    continue
                
                # Extract valid points and compute optical flow
                points_t = tracks[b, valid_both, t]  # [valid_N, 2]
                points_t1 = tracks[b, valid_both, t+1]  # [valid_N, 2]
                
                # Normalize coordinates to [0, 1]
                points_t_norm = points_t.clone()
                points_t_norm[:, 0] /= W  # x
                points_t_norm[:, 1] /= H  # y
                
                points_t1_norm = points_t1.clone()
                points_t1_norm[:, 0] /= W  # x
                points_t1_norm[:, 1] /= H  # y
                
                # Compute optical flow
                flow = points_t1_norm - points_t_norm  # [valid_N, 2]
                
                # Interpolate sparse flow to dense grid
                div_map, conf_map = self._interpolate_and_compute_divergence(
                    points_t_norm, flow, device, dtype
                )
                
                divergence_maps[b, t] = div_map
                confidence_maps[b, t] = conf_map
        
        return divergence_maps, confidence_maps
    
    def _interpolate_and_compute_divergence(self, points, flow, device, dtype):
        """
        Interpolate sparse flow to grid and compute divergence using finite differences
        """
        grid_h, grid_w = self.grid_size
        
        # Convert to numpy for scipy interpolation
        points_np = points.cpu().numpy()
        flow_np = flow.cpu().numpy()
        
        # Create grid coordinates
        grid_coords = self.grid_coords.cpu().numpy()  # [H, W, 2]
        grid_points = grid_coords.reshape(-1, 2)  # [H*W, 2]
        
        # Check if we have enough points for interpolation
        if len(points_np) < 3:
            return torch.zeros(grid_h, grid_w, device=device, dtype=dtype), \
                   torch.zeros(grid_h, grid_w, device=device, dtype=dtype)
        
        try:
            # Interpolate flow components
            flow_x = griddata(points_np, flow_np[:, 0], grid_points, 
                            method='linear', fill_value=0.0)
            flow_y = griddata(points_np, flow_np[:, 1], grid_points, 
                            method='linear', fill_value=0.0)
            
            # Reshape to grid
            flow_x_grid = flow_x.reshape(grid_h, grid_w)
            flow_y_grid = flow_y.reshape(grid_h, grid_w)
            
            # Apply Gaussian smoothing
            flow_x_smooth = gaussian_filter(flow_x_grid, sigma=self.smoothing_sigma)
            flow_y_smooth = gaussian_filter(flow_y_grid, sigma=self.smoothing_sigma)
            
            # Compute divergence using finite differences
            # div = ∂u/∂x + ∂v/∂y
            du_dx = np.gradient(flow_x_smooth, axis=1)
            dv_dy = np.gradient(flow_y_smooth, axis=0)
            divergence = du_dx + dv_dy
            
            # Normalize divergence to [-1, 1] using tanh
            divergence_norm = np.tanh(divergence)
            
            # Compute confidence based on track density (vectorized on device)
            conf_tensor = self._compute_confidence_torch(points, radius=0.1)
            
            # Convert divergence back to torch tensor
            div_tensor = torch.from_numpy(divergence_norm).to(device=device, dtype=dtype)
            
            return div_tensor, conf_tensor
            
        except Exception as e:
            print(f"Warning: Interpolation failed: {e}")
            return torch.zeros(grid_h, grid_w, device=device, dtype=dtype), \
                   torch.zeros(grid_h, grid_w, device=device, dtype=dtype)

    def _compute_confidence_torch(self, points: torch.Tensor, radius: float = 0.1) -> torch.Tensor:
        """
        Vectorized confidence based on track density around each grid point.
        Args:
            points: [P, 2] normalized (0..1) coordinates on same device as buffers
            radius: scalar radius in normalized coordinate space
        Returns:
            confidence map [H, W] as torch.Tensor on same device/dtype as points
        """
        grid_h, grid_w = self.grid_size
        if points.numel() == 0:
            return torch.zeros(grid_h, grid_w, device=points.device, dtype=points.dtype)

        # [H, W, 2] -> [HW, 2]
        grid = self.grid_coords.reshape(-1, 2)  # buffer already on correct device
        # Pairwise distances: [HW, P, 2] -> [HW, P]
        diff = grid.unsqueeze(1) - points.unsqueeze(0)
        dist2 = (diff * diff).sum(dim=-1)
        nearby = (dist2 < (radius * radius)).sum(dim=1).to(points.dtype)
        conf = torch.sigmoid(nearby - 2.0)  # same shaping as numpy version
        return conf.view(grid_h, grid_w)


class TemporalDivergenceAggregator(nn.Module):
    """
    Aggregates divergence maps across temporal windows for stability
    """
    def __init__(self, temporal_window=3):
        super().__init__()
        self.temporal_window = temporal_window
        
        # Temporal convolution for smoothing
        self.temporal_conv = nn.Conv1d(1, 1, kernel_size=temporal_window, 
                                     padding=temporal_window//2, bias=False)
        
        # Initialize with Gaussian kernel
        with torch.no_grad():
            kernel = torch.tensor([0.25, 0.5, 0.25]).unsqueeze(0).unsqueeze(0)
            if temporal_window > 3:
                # Create larger Gaussian kernel
                sigma = temporal_window / 6.0
                x = torch.arange(temporal_window) - temporal_window // 2
                kernel = torch.exp(-0.5 * (x / sigma) ** 2)
                kernel = kernel / kernel.sum()
                kernel = kernel.unsqueeze(0).unsqueeze(0)
            self.temporal_conv.weight.copy_(kernel)
    
    def forward(self, divergence_maps):
        """
        Args:
            divergence_maps: [B, T, H, W] divergence maps
        Returns:
            smoothed_maps: [B, T, H, W] temporally smoothed divergence maps
        """
        B, T, H, W = divergence_maps.shape
        
        # Reshape for temporal convolution: [B, T, H, W] -> [B*H*W, 1, T]
        div_flat = divergence_maps.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, 1, T)
        
        # Apply temporal smoothing
        smoothed_flat = self.temporal_conv(div_flat)
        
        # Reshape back: [B*H*W, 1, T] -> [B, T, H, W]
        smoothed_maps = smoothed_flat.reshape(B, H, W, T).permute(0, 3, 1, 2)
        
        return smoothed_maps 
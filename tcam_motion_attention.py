#!/usr/bin/env python3
"""
TCAM Motion Attention Implementation
Motion attention where motion relationships define attention patterns.

Core Principle: "Given how things are moving, what should pay attention to what?"
- Q: "Where am I going?" (future motion)
- K: "Where did you come from?" (past motion)  
- V: "What motion information do you carry?" (motion features)

Text-guided attention to focus on subject-relevant motion patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from tcam_divergence_extractor import TCAMDivergenceExtractor


class TCAMTrajectoryEmbedding(nn.Module):
    """
    Trajectory-based embeddings that encode the unique spatiotemporal path of each track.
    This allows distinction between identical objects with similar motion patterns.
    """
    def __init__(self, d_model, temporal_window=8, max_position=1000):
        super().__init__()
        self.d_model = d_model
        self.temporal_window = temporal_window
        self.max_position = max_position
        
        # Spatial position embeddings for absolute positions
        self.spatial_embed_x = nn.Embedding(max_position, d_model // 4)
        self.spatial_embed_y = nn.Embedding(max_position, d_model // 4)
        
        # Temporal position embeddings
        self.temporal_embed = nn.Embedding(temporal_window * 2, d_model // 4)
        
        # Trajectory pattern encoding using CNN over trajectory
        self.trajectory_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, padding=1),  # Input: [B*N, 2, T]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Pool to single value per channel
            nn.Flatten(),  # [B*N, 64]
            nn.Linear(64, d_model // 4)  # Project to correct dimension
        )
        
        # Combine all embeddings
        self.projection = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tracks, visibility=None, current_time=None):
        """
        Generate trajectory embeddings from point tracks
        
        Args:
            tracks: [B, N_tracks, T, 2] point tracks
            visibility: [B, N_tracks, T] visibility masks
            current_time: int, current frame index for temporal encoding
            
        Returns:
            trajectory_embeddings: [B, N_tracks, d_model]
        """
        B, N_tracks, T, _ = tracks.shape
        device = tracks.device
        
        # 1. Spatial position embeddings
        # Discretize continuous positions to embedding indices
        tracks_norm = tracks.clone()
        tracks_norm[:, :, :, 0] = torch.clamp(tracks_norm[:, :, :, 0] / 640.0 * self.max_position, 0, self.max_position - 1)
        tracks_norm[:, :, :, 1] = torch.clamp(tracks_norm[:, :, :, 1] / 480.0 * self.max_position, 0, self.max_position - 1)
        
        # Get embeddings for start, middle, and end positions
        start_pos = tracks_norm[:, :, 0].long()  # [B, N, 2]
        mid_pos = tracks_norm[:, :, T//2].long() if T > 1 else start_pos
        end_pos = tracks_norm[:, :, -1].long() if T > 1 else start_pos
        
        # Spatial embeddings - each should be d_model//8 to total d_model//4 when concatenated
        spatial_start_x = self.spatial_embed_x(start_pos[:, :, 0])[:, :, :self.d_model//8]  # Truncate to d_model//8
        spatial_start_y = self.spatial_embed_y(start_pos[:, :, 1])[:, :, :self.d_model//8]  # Truncate to d_model//8
        spatial_start = torch.cat([spatial_start_x, spatial_start_y], dim=-1)  # [B, N, d_model//4]
        
        spatial_end_x = self.spatial_embed_x(end_pos[:, :, 0])[:, :, :self.d_model//8]  # Truncate to d_model//8
        spatial_end_y = self.spatial_embed_y(end_pos[:, :, 1])[:, :, :self.d_model//8]  # Truncate to d_model//8
        spatial_end = torch.cat([spatial_end_x, spatial_end_y], dim=-1)  # [B, N, d_model//4]
        
        # 2. Temporal embeddings (encode temporal context)
        if current_time is not None:
            # Distance from current time
            time_indices = torch.full((B, N_tracks), current_time, device=device).long()
            time_indices = torch.clamp(time_indices, 0, self.temporal_window * 2 - 1)
        else:
            # Use middle of sequence as reference
            time_indices = torch.full((B, N_tracks), self.temporal_window, device=device).long()
        
        temporal_embeds = self.temporal_embed(time_indices)  # [B, N, d_model//4]
        
        # 3. Trajectory pattern encoding
        # Reshape tracks for CNN: [B*N, 2, T]
        tracks_for_cnn = tracks.transpose(-1, -2).reshape(B * N_tracks, 2, T)
        
        # Apply visibility mask if provided
        if visibility is not None:
            vis_mask = visibility.unsqueeze(-1).transpose(-1, -2).reshape(B * N_tracks, 1, T)
            tracks_for_cnn = tracks_for_cnn * vis_mask
        
        # Encode trajectory patterns
        traj_features = self.trajectory_encoder(tracks_for_cnn)  # [B*N, d_model//4]
        traj_features = traj_features.reshape(B, N_tracks, -1)
        
        # 4. Combine all embeddings
        # Use different combinations for start/end to capture direction
        combined_embeds = torch.cat([
            spatial_start,      # Where did it start?
            spatial_end,        # Where did it end?
            temporal_embeds,    # When in the sequence?
            traj_features,      # What's the trajectory pattern?
        ], dim=-1)  # [B, N, d_model]
        
        # Final projection and normalization
        trajectory_embeddings = self.projection(combined_embeds)
        trajectory_embeddings = self.norm(trajectory_embeddings)
        
        return trajectory_embeddings


class TCAMMotionAttention(nn.Module):
    """
    Enhanced Motion Field Attention with trajectory-based token embeddings.
    Now supports distinction between identical objects with similar motion patterns.
    NEW: Text-guided cross-attention for subject-aware motion focus.
    """
    def __init__(self, d_model, n_heads, temporal_window=5, dropout=0.1, use_trajectory_tokens=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.temporal_window = temporal_window
        self.d_k = d_model // n_heads
        self.use_trajectory_tokens = use_trajectory_tokens
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Motion descriptor dimension: vx, vy, ax, ay, div, curl = 6
        self.motion_dim = 6
        
        # NEW: Trajectory embedding for unique object identification
        if use_trajectory_tokens:
            self.trajectory_embedding = TrajectoryEmbedding(d_model, temporal_window)
            
            # Enhanced motion+trajectory to attention projections
            self.motion_traj_to_q = nn.Linear(self.motion_dim + d_model, d_model)
            self.motion_traj_to_k = nn.Linear(self.motion_dim + d_model, d_model)
            self.motion_traj_to_v = nn.Linear(self.motion_dim + d_model, d_model)
        else:
            # Original motion-only projections
            print("🚀 USING MOTION ONLY PATH!")
            self.motion_to_q = nn.Linear(self.motion_dim, d_model)
            self.motion_to_k = nn.Linear(self.motion_dim, d_model)
            self.motion_to_v = nn.Linear(self.motion_dim, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Reuse divergence computation from existing TCAMDivergenceExtractor
        self.div_extractor = TCAMDivergenceExtractor(
            grid_size=(96, 96),  # Match existing
            temporal_window=temporal_window,
            smoothing_sigma=1.5
        )
        
        # Scale factor for attention scores
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Layer normalization for stability
        self.norm_input = nn.LayerNorm(d_model)
        
        # NEW: Track-level attention for fine-grained object distinction
        self.track_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads//2 if n_heads > 1 else 1,
            dropout=dropout,
            batch_first=True
        )
        
        # DISABLED: Text-guided cross-attention components (moved to TextTrackCrossAttention)
        # self.text_guided_attention = nn.MultiheadAttention(
        #     embed_dim=self.motion_dim,  # Motion descriptors as queries
        #     num_heads=min(2, n_heads//2),  # Small number of heads for efficiency  
        #     dropout=dropout,
        #     batch_first=True
        # )
        # 
        # # Project text embeddings to motion descriptor space for compatibility
        # self.text_to_motion_proj = nn.Linear(d_model, self.motion_dim)
        # 
        # # Learnable weighting for text guidance (starts small)
        # self.text_guidance_weight = nn.Parameter(torch.tensor(0.1))  # Small initial weight
        
    def forward(self, x, divergence=None, depth_logits=None, tracks=None, visibility=None, 
                mask=None, current_time=None, return_attention=False, text_embeddings=None):
        """
        Enhanced forward pass with text-guided attention
        
        Args:
            x: [B, N, D] input features (N = H*W flattened)
            divergence: [B, H, W] divergence map (IGNORED - computed from tracks)
            depth_logits: [B, L, H, W] depth stratification logits (IGNORED - not used)
            tracks: [B, N_tracks, T, 2] point tracks from CoTracker (REQUIRED for motion)
            visibility: [B, N_tracks, T] visibility masks (REQUIRED for motion)
            mask: [B, N, N] attention mask (optional)
            current_time: int, current frame index for temporal weighting (optional)
            return_attention: whether to return attention maps
            text_embeddings: [B, d_model] text embeddings for guidance (NEW)
            
        Returns:
            out: [B, N, D] attention output
            attention_maps: [B, H, N, N] if return_attention=True
        """
        # Check if we have tracks data for motion field attention
        if tracks is None or visibility is None:
            # Fallback: use a simple self-attention if no tracks provided
            return self.fallback_attention(x, return_attention)
        
        B, N, D = x.shape
        
        # Normalize input features
        x_norm = self.norm_input(x)
        
        if self.use_trajectory_tokens:
            # NEW: Extract both motion descriptors AND trajectory embeddings
            motion_descriptors = self.compute_motion_descriptors(tracks, visibility, current_time)
            trajectory_embeddings = self.trajectory_embedding(tracks, visibility, current_time)
            
            # DISABLED: Broken text guidance - let TextTrackCrossAttention handle text-track relationships
            # if text_embeddings is not None:
            #     motion_descriptors = self.apply_text_guidance(motion_descriptors, text_embeddings)
            
            # Combine motion and trajectory information
            combined_features = torch.cat([motion_descriptors, trajectory_embeddings], dim=-1)
            
            # Generate Q, K from combined motion+trajectory features
            Q = self.motion_traj_to_q(combined_features)  # [B, N_tracks, d_model]
            K = self.motion_traj_to_k(combined_features)  # [B, N_tracks, d_model]
            
            # V still comes from trajectory embeddings (what unique info each track carries)
            V_tracks = self.motion_traj_to_v(combined_features)  # [B, N_tracks, d_model]
            
        else:
            # Original motion-only approach
            print("🚀 USING MOTION ONLY PATH!")
            motion_descriptors = self.compute_motion_descriptors(tracks, visibility, current_time)
            
            # DISABLED: Broken text guidance - let TextTrackCrossAttention handle text-track relationships
            # if text_embeddings is not None:
            #     motion_descriptors = self.apply_text_guidance(motion_descriptors, text_embeddings)
            
            Q = self.motion_to_q(motion_descriptors)
            K = self.motion_to_k(motion_descriptors)
            V_tracks = motion_descriptors
        
        # NEW: Apply track-level attention to refine track representations
        if self.use_trajectory_tokens:
            V_tracks_refined, track_attention = self.track_attention(V_tracks, V_tracks, V_tracks)
            V_tracks = V_tracks + V_tracks_refined  # Residual connection
        
        # Use refined track features as V, original features for spatial features
        V = x_norm
        
        # If track resolution differs from feature resolution, interpolate
        if Q.shape[1] != N:
            Q = self.interpolate_to_feature_grid(Q, N)
            K = self.interpolate_to_feature_grid(K, N)
            # Also interpolate the track values to spatial grid
            V_tracks_interp = self.interpolate_to_feature_grid(V_tracks, N)
            # Combine spatial and track information
            V = V + V_tracks_interp
        else:
            # Ensure dimensions match before adding
            if V_tracks.shape[-1] != V.shape[-1]:
                # Project V_tracks to match V's dimension
                if not hasattr(self, 'v_tracks_proj'):
                    self.v_tracks_proj = nn.Linear(V_tracks.shape[-1], V.shape[-1]).to(V.device)
                V_tracks = self.v_tracks_proj(V_tracks)
            V = V + V_tracks
        
        # Multi-head attention computation
        Q = Q.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = K.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        V = V.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        
        # Attention scores based on motion+trajectory relationships
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Apply enhanced motion-trajectory modulation
        if self.use_trajectory_tokens:
            scores = self.apply_trajectory_motion_modulation(scores, motion_descriptors, trajectory_embeddings, N)
        else:
            print("🚀 USING MOTION ONLY PATH!")
            scores = self.apply_motion_modulation(scores, motion_descriptors, N)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to propagate features
        out = torch.matmul(attn_weights, V)  # [B, H, N, d_k]
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)
        out = self.out_proj(out)
        
        # Residual connection
        out = out + x
        
        if return_attention:
            return out, attn_weights
        return out
    
    def compute_motion_descriptors(self, tracks, visibility, current_time=None):
        """
        NEW: Learned motion embeddings instead of hand-engineered features
        Converts point trajectories into rich learned representations
        
        Args:
            tracks: [B, N_tracks, T, 2] point tracks
            visibility: [B, N_tracks, T] visibility masks
            current_time: int, current frame index for temporal weighting (optional)
        """
        B, N_tracks, T, _ = tracks.shape
        device = tracks.device
        dtype = tracks.dtype
        
        # Initialize learned motion encoders if not exists
        if not hasattr(self, 'track_embedding'):
            # Use CLIP embedding dimension for alignment
            clip_dim = 512  # Standard CLIP embedding dimension
            hidden_dim = self.d_model // 2
            
            # Position embedding
            self.track_embedding = nn.Sequential(
                nn.Linear(2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim)
            ).to(device)
            
            # Velocity embedding
            self.velocity_embedding = nn.Sequential(
                nn.Linear(2, hidden_dim // 2),
                nn.ReLU(), 
                nn.Linear(hidden_dim // 2, hidden_dim)
            ).to(device)
            
            # Confidence embedding
            self.confidence_embedding = nn.Linear(1, hidden_dim).to(device)
            
            # Spatial transformer for track interactions
            self.spatial_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=2
            ).to(device)
            
            # Temporal transformer for motion dynamics
            self.temporal_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 2,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=2
            ).to(device)
            
            # Positional embeddings for transformers
            self.temporal_pos_embedding = nn.Embedding(100, hidden_dim).to(device)  # Max 100 frames
            self.spatial_pos_embedding = nn.Embedding(1000, hidden_dim).to(device)  # Max 1000 tracks
            
            # CLIP-aligned motion projection
            self.motion_clip_projector = nn.Sequential(
                nn.Linear(hidden_dim * 3, clip_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(clip_dim, clip_dim),
                nn.LayerNorm(clip_dim)
            ).to(device)
            
            # Final projection to motion descriptor space (for backward compatibility)
            self.motion_projection = nn.Sequential(
                nn.Linear(clip_dim, self.motion_dim),
                nn.LayerNorm(self.motion_dim)
            ).to(device)
        
        # Compute velocities (temporal differences)
        velocities = torch.zeros_like(tracks)
        if T > 1:
            velocities[:, :, 1:] = tracks[:, :, 1:] - tracks[:, :, :-1]
        
        # Use visibility as confidence scores
        confidence = visibility.float().mean(dim=2, keepdim=True)  # [B, N_tracks, 1]
        
        # OPTIMIZED: Process tracks with hierarchical transformers using motion aware CLIP approach
        # Key insight: Loop over time (16 iterations) instead of tracks (576 iterations)
        
        # Convert to time-first format: [B, T, N_tracks, 2] for efficient processing
        tracks_time_first = tracks.transpose(1, 2)  # [B, T, N_tracks, 2]
        velocities_time_first = velocities.transpose(1, 2)  # [B, T, N_tracks, 2]
        visibility_time_first = visibility.transpose(1, 2)  # [B, T, N_tracks]
        
        
        # Embed ALL positions and velocities at once (parallel processing)
        # Reshape for batch processing: [B*T*N_tracks, 2]
        tracks_flat = tracks_time_first.reshape(B * T * N_tracks, 2)
        velocities_flat = velocities_time_first.reshape(B * T * N_tracks, 2)
        
        
        # Parallel embedding computation
        track_emb_flat = self.track_embedding(tracks_flat)  # [B*T*N_tracks, hidden_dim]
        velocity_emb_flat = self.velocity_embedding(velocities_flat)  # [B*T*N_tracks, hidden_dim]
        
        
        # Reshape back to time-first format: [B, T, N_tracks, hidden_dim]
        track_emb = track_emb_flat.reshape(B, T, N_tracks, -1)
        velocity_emb = velocity_emb_flat.reshape(B, T, N_tracks, -1)
        
        
        # Add positional embeddings
        temporal_pos = self.temporal_pos_embedding(torch.arange(T, device=device, dtype=torch.long))  # [T, hidden_dim]
        spatial_pos = self.spatial_pos_embedding(torch.arange(N_tracks, device=device, dtype=torch.long))  # [N_tracks, hidden_dim]
        
        # Broadcast positional embeddings
        temporal_pos = temporal_pos.unsqueeze(0).unsqueeze(2)  # [1, T, 1, hidden_dim]
        spatial_pos = spatial_pos.unsqueeze(0).unsqueeze(1)    # [1, 1, N_tracks, hidden_dim]
        
        # Combine embeddings with positional information
        point_features = track_emb + velocity_emb + temporal_pos + spatial_pos  # [B, T, N_tracks, hidden_dim]
        
        
        # 1. SPATIAL PROCESSING: Process each timestep (16 iterations instead of 576)
        spatial_features = []
        for t in range(T):
            # Get features for this timestep: [B, N_tracks, hidden_dim]
            timestep_features = point_features[:, t, :, :]  # [B, N_tracks, hidden_dim]
            timestep_visibility = visibility_time_first[:, t, :]  # [B, N_tracks]
            
            # Check if any points are visible in this timestep
            num_visible = timestep_visibility.sum(dim=1)  # [B]
            
            if (num_visible == 0).all():
                # No visible points - use zero features
                spatial_out = torch.zeros_like(timestep_features)
            else:
                # Apply spatial transformer with visibility masking
                spatial_mask = (timestep_visibility == 0)  # [B, N_tracks] - True for invisible
                
                try:
                    spatial_out = self.spatial_transformer(
                        timestep_features,
                        src_key_padding_mask=spatial_mask
                    )  # [B, N_tracks, hidden_dim]
                except Exception as e:
                    spatial_out = timestep_features
            
            spatial_features.append(spatial_out)
        
        # Stack spatial features: [B, T, N_tracks, hidden_dim]
        spatial_features = torch.stack(spatial_features, dim=1)
        
        # 2. TEMPORAL AGGREGATION: Aggregate across points using visibility weighting
        # Create confidence weights from visibility
        confidence_time_first = confidence.transpose(1, 2).unsqueeze(-1)  # [B, T, N_tracks, 1]
        weights = visibility_time_first.float().unsqueeze(-1) * confidence_time_first  # [B, T, N_tracks, 1]
        weights_sum = weights.sum(dim=2, keepdim=True)  # [B, T, 1, 1]
        
        
        # Handle case where all points are invisible
        has_visible_points = (weights_sum.squeeze(-1) > 1e-8).float()  # [B, T, 1]
        uniform_weights = torch.ones_like(weights) / N_tracks  # [B, T, N_tracks, 1]
        
        # Safe weight normalization
        weights_sum_safe = torch.clamp(weights_sum, min=1e-8)
        weights_norm = weights / weights_sum_safe  # [B, T, N_tracks, 1]
        final_weights = has_visible_points.unsqueeze(-1) * weights_norm + (1 - has_visible_points.unsqueeze(-1)) * uniform_weights
        
        # Weighted aggregation across tracks
        aggregated_features = (spatial_features * final_weights).sum(dim=2)  # [B, T, hidden_dim]
        
        # 3. TEMPORAL PROCESSING: Apply temporal transformer
        temporal_mask = (weights_sum.squeeze(-1).squeeze(-1) > 1e-8)  # [B, T] - True for valid frames
        
        if temporal_mask.sum() == 0:
            # All frames invalid - return zero features
            temporal_out = torch.zeros_like(aggregated_features)
        else:
            # Apply temporal transformer
            temporal_padding_mask = (~temporal_mask)  # [B, T] - True for invalid frames
            
            try:
                temporal_out = self.temporal_transformer(
                    aggregated_features,
                    src_key_padding_mask=temporal_padding_mask
                )  # [B, T, hidden_dim]
            except Exception as e:
                temporal_out = aggregated_features
        
        # 4. FINAL COMBINATION: Convert back to track-first format and combine features
        # We need to expand temporal features back to per-track format for compatibility
        # Use the aggregated temporal features as a base and combine with track-specific info
        
        # Expand temporal features to all tracks: [B, T, hidden_dim] -> [B, N_tracks, hidden_dim]
        # Take mean over time for each track, weighted by visibility
        track_visibility_weights = visibility.float().sum(dim=2, keepdim=True)  # [B, N_tracks, 1]
        track_visibility_weights = torch.clamp(track_visibility_weights, min=1e-8)
        
        # CRITICAL FIX: Safe division to prevent NaN values
        track_weights_sum = track_visibility_weights.sum(dim=1, keepdim=True)  # [B, 1, 1]
        track_weights_sum_safe = torch.clamp(track_weights_sum, min=1e-8)  # Prevent division by zero
        track_visibility_weights = track_visibility_weights / track_weights_sum_safe
        
        # Fallback to uniform weights if NaN detected
        if torch.isnan(track_visibility_weights).any():
            track_visibility_weights = torch.ones_like(track_visibility_weights) / N_tracks
        
        # Aggregate temporal features per track using visibility
        temporal_per_track = []
        for track_idx in range(N_tracks):
            track_vis = visibility[:, track_idx, :].float().unsqueeze(-1)  # [B, T, 1]
            track_vis_sum = torch.clamp(track_vis.sum(dim=1, keepdim=True), min=1e-8)  # [B, 1, 1]
            track_vis_norm = track_vis / track_vis_sum  # [B, T, 1]
            
            track_temporal = (temporal_out * track_vis_norm).sum(dim=1)  # [B, hidden_dim]
            temporal_per_track.append(track_temporal)
        
        temporal_features = torch.stack(temporal_per_track, dim=1)  # [B, N_tracks, hidden_dim]
        
        # Get spatial features by taking mean over time for each track
        spatial_per_track = []
        for track_idx in range(N_tracks):
            track_vis = visibility[:, track_idx, :].float().unsqueeze(-1)  # [B, T, 1]
            track_vis_sum = torch.clamp(track_vis.sum(dim=1, keepdim=True), min=1e-8)  # [B, 1, 1]
            track_vis_norm = track_vis / track_vis_sum  # [B, T, 1]
            
            # Get spatial features for this track across time
            track_spatial_features = spatial_features[:, :, track_idx, :]  # [B, T, hidden_dim]
            track_spatial = (track_spatial_features * track_vis_norm).sum(dim=1)  # [B, hidden_dim]
            spatial_per_track.append(track_spatial)
        
        spatial_features_final = torch.stack(spatial_per_track, dim=1)  # [B, N_tracks, hidden_dim]
        
        # Confidence embedding (per track)
        conf_emb = self.confidence_embedding(confidence)  # [B, N_tracks, hidden_dim]
        
        # Final feature combination
        combined_features = torch.cat([
            spatial_features_final,  # Spatial-temporal processed features
            temporal_features,       # Pure temporal features  
            conf_emb                 # Confidence features
        ], dim=-1)  # [B, N_tracks, 3*hidden_dim]
        
        # Project to CLIP-aligned space first
        motion_clip_features = self.motion_clip_projector(combined_features)  # [B, N_tracks, clip_dim]
        
        # Normalize for cosine similarity with CLIP text embeddings
        # Add epsilon for numerical stability and handle NaN
        motion_clip_features = F.normalize(motion_clip_features + 1e-8, p=2, dim=-1)
        
        # Handle NaN after normalization
        if torch.isnan(motion_clip_features).any():
            motion_clip_features = torch.zeros_like(motion_clip_features)
        
        # Project to motion descriptor space for backward compatibility
        motion_desc = self.motion_projection(motion_clip_features)  # [B, N_tracks, motion_dim]
        
        # Store CLIP-aligned features for potential use
        if not hasattr(self, '_last_motion_clip_features'):
            self._last_motion_clip_features = motion_clip_features
        else:
            self._last_motion_clip_features = motion_clip_features
        
        return motion_desc
    
    def sample_divergence_at_tracks(self, div_maps, tracks):
        """
        Sample divergence values at track locations
        """
        B, N_tracks, T, _ = tracks.shape
        device = tracks.device
        dtype = tracks.dtype
        
        if div_maps.shape[1] == 0:  # No temporal frames
            return torch.zeros(B, N_tracks, 1, device=device, dtype=dtype)
        
        # Use the most recent divergence map
        div_map = div_maps[:, -1]  # [B, H_div, W_div]
        _, H_div, W_div = div_map.shape
        
        # Get most recent track positions
        track_pos = tracks[:, :, -1]  # [B, N_tracks, 2]
        
        # Normalize track coordinates to divergence map space
        track_pos_norm = track_pos.clone()
        track_pos_norm[:, :, 0] = track_pos_norm[:, :, 0] / 640.0 * W_div  # x
        track_pos_norm[:, :, 1] = track_pos_norm[:, :, 1] / 480.0 * H_div  # y
        
        # Clamp to valid range
        track_pos_norm[:, :, 0] = torch.clamp(track_pos_norm[:, :, 0], 0, W_div - 1)
        track_pos_norm[:, :, 1] = torch.clamp(track_pos_norm[:, :, 1], 0, H_div - 1)
        
        # Bilinear sampling
        div_values = []
        for b in range(B):
            # Convert to grid_sample format: [B, C, H, W] and [B, H_out, W_out, 2]
            div_grid = div_map[b:b+1].unsqueeze(0)  # [1, 1, H_div, W_div]
            
            # Normalize coordinates to [-1, 1] for grid_sample
            sample_coords = track_pos_norm[b].unsqueeze(0)  # [1, N_tracks, 2]
            sample_coords[:, :, 0] = 2.0 * sample_coords[:, :, 0] / (W_div - 1) - 1.0  # x
            sample_coords[:, :, 1] = 2.0 * sample_coords[:, :, 1] / (H_div - 1) - 1.0  # y
            
            # Add spatial dimensions for grid_sample
            sample_coords = sample_coords.unsqueeze(2)  # [1, N_tracks, 1, 2]
            
            # Sample divergence values
            sampled = F.grid_sample(div_grid, sample_coords, mode='bilinear', 
                                  padding_mode='border', align_corners=True)
            sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # [N_tracks, 1]
            div_values.append(sampled)
        
        div_at_tracks = torch.stack(div_values, dim=0)  # [B, N_tracks, 1]
        return div_at_tracks
    
    def compute_curl_from_tracks(self, tracks, velocity):
        """
        Compute curl (rotation) from velocity field
        """
        B, N_tracks, _ = velocity.shape
        device = tracks.device
        dtype = tracks.dtype
        
        # Simple curl approximation based on local velocity differences
        # For a proper curl, we'd need spatial derivatives, but this gives a rotation proxy
        
        # Compute velocity magnitude as rotation proxy
        vel_mag = torch.norm(velocity, dim=-1, keepdim=True)  # [B, N_tracks, 1]
        
        # Normalize to reasonable range
        curl = torch.tanh(vel_mag)  # [B, N_tracks, 1]
        
        return curl
    
    def interpolate_to_feature_grid(self, motion_features, target_N):
        """
        Interpolate motion features to match feature grid resolution
        """
        B, N_tracks, D = motion_features.shape
        
        # If sizes already match, no interpolation needed
        if N_tracks == target_N:
            return motion_features
        
        # If target size is larger than motion, use simple linear interpolation
        if target_N > N_tracks:
            # Use linear interpolation along the spatial dimension
            indices = torch.linspace(0, N_tracks - 1, target_N, device=motion_features.device)
            indices_floor = torch.floor(indices).long()
            indices_ceil = torch.ceil(indices).long()
            
            # Clamp indices to valid range
            indices_floor = torch.clamp(indices_floor, 0, N_tracks - 1)
            indices_ceil = torch.clamp(indices_ceil, 0, N_tracks - 1)
            
            # Compute weights for interpolation
            weights = indices - indices_floor.float()
            weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, target_N, 1]
            
            # Interpolate
            motion_floor = motion_features.gather(1, indices_floor.unsqueeze(0).unsqueeze(-1).expand(B, -1, D))
            motion_ceil = motion_features.gather(1, indices_ceil.unsqueeze(0).unsqueeze(-1).expand(B, -1, D))
            
            motion_interp = motion_floor * (1 - weights) + motion_ceil * weights
            return motion_interp
        
        else:
            # If target size is smaller, use average pooling
            # Simple downsampling by selecting evenly spaced points
            indices = torch.linspace(0, N_tracks - 1, target_N, device=motion_features.device).long()
            return motion_features.gather(1, indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, D))
    
    def apply_motion_modulation(self, scores, motion_descriptors, N):
        """
        Apply motion-based modulation to attention scores
        """
        B, H, _, _ = scores.shape
        
        # Ensure motion descriptors match spatial resolution
        if motion_descriptors.shape[1] != N:
            motion_descriptors = self.interpolate_to_feature_grid(motion_descriptors, N)
        
        # Extract velocity components for motion coherence
        vel_x = motion_descriptors[:, :, 0]  # [B, N]
        vel_y = motion_descriptors[:, :, 1]  # [B, N]
        
        # Compute motion similarity matrix
        vel_mag = torch.sqrt(vel_x**2 + vel_y**2)  # [B, N]
        
        # Pairwise velocity magnitude differences
        vel_diff = torch.abs(vel_mag.unsqueeze(-1) - vel_mag.unsqueeze(-2))  # [B, N, N]
        
        # Convert to similarity (Gaussian kernel)
        motion_similarity = torch.exp(-vel_diff**2 / (2 * 0.1**2))  # [B, N, N]
        
        # Apply to all heads
        motion_similarity = motion_similarity.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, N, N]
        
        # Modulate attention scores
        scores = scores * motion_similarity
        
        return scores

    def apply_trajectory_motion_modulation(self, scores, motion_descriptors, trajectory_embeddings, N):
        """
        Enhanced modulation using both motion and trajectory information
        """
        B, H, _, _ = scores.shape
        
        # Ensure descriptors match spatial resolution
        if motion_descriptors.shape[1] != N:
            motion_descriptors = self.interpolate_to_feature_grid(motion_descriptors, N)
            trajectory_embeddings = self.interpolate_to_feature_grid(trajectory_embeddings, N)
        
        # Motion similarity (as before)
        vel_x = motion_descriptors[:, :, 0]  # [B, N]
        vel_y = motion_descriptors[:, :, 1]  # [B, N]
        vel_mag = torch.sqrt(vel_x**2 + vel_y**2)  # [B, N]
        vel_diff = torch.abs(vel_mag.unsqueeze(-1) - vel_mag.unsqueeze(-2))  # [B, N, N]
        motion_similarity = torch.exp(-vel_diff**2 / (2 * 0.1**2))  # [B, N, N]
        
        # NEW: Trajectory similarity using cosine similarity
        # Add epsilon for numerical stability and handle NaN
        traj_norm = F.normalize(trajectory_embeddings + 1e-8, p=2, dim=-1)  # [B, N, D]
        
        # Handle NaN in trajectory normalization
        if torch.isnan(traj_norm).any():
            traj_norm = torch.zeros_like(traj_norm)
        
        trajectory_similarity = torch.bmm(traj_norm, traj_norm.transpose(-2, -1))  # [B, N, N]
        
        # Combine motion and trajectory similarities
        # Motion captures "what kind of movement", trajectory captures "which specific object"
        combined_similarity = 0.3 * motion_similarity + 0.7 * trajectory_similarity
        
        # Apply to all heads
        combined_similarity = combined_similarity.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, N, N]
        
        # Modulate attention scores
        scores = scores * combined_similarity
        
        return scores

    def fallback_attention(self, x, return_attention=False):
        """
        Simple self-attention fallback when tracks are not available
        """
        B, N, D = x.shape
        
        # Normalize input
        x_norm = self.norm_input(x)
        
        if self.use_trajectory_tokens:
            # Create zero motion descriptors and trajectory embeddings
            zero_motion = torch.zeros(B, N, self.motion_dim, device=x.device, dtype=x.dtype)
            zero_trajectory = torch.zeros(B, N, self.d_model, device=x.device, dtype=x.dtype)
            combined_features = torch.cat([zero_motion, zero_trajectory], dim=-1)
            
            Q = self.motion_traj_to_q(combined_features)
            K = self.motion_traj_to_k(combined_features) 
            V = x_norm
        else:
            print("🚀 USING MOTION ONLY PATH!")
            # Use motion projections as Q, K, V (with zero motion features)
            zero_motion = torch.zeros(B, N, self.motion_dim, device=x.device, dtype=x.dtype)
            
            Q = self.motion_to_q(zero_motion)  # [B, N, d_model]
            K = self.motion_to_k(zero_motion)  # [B, N, d_model]
            V = x_norm  # Use actual features as values
        
        # Multi-head attention computation
        Q = Q.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = K.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        V = V.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        
        # Simple attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().reshape(B, N, D)
        out = self.out_proj(out)
        
        # Residual connection
        out = out + x
        
        if return_attention:
            return out, attn_weights
        return out
        
    def compute_motion_affinity_mask(self, motion_descriptors, top_k=50):
        """
        Create sparse attention mask based on motion similarity
        Only attend to top-k most motion-similar points
        """
        B, N, D = motion_descriptors.shape
        
        # Compute pairwise motion similarity using cosine similarity
        motion_norm = F.normalize(motion_descriptors, p=2, dim=-1)  # [B, N, D]
        similarity = torch.bmm(motion_norm, motion_norm.transpose(-2, -1))  # [B, N, N]
        
        # Create top-k mask
        _, top_indices = torch.topk(similarity, k=min(top_k, N), dim=-1)
        mask = torch.zeros_like(similarity)
        mask.scatter_(-1, top_indices, 1.0)
        
        return mask

    def apply_causal_motion_mask(self, attention_weights, tracks):
        """
        Mask attention to only allow causal relationships
        Points can only attend to where they came from
        """
        B, H, N, _ = attention_weights.shape
        
        # For now, return original attention weights
        # Can implement trajectory-based causality here
        return attention_weights

    def apply_text_guidance(self, motion_descriptors, text_embeddings):
        """
        DEPRECATED: This method is disabled because it produces uniform attention weights.
        
        ISSUE: All tracks query the same single text vector, causing identical attention scores.
        SOLUTION: Use TextTrackCrossAttention in TCAMVideoEncoder instead.
        
        Args:
            motion_descriptors: [B, N_tracks, motion_dim] motion features
            text_embeddings: [B, d_model] text embeddings
            
        Returns:
            guided_motion_descriptors: [B, N_tracks, motion_dim] text-guided motion features
        """
        # DISABLED: Return original motion descriptors unchanged
        # Text-track relationships are now handled by TextTrackCrossAttention
        return motion_descriptors


class TCAMTransformerBlock(nn.Module):
    """
    Transformer block using Motion Field Attention
    Compatible transformer block interface
    """
    def __init__(self, d_model, n_heads, temporal_window=5, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.attention = MotionFieldAttention(d_model, n_heads, temporal_window, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, divergence=None, depth_logits=None, tracks=None, visibility=None, mask=None, return_attention=False, text_embeddings=None):
        """
        Transformer block interface
        
        Args:
            x: [B, N, D] input features
            divergence: [B, H, W] divergence map (IGNORED)
            depth_logits: [B, L, H, W] depth stratification (IGNORED)
            tracks: [B, N_tracks, T, 2] point tracks from CoTracker
            visibility: [B, N_tracks, T] visibility masks
            mask: attention mask
            return_attention: whether to return attention weights
            text_embeddings: [B, d_model] text embeddings for guidance (NEW)
        """
        # Self-attention with residual connection
        if return_attention:
            attn_out, attn_weights = self.attention(
                self.norm1(x), divergence, depth_logits, tracks, visibility, mask, 
                return_attention=True, text_embeddings=text_embeddings
            )
            x = attn_out  # Residual already applied in attention
        else:
            x = self.attention(self.norm1(x), divergence, depth_logits, tracks, visibility, mask, text_embeddings=text_embeddings)
            attn_weights = None
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x


class TCAMEncoder(nn.Module):
    """
    Multi-layer encoder using Motion Field Attention blocks
    Compatible encoder interface
    """
    def __init__(self, num_layers, d_model, n_heads, temporal_window=5,
                 d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MotionFieldTransformerBlock(d_model, n_heads, temporal_window, 
                                      d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, divergence=None, depth_logits=None, tracks=None, visibility=None, mask=None, return_attention=False, text_embeddings=None):
        """
        Encoder interface
        
        Args:
            x: [B, N, D] input features
            divergence: [B, H, W] divergence map (IGNORED)
            depth_logits: [B, L, H, W] depth stratification (IGNORED)
            tracks: [B, N_tracks, T, 2] point tracks from CoTracker
            visibility: [B, N_tracks, T] visibility masks
            mask: attention mask
            return_attention: whether to return attention weights from all layers
            text_embeddings: [B, d_model] text embeddings for guidance (NEW)
        """
        attention_weights = [] if return_attention else None
        
        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, divergence, depth_logits, tracks, visibility, mask, return_attention=True, text_embeddings=text_embeddings)
                attention_weights.append(attn)
            else:
                x = layer(x, divergence, depth_logits, tracks, visibility, mask, return_attention=False, text_embeddings=text_embeddings)
        
        x = self.norm(x)
        
        if return_attention:
            return x, attention_weights
        return x


class TCAMHybridAttention(TCAMMotionAttention):
    """
    Hybrid attention that combines motion attention with content attention
    Fallback plan if pure motion attention doesn't train well
    """
    def __init__(self, d_model, n_heads, temporal_window=5, dropout=0.1, 
                 content_weight=0.3, motion_weight=0.7):
        super().__init__(d_model, n_heads, temporal_window, dropout)
        
        self.content_weight = content_weight
        self.motion_weight = motion_weight
        
        # Additional projections for content-based attention
        self.content_q = nn.Linear(d_model, d_model)
        self.content_k = nn.Linear(d_model, d_model)
        
    def forward(self, x, tracks, visibility, return_attention=False):
        """
        Forward pass combining motion and content attention
        """
        B, N, D = x.shape
        
        # Get motion-based attention
        motion_out, motion_attn = super().forward(x, tracks, visibility, return_attention=True)
        
        # Get content-based attention (simplified)
        x_norm = self.norm_input(x)
        Q_content = self.content_q(x_norm).reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K_content = self.content_k(x_norm).reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V_content = x_norm.reshape(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        content_scores = torch.matmul(Q_content, K_content.transpose(-2, -1)) * self.scale
        content_attn = F.softmax(content_scores, dim=-1)
        content_out = torch.matmul(content_attn, V_content)
        content_out = content_out.transpose(1, 2).contiguous().reshape(B, N, D)
        content_out = self.out_proj(content_out) + x
        
        # Combine motion and content attention
        combined_out = self.motion_weight * motion_out + self.content_weight * content_out
        combined_attn = self.motion_weight * motion_attn + self.content_weight * content_attn
        
        if return_attention:
            return combined_out, combined_attn
        return combined_out 
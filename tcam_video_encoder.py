#!/usr/bin/env python3
"""
TCAM Video Encoder
Core Components:
1. TCAMVideoEncoder - Combines grid tracking + motion attention + visual features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import cv2
from PIL import Image
from pycocotools import mask as coco_mask
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# CoTracker imports
import sys
sys.path.append('co-tracker')
from cotracker.predictor import CoTrackerPredictor

# Import our motion attention layer
from tcam_motion_attention import TCAMMotionAttention, TCAMTrajectoryEmbedding
from tcam_spatial_grounding import TCAMSpatialGrounding

# CoTracker imports - use official PyTorch Hub API
import torch.hub
from transformers import CLIPVisionModel



class SimpleVisualEncoder(nn.Module):
    """Simple CNN backbone for visual features"""
    def __init__(self, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # Input: [B, 3, H, W]
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv block 1
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Conv block 2
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Final projection
            nn.Conv2d(256, d_model, kernel_size=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling for video-level features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, frames):
        """
        Args:
            frames: [B, T, 3, H, W] video frames
        Returns:
            features: [B, T, d_model, H', W'] spatial features
            global_features: [B, T, d_model] global features per frame
        """
        B, T, C, H, W = frames.shape
        
        # Process each frame
        frames_flat = frames.reshape(B * T, C, H, W)
        spatial_features = self.backbone(frames_flat)  # [B*T, d_model, H', W']
        
        # Reshape back to video format
        _, d_model, H_feat, W_feat = spatial_features.shape
        spatial_features = spatial_features.reshape(B, T, d_model, H_feat, W_feat)
        
        # Global pooling for frame-level features
        global_features = self.global_pool(spatial_features.reshape(B * T, d_model, H_feat, W_feat))
        global_features = global_features.reshape(B, T, d_model)
        
        return spatial_features, global_features


class TextTrackCrossAttention(nn.Module):
    """
    Cross-attention module that allows text to guide track attention.
    Text queries attend to track motion descriptors to identify relevant motion patterns.
    
    Supports both single-event (backward compatible) and multi-event discovery modes.
    """
    def __init__(self, d_model=128, motion_dim=6, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.motion_dim = motion_dim
        self.n_heads = n_heads
        
        # Project motion descriptors to d_model space
        self.motion_projector = nn.Linear(motion_dim, d_model)
        
        # Cross-attention: Text queries attend to track motion patterns
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and feedforward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Small feedforward network for text enhancement
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Single-event relevance computation (backward compatibility)
        self.relevance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Multi-event components - NEW
        # Learnable confidence thresholds per head (for automatic event detection)
        self.confidence_thresholds = nn.Parameter(torch.ones(n_heads) * 0.5)
        
        # Per-head relevance computation for multi-event mode
        self.per_head_relevance = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(), 
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
            ) for _ in range(n_heads)
        ])
        
        # Manual multi-head attention for per-head access
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        
        # Multi-event query, key, value projections (separate from main cross_attention)
        self.multi_q_proj = nn.Linear(d_model, d_model)
        self.multi_k_proj = nn.Linear(d_model, d_model)
        self.multi_v_proj = nn.Linear(d_model, d_model)
        self.multi_out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, text_features, motion_descriptors, return_all_events=True):
        """
        Args:
            text_features: [B, d_model] text embeddings
            motion_descriptors: [B, N_tracks, motion_dim] motion features per track
            return_all_events: bool, if True returns per-head multi-event output
            
        Returns:
            If return_all_events=False (default, backward compatible):
                relevance_scores: [B, N_tracks] relevance score per track
                attention_weights: [B, N_tracks] cross-attention weights
                
            If return_all_events=True (multi-event mode):
                dict containing:
                    'relevance_scores': [B, n_heads, N_tracks] per-head relevance
                    'attention_weights': [B, n_heads, N_tracks] per-head attention  
                    'confidence_scores': [B, n_heads] confidence per head/event
                    'active_events': [B, n_heads] boolean mask of active events
                    'aggregated_relevance': [B, N_tracks] backward-compatible aggregated scores
                    'aggregated_attention': [B, N_tracks] backward-compatible aggregated attention
        """
        B, N_tracks, _ = motion_descriptors.shape
        
        # Project motion descriptors to same dimension as text
        track_features = self.motion_projector(motion_descriptors)  # [B, N_tracks, d_model]
        
        # Prepare text as query (add sequence dimension)
        text_query = text_features.unsqueeze(1)  # [B, 1, d_model]
        
        if not return_all_events:
            # SIMPLIFIED APPROACH: Direct cosine similarity (more reliable than complex cross-attention)
            
            # Normalize features for cosine similarity
            text_norm = F.normalize(text_query, dim=-1)  # [B, 1, d_model]
            tracks_norm = F.normalize(track_features, dim=-1)  # [B, N_tracks, d_model]
            
            # Direct cosine similarity between text and each track
            similarity_scores = torch.matmul(text_norm, tracks_norm.transpose(-1, -2))  # [B, 1, N_tracks]
            track_relevance = similarity_scores.squeeze(1)  # [B, N_tracks]
            
            # Scale to reasonable range and add learnable bias
            track_relevance = track_relevance * 2.0  # Expand range from [-1,1] to [-2,2]
            
            # Create attention weights for backward compatibility (just softmax of relevance)
            attention_weights = F.softmax(track_relevance, dim=-1)  # [B, N_tracks]
            
            return track_relevance, attention_weights
        
        else:
            # MULTI-EVENT PATH - new functionality
            # Manual multi-head attention to access per-head information
            
            # Project queries, keys, values
            q = self.multi_q_proj(text_query)  # [B, 1, d_model]
            k = self.multi_k_proj(track_features)  # [B, N_tracks, d_model] 
            v = self.multi_v_proj(track_features)  # [B, N_tracks, d_model]
            
            # Reshape for multi-head processing
            q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, 1, head_dim]
            k = k.view(B, N_tracks, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N_tracks, head_dim]
            v = v.view(B, N_tracks, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, N_tracks, head_dim]
            
            # Compute per-head attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, n_heads, 1, N_tracks]
            per_head_attention = F.softmax(scores, dim=-1)  # [B, n_heads, 1, N_tracks]
            
            # Apply attention to values
            attended_per_head = torch.matmul(per_head_attention, v)  # [B, n_heads, 1, head_dim]
            
            # Reshape and project output
            attended_concat = attended_per_head.transpose(1, 2).contiguous().view(B, 1, self.d_model)  # [B, 1, d_model]
            attended_features = self.multi_out_proj(attended_concat)  # [B, 1, d_model]
            
            # Text enhancement (shared path)
            text_enhanced = self.norm1(text_query + attended_features)
            text_final = self.norm2(text_enhanced + self.ffn(text_enhanced))
            
            # Per-head relevance computation
            per_head_relevance = []
            per_head_confidence = []
            
            for head_idx in range(self.n_heads):
                # Get attended features for this head
                head_attended = attended_per_head[:, head_idx, :, :]  # [B, 1, head_dim]
                
                # Project to full dimension for relevance computation
                head_features = torch.cat([
                    text_final.squeeze(1),  # [B, d_model] - enhanced text
                    head_attended.squeeze(1).repeat(1, self.d_model // self.head_dim)[:, :self.d_model - self.head_dim],  # Pad to d_model
                    head_attended.squeeze(1)  # [B, head_dim]
                ], dim=1)[:, :self.d_model]  # [B, d_model]
                
                # Compute head-specific relevance
                head_rel_single = self.per_head_relevance[head_idx](head_features.unsqueeze(1)).squeeze(-1).squeeze(-1)  # [B]
                
                # Head-specific cosine similarity  
                head_text_norm = F.normalize(head_features.unsqueeze(1), dim=-1)  # [B, 1, d_model]
                tracks_norm = F.normalize(track_features, dim=-1)  # [B, N_tracks, d_model]
                head_similarity = torch.matmul(head_text_norm, tracks_norm.transpose(-1, -2)).squeeze(1)  # [B, N_tracks]
                
                # Combine approaches for this head
                head_rel_expanded = head_rel_single.unsqueeze(1).expand(-1, N_tracks)  # [B, N_tracks]
                head_final_relevance = 0.7 * head_similarity + 0.3 * head_rel_expanded  # [B, N_tracks]
                
                per_head_relevance.append(head_final_relevance)
                
                # FIXED: Compute confidence score for this head with sigmoid to ensure positive values
                # Use mean of top-k relevance scores instead of just max for stability
                top_k_relevance, _ = torch.topk(head_final_relevance, k=min(10, N_tracks), dim=1)
                head_confidence_raw = top_k_relevance.mean(dim=1)  # [B] - mean of top-k
                head_confidence = torch.sigmoid(head_confidence_raw)  # [B] - ensure 0-1 range
                per_head_confidence.append(head_confidence)
            
            # Stack per-head results
            all_head_relevance = torch.stack(per_head_relevance, dim=1)  # [B, n_heads, N_tracks]
            all_head_confidence = torch.stack(per_head_confidence, dim=1)  # [B, n_heads]
            all_head_attention = per_head_attention.squeeze(2)  # [B, n_heads, N_tracks]
            
            # Determine active events based on learnable thresholds
            active_events = all_head_confidence > self.confidence_thresholds.unsqueeze(0)  # [B, n_heads]
            
            # Compute backward-compatible aggregated outputs
            # Weight by confidence for aggregation
            confidence_weights = F.softmax(all_head_confidence, dim=1)  # [B, n_heads]
            
            # Aggregate raw logits (no softmax) for BCE loss compatibility
            # BCE expects raw logits and will apply sigmoid internally
            # Weighted sum across heads based on confidence
            aggregated_relevance_raw = torch.sum(all_head_relevance * confidence_weights.unsqueeze(2), dim=1)  # [B, N_tracks]
            
            # CRITICAL: Scale up the logits to expand the range for effective BCE learning
            # Cosine similarity is naturally in [-1, 1], but clusters near 0 when untrained
            # Scale by 4.0 to achieve wider range (e.g., [-4, 4]) → sigmoid produces varied outputs
            aggregated_relevance = aggregated_relevance_raw * 4.0  # [B, N_tracks] - scaled logits for BCE
            
            aggregated_attention = torch.sum(all_head_attention * confidence_weights.unsqueeze(2), dim=1)  # [B, N_tracks]
            
            return {
                'relevance_scores': all_head_relevance,        # [B, n_heads, N_tracks]
                'attention_weights': all_head_attention,       # [B, n_heads, N_tracks]  
                'confidence_scores': all_head_confidence,      # [B, n_heads]
                'active_events': active_events,                # [B, n_heads]
                'aggregated_relevance': aggregated_relevance,  # [B, N_tracks] - backward compatible
                'aggregated_attention': aggregated_attention   # [B, N_tracks] - backward compatible
            }


class TCAMVideoEncoder(nn.Module):
    """
    TCAM Video Encoder combining:
    1. CoTracker for dense grid tracking
    2. Motion Attention for motion understanding
    3. Visual CNN for appearance features
    4. Multi-modal fusion for final representation
    """
    
    def __init__(self, 
                 d_model=256,
                 n_heads=8,
                 temporal_window=8,
                 grid_size=20,
                 dropout=0.1,
                 use_cross_attention=True):
        super().__init__()
        
        self.d_model = d_model
        self.temporal_window = temporal_window
        self.grid_size = grid_size
        self.use_cross_attention = use_cross_attention
        
        # 1. CoTracker for motion tracking
        self.cotracker = None  # Will be loaded separately
        
        # 2. Motion Attention (with trajectory tokens for object distinction)
        self.motion_attention = TCAMMotionAttention(
            d_model=d_model,
            n_heads=n_heads,
            temporal_window=temporal_window,
            dropout=dropout,
            use_trajectory_tokens=True  # Enable trajectory tokens for object distinction
        )
        
        # 3. CLIP Visual encoder (pre-trained on 400M image-text pairs)
        self.clip_vision_model = CLIPVisionModel.from_pretrained(
            'openai/clip-vit-base-patch32', 
            use_safetensors=True
        )
        self.visual_projector_clip = nn.Linear(768, d_model)  # CLIP-ViT outputs 768-dim
        
        # Freeze CLIP vision model for transfer learning
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False
        print("🔒 CLIP Vision model frozen for transfer learning")
        
        # 4. Spatial Grounding for subject localization
        if use_cross_attention:
            self.spatial_grounding = TCAMSpatialGrounding(
                d_model=d_model,
                motion_dim=6,  # vx, vy, ax, ay, div, curl
                dropout=dropout
            )
            print("✨ Using TCAMSpatialGrounding for track relevance")
        else:
            # Fallback: simple projection for backward compatibility
            self.motion_projector_simple = nn.Linear(6, d_model)  # motion_dim=6 to d_model
        
        # 5. Feature fusion
        self.motion_projector = nn.Linear(d_model, d_model)
        self.visual_projector = nn.Linear(d_model, d_model)
        
        # 6. Multi-modal fusion
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads//2,
            dropout=dropout,
            batch_first=True
        )
        
        # 7. Video representation
        self.temporal_pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # 8. SIMPLE temporal aggregation (research-proven)
        # Based on CLIP-Hitchhiker's Guide: simple query-scoring beats complex methods
        self.temporal_temperature = nn.Parameter(torch.tensor(0.1))  # Learnable temperature
        self.frame_importance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )  # Simple learned frame weighting (when no text available)
        
        # 9. Final video embedding
        self.video_projector = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def load_cotracker(self):
        """Load CoTracker model using official API"""
        print("🔄 Loading CoTracker (official API)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Use official CoTracker3 (as in their demo)
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.cotracker = self.cotracker.to(device)
        print(f"✅ CoTracker loaded on {device}")
        
    def extract_grid_tracks(self, video_tensor, mask=None):
        """
        Extract grid tracks from video using official CoTracker API
        
        Args:
            video_tensor: [B, T, 3, H, W] video frames (normalized 0-1)
            mask: [B, H, W] optional mask for object-focused tracking
            
        Returns:
            tracks: [B, N_tracks, T, 2] point tracks
            visibility: [B, N_tracks, T] visibility masks
        """
        if self.cotracker is None:
            raise ValueError("CoTracker not loaded. Call load_cotracker() first.")
        
        B, T, C, H, W = video_tensor.shape
        device = video_tensor.device
        
        # Process each video in the batch individually (CoTracker limitation)
        all_tracks = []
        all_visibility = []
        
        for b in range(B):
            # Get single video: [1, T, C, H, W]
            single_video = video_tensor[b:b+1]
            
            # Use standard CoTracker grid tracking with proper coordinate handling
            with torch.no_grad():
                pred_tracks, pred_visibility = self.cotracker(
                    single_video,
                    grid_size=self.grid_size,
                    grid_query_frame=0,  # Start from first frame for consistency
                    backward_tracking=False
                )
                
            
            # CRITICAL: CoTracker returns pixel coordinates by default (not normalized)
            # Clamp to valid bounds and handle negative coordinates
            pred_tracks[..., 0] = torch.clamp(pred_tracks[..., 0], 0, W - 1)
            pred_tracks[..., 1] = torch.clamp(pred_tracks[..., 1], 0, H - 1)
            
            # Mark tracks outside bounds as invisible (safety check)
            out_of_bounds = ((pred_tracks[..., 0] < 0) | (pred_tracks[..., 0] >= W) |
                           (pred_tracks[..., 1] < 0) | (pred_tracks[..., 1] >= H))
            pred_visibility = pred_visibility & (~out_of_bounds)
            
            # CoTracker returns [1, T, N, 2] and [1, T, N]
            # We need [1, N, T, 2] and [1, N, T]
            tracks = pred_tracks.permute(0, 2, 1, 3)  # [1, N, T, 2]
            visibility = pred_visibility.permute(0, 2, 1)  # [1, N, T]
            
            all_tracks.append(tracks)
            all_visibility.append(visibility)
        
        # Concatenate batch results
        tracks = torch.cat(all_tracks, dim=0)  # [B, N, T, 2]
        visibility = torch.cat(all_visibility, dim=0)  # [B, N, T]
        
        return tracks, visibility
    
    def get_track_motion_descriptors(self, video_frames, text_features=None):
        """
        Extract motion descriptors for tracks (used by test scripts)
        
        Args:
            video_frames: [B, T, 3, H, W] video frames
            text_features: [B, d_model] optional text features for cross-attention
            
        Returns:
            tracks: [B, N_tracks, T, 2] point tracks
            relevance_scores: [B, N_tracks] relevance per track
        """
        with torch.no_grad():
            # Extract tracks
            tracks, visibility = self.extract_grid_tracks(video_frames)
            
            if text_features is not None and self.use_cross_attention:
                # Get motion descriptors and apply simplified spatial grounding
                motion_descriptors = self.motion_attention.compute_motion_descriptors(tracks, visibility)
                relevance_scores = self.spatial_grounding(text_features, motion_descriptors)
                return tracks, relevance_scores
            else:
                # Fallback: compute motion descriptors and simple projection
                motion_descriptors = self.motion_attention.compute_motion_descriptors(tracks, visibility)
                if hasattr(self, 'motion_projector_simple'):
                    projected = self.motion_projector_simple(motion_descriptors)
                    # Simple cosine similarity with text
                    if text_features is not None:
                        text_norm = F.normalize(text_features.unsqueeze(1), dim=-1)
                        proj_norm = F.normalize(projected, dim=-1)
                        relevance_scores = torch.matmul(text_norm, proj_norm.transpose(-1, -2)).squeeze(1)
                    else:
                        relevance_scores = torch.zeros(tracks.shape[0], tracks.shape[1], device=tracks.device)
                else:
                    relevance_scores = torch.zeros(tracks.shape[0], tracks.shape[1], device=tracks.device)
                
                return tracks, relevance_scores
    
    def forward(self, video_frames, object_mask=None, return_intermediate=False, text_embeddings=None):
        """
        Forward pass for motion-aware video encoding with optional text-guided track attention
        
        Args:
            video_frames: [B, T, 3, H, W] video frames (0-1 normalized)
            object_mask: [B, H, W] optional object mask for focused tracking
            return_intermediate: bool, whether to return intermediate features
            text_embeddings: [B, d_model] optional text embeddings for text-guided attention
            
        Returns:
            If text_embeddings provided:
                video_embedding: [B, d_model] final video representation
                track_relevance: [B, N_tracks] relevance scores per track
                intermediate_dict: dict with intermediate features if requested
            Else:
                video_embedding: [B, d_model] final video representation
                intermediate_dict: dict with intermediate features if requested
        """
        B, T, C, H, W = video_frames.shape
        device = video_frames.device
        
        intermediate = {} if return_intermediate else None
        
        # 1. Extract motion tracks
        tracks, visibility = self.extract_grid_tracks(video_frames, object_mask)
        # tracks: [B, N_tracks, T, 2], visibility: [B, N_tracks, T]
        
        if return_intermediate:
            intermediate['tracks'] = tracks
            intermediate['visibility'] = visibility
        
        # 2. Extract CLIP visual features (pre-trained on 400M image-text pairs)
        B, T, C, H, W = video_frames.shape
        frames_flat = video_frames.reshape(B * T, C, H, W)  # Flatten temporal dimension
        
        # Resize frames to CLIP's expected input size (224x224)
        frames_resized = F.interpolate(frames_flat, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract CLIP features for all frames
        with torch.no_grad():  # CLIP frozen, no gradients needed
            clip_outputs = self.clip_vision_model(pixel_values=frames_resized)
            # last_hidden_state: [B*T, 1+N_patches, 768] (CLS + patch tokens)
            # pooler_output: [B*T, 768] (global frame embedding)
            last_hidden = clip_outputs.last_hidden_state
            clip_features = clip_outputs.pooler_output
        
        # Global (CLS) features -> project to d_model and reshape by time
        global_features = self.visual_projector_clip(clip_features)  # [B*T, d_model]
        global_features = global_features.reshape(B, T, self.d_model)  # [B, T, d_model]
        
        # Use CLIP ViT patch tokens as real spatial features (exclude CLS token)
        patch_tokens = last_hidden[:, 1:, :]  # [B*T, N_patches, 768]
        N_patches = patch_tokens.shape[1]
        # Project patch tokens to d_model
        patch_tokens_proj = self.visual_projector_clip(patch_tokens.reshape(-1, patch_tokens.shape[-1]))  # [B*T*N_patches, d_model]
        patch_tokens_proj = patch_tokens_proj.reshape(B * T, N_patches, self.d_model)  # [B*T, N_patches, d_model]
        # Derive spatial grid (H', W') dynamically
        H_feat = W_feat = int(round(N_patches ** 0.5))
        if H_feat * W_feat != N_patches:
            # Fallback: compute from model's patch size and resized resolution
            try:
                patch_size = getattr(self.clip_vision_model.config.vision_config, 'patch_size', None)
                if patch_size is None:
                    raise AttributeError
                H_feat = frames_resized.shape[-2] // patch_size
                W_feat = frames_resized.shape[-1] // patch_size
            except Exception:
                raise ValueError(f"Cannot derive grid from N_patches={N_patches}; and model patch_size unavailable.")
            if H_feat * W_feat != N_patches:
                raise ValueError(f"Derived grid {(H_feat, W_feat)} does not match N_patches={N_patches}")
        # Reshape to spatial grid
        patch_bt = patch_tokens_proj.reshape(B, T, N_patches, self.d_model)  # [B, T, N_patches, d_model]
        patch_grid = patch_bt.reshape(B, T, H_feat, W_feat, self.d_model)  # [B, T, H', W', d_model]
        spatial_features = patch_grid.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, d_model, H', W']
        # spatial_features: [B, T, d_model, H', W']
        # global_features: [B, T, d_model]
        if return_intermediate:
            intermediate['visual_features'] = global_features
        
        # 3. Process motion with MFA - FULL TEMPORAL SEQUENCE
        # Reshape spatial features to spatiotemporal sequence
        B, T, d_model, H_feat, W_feat = spatial_features.shape
        # Convert to spatiotemporal tokens: [B, T*H'*W', d_model]
        spatial_features_reshaped = spatial_features.permute(0, 1, 3, 4, 2)  # [B, T, H', W', d_model]
        spatiotemporal_tokens = spatial_features_reshaped.reshape(B, T * H_feat * W_feat, d_model)
        
        # Apply Motion Field Attention on FULL temporal sequence
        try:
            motion_attended = self.motion_attention(
                spatiotemporal_tokens,  # Full spatiotemporal sequence
                tracks=tracks,          # Full temporal tracks
                visibility=visibility,  # Full temporal visibility
                text_embeddings=text_embeddings  # NEW: Text guidance for attention
                # No current_time - process entire sequence!
            )  # [B, T*H'*W', d_model]
        except Exception as e:
            # Fallback - just use spatiotemporal tokens as-is
            motion_attended = spatiotemporal_tokens
        
        # Reshape back to temporal format and pool spatial dimensions
        motion_attended_reshaped = motion_attended.reshape(B, T, H_feat * W_feat, d_model)
        motion_features = motion_attended_reshaped.mean(dim=2)  # [B, T, d_model] - pool spatial
        
        # NEW: Text-guided track attention for subject localization
        track_relevance = None
        cross_attention_weights = None
        
        if text_embeddings is not None and self.use_cross_attention:
            # ✨ SIMPLIFIED: Direct spatial grounding via cosine similarity
            try:
                # Get motion descriptors from MFA
                motion_descriptors = self.motion_attention.compute_motion_descriptors(tracks, visibility)
                
                # Apply simplified spatial grounding (works for both single and multi-expression)
                track_relevance = self.spatial_grounding(
                    text_embeddings,      # [B, d_model] - text features (projected+normalized)
                    motion_descriptors    # [B, N_tracks, 6] - motion features
                )  # Returns: [B, N_tracks] - relevance logits
                
                # Placeholder for attention weights (for backward compatibility)
                cross_attention_weights = torch.softmax(track_relevance, dim=-1)
                
                if return_intermediate:
                    intermediate['motion_descriptors'] = motion_descriptors
                    intermediate['track_relevance'] = track_relevance
                    intermediate['cross_attention_weights'] = cross_attention_weights
                    
            except Exception as e:
                # Continue without cross-attention
                pass
        
        if return_intermediate:
            intermediate['motion_features'] = motion_features
        
        # 4. Project features
        motion_proj = self.motion_projector(motion_features)  # [B, T, d_model]
        visual_proj = self.visual_projector(global_features)  # [B, T, d_model]
        
        # 5. Multi-modal fusion
        # Concatenate motion and visual features
        combined_features = motion_proj + visual_proj  # Element-wise combination
        
        # Apply cross-attention between motion and visual
        fused_features, _ = self.fusion_attention(
            combined_features, combined_features, combined_features
        )  # [B, T, d_model]
        
        # 6. SIMPLE temporal aggregation (research-proven approach)
        # Based on CLIP-Hitchhiker's Guide: query-scoring outperforms complex methods
        B, T, d_model = fused_features.shape
        
        if text_embeddings is not None:
            # Method 1: Query-scoring (SOTA approach - 0 parameters!)
            # Compute frame-text similarity for each frame
            text_norm = F.normalize(text_embeddings, dim=-1)  # [B, d_model]
            frames_norm = F.normalize(fused_features, dim=-1)  # [B, T, d_model]
            
            # Frame-text similarity scores
            frame_scores = torch.matmul(frames_norm, text_norm.unsqueeze(-1)).squeeze(-1)  # [B, T]
            
            # Apply temperature scaling (learnable)
            weighted_scores = F.softmax(frame_scores / self.temporal_temperature, dim=1)  # [B, T]
            
            # Weighted average of frames (preserves motion patterns!)
            temporal_processed = torch.sum(
                fused_features * weighted_scores.unsqueeze(-1), dim=1
            )  # [B, d_model]
            
        else:
            # Method 2: Learned frame importance (when no text available)
            frame_importance = self.frame_importance_head(fused_features).squeeze(-1)  # [B, T]
            importance_weights = F.softmax(frame_importance / self.temporal_temperature, dim=1)  # [B, T]
            
            # Weighted average based on learned importance
            temporal_processed = torch.sum(
                fused_features * importance_weights.unsqueeze(-1), dim=1
            )  # [B, d_model]
        
        # Add residual connection with max pooling (captures peak motion events)
        max_pooled, _ = fused_features.max(dim=1)  # [B, d_model]
        temporal_processed = 0.8 * temporal_processed + 0.2 * max_pooled
        
        # 7. Final video embedding
        video_embedding = self.video_projector(temporal_processed)
        video_embedding = self.norm(video_embedding)
        
        if return_intermediate:
            intermediate['features'] = video_embedding  # For compatibility
            intermediate['fused_features'] = fused_features
            intermediate['video_embedding'] = video_embedding
            intermediate['tracks'] = tracks
            intermediate['visibility'] = visibility
            intermediate['track_relevance'] = track_relevance
            intermediate['cross_attention_weights'] = cross_attention_weights
            return intermediate
        
        # Return based on whether text guidance was used
        if track_relevance is not None:
            # Store tracks for BCE loss computation
            self._last_tracks = tracks
            return video_embedding, track_relevance
        else:
            return video_embedding
    
    def get_last_tracks(self):
        """Get the last computed tracks for BCE loss computation"""
        return getattr(self, '_last_tracks', None)
    


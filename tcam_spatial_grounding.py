"""
TCAM Spatial Grounding Module

Clean, interpretable approach:
1. Project motion descriptors to same space as text
2. Compute cosine similarity (text similarity already works for retrieval!)
3. Apply learnable scaling/bias to get track relevance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCAMSpatialGrounding(nn.Module):
    """
    Simple spatial grounding: direct cosine similarity + learnable scaling
    
    Philosophy: If text-video retrieval works well, text-track matching should be easy!
    We already know the model can match text to video features effectively.
    So we just need to match text to motion features the same way.
    """
    
    def __init__(self, d_model=256, motion_dim=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.motion_dim = motion_dim
        
        # Project motion descriptors to same space as text (d_model)
        self.motion_projector = nn.Sequential(
            nn.Linear(motion_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Learnable scaling and bias for relevance scores
        # This allows the model to adjust the sensitivity of matching
        self.relevance_scale = nn.Parameter(torch.tensor(4.0))  # Initialize to 4.0 (our empirical scaling)
        self.relevance_bias = nn.Parameter(torch.tensor(0.0))   # Start with no bias
        
        # Optional: Small MLP for non-linear refinement
        self.relevance_refine = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, text_features, motion_descriptors, return_dict=False):
        """
        Args:
            text_features: [B, d_model] - already projected and normalized text embeddings
            motion_descriptors: [B, N_tracks, motion_dim] - raw motion features from MFA
            return_dict: if True, return dict with intermediate values for debugging
            
        Returns:
            track_relevance: [B, N_tracks] - relevance scores (logits for BCE/ranking loss)
        """
        B, N_tracks, _ = motion_descriptors.shape
        
        # 1. Project motion descriptors to text space
        motion_features = self.motion_projector(motion_descriptors)  # [B, N_tracks, d_model]
        
        # 2. Normalize for cosine similarity (same as text features)
        motion_features_norm = F.normalize(motion_features, dim=-1)  # [B, N_tracks, d_model]
        text_features_norm = F.normalize(text_features, dim=-1)     # [B, d_model]
        
        # 3. Compute cosine similarity (same approach as retrieval!)
        # This is the core: if text matches video globally, it should match relevant motion locally
        text_expanded = text_features_norm.unsqueeze(1)  # [B, 1, d_model]
        cosine_sim = torch.sum(text_expanded * motion_features_norm, dim=-1)  # [B, N_tracks]
        
        # 4. Apply learnable scaling and bias
        # Scaling: amplifies differences for better discrimination
        # Bias: shifts the distribution to handle class imbalance
        scaled_sim = cosine_sim * self.relevance_scale + self.relevance_bias  # [B, N_tracks]
        
        # 5. Optional refinement (small non-linear adjustment)
        # This can help learn more complex decision boundaries
        refined = self.relevance_refine(scaled_sim.unsqueeze(-1)).squeeze(-1)  # [B, N_tracks]
        
        # Final relevance scores (logits)
        track_relevance = refined
        
        if return_dict:
            return {
                'track_relevance': track_relevance,      # [B, N_tracks] - final scores
                'cosine_similarity': cosine_sim,         # [B, N_tracks] - raw similarity
                'scaled_similarity': scaled_sim,         # [B, N_tracks] - after scaling
                'motion_features': motion_features_norm  # [B, N_tracks, d_model] - for analysis
            }
        
        return track_relevance




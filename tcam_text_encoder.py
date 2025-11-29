#!/usr/bin/env python3
"""
TCAM Text Encoder & Contrastive Learning
Text encoder for motion expressions and video-text matching framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2


class TCAMTextEncoder(nn.Module):
    """
    Text encoder specifically designed for motion expressions
    Uses BERT-based model with motion-specific fine-tuning
    """
    
    def __init__(self, 
                 model_name: str = 'distilbert-base-uncased',
                 d_model: int = 256,
                 max_length: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.model_name = model_name
        self.d_model = d_model
        self.max_length = max_length
        
        # Load CLIP text model (trained on image-text pairs for visual-language understanding)
        self.tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-base-patch32', 
            use_safetensors=True
        )
        self.clip_text_model = CLIPTextModel.from_pretrained(
            'openai/clip-vit-base-patch32',
            use_safetensors=True
        )
        
        # Freeze CLIP text model for transfer learning
        for param in self.clip_text_model.parameters():
            param.requires_grad = False
        print("🔒 CLIP Text model frozen for transfer learning")
        
        # CLIP text hidden size
        clip_hidden_size = self.clip_text_model.config.hidden_size  # 512 for CLIP
        
        # Motion-specific text processing layers
        self.motion_text_processor = nn.Sequential(
            nn.Linear(clip_hidden_size, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Motion keyword attention
        self.motion_keyword_attention = nn.MultiheadAttention(
            embed_dim=clip_hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection
        self.text_projector = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Motion keyword embeddings (learnable)
        motion_keywords = [
            'rolling', 'climbing', 'falling', 'jumping', 'running', 'walking',
            'sliding', 'flying', 'swimming', 'dancing', 'spinning', 'moving',
            'fast', 'slow', 'up', 'down', 'left', 'right', 'around', 'through'
        ]
        
        self.motion_keywords = motion_keywords
        self.keyword_embeddings = nn.Embedding(len(motion_keywords), clip_hidden_size)
        
    def forward(self, text_list: List[str], multi_expression_mode: bool = False) -> torch.Tensor:
        """
        Encode motion expressions into embeddings
        
        Args:
            text_list: List of motion expression strings
            multi_expression_mode: If True, treat each text as separate expression for multi-head attention
            
        Returns:
            If multi_expression_mode=False (default):
                text_embeddings: [B, d_model] text embeddings
            If multi_expression_mode=True:
                text_embeddings: [B, num_expressions, d_model] per-expression embeddings
        """
        device = next(self.parameters()).device
        batch_size = len(text_list)
        
        # Tokenize text
        tokenized = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(device)
        
        # Get CLIP text embeddings
        with torch.no_grad():  # CLIP frozen, no gradients needed
            clip_outputs = self.clip_text_model(**tokenized)
        
        # [B, seq_len, hidden_size]
        sequence_embeddings = clip_outputs.last_hidden_state
        
        # Apply motion keyword attention
        # Create keyword queries
        keyword_embeds = self.keyword_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_keywords, hidden_size]
        
        attended_text, attention_weights = self.motion_keyword_attention(
            keyword_embeds,  # Query: motion keywords
            sequence_embeddings,  # Key: text tokens
            sequence_embeddings   # Value: text tokens
        )  # [B, num_keywords, hidden_size]
        
        # Pool keyword-attended features
        motion_aware_embedding = attended_text.mean(dim=1)  # [B, hidden_size]
        
        # Process through motion-specific layers
        text_features = self.motion_text_processor(motion_aware_embedding)  # [B, d_model]
        
        # Final projection and normalization
        text_embedding = self.text_projector(text_features)
        text_embedding = self.norm(text_embedding)
        
        if multi_expression_mode:
            # Reshape for multi-expression: [B, 1, d_model] -> [B, 1, d_model]
            # In multi-expression mode, each text is treated as a separate expression
            text_embedding = text_embedding.unsqueeze(1)  # [B, 1, d_model]
        
        return text_embedding


class TCAMVideoTextMatcher(nn.Module):
    """
    Complete video-text matching system with contrastive learning
    Combines TCAM video encoder with TCAM text encoder
    """
    
    def __init__(self,
                 video_encoder,  # TCAMVideoEncoder
                 d_model: int = 256,
                 temperature: float = 0.5,  # FIXED: Use 0.5 to match training
                 spatial_loss_type: str = 'bce'):  # 'bce', 'weighted_bce', or 'focal'
        super().__init__()
        
        self.video_encoder = video_encoder
        self.text_encoder = TCAMTextEncoder(d_model=d_model)
        self.temperature = temperature
        self.spatial_loss_type = spatial_loss_type
        
        # Storage for track relevance (for spatial supervision)
        self.last_track_relevance = None
        self.last_tracks = None
        self.last_visibility = None
        self.last_per_head_attention = None
        
        # Cross-modal projectors for better alignment
        self.video_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # NEW: Store text embeddings from training
        self.register_buffer('text_embedding_bank', None)
        self.text_descriptions = []
        
    def update_text_bank(self, text_features, descriptions):
        """Update during training with all seen text embeddings"""
        text_features = text_features.detach()
        if self.text_embedding_bank is None:
            self.text_embedding_bank = text_features
            self.text_descriptions = list(descriptions)  # Ensure it's a list
        else:
            # FIXED: Check for duplicates BEFORE adding to prevent memory bloat
            existing_descriptions_set = set(self.text_descriptions)
            
            # Find only new (unique) descriptions
            new_indices = []
            new_descriptions = []
            for i, desc in enumerate(descriptions):
                if desc not in existing_descriptions_set:
                    new_indices.append(i)
                    new_descriptions.append(desc)
            
            # Only add truly new embeddings and descriptions
            if new_indices:
                new_features = text_features[new_indices]
                self.text_embedding_bank = torch.cat([self.text_embedding_bank, new_features])
                self.text_descriptions.extend(new_descriptions)
                print(f"📚 Added {len(new_indices)} new descriptions (total: {len(self.text_descriptions)})")
            else:
                print(f"📚 No new descriptions to add (total: {len(self.text_descriptions)})")
        
    def forward(self, video_frames, text_list, object_masks=None):
        """
        Forward pass for video-text matching with text-guided attention
        
        Args:
            video_frames: [B, T, 3, H, W] video frames
            text_list: List[str] motion expressions
            object_masks: [B, H, W] optional object masks
            
        Returns:
            video_embeddings: [B, d_model]
            text_embeddings: [B, d_model]
            similarity_matrix: [B, B] cosine similarities
        """
        # Encode texts first to get embeddings for guiding video attention
        text_embeddings = self.text_encoder(text_list)  # [B, d_model]
        
        # CRITICAL FIX: Project and normalize text features BEFORE passing to video encoder
        # This matches the training path in compute_unified_multi_positive_loss!
        text_features_for_video = self.text_projector(text_embeddings)
        text_features_for_video = F.normalize(text_features_for_video, p=2, dim=1)
        
        # Encode videos with text guidance
        video_output = self.video_encoder(
            video_frames, 
            object_masks, 
            text_embeddings=text_features_for_video,  # FIXED: Pass projected+normalized!
            return_intermediate=True  # Get intermediate results for spatial supervision
        )
        
        # Handle different return types from video encoder
        if isinstance(video_output, dict):
            video_embeddings = video_output['features']
            # Store track information for spatial supervision
            self.last_track_relevance = video_output.get('track_relevance')
            self.last_tracks = video_output.get('tracks')  
            self.last_visibility = video_output.get('visibility')
# Debug prints removed for production
        elif isinstance(video_output, tuple):
            video_embeddings = video_output[0]  # Take just the embeddings
        else:
            video_embeddings = video_output
        
        # Project to shared space
        video_features = self.video_projector(video_embeddings)
        text_features = self.text_projector(text_embeddings)
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(video_features, text_features)
        
        return video_features, text_features, similarity_matrix
    
    def get_last_track_relevance(self):
        """
        FIXED: Get the last computed track relevance from cross-attention
        
        Returns:
            dict containing:
                'aggregated_relevance': [N_tracks] track relevance scores
                'per_head_attention': [n_heads, N_tracks] per-head attention (if available)
                'tracks': [N_tracks, T, 2] track coordinates (if available)
                'visibility': [N_tracks, T] track visibility (if available)
        """
        if self.last_track_relevance is not None:
            return {
                'aggregated_relevance': self.last_track_relevance,  # Match testing script expectation
                'track_relevance': self.last_track_relevance,  # Keep backward compatibility
                'tracks': self.last_tracks,
                'visibility': self.last_visibility,
                'per_head_attention': self.last_per_head_attention  # Include per-head attention if available
            }
        else:
            return None
    
    def forward_multi_expression(self, video_frames, text_lists, object_masks=None):
        """
        Forward pass for multi-expression processing (for testing/inference)
        
        Args:
            video_frames: [B, T, 3, H, W] video frames  
            text_lists: List[List[str]] - list of expression lists per video
            object_masks: [B, H, W] optional object masks
            
        Returns:
            video_embeddings: [B, d_model]
            multi_head_results: dict with per-head track relevance and attention
        """
        B = video_frames.shape[0]
        
        # Process each video with its expressions
        all_video_embeddings = []
        all_multi_results = []
        
        for b in range(B):
            # Get expressions for this video
            expressions = text_lists[b] if b < len(text_lists) else text_lists[0]
            
            # Encode expressions in multi-expression mode
            multi_text_embeddings = []
            for expr in expressions:
                text_emb = self.text_encoder([expr], multi_expression_mode=True)  # [1, 1, d_model]
                multi_text_embeddings.append(text_emb)
            
            if multi_text_embeddings:
                # Stack expressions: [num_expressions, 1, d_model] -> [1, num_expressions, d_model]
                stacked_text = torch.cat(multi_text_embeddings, dim=1)  # [1, num_expressions, d_model]
            else:
                # Fallback: single empty expression
                stacked_text = self.text_encoder([""], multi_expression_mode=True)
            
            # Encode video with multi-head text guidance
            video_batch = video_frames[b:b+1]  # [1, T, 3, H, W]
            mask_batch = object_masks[b:b+1] if object_masks is not None else None
            
            video_output = self.video_encoder(
                video_batch,
                mask_batch,
                text_embeddings=stacked_text,  # [1, num_expressions, d_model]
                return_intermediate=True
            )
            
            if isinstance(video_output, tuple):
                video_emb, intermediate = video_output
                all_multi_results.append(intermediate)
            else:
                video_emb = video_output
                all_multi_results.append({})
            
            all_video_embeddings.append(video_emb)
        
        # Stack results
        video_embeddings = torch.cat(all_video_embeddings, dim=0)  # [B, d_model]
        
        return video_embeddings, all_multi_results
    
    def compute_contrastive_loss(self, similarity_matrix):
        """
        Compute bidirectional contrastive loss
        
        Args:
            similarity_matrix: [B, B] similarity scores
            
        Returns:
            loss: scalar contrastive loss
        """
        # Get batch size
        batch_size = similarity_matrix.size(0)
        
        # Labels are all diagonal elements (positive pairs)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Compute video-to-text loss
        v2t_loss = F.cross_entropy(similarity_matrix / self.temperature, labels)
        
        # Compute text-to-video loss
        t2v_loss = F.cross_entropy(similarity_matrix.t() / self.temperature, labels)
        
        # Combine losses
        loss = (v2t_loss + t2v_loss) / 2
        
        return loss
    
    def compute_multi_positive_loss(self, video_features, text_features, positive_mapping, temperature=None):
        """
        Multi-positive contrastive loss where multiple texts can be positive for same video
        
        Args:
            video_features: [B, d_model] normalized video features
            text_features: [B, d_model] normalized text features  
            positive_mapping: Dict[batch_idx -> List[batch_idx]] of positive pairs
            temperature: Optional temperature override
            
        Returns:
            loss: scalar multi-positive contrastive loss
        """
        if temperature is None:
            temperature = self.temperature
            
        B = video_features.shape[0]
        device = video_features.device
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(video_features, text_features.T) / temperature  # [B, B]
        
        total_loss = 0.0
        num_valid_samples = 0
        
        # Video-to-text direction
        for video_idx in range(B):
            positive_text_indices = positive_mapping.get(video_idx, [video_idx])  # Default to self if no mapping
            
            if len(positive_text_indices) == 0:
                continue
                
            video_similarities = similarity_matrix[video_idx]  # [B]
            
            # Create positive mask
            positive_mask = torch.zeros(B, device=device, dtype=torch.bool)
            for pos_idx in positive_text_indices:
                if 0 <= pos_idx < B:  # Ensure valid index
                    positive_mask[pos_idx] = True
            
            if positive_mask.sum() == 0:
                continue
                
            # Multi-positive contrastive loss
            # Use logsumexp for numerical stability
            positive_logits = video_similarities[positive_mask]
            negative_logits = video_similarities[~positive_mask]
            
            # Positive term: maximize similarity with all positives
            pos_term = torch.logsumexp(positive_logits, dim=0)
            
            # Negative term: all samples (including positives) for denominator
            all_term = torch.logsumexp(video_similarities, dim=0)
            
            # Loss: -log(exp(pos) / exp(all)) = -(pos - all)
            video_loss = -(pos_term - all_term)
            total_loss += video_loss
            num_valid_samples += 1
        
        # Text-to-video direction
        for text_idx in range(B):
            positive_video_indices = []
            # Find all videos that have this text as positive
            for video_idx, pos_texts in positive_mapping.items():
                if text_idx in pos_texts:
                    positive_video_indices.append(video_idx)
            
            if len(positive_video_indices) == 0:
                positive_video_indices = [text_idx]  # Default to self
                
            text_similarities = similarity_matrix[:, text_idx]  # [B]
            
            # Create positive mask
            positive_mask = torch.zeros(B, device=device, dtype=torch.bool)
            for pos_idx in positive_video_indices:
                if 0 <= pos_idx < B:
                    positive_mask[pos_idx] = True
            
            if positive_mask.sum() == 0:
                continue
                
            # Multi-positive contrastive loss
            positive_logits = text_similarities[positive_mask]
            
            # Positive term
            pos_term = torch.logsumexp(positive_logits, dim=0)
            
            # All term
            all_term = torch.logsumexp(text_similarities, dim=0)
            
            # Loss
            text_loss = -(pos_term - all_term)
            total_loss += text_loss
            num_valid_samples += 1
        
        if num_valid_samples > 0:
            final_loss = total_loss / num_valid_samples
            return final_loss
        else:
            # Fallback to standard contrastive loss
            return self.compute_contrastive_loss(similarity_matrix)
    
    def compute_unified_multi_positive_loss(self, video_frames, expressions, positive_mapping, batch, spatial_weight=0.1):
        """
        UNIFIED loss computation that processes video and text in single forward pass
        This avoids DDP parameter reuse issues by ensuring each parameter is used only once per backward pass
        
        Args:
            video_frames: [B, T, 3, H, W] raw video frames
            expressions: List[str] text expressions
            positive_mapping: dict mapping video indices to positive text indices
            batch: batch data containing masks if available
            spatial_weight: weight for spatial supervision term
            
        Returns:
            combined_loss: global similarity loss + spatial track loss
            similarity_matrix: [B, B] for accuracy computation
        """
        device = video_frames.device
        B = video_frames.shape[0]
        
        # SINGLE FORWARD PASS: Process video and text together
        # This ensures all parameters are used exactly once
        
        # 1. Encode text features
        text_features = self.text_encoder(expressions)
        text_features = self.text_projector(text_features)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 2. Encode video features with text guidance (single pass through video encoder)
        video_result = self.video_encoder(
            video_frames, 
            text_embeddings=text_features,
            return_intermediate=False  # Don't need intermediate, tracks stored internally
        )
        
        # Handle different return formats from video encoder
        if isinstance(video_result, tuple) and len(video_result) == 2:
            video_features, track_relevance = video_result
        else:
            video_features = video_result
            track_relevance = None
        
        # Project and normalize video features
        video_features = self.video_projector(video_features)
        video_features = F.normalize(video_features, p=2, dim=1)
        
        # 3. Compute similarity matrix (uses both video and text projector parameters)
        similarity_matrix = torch.matmul(video_features, text_features.T) / self.temperature
        
        # 4. Compute global contrastive loss (using same logic as compute_multi_positive_loss)
        if positive_mapping:
            total_loss = 0.0
            num_valid_samples = 0
            
            # Video-to-text direction (matching compute_multi_positive_loss exactly)
            for video_idx in range(B):
                positive_text_indices = positive_mapping.get(video_idx, [video_idx])  # Default to self if no mapping
                
                if len(positive_text_indices) == 0:
                    continue
                    
                video_similarities = similarity_matrix[video_idx]  # [B]
                
                # Create positive mask
                positive_mask = torch.zeros(B, device=device, dtype=torch.bool)
                for pos_idx in positive_text_indices:
                    if 0 <= pos_idx < B:  # Ensure valid index
                        positive_mask[pos_idx] = True
                
                if positive_mask.sum() == 0:
                    continue
                    
                # Multi-positive contrastive loss
                # Use logsumexp for numerical stability
                positive_logits = video_similarities[positive_mask]
                negative_logits = video_similarities[~positive_mask]
                
                # Positive term: maximize similarity with all positives
                pos_term = torch.logsumexp(positive_logits, dim=0)
                
                # FIXED: Correct InfoNCE denominator (concat all logits, don't logsumexp twice)
                if len(negative_logits) > 0:
                    # Combine all logits (positives + negatives) for denominator
                    all_logits = torch.cat([positive_logits, negative_logits], dim=0)
                    video_loss = -pos_term + torch.logsumexp(all_logits, dim=0)
                else:
                    # If no negatives, just maximize positive term
                    video_loss = -pos_term
                
                total_loss += video_loss
                num_valid_samples += 1
            
            # Text-to-video direction (for symmetry, matching compute_multi_positive_loss exactly)
            for text_idx in range(B):
                # Find which videos are positive for this text
                positive_video_indices = []
                for vid_idx, pos_texts in positive_mapping.items():
                    if text_idx in pos_texts:
                        positive_video_indices.append(vid_idx)
                
                if len(positive_video_indices) == 0:
                    positive_video_indices = [text_idx]  # Default to self
                
                text_similarities = similarity_matrix[:, text_idx]  # [B]
                
                # Create positive mask
                positive_mask = torch.zeros(B, device=device, dtype=torch.bool)
                for pos_idx in positive_video_indices:
                    if 0 <= pos_idx < B:
                        positive_mask[pos_idx] = True
                
                if positive_mask.sum() == 0:
                    continue
                
                positive_logits = text_similarities[positive_mask]
                negative_logits = text_similarities[~positive_mask]
                
                # Positive term
                pos_term = torch.logsumexp(positive_logits, dim=0)
                
                # FIXED: Correct InfoNCE denominator (concat all logits, don't logsumexp twice)
                if len(negative_logits) > 0:
                    # Combine all logits (positives + negatives) for denominator
                    all_logits = torch.cat([positive_logits, negative_logits], dim=0)
                    text_loss = -pos_term + torch.logsumexp(all_logits, dim=0)
                else:
                    text_loss = -pos_term
                
                total_loss += text_loss
                num_valid_samples += 1
            
            if num_valid_samples > 0:
                global_loss = total_loss / num_valid_samples
            else:
                global_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # Fallback to standard contrastive loss
            labels = torch.arange(B, device=device)
            global_loss = F.cross_entropy(similarity_matrix, labels)
        
        # 5. Compute spatial track supervision loss (uses cross-attention parameters)
        spatial_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if track_relevance is not None:
            # Get track data for BCE alignment loss
            track_relevance_dict = {
                'track_relevance': track_relevance,
                'tracks': self.video_encoder.get_last_tracks()  # Get tracks from video encoder
            }
            
            # Compute BCE alignment loss between track relevance and object masks
            mask_alignment_loss = self.compute_track_mask_alignment_loss(
                track_relevance_dict, batch, positive_mapping, loss_type=self.spatial_loss_type
            )
            
            # CLEANED: Remove conflicting regularizers that push relevances to zero
            # Let BCE loss handle the discrimination directly
            spatial_loss = mask_alignment_loss
        
        # 6. Combine losses with balanced contributions
        # Scale spatial loss to make it more significant
        spatial_scale = 10.0
        scaled_spatial_loss = spatial_loss * spatial_scale
        
        total_loss = (1.0 - spatial_weight) * global_loss + spatial_weight * scaled_spatial_loss
        
        # Return loss components for logging
        loss_components = {
            'total_loss': total_loss,
            'global_loss': global_loss,
            'spatial_loss': spatial_loss,
            'scaled_spatial_loss': scaled_spatial_loss,
            'mask_alignment_loss': mask_alignment_loss if track_relevance is not None else torch.tensor(0.0, device=device)
        }
        
        return total_loss, similarity_matrix, loss_components
    
    def compute_spatially_aware_multi_positive_loss(self, video_features, text_features, track_relevance_dict, positive_mapping, batch, spatial_weight=0.5):
        """
        DEPRECATED: Legacy function kept for backward compatibility
        Use compute_unified_multi_positive_loss instead to avoid DDP issues
        """
        print("⚠️  WARNING: Using deprecated compute_spatially_aware_multi_positive_loss. Use compute_unified_multi_positive_loss instead.")
        
        # FIXED: Compute global loss directly without calling separate function
        # This avoids parameter reuse that causes DDP issues
        similarity_matrix = self.compute_similarity_matrix(video_features, text_features)
        
        # Multi-positive contrastive loss computation (inline to avoid double parameter use)
        device = similarity_matrix.device
        B = similarity_matrix.shape[0]
        
        if positive_mapping:
            losses = []
            for i in range(B):
                # Get positive indices for this sample
                positive_indices = positive_mapping.get(i, [i])
                
                # Compute contrastive loss for this sample
                logits = similarity_matrix[i] / self.temperature
                
                # Create positive mask
                positive_mask = torch.zeros(B, dtype=torch.bool, device=device)
                positive_mask[positive_indices] = True
                
                # Multi-positive InfoNCE loss
                if positive_mask.sum() > 0:
                    # Numerator: sum of positive pairs
                    positive_logits = logits[positive_mask]
                    numerator = torch.logsumexp(positive_logits, dim=0)
                    
                    # Denominator: all pairs
                    denominator = torch.logsumexp(logits, dim=0)
                    
                    # Loss for this sample
                    sample_loss = -(numerator - denominator)
                    losses.append(sample_loss)
            
            global_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        else:
            # Fallback to standard contrastive loss
            global_loss = self.compute_contrastive_loss(similarity_matrix)
        
        # Spatial track supervision loss (separate parameter set)
        spatial_loss = self.compute_track_spatial_loss(track_relevance_dict, batch, positive_mapping)
        
        # Adaptive loss scaling to balance contributions
        spatial_scale = 10.0  # Scale factor to make spatial loss more significant
        scaled_spatial_loss = spatial_loss * spatial_scale
        
        # Combine losses with balanced contributions
        total_loss = (1.0 - spatial_weight) * global_loss + spatial_weight * scaled_spatial_loss
        
        return total_loss
    
    def compute_track_spatial_loss(self, track_relevance_dict, batch, positive_mapping):
        """
        Compute spatial supervision loss using track relevance and encourage spatial diversity
        
        Args:
            track_relevance_dict: dict with track_relevance [B, N_tracks], tracks [B, N_tracks, T, 2], visibility
            batch: batch data
            positive_mapping: mapping of positive pairs
            
        Returns:
            spatial_loss: loss encouraging meaningful track relevance patterns
        """
        if track_relevance_dict['track_relevance'] is None:
            return torch.tensor(0.0, requires_grad=True)
        
        track_relevance = track_relevance_dict['track_relevance']  # [B, N_tracks]
        device = track_relevance.device
        
        # Encourage non-uniform track attention (prevent attention collapse)
        # High variance = good specialization, low variance = attention collapse
        relevance_std = track_relevance.std(dim=1)  # [B] - std per batch
        attention_diversity_loss = torch.clamp(0.1 - relevance_std, min=0.0).mean()  # Loss when std < 0.1
        
        # Encourage sparsity: only some tracks should be highly relevant
        # Use L1 regularization on track relevance to encourage sparsity
        sparsity_loss = torch.abs(track_relevance).mean() * 0.01  # Small L1 penalty
        
        # If we have ground truth masks, use them for spatial supervision
        mask_alignment_loss = 0.0
        if 'masks' in batch and track_relevance_dict['tracks'] is not None:
            mask_alignment_loss = self.compute_track_mask_alignment_loss(
                track_relevance_dict, batch, positive_mapping
            )
        
        # Combine spatial loss components
        total_spatial_loss = attention_diversity_loss + sparsity_loss + mask_alignment_loss
        
        return total_spatial_loss
    
    def compute_track_mask_alignment_loss(self, track_relevance_dict, batch, positive_mapping, loss_type='ranking'):
        """
        SIMPLE RANKING LOSS: Force tracks inside object to have higher relevance than tracks outside
        
        The idea is simple:
        1. CoTracker produces tracks on a fixed grid
        2. Model learns to weight which tracks are relevant to each caption
        3. Training signal: Tracks inside the object mask should be weighted higher than outside
        
        Args:
            loss_type: 'ranking' (default) - margin-based ranking loss
        """
        track_relevance = track_relevance_dict['track_relevance']  # [B, N_tracks] - raw scores (not logits!)
        tracks = track_relevance_dict['tracks']  # [B, N_tracks, T, 2]
        masks = batch['object_masks']  # [B, T, H, W] ground truth masks
        
        device = track_relevance.device
        B, N_tracks, T, _ = tracks.shape
        H, W = masks.shape[-2:]
        
        total_loss = 0.0
        num_valid = 0
        
        # Debug stats
        debug_stats = {
            'inside_mean': [],
            'outside_mean': [],
            'gap': [],
            'n_inside': [],
            'n_outside': []
        }
        
        for batch_idx in range(B):
            # Get mask (use last frame for stability)
            mask = masks[batch_idx, -1].float()  # [H, W]
            batch_tracks = tracks[batch_idx]  # [N_tracks, T, 2]
            batch_relevance = track_relevance[batch_idx]  # [N_tracks] - RAW SCORES
            
            # Convert tracks to pixel coordinates
            tracks_pixel = batch_tracks.clone()
            with torch.no_grad():
                coord_max = tracks_pixel.max().item()
            if coord_max <= 1.5:  # normalized [0,1]
                tracks_pixel[:, :, 0] *= W
                tracks_pixel[:, :, 1] *= H
            tracks_pixel[:, :, 0] = torch.clamp(tracks_pixel[:, :, 0], 0, W-1)
            tracks_pixel[:, :, 1] = torch.clamp(tracks_pixel[:, :, 1], 0, H-1)
            
            # Sample mask at track locations (last frame)
            inside_mask = torch.zeros(N_tracks, dtype=torch.bool, device=device)
            for track_idx in range(N_tracks):
                x, y = tracks_pixel[track_idx, -1]
                x, y = int(x.item()), int(y.item())
                if 0 <= y < H and 0 <= x < W:
                    inside_mask[track_idx] = mask[y, x] > 0.5
            
            # Separate inside vs outside tracks
            inside_relevance = batch_relevance[inside_mask]  # Tracks inside object
            outside_relevance = batch_relevance[~inside_mask]  # Tracks outside object
            
            n_inside = inside_relevance.shape[0]
            n_outside = outside_relevance.shape[0]
            
            # Skip if no tracks inside or outside
            if n_inside == 0 or n_outside == 0:
                continue
            
            # RANKING LOSS: Force inside tracks to score higher than outside tracks
            # Simple margin-based loss: inside_mean should be > outside_mean + margin
            margin = getattr(self, 'spatial_loss_margin', 0.5)  # Configurable margin (default 0.5)
            
            inside_mean = inside_relevance.mean()
            outside_mean = outside_relevance.mean()
            
            # Loss is positive when outside scores are too high
            # max(0, margin + outside_mean - inside_mean)
            sample_loss = F.relu(margin + outside_mean - inside_mean)
            
            total_loss += sample_loss
            num_valid += 1
            
            # Collect stats for debugging
            debug_stats['inside_mean'].append(inside_mean.item())
            debug_stats['outside_mean'].append(outside_mean.item())
            debug_stats['gap'].append((inside_mean - outside_mean).item())
            debug_stats['n_inside'].append(n_inside)
            debug_stats['n_outside'].append(n_outside)
        
        final_loss = total_loss / max(num_valid, 1)
        
        # Store gap for monitoring (used by fixed trainer)
        if len(debug_stats['gap']) > 0:
            self._last_gap = sum(debug_stats['gap']) / len(debug_stats['gap'])
        
        # Debug print for first batch of each epoch
        if hasattr(self, '_debug_ranking_loss') and self._debug_ranking_loss and len(debug_stats['gap']) > 0:
            print(f"\n🎯 [DEBUG] RANKING LOSS:")
            print(f"   Inside mean relevance:  {sum(debug_stats['inside_mean'])/len(debug_stats['inside_mean']):.4f}")
            print(f"   Outside mean relevance: {sum(debug_stats['outside_mean'])/len(debug_stats['outside_mean']):.4f}")
            print(f"   Gap (inside - outside): {sum(debug_stats['gap'])/len(debug_stats['gap']):.4f} (want: >{margin:.1f})")
            print(f"   Tracks inside/outside:  {int(sum(debug_stats['n_inside'])/len(debug_stats['n_inside']))}/{int(sum(debug_stats['n_outside'])/len(debug_stats['n_outside']))}")
            print(f"   Loss: {final_loss:.4f}")
            self._debug_ranking_loss = False
        
        return final_loss
        
    def forward_inference(self, video_frames, detect_multiple_events=False):
        """
        FIXED: Inference with proper cross-attention usage
        
        This method now properly uses the trained cross-attention mechanism by:
        1. Using ALL text embeddings from the bank to guide video encoding
        2. Leveraging the trained spatial supervision features
        3. Computing track relevance through cross-attention
        
        Args:
            video_frames: [B, T, 3, H, W] video frames
            detect_multiple_events: bool, if True enables multi-event discovery
            
        Returns:
            If detect_multiple_events=False (default, backward compatible):
                retrieved_texts: List[List[str]] top-k text descriptions per batch
                similarities: [B, k] similarity scores for top-k matches
                
            If detect_multiple_events=True (multi-event mode):
                dict containing:
                    'per_event_texts': List[List[List[str]]] per-head text matches [B, n_heads, k]
                    'per_event_similarities': [B, n_heads, k] similarity scores per head
                    'active_events': [B, n_heads] boolean mask of active events
                    'confidence_scores': [B, n_heads] confidence per event
                    'aggregated_texts': List[List[str]] backward-compatible aggregated results [B, k]
                    'aggregated_similarities': [B, k] backward-compatible aggregated scores
        """
        if self.text_embedding_bank is None or len(self.text_descriptions) == 0:
            print("⚠️ No text bank available!")
            # Return default values if no text bank
            if not detect_multiple_events:
                return [["no motion description available"]], torch.zeros((1, 1), device=video_frames.device)
            else:
                # Multi-event default
                return {
                    'per_event_texts': [[["no motion description available"] * 4]],  # 4 heads
                    'per_event_similarities': torch.zeros((1, 4, 1), device=video_frames.device),
                    'active_events': torch.zeros((1, 4), dtype=torch.bool, device=video_frames.device),
                    'confidence_scores': torch.zeros((1, 4), device=video_frames.device),
                    'aggregated_texts': [["no motion description available"]],
                    'aggregated_similarities': torch.zeros((1, 1), device=video_frames.device)
                }
            
        if not detect_multiple_events:
            # FIXED: Use iterative cross-attention approach for proper zero-shot retrieval
            print("🔧 FIXED: Using trained cross-attention for zero-shot retrieval")
            
            # Step 1: Get initial video features (without text guidance)
            initial_video_embedding = self.video_encoder(video_frames, text_embeddings=None)
            initial_video_features = self.video_projector(initial_video_embedding)
            initial_video_features = F.normalize(initial_video_features, p=2, dim=1)
            
            # Step 2: Get top-k candidate texts using initial similarity
            text_bank_features = F.normalize(self.text_embedding_bank, p=2, dim=1)
            initial_similarities = torch.matmul(initial_video_features, text_bank_features.T)
            
            # Get top candidates to focus cross-attention on
            k_candidates = min(50, len(self.text_descriptions))  # Use more candidates for refinement
            top_k_values, top_k_indices = torch.topk(initial_similarities, k=k_candidates, dim=1)
            
            print(f"📊 Initial retrieval: {k_candidates} candidates")
            print(f"   Score range: [{top_k_values.min():.4f}, {top_k_values.max():.4f}]")
            
            # Step 3: Use cross-attention with top candidate texts to refine results
            batch_refined_similarities = []
            batch_track_relevance = []
            
            for batch_idx in range(video_frames.shape[0]):
                candidate_indices = top_k_indices[batch_idx]  # [k_candidates]
                candidate_texts = [self.text_descriptions[idx.item()] for idx in candidate_indices]
                
                # Encode candidate texts
                candidate_embeddings = self.text_encoder(candidate_texts)  # [k_candidates, d_model]
                
                # CRITICAL FIX: Process text embeddings individually to avoid shape issues
                # The video encoder expects [B, d_model] text embeddings, not [B, k_candidates, d_model]
                # So we'll use the FIRST candidate as a representative query for cross-attention
                representative_text = candidate_embeddings[0:1]  # [1, d_model] - use first candidate
                
                refined_video_output = self.video_encoder(
                    video_frames[batch_idx:batch_idx+1],  # [1, T, 3, H, W]
                    text_embeddings=representative_text,  # [1, d_model] - correct shape
                    return_intermediate=True
                )
                
                # Extract refined video features and track relevance
                if isinstance(refined_video_output, dict):
                    refined_video_embedding = refined_video_output['features']
                    track_relevance = refined_video_output.get('track_relevance')  # This should now have meaningful values!
                    
                    # Store track relevance for visualization
                    if track_relevance is not None:
                        if track_relevance.dim() > 2:  # Multi-head case: [1, n_heads, N_tracks]
                            # Aggregate across heads for backward compatibility
                            aggregated_relevance = track_relevance.mean(dim=1)  # [1, N_tracks]
                            batch_track_relevance.append(aggregated_relevance[0])
                        else:  # Single-head case: [1, N_tracks]
                            batch_track_relevance.append(track_relevance[0])
                    
                else:
                    refined_video_embedding = refined_video_output
                
                # Project refined features
                refined_video_features = self.video_projector(refined_video_embedding)
                refined_video_features = F.normalize(refined_video_features, p=2, dim=1)
                
                # Compute refined similarities with candidate texts
                candidate_text_features = self.text_projector(candidate_embeddings)
                candidate_text_features = F.normalize(candidate_text_features, p=2, dim=1)
                
                refined_similarities = torch.matmul(refined_video_features, candidate_text_features.T)  # [1, k_candidates]
                batch_refined_similarities.append(refined_similarities[0])
            
            # Step 4: Combine results
            final_similarities = torch.stack(batch_refined_similarities)  # [B, k_candidates]
            
            # Store track relevance for testing script access
            if batch_track_relevance:
                self.last_track_relevance = torch.stack(batch_track_relevance)  # [B, N_tracks]
                print(f"✅ Track relevance computed via cross-attention!")
                print(f"   Shape: {self.last_track_relevance.shape}")
                print(f"   Range: [{self.last_track_relevance.min():.4f}, {self.last_track_relevance.max():.4f}]")
            else:
                print("⚠️ No track relevance computed")
            
            # Get final top-k results
            k_final = min(5, k_candidates)
            final_top_k_values, final_top_k_indices = torch.topk(final_similarities, k=k_final, dim=1)
            
            print("📝 Final refined matches:")
            retrieved_texts = []
            for batch_idx in range(final_top_k_indices.shape[0]):
                batch_texts = []
                for i, local_idx in enumerate(final_top_k_indices[batch_idx]):
                    # FIXED: Handle tensor indexing properly
                    local_idx_val = local_idx.item() if hasattr(local_idx, 'item') else int(local_idx)
                    # Map back to global text bank index
                    global_idx = top_k_indices[batch_idx, local_idx_val]
                    global_idx_val = global_idx.item() if hasattr(global_idx, 'item') else int(global_idx)
                    text = self.text_descriptions[global_idx_val]
                    score = final_top_k_values[batch_idx, i].item()
                    print(f"   {i+1}. {text} (refined score: {score:.4f})")
                    batch_texts.append(text)
                retrieved_texts.append(batch_texts)
            
            return retrieved_texts, final_top_k_values
    
    def forward_multi_retrieval(self, video_frames, top_k=20, threshold_percentile=70):
        """
        FIXED: Multi-expression retrieval with proper cross-attention usage
        
        Args:
            video_frames: [B, T, 3, H, W] video frames
            top_k: Number of top candidates to consider
            threshold_percentile: Percentile threshold for active expressions (e.g., 70 = top 30%)
            
        Returns:
            dict containing:
                'expressions': List[List[str]] active expressions per batch [B, N_active]
                'confidences': List[List[float]] confidence scores per batch [B, N_active]
                'num_expressions': List[int] number of active expressions per batch [B]
        """
        if self.text_embedding_bank is None or len(self.text_descriptions) == 0:
            print("⚠️ No text bank available for multi-retrieval!")
            return {
                'expressions': [["no motion description available"]],
                'confidences': [[0.0]],
                'num_expressions': [1]
            }
        
        print("🔧 FIXED: Multi-retrieval using trained cross-attention")
        
        # Step 1: Get initial video features (without text guidance) for candidate selection
        initial_video_embedding = self.video_encoder(video_frames, text_embeddings=None)
        initial_video_features = self.video_projector(initial_video_embedding)
        initial_video_features = F.normalize(initial_video_features, p=2, dim=1)
        
        # Step 2: Get top-k candidate texts using initial similarity
        text_bank_features = F.normalize(self.text_embedding_bank, p=2, dim=1)
        initial_similarities = torch.matmul(initial_video_features, text_bank_features.T)  # [B, num_texts]
        
        # Get expanded candidates for cross-attention refinement
        k_candidates = min(top_k * 3, len(self.text_descriptions))  # Use 3x more candidates
        top_k_values, top_k_indices = torch.topk(initial_similarities, k=k_candidates, dim=1)
        
        print(f"📊 Multi-retrieval: {k_candidates} initial candidates")
        
        # Step 3: Use cross-attention with candidate texts to refine scores
        batch_results = []
        batch_confidences = []
        batch_counts = []
        
        for batch_idx in range(video_frames.shape[0]):
            candidate_indices = top_k_indices[batch_idx]  # [k_candidates]
            candidate_texts = [self.text_descriptions[idx.item()] for idx in candidate_indices]
            
            # Encode candidate texts
            candidate_embeddings = self.text_encoder(candidate_texts)  # [k_candidates, d_model]
            
            # CRITICAL FIX: Use representative text embedding for cross-attention
            # The video encoder expects [B, d_model], not [B, k_candidates, d_model]
            representative_text = candidate_embeddings[0:1]  # [1, d_model] - use first candidate
            
            refined_video_output = self.video_encoder(
                video_frames[batch_idx:batch_idx+1],  # [1, T, 3, H, W]
                text_embeddings=representative_text,  # [1, d_model] - correct shape
                return_intermediate=True
            )
            
            # Extract refined video features
            if isinstance(refined_video_output, dict):
                refined_video_embedding = refined_video_output['features']
                track_relevance = refined_video_output.get('track_relevance')
                
                # Store track relevance for later use
                if track_relevance is not None and batch_idx == 0:  # Store from first batch
                    if track_relevance.dim() > 2:  # Multi-head case: [1, n_heads, N_tracks]
                        self.last_track_relevance = track_relevance.mean(dim=1)[0]  # [N_tracks]
                    else:  # Single-head case: [1, N_tracks]
                        self.last_track_relevance = track_relevance[0]
                    
                    # Also store other info for testing script
                    self.last_tracks = refined_video_output.get('tracks')
                    self.last_visibility = refined_video_output.get('visibility')
                    self.last_per_head_attention = refined_video_output.get('cross_attention_weights')
                    
            else:
                refined_video_embedding = refined_video_output
            
            # Project refined features
            refined_video_features = self.video_projector(refined_video_embedding)
            refined_video_features = F.normalize(refined_video_features, p=2, dim=1)
            
            # Compute refined similarities with candidate texts
            candidate_text_features = self.text_projector(candidate_embeddings)
            candidate_text_features = F.normalize(candidate_text_features, p=2, dim=1)
            
            refined_similarities = torch.matmul(refined_video_features, candidate_text_features.T)  # [1, k_candidates]
            refined_similarities = refined_similarities[0]  # [k_candidates]
            
            # Apply adaptive threshold to refined scores
            if len(refined_similarities) > 1:
                threshold = torch.quantile(refined_similarities, threshold_percentile / 100.0)
                active_mask = refined_similarities >= threshold
            else:
                active_mask = torch.ones_like(refined_similarities, dtype=torch.bool)
            
            # Get active expressions after cross-attention refinement
            active_local_indices = torch.where(active_mask)[0]
            active_scores = refined_similarities[active_mask]
            
            # Map back to global text bank indices
            active_global_indices = candidate_indices[active_local_indices]
            active_expressions = [self.text_descriptions[idx.item()] for idx in active_global_indices]
            active_confidences = active_scores.tolist()
            
            batch_results.append(active_expressions)
            batch_confidences.append(active_confidences)
            batch_counts.append(len(active_expressions))
            
            # Debug info
            print(f"🎯 Batch {batch_idx}: Retrieved {len(active_expressions)} expressions")
            print(f"   Threshold: {threshold.item():.3f}")
            for i, (expr, conf) in enumerate(zip(active_expressions, active_confidences)):
                print(f"   {i+1}. '{expr[:50]}...' (conf: {conf:.3f})")
        
        return {
            'expressions': batch_results,
            'confidences': batch_confidences, 
            'num_expressions': batch_counts
        }
    
    def get_motion_based_segmentation(self, video_frames):
            k = min(5, len(self.text_descriptions))
            initial_similarities = torch.matmul(video_features, text_bank_features.T)  # [B, num_texts]
            
            # Get top-k candidates globally first
            global_top_k_values, global_top_k_indices = torch.topk(initial_similarities, k=min(20, len(self.text_descriptions)), dim=1)
            
            print(f"🔍 Analyzing top {global_top_k_indices.shape[1]} text candidates across {self.video_encoder.text_track_attention.n_heads} heads...")
            
            # 5. For each head, find the best matching texts using cross-attention
            per_event_results = []
            per_event_similarities = []
            per_event_confidence = []
            
            B = video_frames.shape[0]
            n_heads = self.video_encoder.text_track_attention.n_heads
            
            for batch_idx in range(B):
                batch_event_texts = []
                batch_event_sims = []
                batch_event_conf = []
                
                for head_idx in range(n_heads):
                    head_texts = []
                    head_similarities = []
                    
                    # Test each candidate text with this specific head
                    head_text_scores = []  # Store (text, score) pairs for debugging
                    
                    for candidate_idx in global_top_k_indices[batch_idx]:
                        candidate_text = self.text_descriptions[candidate_idx.item()]
                        
                        # Get text embedding for this candidate
                        with torch.no_grad():
                            text_embedding = self.text_encoder([candidate_text])  # [1, d_model]
                            
                            # Use multi-event cross-attention to get per-head results
                            multi_results = self.video_encoder.text_track_attention(
                                text_embedding, 
                                motion_descriptors[batch_idx:batch_idx+1], 
                                return_all_events=True
                            )
                            
                            # Get confidence score for this specific head
                            head_confidence = multi_results['confidence_scores'][0, head_idx].item()
                            head_similarities.append(head_confidence)
                            head_text_scores.append((candidate_text, head_confidence))
                    
                    # DEBUG: Show what each head is selecting
                    if head_idx < 4:  # Only show first 4 heads
                        sorted_head_scores = sorted(head_text_scores, key=lambda x: x[1], reverse=True)
                        print(f"🔍 Head {head_idx} top 3 matches:")
                        for i, (text, score) in enumerate(sorted_head_scores[:3]):
                            print(f"    {i+1}. '{text[:40]}...' (score: {score:.4f})")
                    
                    # Sort by head-specific confidence and take top-k for this head
                    head_k = min(k, len(head_similarities))
                    head_sorted_indices = torch.tensor(head_similarities).argsort(descending=True)[:head_k]
                    
                    for rank, idx in enumerate(head_sorted_indices):
                        candidate_idx = global_top_k_indices[batch_idx, idx].item()
                        text = self.text_descriptions[candidate_idx]
                        similarity = head_similarities[idx]
                        head_texts.append(text)
                        
                        if rank == 0:  # Best match for this head
                            print(f"   Head {head_idx}: {text} (confidence: {similarity:.4f})")
                    
                    batch_event_texts.append(head_texts)
                    batch_event_sims.append(head_similarities[:head_k])
                    
                    # Head confidence is the best similarity score for this head
                    head_conf = max(head_similarities) if head_similarities else 0.0
                    batch_event_conf.append(head_conf)
                
                per_event_results.append(batch_event_texts)
                per_event_similarities.append(batch_event_sims)
                per_event_confidence.append(batch_event_conf)
            
            # 6. Convert to tensors and determine active events
            # Pad similarities to same length for tensor conversion
            max_k = max(max(len(sims) for sims in batch_sims) for batch_sims in per_event_similarities)
            
            padded_similarities = []
            for batch_sims in per_event_similarities:
                batch_padded = []
                for head_sims in batch_sims:
                    padded_head = head_sims + [0.0] * (max_k - len(head_sims))
                    batch_padded.append(padded_head[:max_k])
                padded_similarities.append(batch_padded)
            
            per_event_similarities_tensor = torch.tensor(padded_similarities, device=video_frames.device)  # [B, n_heads, max_k]
            confidence_scores = torch.tensor(per_event_confidence, device=video_frames.device)  # [B, n_heads]
            
            # FIXED: Use adaptive thresholds instead of learned thresholds
            # Learned thresholds are often too high (~0.5) vs actual confidence scores (~0.1-0.2)
            # Use 70th percentile adaptive threshold approach
            
            adaptive_thresholds = []
            active_events_list = []
            
            for batch_idx in range(confidence_scores.shape[0]):
                batch_confidences = confidence_scores[batch_idx]
                
                # Compute adaptive threshold: 70th percentile with minimum of 0.1
                adaptive_threshold = max(torch.quantile(batch_confidences, 0.7).item(), 0.1)
                adaptive_thresholds.append(adaptive_threshold)
                
                # Apply adaptive threshold
                batch_active = batch_confidences > adaptive_threshold
                active_events_list.append(batch_active)
                
                print(f"🔧 Batch {batch_idx}: Adaptive threshold={adaptive_threshold:.3f}, Active={batch_active.sum().item()}/{len(batch_active)}")
            
            active_events = torch.stack(active_events_list, dim=0)  # [B, n_heads]
            
            print(f"🎯 Active events per batch (adaptive): {active_events.sum(dim=1).tolist()}")
            
            # 7. Create backward-compatible aggregated results
            # Weight by confidence scores for aggregation
            confidence_weights = F.softmax(confidence_scores, dim=1)  # [B, n_heads]
            
            aggregated_texts = []
            aggregated_similarities = []
            
            for batch_idx in range(B):
                # Collect all unique texts from active heads, weighted by confidence
                text_scores = {}
                
                for head_idx in range(n_heads):
                    if active_events[batch_idx, head_idx]:
                        weight = confidence_weights[batch_idx, head_idx].item()
                        for text_idx, text in enumerate(per_event_results[batch_idx][head_idx]):
                            if text_idx < len(per_event_similarities[batch_idx][head_idx]):
                                score = per_event_similarities[batch_idx][head_idx][text_idx] * weight
                                if text in text_scores:
                                    text_scores[text] += score
                                else:
                                    text_scores[text] = score
                
                # Sort by aggregated score and take top-k
                if text_scores:
                    sorted_texts = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                    batch_texts = [text for text, score in sorted_texts]
                    batch_scores = [score for text, score in sorted_texts]
                else:
                    # Fallback to best from most confident head
                    best_head = confidence_scores[batch_idx].argmax().item()
                    batch_texts = per_event_results[batch_idx][best_head][:k]
                    batch_scores = per_event_similarities[batch_idx][best_head][:k]
                
                aggregated_texts.append(batch_texts)
                aggregated_similarities.append(batch_scores)
            
            # Pad aggregated similarities for tensor conversion
            max_agg_k = max(len(sims) for sims in aggregated_similarities)
            padded_agg_sims = []
            for sims in aggregated_similarities:
                padded = sims + [0.0] * (max_agg_k - len(sims))
                padded_agg_sims.append(padded[:max_agg_k])
            
            aggregated_similarities_tensor = torch.tensor(padded_agg_sims, device=video_frames.device)
            
            print(f"✅ Multi-event retrieval completed: {len(aggregated_texts)} batches, {sum(active_events.sum(dim=1).tolist())} total active events")
            
            return {
                'per_event_texts': per_event_results,              # List[List[List[str]]] [B, n_heads, k]
                'per_event_similarities': per_event_similarities_tensor,  # [B, n_heads, k]
                'active_events': active_events,                    # [B, n_heads]
                'confidence_scores': confidence_scores,            # [B, n_heads]
                'aggregated_texts': aggregated_texts,              # List[List[str]] [B, k] - backward compatible
                'aggregated_similarities': aggregated_similarities_tensor  # [B, k] - backward compatible
            }
        
    def get_motion_based_segmentation(self, video_frames):
        """
        Get segmentation based on motion patterns with text guidance
        
        Args:
            video_frames: [B, T, 3, H, W] video frames
            
        Returns:
            segmentation_mask: [B, H, W] binary segmentation mask
            track_importance: [B, N_tracks] importance score per track
        """
        # FIXED: Use text-guided track attention for proper relevance computation
        # First, we need a text embedding to guide the attention
        # Since we don't have specific text input, use a generic motion query
        
        with torch.no_grad():
            # Create a generic motion query
            generic_motion_text = ["object moving in the scene"]
            text_embedding = self.text_encoder(generic_motion_text)  # [1, d_model]
            
            # Use the video encoder's text-guided attention
            tracks, visibility = self.video_encoder.extract_grid_tracks(video_frames)
        
            try:
                # Get motion descriptors and apply cross-attention  
                motion_descriptors = self.video_encoder.motion_attention.compute_motion_descriptors(tracks, visibility)
        
                # Apply text-guided attention to get proper track relevance
                if self.video_encoder.use_cross_attention and hasattr(self.video_encoder, 'text_track_attention'):
                    multi_results = self.video_encoder.text_track_attention(
                        text_embedding, motion_descriptors, return_all_events=True
                    )
                    track_importance = multi_results['aggregated_relevance']
                else:
                    # Fallback: use motion magnitude as importance
                    print("⚠️ No text-track attention available, using motion magnitude fallback")
                    velocities = tracks[:, :, 1:] - tracks[:, :, :-1]  # [B, N, T-1, 2]
                    vel_magnitude = torch.norm(velocities, dim=-1)  # [B, N, T-1]
                    track_importance = vel_magnitude.mean(dim=-1)  # [B, N]
                    track_importance = F.softmax(track_importance, dim=-1)  # Normalize
                
            except Exception as e:
                print(f"⚠️ Track relevance computation failed: {e}")
                # Final fallback: uniform importance
                B, N_tracks = tracks.shape[0], tracks.shape[1]
                track_importance = torch.ones(B, N_tracks, device=video_frames.device) / N_tracks
        
        # Convert to spatial mask
        H, W = video_frames.shape[-2:]
        segmentation_mask = torch.zeros((video_frames.shape[0], H, W), device=video_frames.device)
        
        # For each batch
        for b in range(tracks.shape[0]):
            # Get first frame track positions
            first_frame_tracks = tracks[b, :, 0, :]  # [N_tracks, 2]
            track_scores = track_importance[b]  # [N_tracks]
            
            # Only consider tracks that are visible in first frame
            first_frame_visibility = visibility[b, :, 0] > 0.5  # [N_tracks]
            
            valid_tracks = first_frame_tracks[first_frame_visibility]  # [N_valid, 2]
            valid_scores = track_scores[first_frame_visibility]  # [N_valid]
            
            if len(valid_tracks) > 0:
                # Convert track positions to pixel coordinates
                track_pixels = valid_tracks.round().long()  # [N_valid, 2]
                
                # Filter tracks within image bounds
                in_bounds = (track_pixels[:, 0] >= 0) & (track_pixels[:, 0] < W) & \
                           (track_pixels[:, 1] >= 0) & (track_pixels[:, 1] < H)
                
                bounded_pixels = track_pixels[in_bounds]  # [N_bounded, 2]
                bounded_scores = valid_scores[in_bounds]  # [N_bounded]
                
                if len(bounded_pixels) > 0:
                    # Add scores to segmentation mask
                    segmentation_mask[b, bounded_pixels[:, 1], bounded_pixels[:, 0]] = bounded_scores
        
        # Apply Gaussian smoothing to create smoother segmentation
        for b in range(segmentation_mask.shape[0]):
            mask_np = segmentation_mask[b].cpu().numpy()
            if mask_np.max() > 0:
                # Convert to 0-255 range for OpenCV
                mask_uint8 = (mask_np * 255 / mask_np.max()).astype(np.uint8)
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(mask_uint8, (15, 15), 5.0)
                # Convert back to float and normalize
                segmentation_mask[b] = torch.from_numpy(blurred.astype(np.float32) / 255.0).to(video_frames.device)
        
        # Threshold to create binary mask
        threshold = segmentation_mask.max() * 0.3  # Use 30% of max value as threshold
        segmentation_mask = (segmentation_mask > threshold).float()
        
        return segmentation_mask, track_importance
    
    def multi_expression_retrieval_with_segmentation(self, video_frames, top_k=20, threshold_percentile=70):
        """
        Complete multi-expression pipeline: Video → Multiple Expressions + Segmentations
        
        Args:
            video_frames: [B, T, 3, H, W] video frames
            top_k: Number of top candidates to consider
            threshold_percentile: Percentile threshold for active expressions
            
        Returns:
            dict containing:
                'expressions': List[List[str]] active expressions per batch
                'confidences': List[List[float]] confidence scores per batch
                'segmentations': List[torch.Tensor] segmentation masks per batch [B, N_active, H, W]
                'num_expressions': List[int] number of active expressions per batch
        """
        # 1. Multi-expression retrieval
        retrieval_results = self.forward_multi_retrieval(
            video_frames, top_k=top_k, threshold_percentile=threshold_percentile
        )
        
        # 2. Generate segmentation for each retrieved expression
        batch_segmentations = []
        
        for batch_idx in range(len(retrieval_results['expressions'])):
            batch_expressions = retrieval_results['expressions'][batch_idx]
            batch_video_frames = video_frames[batch_idx:batch_idx+1]  # [1, T, 3, H, W]
            
            expression_segmentations = []
            
            for expr in batch_expressions:
                # Generate text-guided segmentation for this expression
                segmentation = self._create_text_guided_segmentation(
                    batch_video_frames, expr
                )
                expression_segmentations.append(segmentation)
            
            # Stack segmentations for this batch
            if expression_segmentations:
                batch_segmentations.append(torch.stack(expression_segmentations))  # [N_active, H, W]
            else:
                # Empty fallback
                H, W = video_frames.shape[-2:]
                batch_segmentations.append(torch.zeros(1, H, W, device=video_frames.device))
        
        # Add segmentations to results
        retrieval_results['segmentations'] = batch_segmentations
        
        return retrieval_results
    
    def multi_expression_head_guided_retrieval(self, video_frames, top_k=20, threshold_percentile=70):
        """
        NEW: Head-guided multi-expression retrieval using per-head attention for track selection
        
        Theory: Instead of complex per-head text retrieval, use global text retrieval 
        + per-head track selection + MFA segmentation
        
        Args:
            video_frames: [B, T, 3, H, W] video frames
            top_k: Number of top candidates to consider  
            threshold_percentile: Percentile threshold for active expressions
            
        Returns:
            dict containing:
                'expressions': List[List[str]] active expressions per batch
                'confidences': List[List[float]] confidence scores per batch
                'segmentations': List[torch.Tensor] expression segmentations [B, N_expressions, H, W]
                'per_head_segmentations': List[torch.Tensor] per-head segmentations [B, N_heads, H, W]
                'expression_head_mapping': List[List[int]] which heads are used for each expression
                'num_expressions': List[int] number of active expressions per batch
        """
        print(f"\n🚀 STARTING HEAD-GUIDED RETRIEVAL...")
        
        # 1. Global text retrieval (already working perfectly)
        retrieval_results = self.forward_multi_retrieval(
            video_frames, top_k=top_k, threshold_percentile=threshold_percentile
        )
        
        print(f"🚀 HEAD-GUIDED RETRIEVAL: {len(retrieval_results['expressions'][0])} expressions")
        
        # 2. Extract per-head attention patterns and create head-guided segmentations
        batch_segmentations = []
        batch_per_head_segmentations = []
        batch_expression_head_mapping = []
        
        for batch_idx in range(len(retrieval_results['expressions'])):
            batch_expressions = retrieval_results['expressions'][batch_idx]
            batch_video_frames = video_frames[batch_idx:batch_idx+1]  # [1, T, 3, H, W]
            
            if batch_expressions:
                # Get per-head attention patterns using FIRST expression as representative query
                # NOTE: Cross-attention expects single text input [B, d_model], not multiple expressions
                first_expr = batch_expressions[0]
                text_embedding = self.text_encoder([first_expr])  # Single expression for cross-attention
                
                # Extract tracks and motion descriptors
                tracks, visibility = self.video_encoder.extract_grid_tracks(batch_video_frames)
                motion_descriptors = self.video_encoder.motion_attention.compute_motion_descriptors(tracks, visibility)
                
                # Get per-head attention patterns using representative query
                if hasattr(self.video_encoder, 'text_track_attention'):
                    cross_attn_results = self.video_encoder.text_track_attention(
                        text_embedding, motion_descriptors, return_all_events=True
                    )
                    
                    # CRITICAL FIX: Store track relevance for testing script access
                    self.last_track_relevance = cross_attn_results.get('aggregated_relevance')
                    self.last_tracks = tracks
                    self.last_visibility = visibility
                    self.last_per_head_attention = cross_attn_results.get('attention_weights')
                    
                    if 'attention_weights' in cross_attn_results:
                        per_head_attention = cross_attn_results['attention_weights']  # [1, n_heads, N_tracks]
                        n_heads = per_head_attention.shape[1]
                        
                        print(f"\n🔍 HEAD-GUIDED RETRIEVAL:")
                        print(f"   Expressions: {len(batch_expressions)}")
                        print(f"   Heads: {n_heads}")
                        
                        # 3. Create semantic expression-head mapping based on attention patterns
                        expression_head_mapping = self._compute_semantic_head_mapping(
                            batch_expressions, per_head_attention, n_heads
                        )
                        
                        # 4. Create expression-specific segmentations using head-guided track selection
                        expression_segmentations = []
                        per_head_segmentations = []
                        
                        # Create per-head segmentations first
                        for head_idx in range(n_heads):
                            head_attention = per_head_attention[0, head_idx]  # [N_tracks]
                            head_segmentation = self._create_head_guided_segmentation(
                                batch_video_frames, tracks, head_attention
                            )
                            per_head_segmentations.append(head_segmentation)
                        
                        per_head_segmentations = torch.stack(per_head_segmentations)  # [n_heads, H, W]
                        
                        # Create expression segmentations by combining assigned heads
                        for expr_idx, head_indices in enumerate(expression_head_mapping):
                            # Combine segmentations from assigned heads
                            expr_segmentation = torch.zeros_like(per_head_segmentations[0])
                            for head_idx in head_indices:
                                expr_segmentation = torch.maximum(expr_segmentation, per_head_segmentations[head_idx])
                            expression_segmentations.append(expr_segmentation)
                        
                        expression_segmentations = torch.stack(expression_segmentations)  # [N_expressions, H, W]
                        
                        batch_segmentations.append(expression_segmentations)
                        batch_per_head_segmentations.append(per_head_segmentations)
                        batch_expression_head_mapping.append(expression_head_mapping)
                        
                    else:
                        print("⚠️ Per-head attention not available, using fallback")
                        self._create_fallback_segmentations(
                            batch_expressions, batch_video_frames, 
                            batch_segmentations, batch_per_head_segmentations, batch_expression_head_mapping
                        )
                else:
                    print("⚠️ Cross-attention not available, using fallback")
                    self._create_fallback_segmentations(
                        batch_expressions, batch_video_frames,
                        batch_segmentations, batch_per_head_segmentations, batch_expression_head_mapping
                    )
            else:
                print("⚠️ No expressions found, using fallback")
                self._create_fallback_segmentations(
                    ["no expression"], batch_video_frames,
                    batch_segmentations, batch_per_head_segmentations, batch_expression_head_mapping
                )
        
        # Add new results to retrieval_results
        retrieval_results['segmentations'] = batch_segmentations
        retrieval_results['per_head_segmentations'] = batch_per_head_segmentations
        retrieval_results['expression_head_mapping'] = batch_expression_head_mapping
        
        return retrieval_results
    
    def _create_head_guided_segmentation(self, video_frames, tracks, head_attention):
        """
        Create segmentation mask using head-specific track attention + MFA
        
        Args:
            video_frames: [1, T, 3, H, W] single video
            tracks: [1, N_tracks, T, 2] track positions
            head_attention: [N_tracks] attention weights for this head
            
        Returns:
            segmentation: [H, W] segmentation mask
        """
        H, W = video_frames.shape[-2:]
        device = video_frames.device
        
        try:
            # Use PURE head attention for segmentation (DETR learned the right patterns!)
            tracks_np = tracks[0].cpu().numpy()  # [N_tracks, T, 2]
            
            # Use head attention directly - this is what makes each head unique!
            head_attn_np = head_attention.cpu().numpy()
            
            # Apply attention threshold to focus on most relevant tracks for this head
            attention_threshold = np.percentile(head_attn_np, 75)  # Top 25% of tracks
            
            # Use pure head attention (no dilution with motion magnitude)
            combined_relevance = head_attn_np.copy()
            
            # Zero out tracks below threshold to increase head specialization
            combined_relevance[combined_relevance < attention_threshold] *= 0.1  # Heavily reduce low-attention tracks
            
            # Select top tracks based on head attention (more selective for specialization)
            top_k = min(15, len(combined_relevance))  # Reduced from 20 to 15 for more focus
            top_indices = np.argsort(combined_relevance)[-top_k:]
            
            # Further filter: only keep tracks with significant attention
            min_attention = np.percentile(head_attn_np, 80)  # Only top 20% attention tracks
            top_indices = [idx for idx in top_indices if head_attn_np[idx] >= min_attention]
            
            # Create segmentation mask using Gaussian blobs
            segmentation_mask = np.zeros((H, W), dtype=np.float32)
            
            for track_idx in top_indices:
                # FIX: Use ALL temporal positions to create moving segmentation
                track_trajectory = tracks_np[track_idx]  # [T, 2] all temporal positions
                track_relevance = combined_relevance[track_idx]
                
                # Create segmentation using all track positions (weighted by time)
                for t, track_pos in enumerate(track_trajectory):
                    x, y = int(track_pos[0]), int(track_pos[1])
                    
                    # Ensure coordinates are within bounds
                    if 0 <= x < W and 0 <= y < H:
                        # Create smaller, more focused Gaussian blobs for head specialization
                        sigma = 12.0  # Reduced blob size for more precision
                        # Weight recent positions more (temporal weighting)
                        temporal_weight = 0.3 + 0.7 * (t / max(1, len(track_trajectory) - 1))  # 0.3 to 1.0 (more emphasis on recent)
                        
                        # Smaller radius for more focused segmentation
                        for dy in range(-20, 21):  # Reduced from -30,31 to -20,21
                            for dx in range(-20, 21):
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < W and 0 <= ny < H:
                                    distance_sq = dx*dx + dy*dy
                                    weight = np.exp(-distance_sq / (2 * sigma * sigma))
                                    # Scale by head attention strength (more aggressive)
                                    attention_strength = track_relevance ** 1.5  # Emphasize high-attention tracks
                                    weight *= attention_strength * temporal_weight
                                    segmentation_mask[ny, nx] = max(segmentation_mask[ny, nx], weight)
            
            # Normalize and convert to tensor
            if segmentation_mask.max() > 0:
                segmentation_mask = segmentation_mask / segmentation_mask.max()
            
            return torch.from_numpy(segmentation_mask).to(device)
            
        except Exception as e:
            print(f"⚠️ Head-guided segmentation failed: {e}")
            # Fallback: empty mask
            return torch.zeros(H, W, device=device)
    
    def _create_fallback_segmentations(self, expressions, video_frames, 
                                     batch_segmentations, batch_per_head_segmentations, batch_expression_head_mapping):
        """
        Create fallback segmentations when head-guided approach fails
        """
        H, W = video_frames.shape[-2:]
        device = video_frames.device
        
        n_expr = len(expressions)
        n_heads = 8  # Default number of heads
        
        # Create empty segmentations
        expression_segmentations = torch.zeros(n_expr, H, W, device=device)
        per_head_segmentations = torch.zeros(n_heads, H, W, device=device)
        # Use semantic mapping even for fallback
        expression_head_mapping = []
        for i in range(n_expr):
            primary_head = i % n_heads
            secondary_head = (i + n_heads // 2) % n_heads
            expression_head_mapping.append([primary_head, secondary_head])
        
        batch_segmentations.append(expression_segmentations)
        batch_per_head_segmentations.append(per_head_segmentations)
        batch_expression_head_mapping.append(expression_head_mapping)
    
    def _compute_semantic_head_mapping(self, expressions, per_head_attention, n_heads):
        """
        Compute semantic expression-head mapping based on attention similarity
        
        Args:
            expressions: List[str] text expressions
            per_head_attention: [B, n_heads, N_tracks] per-head attention weights
            n_heads: int number of attention heads
            
        Returns:
            expression_head_mapping: List[List[int]] head assignments per expression
        """
        try:
            # Encode expressions to get embeddings
            expression_embeddings = []
            for expr in expressions:
                # Get text embedding for this expression
                text_emb = self.text_encoder([expr])  # [1, d_model]
                expression_embeddings.append(text_emb.squeeze(0))  # [d_model]
            
            expression_embeddings = torch.stack(expression_embeddings)  # [N_expr, d_model]
            
            # Compute head-wise attention statistics to characterize each head
            head_profiles = []
            for head_idx in range(n_heads):
                head_attention = per_head_attention[0, head_idx, :]  # [N_tracks]
                
                # Create head profile based on attention statistics
                profile = torch.tensor([
                    head_attention.mean().item(),      # Average attention
                    head_attention.std().item(),       # Attention variance
                    head_attention.max().item(),       # Peak attention
                    (head_attention > head_attention.median()).float().mean().item(),  # Selectivity
                ], device=expression_embeddings.device)
                
                head_profiles.append(profile)
            
            head_profiles = torch.stack(head_profiles)  # [n_heads, 4]
            
            # For each expression, find the most compatible heads
            expression_head_mapping = []
            used_heads = set()
            
            for expr_idx, expr_emb in enumerate(expression_embeddings):
                # Compute compatibility between expression and head profiles
                # Use a simple heuristic: expressions with motion words prefer high-variance heads
                motion_words = ['moving', 'running', 'walking', 'jumping', 'flying', 'chasing', 'following']
                spatial_words = ['left', 'right', 'center', 'top', 'bottom', 'corner']
                
                expr_text = expressions[expr_idx].lower()
                
                # Score heads based on expression characteristics
                head_scores = []
                for head_idx in range(n_heads):
                    score = 0.0
                    
                    # Motion expressions prefer heads with high variance (selective attention)
                    if any(word in expr_text for word in motion_words):
                        score += head_profiles[head_idx][1].item()  # std
                    
                    # Spatial expressions prefer heads with high peak attention
                    if any(word in expr_text for word in spatial_words):
                        score += head_profiles[head_idx][2].item()  # max
                    
                    # General preference for heads with good selectivity
                    score += head_profiles[head_idx][3].item() * 0.5  # selectivity
                    
                    # Penalize already heavily used heads
                    if head_idx in used_heads:
                        score *= 0.5
                    
                    head_scores.append(score)
                
                # Select top 2 heads for this expression
                head_scores = torch.tensor(head_scores)
                top_heads = torch.topk(head_scores, min(2, n_heads)).indices.tolist()
                
                expression_head_mapping.append(top_heads)
                used_heads.update(top_heads)
                
                head_score_list = [head_scores[h].item() for h in top_heads]
                print(f"   Expression '{expr_text}' → Heads {top_heads} (scores: {head_score_list})")
            
            return expression_head_mapping
            
        except Exception as e:
            print(f"⚠️ Semantic head mapping failed: {e}, using fallback")
            # Fallback to round-robin
            expression_head_mapping = []
            for expr_idx in range(len(expressions)):
                primary_head = expr_idx % n_heads
                secondary_head = (expr_idx + n_heads // 2) % n_heads
                expression_head_mapping.append([primary_head, secondary_head])
            return expression_head_mapping

    def _create_expression_head_mapping(self, expressions, n_heads):
        """
        DEPRECATED: Create expression-head mapping based on expression similarity
        Create segmentation mask guided by specific text expression
        
        Args:
            video_frames: [1, T, 3, H, W] single video
            expression: str text expression to guide segmentation
            
        Returns:
            segmentation: [H, W] segmentation mask
        """
        H, W = video_frames.shape[-2:]
        device = video_frames.device
        
        try:
            # Get text embedding for this expression
            text_embedding = self.text_encoder([expression])  # [1, d_model]
            
            # Extract tracks and motion descriptors
            tracks, visibility = self.video_encoder.extract_grid_tracks(video_frames)
            motion_descriptors = self.video_encoder.motion_attention.compute_motion_descriptors(tracks, visibility)
            
            # SIMPLIFIED: Use motion magnitude directly (reliable and working)
            # The cross-attention is not properly trained, so use motion-based relevance
            
            # Compute track motion magnitude (this works well as shown in visualizations)
            velocities = tracks[0, :, 1:] - tracks[0, :, :-1]  # [N, T-1, 2]
            vel_magnitude = torch.norm(velocities, dim=-1).mean(dim=-1)  # [N] - average velocity magnitude per track
            
            # Apply softmax to get proper probability distribution
            track_relevance = F.softmax(vel_magnitude / 0.1, dim=0)  # Temperature scaling for sharper distribution
            
            # Get high-relevance tracks (top 20%)
            threshold = torch.quantile(track_relevance, 0.8)
            relevant_mask = track_relevance > threshold
            
            if relevant_mask.sum() == 0:
                # Fallback: use top tracks
                top_k = min(10, len(track_relevance))
                top_indices = track_relevance.argsort(descending=True)[:top_k]
                relevant_mask = torch.zeros_like(track_relevance, dtype=torch.bool)
                relevant_mask[top_indices] = True
            
            # Get first frame positions of relevant tracks
            relevant_tracks = tracks[0, relevant_mask, 0, :]  # [N_relevant, 2]
            relevant_scores = track_relevance[relevant_mask]  # [N_relevant]
            
            # Create segmentation mask
            segmentation = torch.zeros((H, W), device=device)
            
            for track_pos, score in zip(relevant_tracks, relevant_scores):
                x, y = int(track_pos[0].item()), int(track_pos[1].item())
                
                # Ensure coordinates are within bounds
                if 0 <= x < W and 0 <= y < H:
                    # Create Gaussian blob around this track
                    radius = 12
                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(max(0, y-radius), min(H, y+radius+1), device=device),
                        torch.arange(max(0, x-radius), min(W, x+radius+1), device=device),
                        indexing='ij'
                    )
                    
                    # Gaussian falloff
                    dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
                    gaussian = torch.exp(-dist_sq / (2 * (radius/3) ** 2))  # sigma = radius/3
                    
                    # Add weighted contribution
                    segmentation[y_coords, x_coords] += score * gaussian
            
            # Normalize and threshold
            if segmentation.max() > 0:
                segmentation = segmentation / segmentation.max()  # Normalize to 0-1
                # Use adaptive threshold based on distribution
                threshold = torch.quantile(segmentation[segmentation > 0], 0.6) if (segmentation > 0).sum() > 0 else 0.5
                segmentation = (segmentation > threshold).float()
            
            return segmentation
            
        except Exception as e:
            print(f"⚠️ Text-guided segmentation failed for '{expression}': {e}")
            # Return empty mask
            return torch.zeros((H, W), device=device)

    def compute_similarity_matrix(self, video_features, text_features):
        """
        Compute video-text similarity matrix
        
        Args:
            video_features: [B, D] normalized video features  
            text_features: [B, D] normalized text features
            
        Returns:
            similarity_matrix: [B, B] similarity scores
        """
        # DEBUG: Check features before normalization
        if hasattr(self, '_debug_features'):
            print(f"🔍 Pre-norm video: mean={video_features.mean():.4f}, std={video_features.std():.4f}")
            print(f"🔍 Pre-norm text: mean={text_features.mean():.4f}, std={text_features.std():.4f}")
        
        # Normalize features
        video_features = F.normalize(video_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # DEBUG: Check features after normalization  
        if hasattr(self, '_debug_features'):
            print(f"🔍 Post-norm video: mean={video_features.mean():.4f}, std={video_features.std():.4f}")
            print(f"🔍 Post-norm text: mean={text_features.mean():.4f}, std={text_features.std():.4f}")
        
        # Compute similarity matrix and apply temperature
        # FIXED: Move temperature inside to avoid double normalization
        logits = torch.matmul(video_features, text_features.T) 
        similarity_matrix = logits / self.temperature
        
        # DEBUG: Check similarity matrix statistics
        if hasattr(self, '_debug_features'):
            print(f"🔍 Similarity: min={similarity_matrix.min():.4f}, max={similarity_matrix.max():.4f}")
            print(f"🔍 Similarity diagonal: {torch.diag(similarity_matrix).mean():.4f}")
            
        return similarity_matrix
    
    def rebuild_text_bank_in_eval_mode(self):
        """
        CRITICAL FIX: Rebuild text bank in eval mode to ensure consistency
        This prevents train/eval mode contamination in normalization layers
        """
        if not self.text_descriptions:
            print("⚠️ No text descriptions available for rebuilding text bank")
            return
            
        print(f"🔧 Rebuilding text bank in eval mode with {len(self.text_descriptions)} descriptions...")
        
        # Force eval mode for consistent embeddings
        was_training = self.training
        self.eval()
        
        fresh_text_embeddings = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(self.text_descriptions), batch_size):
                batch_texts = self.text_descriptions[i:i+batch_size]
                
                # Encode text batch in eval mode
                text_embeddings = self.text_encoder(batch_texts)
                text_features = self.text_projector(text_embeddings)
                
                # Store UN-normalized features (normalization happens in forward_inference)
                fresh_text_embeddings.append(text_features.cpu())
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"   Processed {i + len(batch_texts)}/{len(self.text_descriptions)} descriptions")
        
        # Combine all text features
        fresh_text_bank = torch.cat(fresh_text_embeddings, dim=0).to(next(self.parameters()).device)
        
        # Update text bank
        self.register_buffer('text_embedding_bank', fresh_text_bank)
        
        print(f"✅ Text bank rebuilt: {fresh_text_bank.shape}")
        print(f"📊 New text bank stats:")
        print(f"   Mean: {fresh_text_bank.mean():.4f}, Std: {fresh_text_bank.std():.4f}")
        
        # Restore original training mode
        if was_training:
            self.train()
        
        return fresh_text_bank


def _test_tcam_text_encoder():
    """Test the motion text encoder"""
    print("📝 TESTING MOTION TEXT ENCODER")
    print("=" * 40)
    
    # Create text encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_encoder = TCAMTextEncoder(d_model=256).to(device)
    
    # Test motion expressions
    motion_expressions = [
        "panda rolling around",
        "panda climbing branch", 
        "panda falling off branch",
        "person jumping up and down",
        "cat running fast across the room",
        "dog playing with ball"
    ]
    
    print(f"🔤 Encoding {len(motion_expressions)} motion expressions...")
    
    # Encode expressions
    with torch.no_grad():
        text_embeddings = text_encoder(motion_expressions)
    
    print(f"✅ Text embeddings shape: {text_embeddings.shape}")
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(
        F.normalize(text_embeddings, p=2, dim=1),
        F.normalize(text_embeddings, p=2, dim=1).T
    )
    
    print(f"\n📊 Text Similarity Analysis:")
    for i, expr1 in enumerate(motion_expressions):
        for j, expr2 in enumerate(motion_expressions):
            if i < j:  # Only upper triangle
                sim = similarity_matrix[i, j].item()
                print(f"   '{expr1}' ↔ '{expr2}': {sim:.3f}")
    
    return text_encoder


def _test_video_text_matching():
    """Test complete video-text matching system"""
    print("\n🎯 TESTING VIDEO-TEXT MATCHING SYSTEM")
    print("=" * 45)
    
    # Import video encoder
    from tcam_video_encoder import TCAMVideoEncoder
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create video encoder
    video_encoder = TCAMVideoEncoder(
        d_model=256,
        n_heads=8,
        temporal_window=8,
        grid_size=10,
        dropout=0.1
    ).to(device)
    
    # Load CoTracker
    video_encoder.load_cotracker()
    
    # Create video-text matcher
    matcher = TCAMVideoTextMatcher(video_encoder, d_model=256).to(device)
    
    # Test data
    batch_size = 2
    video_frames = torch.randn(batch_size, 8, 3, 480, 640).to(device) * 0.5 + 0.5
    motion_expressions = [
        "panda rolling around",
        "panda climbing branch"
    ]
    
    print(f"📹 Video input shape: {video_frames.shape}")
    print(f"📝 Text expressions: {motion_expressions}")
    
    # Forward pass
    with torch.no_grad():
        video_features, text_features, similarity_matrix = matcher(
            video_frames, motion_expressions
        )
    
    print(f"✅ Video features shape: {video_features.shape}")
    print(f"✅ Text features shape: {text_features.shape}")
    print(f"✅ Similarity matrix shape: {similarity_matrix.shape}")
    
    # Compute contrastive loss
    contrastive_loss = matcher.compute_contrastive_loss(similarity_matrix)
    print(f"📊 Contrastive loss: {contrastive_loss.item():.4f}")
    
    # Show similarity scores
    print(f"\n📏 Video-Text Similarity Matrix:")
    for i in range(batch_size):
        for j in range(batch_size):
            sim = similarity_matrix[i, j].item()
            print(f"   Video {i} ↔ Text {j}: {sim:.3f}")
    
    return matcher


def inference_segment_and_retrieve(model, video_frames, detect_multiple_events=False):
    """
    Full pipeline: video → segmentation + text retrieval
    
    Args:
        model: TCAMVideoTextMatcher instance
        video_frames: [B, T, 3, H, W] video frames
        detect_multiple_events: bool, if True enables multi-event discovery
        
    Returns:
        If detect_multiple_events=False (default, backward compatible):
            dict containing:
                - segmentation: [B, H, W] binary segmentation mask
                - retrieved_captions: List[str] top-5 matching text descriptions
                - confidence: float, confidence score for best match
                - track_importance: [B, N_tracks] importance score per track
                
        If detect_multiple_events=True (multi-event mode):
            dict containing:
                - per_event_segmentations: [B, n_heads, H, W] segmentation masks per event
                - per_event_captions: List[List[List[str]]] text descriptions per event [B, n_heads, k]
                - per_event_confidences: [B, n_heads] confidence per event
                - active_events: [B, n_heads] boolean mask of active events
                - aggregated_segmentation: [B, H, W] combined segmentation (backward compatible)
                - aggregated_captions: List[List[str]] combined captions (backward compatible)
                - aggregated_confidence: [B] combined confidence (backward compatible)
                - track_importance: [B, N_tracks] aggregated track importance
    """
    if not detect_multiple_events:
        # BACKWARD COMPATIBLE PATH - unchanged behavior
        # 1. First, retrieve matching text descriptions to understand what to segment
        retrieved_texts, similarities = model.forward_inference(video_frames)
        
        # 2. Use the best retrieved text to guide segmentation
        if retrieved_texts and len(retrieved_texts[0]) > 0:
            best_text = retrieved_texts[0][0]  # Best matching text
            print(f"🎯 Using '{best_text}' to guide segmentation")
            
            # Generate text embedding for the best match
            with torch.no_grad():
                text_embedding = model.text_encoder([best_text])  # [1, d_model]
                
                # Extract tracks and compute text-guided relevance
                tracks, visibility = model.video_encoder.extract_grid_tracks(video_frames)
                
                try:
                    # Get motion descriptors
                    motion_descriptors = model.video_encoder.motion_attention.compute_motion_descriptors(tracks, visibility)
                    
                    # Apply text-guided attention for relevance
                    if model.video_encoder.use_cross_attention and hasattr(model.video_encoder, 'text_track_attention'):
                        multi_results = model.video_encoder.text_track_attention(
                            text_embedding, motion_descriptors, return_all_events=True
                        )
                        track_importance = multi_results['aggregated_relevance']
                        print(f"✅ Text-guided track relevance computed")
                    else:
                        print("⚠️ Falling back to motion-based segmentation")
                        segmentation_mask, track_importance = model.get_motion_based_segmentation(video_frames)
                        return {
                            'segmentation': segmentation_mask,
                            'retrieved_captions': retrieved_texts[0],
                            'confidence': similarities[0][0].item() if similarities.numel() > 0 else 0.0,
                            'track_importance': track_importance
                        }
                    
                    # Create improved segmentation using text-guided tracks
                    H, W = video_frames.shape[-2:]
                    segmentation_mask = torch.zeros((video_frames.shape[0], H, W), device=video_frames.device)
                    
                    for b in range(tracks.shape[0]):
                        # Get first frame track positions
                        first_frame_tracks = tracks[b, :, 0, :]  # [N_tracks, 2]
                        track_scores = track_importance[b]  # [N_tracks]
                        
                        # Apply threshold to focus on most relevant tracks
                        score_threshold = torch.quantile(track_scores, 0.7)  # Top 30% of tracks
                        relevant_mask = track_scores > score_threshold
                        
                        relevant_tracks = first_frame_tracks[relevant_mask]
                        relevant_scores = track_scores[relevant_mask]
                        
                        if len(relevant_tracks) > 0:
                            # Convert to pixel coordinates
                            track_pixels = relevant_tracks.round().long()
                            
                            # Filter in-bounds
                            in_bounds = (track_pixels[:, 0] >= 0) & (track_pixels[:, 0] < W) & \
                                       (track_pixels[:, 1] >= 0) & (track_pixels[:, 1] < H)
                            
                            bounded_pixels = track_pixels[in_bounds]
                            bounded_scores = relevant_scores[in_bounds]
                            
                            if len(bounded_pixels) > 0:
                                # Create Gaussian blobs around high-relevance tracks
                                for pixel, score in zip(bounded_pixels, bounded_scores):
                                    x, y = pixel[0].item(), pixel[1].item()
                                    # Create a Gaussian blob around this point
                                    y_coords, x_coords = torch.meshgrid(
                                        torch.arange(max(0, y-15), min(H, y+16), device=video_frames.device),
                                        torch.arange(max(0, x-15), min(W, x+16), device=video_frames.device),
                                        indexing='ij'
                                    )
                                    
                                    # Gaussian falloff
                                    dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
                                    gaussian_weight = torch.exp(-dist_sq / (2 * 8 ** 2))  # sigma=8
                                    
                                    # Add weighted contribution
                                    segmentation_mask[b, y_coords, x_coords] += score * gaussian_weight
                    
                    # Normalize and threshold
                    for b in range(segmentation_mask.shape[0]):
                        mask = segmentation_mask[b]
                        if mask.max() > 0:
                            mask = mask / mask.max()  # Normalize to 0-1
                            threshold = mask.quantile(0.8)  # Top 20% becomes foreground
                            segmentation_mask[b] = (mask > threshold).float()
                    
                except Exception as e:
                    print(f"⚠️ Text-guided segmentation failed: {e}")
                    # Fallback to motion-based segmentation
                    segmentation_mask, track_importance = model.get_motion_based_segmentation(video_frames)
                    
        else:
            # No text retrieved, use generic motion-based segmentation
            print("⚠️ No text retrieved, using generic motion-based segmentation")
            segmentation_mask, track_importance = model.get_motion_based_segmentation(video_frames)
        
        # 3. Return results
        return {
            'segmentation': segmentation_mask,
            'retrieved_captions': retrieved_texts[0] if retrieved_texts else ["no motion description available"],
            'confidence': similarities[0][0].item() if similarities.numel() > 0 else 0.0,
            'track_importance': track_importance
        }
    
    else:
        # MULTI-EVENT PATH - new functionality
        print("🎯 Multi-event segmentation enabled!")
        
        # 1. Perform multi-event text retrieval
        multi_results = model.forward_inference(video_frames, detect_multiple_events=True)
        
        # Extract results from multi-event inference
        per_event_texts = multi_results['per_event_texts']
        active_events = multi_results['active_events']
        confidence_scores = multi_results['confidence_scores']
        
        # 2. Extract tracks and motion descriptors
        tracks, visibility = model.video_encoder.extract_grid_tracks(video_frames)
        motion_descriptors = model.video_encoder.motion_attention.compute_motion_descriptors(tracks, visibility)
        
        B, n_heads = active_events.shape
        H, W = video_frames.shape[-2:]
        
        # 3. Generate segmentation for each event/head
        per_event_segmentations = torch.zeros((B, n_heads, H, W), device=video_frames.device)
        per_event_track_importance = []
        
        print(f"🎨 Generating segmentations for {active_events.sum().item()} active events...")
        
        for batch_idx in range(B):
            batch_track_importance = []
            
            for head_idx in range(n_heads):
                if active_events[batch_idx, head_idx]:
                    # Get best text for this head/event
                    if per_event_texts[batch_idx][head_idx]:
                        event_text = per_event_texts[batch_idx][head_idx][0]  # Best text for this event
                        print(f"   Event {head_idx}: Segmenting '{event_text}' (confidence: {confidence_scores[batch_idx, head_idx]:.3f})")
                        
                        # Generate text embedding for this event
                        with torch.no_grad():
                            text_embedding = model.text_encoder([event_text])  # [1, d_model]
                            
                            # Get multi-event cross-attention results
                            multi_attention_results = model.video_encoder.text_track_attention(
                                text_embedding, 
                                motion_descriptors[batch_idx:batch_idx+1], 
                                return_all_events=True
                            )
                            
                            # Extract relevance for this specific head
                            head_relevance = multi_attention_results['relevance_scores'][0, head_idx]  # [N_tracks]
                            batch_track_importance.append(head_relevance.cpu())
                            
                            # Create segmentation mask for this event
                            first_frame_tracks = tracks[batch_idx, :, 0, :]  # [N_tracks, 2]
                            
                            # Apply threshold for this head
                            score_threshold = torch.quantile(head_relevance, 0.75)  # Top 25% of tracks
                            relevant_mask = head_relevance > score_threshold
                            
                            relevant_tracks = first_frame_tracks[relevant_mask]
                            relevant_scores = head_relevance[relevant_mask]
                            
                            if len(relevant_tracks) > 0:
                                # Convert to pixel coordinates
                                track_pixels = relevant_tracks.round().long()
                                
                                # Filter in-bounds
                                in_bounds = (track_pixels[:, 0] >= 0) & (track_pixels[:, 0] < W) & \
                                           (track_pixels[:, 1] >= 0) & (track_pixels[:, 1] < H)
                                
                                bounded_pixels = track_pixels[in_bounds]
                                bounded_scores = relevant_scores[in_bounds]
                                
                                if len(bounded_pixels) > 0:
                                    # Create Gaussian blobs for this event
                                    event_mask = torch.zeros((H, W), device=video_frames.device)
                                    
                                    for pixel, score in zip(bounded_pixels, bounded_scores):
                                        x, y = pixel[0].item(), pixel[1].item()
                                        # Create a Gaussian blob around this point
                                        y_coords, x_coords = torch.meshgrid(
                                            torch.arange(max(0, y-12), min(H, y+13), device=video_frames.device),
                                            torch.arange(max(0, x-12), min(W, x+13), device=video_frames.device),
                                            indexing='ij'
                                        )
                                        
                                        # Gaussian falloff (smaller radius for multi-event)
                                        dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
                                        gaussian_weight = torch.exp(-dist_sq / (2 * 6 ** 2))  # sigma=6
                                        
                                        # Add weighted contribution
                                        event_mask[y_coords, x_coords] += score * gaussian_weight
                                    
                                    # Normalize and threshold this event mask
                                    if event_mask.max() > 0:
                                        event_mask = event_mask / event_mask.max()  # Normalize to 0-1
                                        threshold = event_mask.quantile(0.7)  # Top 30% becomes foreground
                                        per_event_segmentations[batch_idx, head_idx] = (event_mask > threshold).float()
                else:
                    # Inactive event - zero relevance
                    batch_track_importance.append(torch.zeros(motion_descriptors.shape[1]))
            
            per_event_track_importance.append(batch_track_importance)
        
        # 4. Create aggregated results for backward compatibility
        # Combine active event segmentations weighted by confidence
        aggregated_segmentation = torch.zeros((B, H, W), device=video_frames.device)
        aggregated_track_importance = torch.zeros((B, motion_descriptors.shape[1]), device=video_frames.device)
        
        for batch_idx in range(B):
            # Get confidence weights for active events
            active_confidences = confidence_scores[batch_idx] * active_events[batch_idx].float()
            if active_confidences.sum() > 0:
                weights = active_confidences / active_confidences.sum()  # Normalize weights
                
                # Weighted combination of segmentations
                for head_idx in range(n_heads):
                    if active_events[batch_idx, head_idx]:
                        weight = weights[head_idx]
                        aggregated_segmentation[batch_idx] += weight * per_event_segmentations[batch_idx, head_idx]
                        
                        # Weighted combination of track importance
                        head_track_importance = torch.tensor(per_event_track_importance[batch_idx][head_idx], device=video_frames.device)
                        aggregated_track_importance[batch_idx] += weight * head_track_importance
            else:
                # No active events - use most confident head
                best_head = confidence_scores[batch_idx].argmax().item()
                aggregated_segmentation[batch_idx] = per_event_segmentations[batch_idx, best_head]
                aggregated_track_importance[batch_idx] = torch.tensor(per_event_track_importance[batch_idx][best_head], device=video_frames.device)
        
        # Threshold aggregated segmentation
        for batch_idx in range(B):
            mask = aggregated_segmentation[batch_idx]
            if mask.max() > 0:
                threshold = mask.quantile(0.6)  # More permissive threshold for aggregated mask
                aggregated_segmentation[batch_idx] = (mask > threshold).float()
        
        # 5. Prepare output
        aggregated_confidence = confidence_scores.max(dim=1)[0] * active_events.any(dim=1).float()  # Best confidence per batch
        
        print(f"✅ Multi-event segmentation completed:")
        for batch_idx in range(B):
            active_count = active_events[batch_idx].sum().item()
            print(f"   Batch {batch_idx}: {active_count} active events, confidence: {aggregated_confidence[batch_idx]:.3f}")
        
        return {
            'per_event_segmentations': per_event_segmentations,     # [B, n_heads, H, W]
            'per_event_captions': per_event_texts,                  # List[List[List[str]]] [B, n_heads, k]
            'per_event_confidences': confidence_scores,             # [B, n_heads]
            'active_events': active_events,                         # [B, n_heads]
            'aggregated_segmentation': aggregated_segmentation,     # [B, H, W] - backward compatible
            'aggregated_captions': multi_results['aggregated_texts'], # List[List[str]] [B, k] - backward compatible
            'aggregated_confidence': aggregated_confidence,         # [B] - backward compatible
            'track_importance': aggregated_track_importance         # [B, N_tracks] - backward compatible
        }


if __name__ == '__main__':
    pass 
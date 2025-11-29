#!/usr/bin/env python3
"""
TCAM Training Pipeline
Trains with multiple expressions per video for multi-retrieval capability
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
import json
import random
import argparse
from datetime import datetime, timedelta
import os
import time
from tqdm import tqdm

# Import our components
from tcam_dataset import TCAMDataset, create_tcam_dataloader
from tcam_video_encoder import TCAMVideoEncoder
from tcam_text_encoder import TCAMVideoTextMatcher


class TCAMTrainer:
    """Trainer for multi-positive video-text retrieval"""
    
    def __init__(self, args):
        self.args = args
        # Distributed setup
        self.distributed = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID', -1)))
        self._maybe_init_distributed()
        
        # Device placement
        if torch.cuda.is_available():
            if self.distributed and self.local_rank >= 0:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device(f'cuda:{self.local_rank}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"tcam_training_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)
        
        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        if self.is_main_process():
            print(f"🏗️ Output directory: {self.output_dir}")
        
        # Setup components
        self._setup_data()
        self._setup_model()
        self._setup_training()
    
    def _setup_data(self):
        """Setup datasets with ALL videos (single and multi-expression)"""
        if self.is_main_process():
            print("📁 Setting up datasets...")
        
        # Training dataset - include ALL videos
        self.train_dataset = TCAMDataset(
            data_root=self.args.data_root,
            split='train',
            num_frames=self.args.num_frames,
            frame_size=(self.args.frame_height, self.args.frame_width),
            max_videos=self.args.max_videos,
            min_expressions_per_video=1  # Include single-expression videos too!
        )
        
        # Validation dataset
        val_max_videos = min(50, self.args.max_videos // 4) if self.args.max_videos else 50
        self.val_dataset = TCAMDataset(
            data_root=self.args.data_root,
            split='train',
            num_frames=self.args.num_frames,
            frame_size=(self.args.frame_height, self.args.frame_width),
            max_videos=val_max_videos,
            min_expressions_per_video=1  # Include all videos
        )
        
        # Dataloaders (use DistributedSampler if distributed)
        if self.distributed:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                sampler=self.train_sampler,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=lambda batch: {
                    'video_frames': torch.stack([item['video_frames'] for item in batch]),
                    'object_masks': torch.stack([item['object_masks'] for item in batch]),
                    'expression': [item['expression'] for item in batch],
                    'video_id': [item['video_id'] for item in batch],
                    'expression_id': [item['expression_id'] for item in batch],
                    'sample_indices': [item['sample_idx'] for item in batch],
                }
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                sampler=self.val_sampler,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=lambda batch: {
                    'video_frames': torch.stack([item['video_frames'] for item in batch]),
                    'object_masks': torch.stack([item['object_masks'] for item in batch]),
                    'expression': [item['expression'] for item in batch],
                    'video_id': [item['video_id'] for item in batch],
                    'expression_id': [item['expression_id'] for item in batch],
                    'sample_indices': [item['sample_idx'] for item in batch],
                }
            )
        else:
            self.train_loader = create_tcam_dataloader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers
            )
            self.val_loader = create_tcam_dataloader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )
        
        # Print statistics
        if self.is_main_process():
            print(f"✅ Train dataset: {len(self.train_dataset)} samples")
            print(f"✅ Val dataset: {len(self.val_dataset)} samples")
        
        # Count single vs multi-expression videos
        single_expr = sum(1 for v in self.train_dataset.video_to_samples.values() if len(v) == 1)
        multi_expr = len(self.train_dataset.video_to_samples) - single_expr
        if self.is_main_process():
            print(f"📊 Training videos: {single_expr} single-expression, {multi_expr} multi-expression")
    
    def _setup_model(self):
        """Setup video encoder and matcher"""
        if self.is_main_process():
            print("🏗️ Setting up TCAM model...")
        
        # Video encoder
        self.video_encoder = TCAMVideoEncoder(
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            temporal_window=self.args.num_frames,
            grid_size=self.args.grid_size,
            dropout=self.args.dropout,
            use_cross_attention=True
        ).to(self.device)
        
        # Load CoTracker
        self.video_encoder.load_cotracker()
        
        # Video-text matcher
        matcher = TCAMVideoTextMatcher(
            self.video_encoder,
            d_model=self.args.d_model,
            temperature=self.args.temperature,
            spatial_loss_type=self.args.spatial_loss_type
        ).to(self.device)
        
        # Wrap with DDP if distributed
        if self.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            # IMPROVED DDP setup to handle dynamic execution paths and parameter reuse
            self.matcher = DDP(
                matcher,
                device_ids=[self.local_rank] if self.device.type == 'cuda' else None,
                output_device=self.local_rank if self.device.type == 'cuda' else None,
                find_unused_parameters=False,  # Set to False since we use unified loss now
                broadcast_buffers=False,       # Disable buffer broadcasting for better performance
                gradient_as_bucket_view=True   # Enable gradient bucketing optimization
            )
            
            # Register forward pre-hook to ensure clean parameter state
            def ddp_forward_pre_hook(module, input):
                """Reset DDP state before each forward pass"""
                # Clear any cached gradients that might cause issues
                if hasattr(module, '_ddp_params_and_buffers_to_ignore'):
                    module._ddp_params_and_buffers_to_ignore.clear()
                return None
            
            self.matcher.register_forward_pre_hook(ddp_forward_pre_hook)
            
            print("✅ DDP setup optimized for unified loss computation")
        else:
            self.matcher = matcher
        
        if self.is_main_process():
            print(f"✅ Model created with d_model={self.args.d_model}, n_heads={self.args.n_heads}")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and training state"""
        if self.is_main_process():
            print("🎯 Setting up TCAM training...")
        
        # Get trainable parameters (exclude CoTracker)
        trainable_params = []
        frozen_params = []
        
        for name, param in self.matcher.named_parameters():
            if 'cotracker' in name.lower():
                param.requires_grad = False
                frozen_params.append(name)
            else:
                param.requires_grad = True
                trainable_params.append(name)
        
        if self.is_main_process():
            print(f"🔥 Trainable: {len(trainable_params)} parameters")
            print(f"🔒 Frozen (CoTracker): {len(frozen_params)} parameters")
        
        # Optimizer
        optimizable_params = [p for p in self.matcher.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            optimizable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.args.learning_rate,
            epochs=self.args.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100
        )
        
        # Training stability
        self.max_grad_norm = 1.0  # Gradient clipping for stability
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')  # Changed from best_val_acc to best_val_loss
        
        # Resume from checkpoint if provided
        if self.args.resume_checkpoint:
            self._load_checkpoint()
        
        if self.is_main_process():
            print("✅ TCAM training setup complete")

    def is_main_process(self):
        return (not self.distributed) or (self.rank == 0)

    def get_model(self):
        """Return the underlying matcher module (unwrap DDP if needed)."""
        return self.matcher.module if hasattr(self.matcher, 'module') else self.matcher

    def _maybe_init_distributed(self):
        """Initialize torch.distributed if environment indicates multi-process."""
        # Prefer torchrun env vars
        world_size_env = os.environ.get('WORLD_SIZE')
        if world_size_env is not None and int(world_size_env) > 1:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
            self.distributed = True
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.distributed = False
            self.rank = 0
            self.world_size = 1
    
    def _load_checkpoint(self):
        """Load checkpoint for resuming training"""
        print(f"🔄 Loading checkpoint from {self.args.resume_checkpoint}")
        
        try:
            checkpoint = torch.load(self.args.resume_checkpoint, map_location=self.device)
            
            # Load model state
            self.matcher.load_state_dict(checkpoint['matcher_state_dict'])
            
            # Load optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.epoch = checkpoint['epoch'] + 1  # Start from next epoch
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Changed from best_val_acc
            
            # Load text bank if available
            if 'text_embedding_bank' in checkpoint:
                self.matcher.register_buffer('text_embedding_bank', checkpoint['text_embedding_bank'])
            if 'text_descriptions' in checkpoint:
                self.matcher.text_descriptions = checkpoint['text_descriptions']
            
            print(f"✅ Resumed from epoch {self.epoch}, best val loss: {self.best_val_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print("🔄 Starting training from scratch...")
            self.epoch = 0
    
    def train_epoch(self):
        """Train one epoch with multi-positive loss"""
        self.matcher.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = len(self.train_loader)
        
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            video_frames = batch['video_frames'].to(self.device)
            expressions = batch['expression']
            sample_indices = batch['sample_indices']
            
            # Get positive mapping for multi-positive loss
            positive_mapping = self.train_dataset.get_batch_positive_mapping(sample_indices)
            
            # UNIFIED forward pass - computes both losses in single pass to avoid DDP parameter reuse
            # Use spatial_weight to control global vs spatial loss balance
            spatial_weight = getattr(self.args, 'spatial_weight', 0.1)  # Default 10% spatial, 90% global
            
            try:
                loss, similarity_matrix, loss_components = self.get_model().compute_unified_multi_positive_loss(
                    video_frames, expressions, positive_mapping, batch, spatial_weight=spatial_weight
                )
            except Exception as e:
                # Fallback to original approach if unified fails
                print(f"⚠️  Unified loss failed ({e}), falling back to separate computation")
                video_features, text_features, similarity_matrix = self.matcher(
                    video_frames, expressions
                )
                
                # Get track relevance for spatial supervision
                track_relevance = None
                if hasattr(self.get_model(), 'get_last_track_relevance'):
                    track_relevance = self.get_model().get_last_track_relevance()
                
                # Multi-positive contrastive loss with spatial supervision
                if track_relevance is not None:
                    loss = self.get_model().compute_spatially_aware_multi_positive_loss(
                        video_features, text_features, track_relevance, positive_mapping, batch
                    )
                else:
                    # Fallback to original loss
                    loss = self.get_model().compute_multi_positive_loss(
                        video_features, text_features, positive_mapping
                    )
                # Create dummy loss components for logging
                loss_components = {
                    'total_loss': loss,
                    'global_loss': loss,
                    'spatial_loss': torch.tensor(0.0),
                    'mask_alignment_loss': torch.tensor(0.0)
                }
            
            # Skip NaN loss batches
            if torch.isnan(loss):
                continue
            
            # Calculate training accuracy
            batch_size = similarity_matrix.shape[0]
            batch_correct = 0
            for i in range(batch_size):
                video_similarities = similarity_matrix[i]
                positive_indices = positive_mapping.get(i, [i])
                
                # Check if any positive has highest similarity
                max_idx = video_similarities.argmax().item()
                if max_idx in positive_indices:
                    batch_correct += 1
            
            total_correct += batch_correct
            total_samples += batch_size
            

            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(self.matcher.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Log progress with accuracy and spatial loss components (rank-0 only)
            if self.is_main_process() and batch_idx % self.args.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                batch_accuracy = batch_correct / batch_size if batch_size > 0 else 0.0
                global_loss_val = loss_components['global_loss'].item() if torch.is_tensor(loss_components['global_loss']) else loss_components['global_loss']
                spatial_loss_val = loss_components['spatial_loss'].item() if torch.is_tensor(loss_components['spatial_loss']) else loss_components['spatial_loss']
                bce_loss_val = loss_components['mask_alignment_loss'].item() if torch.is_tensor(loss_components['mask_alignment_loss']) else loss_components['mask_alignment_loss']
                print(f"Epoch {self.epoch} [{batch_idx}/{num_batches}] "
                      f"Loss: {loss.item():.4f} (G:{global_loss_val:.4f} S:{spatial_loss_val:.4f} BCE:{bce_loss_val:.4f}) | "
                      f"Acc: {batch_accuracy:.3f} | LR: {current_lr:.2e}")
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, avg_accuracy
    

    

    
    def validate(self):
        """Validate with multi-positive evaluation"""
        self.matcher.eval()
        val_losses = []
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                try:
                    video_frames = batch['video_frames'].to(self.device)
                    expressions = batch['expression']
                    sample_indices = batch['sample_indices']
                    
                    # Get positive mapping
                    positive_mapping = self.val_dataset.get_batch_positive_mapping(sample_indices)
                    
                    # UNIFIED forward pass - same as training to ensure consistent loss computation
                    spatial_weight = getattr(self.args, 'spatial_weight', 0.1)  # Same weight as training
                    try:
                        loss, similarity_matrix, loss_components = self.get_model().compute_unified_multi_positive_loss(
                            video_frames, expressions, positive_mapping, batch, spatial_weight=spatial_weight
                        )
                    except Exception as e:
                        print(f"⚠️  Unified loss failed in validation ({e}), falling back to separate computation")
                        # Fallback to original approach if unified fails
                        video_features, text_features, similarity_matrix = self.matcher(
                            video_frames, expressions
                        )
                        
                        # Multi-positive loss (fallback)
                        loss = self.get_model().compute_multi_positive_loss(
                            video_features, text_features, positive_mapping
                        )
                    
                    # Skip NaN loss batches
                    if torch.isnan(loss):
                        continue
                    
                    val_losses.append(loss.item())
                    
                    # Accuracy: any positive match counts as correct
                    batch_size = similarity_matrix.shape[0]
                    for i in range(batch_size):
                        video_similarities = similarity_matrix[i]
                        positive_indices = positive_mapping.get(i, [i])
                        
                        # Check if any positive has highest similarity
                        max_idx = video_similarities.argmax().item()
                        if max_idx in positive_indices:
                            total_correct += 1
                        total_samples += 1
                
                except Exception as e:
                    if self.is_main_process():
                        print(f"❌ Validation error: {e}")
                    continue
        
        val_loss = np.mean(val_losses) if val_losses else float('inf')
        val_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        return val_loss, val_accuracy
    
    def save_checkpoint(self, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'matcher_state_dict': self.get_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,  # Changed from best_val_acc
            'config': vars(self.args)
        }
        
        # Save text bank and descriptions if available
        if hasattr(self.matcher, 'text_embedding_bank') and self.matcher.text_embedding_bank is not None:
            checkpoint['text_embedding_bank'] = self.matcher.text_embedding_bank
        if hasattr(self.matcher, 'text_descriptions'):
            checkpoint['text_descriptions'] = self.matcher.text_descriptions
        
        # Save regular checkpoint
        if self.is_main_process():
            checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            if self.is_main_process():
                best_path = self.output_dir / 'best_checkpoint.pth'
                torch.save(checkpoint, best_path)
                print(f"💾 Best checkpoint saved: {best_path}")
    
    def train(self):
        """Main training loop"""
        if self.is_main_process():
            print(f"🚀 Starting TCAM training for {self.args.num_epochs} epochs...")
        
        # Build text bank from training data
        if self.distributed:
            if self.is_main_process():
                self._build_initial_text_bank()
            # Broadcast text bank to all ranks
            self._broadcast_text_bank()
        else:
            self._build_initial_text_bank()
        
        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            
            if self.distributed:
                # Ensure each epoch shuffles differently across ranks
                if hasattr(self, 'train_sampler') and self.train_sampler is not None:
                    self.train_sampler.set_epoch(epoch)
            if self.is_main_process():
                print(f"\n{'='*60}")
                print(f"📚 EPOCH {epoch+1}/{self.args.num_epochs}")
                print(f"{'='*60}")
            
            # Train
            train_loss, train_accuracy = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            if self.is_main_process():
                print(f"📊 Epoch {epoch+1} Results: Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss  # Changed from accuracy to loss (lower is better)
            if is_best:
                self.best_val_loss = val_loss
                if self.is_main_process():
                    print(f"🏆 New best validation loss: {val_loss:.4f}")
            
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        if self.is_main_process():
            print(f"\n✅ TCAM training completed!")
            print(f"📊 Best validation loss: {self.best_val_loss:.4f}")
    
    def _build_initial_text_bank(self):
        """
        FIXED: Precompute ALL text embeddings before training starts
        This eliminates the moving target problem during training
        OPTIMIZED: Direct text access without loading video frames
        """
        if self.is_main_process():
            print("📚 Precomputing text bank from all training data...")
        
        self.matcher.eval()
        model = self.get_model()
        all_text_features = []
        all_descriptions = []
        
        with torch.no_grad():
            # OPTIMIZED: Direct access to text descriptions without loading video data
            if self.is_main_process():
                print("🔍 Collecting unique text descriptions...")
            unique_descriptions = set()
            for sample in tqdm(self.train_dataset.samples, desc="Scanning samples"):
                unique_descriptions.add(sample['expression'])
            
            unique_descriptions = list(unique_descriptions)
            if self.is_main_process():
                print(f"📝 Found {len(unique_descriptions)} unique text descriptions")
            
            # Process in batches to avoid memory issues
            batch_size = 32
            if self.is_main_process():
                print("🤖 Encoding text descriptions with CLIP...")
            for i in tqdm(range(0, len(unique_descriptions), batch_size), desc="Encoding batches"):
                batch_descriptions = unique_descriptions[i:i + batch_size]
                
                # Encode text batch
                text_embeddings = model.text_encoder(batch_descriptions)
                text_features = model.text_projector(text_embeddings)
                text_features = F.normalize(text_features, p=2, dim=1)
                
                all_text_features.append(text_features)
                all_descriptions.extend(batch_descriptions)
            
            # Combine all text features
            if all_text_features:
                text_bank = torch.cat(all_text_features, dim=0)
                
                # Store in matcher
                model.text_embedding_bank = text_bank
                model.text_descriptions = all_descriptions
                
                if self.is_main_process():
                    print(f"✅ Text bank precomputed: {text_bank.shape[0]} embeddings")
                    print(f"📊 Text bank statistics:")
                    print(f"   - Mean: {text_bank.mean():.4f}")
                    print(f"   - Std: {text_bank.std():.4f}")
                    print(f"   - Min: {text_bank.min():.4f}")
                    print(f"   - Max: {text_bank.max():.4f}")
            else:
                if self.is_main_process():
                    print("❌ No text descriptions found!")
        
        # Return to training mode
        self.matcher.train()

    def _broadcast_text_bank(self):
        """Broadcast precomputed text bank from rank-0 to all ranks."""
        if not self.distributed:
            return
        # Determine size from rank-0
        if self.is_main_process():
            model = self.get_model()
            bank = getattr(model, 'text_embedding_bank', None)
            size = torch.tensor([-1, -1], dtype=torch.long, device=self.device)
            if bank is not None:
                size[0] = bank.shape[0]
                size[1] = bank.shape[1]
        else:
            size = torch.tensor([-1, -1], dtype=torch.long, device=self.device)
        dist.broadcast(size, src=0)
        rows, cols = int(size[0].item()), int(size[1].item())
        if rows <= 0 or cols <= 0:
            return
        # Prepare tensor on non-main ranks
        model = self.get_model()
        if not self.is_main_process():
            setattr(model, 'text_embedding_bank', torch.empty(rows, cols, device=self.device))
        # Broadcast data
        dist.broadcast(getattr(model, 'text_embedding_bank'), src=0)
    



def main():
    parser = argparse.ArgumentParser(description='TCAM Training')
    
    # Data args
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='Maximum number of videos to use (None = all)')
    
    # Model args
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames per video')
    parser.add_argument('--frame_height', type=int, default=256,
                        help='Frame height')
    parser.add_argument('--frame_width', type=int, default=320,
                        help='Frame width')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=16,
                        help='Number of attention heads')
    parser.add_argument('--grid_size', type=int, default=20,
                        help='CoTracker grid size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Contrastive loss temperature')
    parser.add_argument('--spatial_weight', type=float, default=0.1,
                        help='Weight for spatial track supervision loss (0.1 = 10% spatial, 90% global)')
    parser.add_argument('--spatial_loss_type', type=str, default='ranking',
                        choices=['ranking'],
                        help='Type of loss for spatial grounding: ranking (margin-based, class-imbalance robust)')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loader workers')
    
    # Logging args
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log interval in batches')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save interval in epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed (will be offset by rank)')
    
    # Resume training args
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    
    # Distributed args (torchrun will set env vars; --local_rank kept for compatibility)
    parser.add_argument('--distributed', action='store_true', help='Force distributed initialization if >1 ranks available')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by torchrun')
    
    args = parser.parse_args()
    
    # Print configuration
    print("🔧 TCAM TRAINING")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key:20}: {value}")
    print("=" * 60)
    
    # Seeding per-rank for reproducibility
    def seed_all(seed: int, rank_offset: int = 0):
        s = int(seed) + int(rank_offset)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        # Deterministic behavior for cuDNN
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    rank_env = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
    seed_all(args.seed, rank_env)
    
    # Create trainer and train
    trainer = TCAMTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
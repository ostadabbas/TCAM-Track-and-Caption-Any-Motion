#!/usr/bin/env python3
"""
TCAM Dataset for Multi-Expression Training
Groups multiple expressions per video for multi-positive contrastive learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import cv2
from PIL import Image
from pycocotools import mask as coco_mask
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict
import hashlib  # For deterministic hashing


class TCAMDataset(Dataset):
    """
    TCAM Dataset that supports multi-positive training
    Same video can match multiple expressions during training
    """
    
    def __init__(self, 
                 data_root: str = './data',
                 split: str = 'train',
                 num_frames: int = 16,
                 frame_size: Tuple[int, int] = (256, 320),
                 max_videos: Optional[int] = None,
                 min_expressions_per_video: int = 1):  # Changed from 2 to 1!
        """
        Args:
            data_root: Path to dataset
            split: 'train' or 'val' 
            num_frames: Number of frames to load per video
            frame_size: (H, W) frame resolution
            max_videos: Limit number of videos for testing (None = all)
            min_expressions_per_video: Set to 1 to include ALL videos
        """
        self.data_root = Path(data_root)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.min_expressions_per_video = min_expressions_per_video
        
        # Load dataset metadata
        self._load_metadata()
        
        # Load mask dictionary
        self._load_masks()
        
        # Create multi-positive samples
        self._create_multi_positive_samples(max_videos)
        
        print(f"📺 Loaded {len(self.samples)} samples from {len(self.video_to_samples)} videos")
        if len(self.video_to_samples) > 0:
            print(f"🔗 Average {len(self.samples)/len(self.video_to_samples):.1f} expressions per video")
        else:
            print("🔗 No videos loaded")
        
    def _load_metadata(self):
        """Load dataset metadata"""
        if self.split == 'train':
            ann_file = self.data_root / 'train' / 'meta_expressions.json'
            self.img_folder = self.data_root / 'train'
        else:
            # For now, use train set (val annotations not available)
            ann_file = self.data_root / 'train' / 'meta_expressions.json'
            self.img_folder = self.data_root / 'train'
            
        with open(ann_file, 'r') as f:
            data = json.load(f)
            
        self.videos_data = data['videos']
        print(f"📁 Found {len(self.videos_data)} videos in metadata")
        
    def _load_masks(self):
        """Load mask dictionary"""
        mask_file = self.img_folder / 'mask_dict.json'
        with open(mask_file, 'r') as f:
            self.mask_dict = json.load(f)
        print(f"🎭 Loaded {len(self.mask_dict)} mask annotations")
        
    def _create_multi_positive_samples(self, max_videos):
        """Create samples supporting multi-positive training"""
        self.samples = []
        self.video_to_samples = defaultdict(list)  # Map video_id to sample indices
        
        video_list = list(self.videos_data.keys())
        if max_videos is not None:
            video_list = video_list[:max_videos]
            
        videos_with_multiple_expressions = 0
        total_expressions = 0
        
        for video_id in video_list:
            video_data = self.videos_data[video_id]
            frames = video_data['frames']
            
            # Skip videos that are too short
            if len(frames) < self.num_frames:
                continue
            
            # Collect all expressions for this video
            expressions = []
            for exp_id, exp_data in video_data['expressions'].items():
                expressions.append({
                    'exp_id': exp_id,
                    'expression': exp_data['exp'],
                    'obj_ids': exp_data['obj_id'],
                    'anno_ids': [str(x) for x in exp_data['anno_id']]
                })
            
            # Only keep videos with minimum number of expressions
            if len(expressions) < self.min_expressions_per_video:
                continue
                
            videos_with_multiple_expressions += 1
            total_expressions += len(expressions)
            
            # FIXED: Deterministic frame sampling using MD5 hash (not Python's hash which is random!)
            seed_hash = int(hashlib.md5(video_id.encode()).hexdigest(), 16) % 1000000
            np.random.seed(abs(seed_hash))
            max_start = len(frames) - self.num_frames
            start_frame = np.random.randint(0, max(1, max_start + 1))
            selected_frames = frames[start_frame:start_frame + self.num_frames]
            
            # Store the seed for debugging
            # print(f"Video {video_id}: seed={seed_hash}, start_frame={start_frame}")  # Debug
            
            # Create individual samples for each expression (for dataloader compatibility)
            video_sample_indices = []
            for expr_data in expressions:
                sample_idx = len(self.samples)
                sample = {
                    'video_id': video_id,
                    'expression_id': expr_data['exp_id'],
                    'expression': expr_data['expression'].strip().lower(),
                    'obj_ids': expr_data['obj_ids'],
                    'anno_ids': expr_data['anno_ids'],
                    'frames': selected_frames,
                    'start_frame': start_frame
                }
                
                self.samples.append(sample)
                video_sample_indices.append(sample_idx)
            
            # Store mapping from video to all its samples
            self.video_to_samples[video_id] = video_sample_indices
        
        print(f"✅ Created {len(self.samples)} samples from {videos_with_multiple_expressions} videos")
        if videos_with_multiple_expressions > 0:
            print(f"📊 Average {total_expressions/videos_with_multiple_expressions:.1f} expressions per video")
        else:
            print("📊 No videos with multiple expressions found")
        
        # Reset random seed
        np.random.seed(None)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a video sample with multi-positive information"""
        sample = self.samples[idx]
        
        # Load video frames
        frames = self._load_video_frames(sample)
        
        # Load object masks
        masks = self._load_object_masks(sample)
        
        return {
            'video_frames': frames,  # [T, 3, H, W] tensor 0-1 normalized
            'object_masks': masks,   # [T, H, W] tensor 0-1
            'expression': sample['expression'],  # str
            'video_id': sample['video_id'],
            'expression_id': sample['expression_id'],
            'sample_idx': idx,  # For getting positive mapping
            'metadata': sample
        }
    
    def get_batch_positive_mapping(self, batch_sample_indices):
        """
        Get mapping of which samples in batch are positive for each other
        
        Args:
            batch_sample_indices: List of sample indices in current batch
            
        Returns:
            positive_mapping: Dict[batch_idx -> List[batch_idx]] of positive pairs
        """
        positive_mapping = {}
        
        # Group batch samples by video_id
        video_to_batch_indices = defaultdict(list)
        for batch_idx, sample_idx in enumerate(batch_sample_indices):
            video_id = self.samples[sample_idx]['video_id']
            video_to_batch_indices[video_id].append(batch_idx)
        
        # Create positive mapping: samples from same video are positive
        for batch_idx, sample_idx in enumerate(batch_sample_indices):
            video_id = self.samples[sample_idx]['video_id']
            positive_mapping[batch_idx] = video_to_batch_indices[video_id]
        
        return positive_mapping
    
    def _load_video_frames(self, sample):
        """Load and preprocess video frames"""
        frames = []
        video_path = self.img_folder / 'JPEGImages' / sample['video_id']
        
        for frame_name in sample['frames']:
            frame_path = video_path / f'{frame_name}.jpg'  # Add .jpg extension like original
            
            try:
                # Use PIL like original (better error handling)
                img = Image.open(frame_path).convert('RGB')
                img = img.resize((self.frame_size[1], self.frame_size[0]))  # (W, H)
                
                # Convert to tensor and normalize
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [3, H, W]
                
                frames.append(img_tensor)
                
            except Exception as e:
                print(f"⚠️ Failed to load frame {frame_path}: {e}")
                # Create black frame as fallback
                black_frame = torch.zeros(3, self.frame_size[0], self.frame_size[1])
                frames.append(black_frame)
        
        return torch.stack(frames, dim=0)  # [T, 3, H, W]
    
    def _load_object_masks(self, sample):
        """Load object masks for each frame"""
        masks = []
        anno_ids = sample['anno_ids']
        start_frame = sample['start_frame']
        
        for i, frame_name in enumerate(sample['frames']):
            frame_idx = start_frame + i
            
            # Combine masks from all annotation IDs
            combined_mask = np.zeros(self.frame_size, dtype=np.float32)
            
            for anno_id in anno_ids:
                if anno_id in self.mask_dict and frame_idx < len(self.mask_dict[anno_id]):
                    frame_anno = self.mask_dict[anno_id][frame_idx]
                    if frame_anno is not None:
                        try:
                            decoded_mask = coco_mask.decode(frame_anno)
                            # Resize mask to match frame size
                            decoded_mask = cv2.resize(decoded_mask, 
                                                   (self.frame_size[1], self.frame_size[0]),
                                                   interpolation=cv2.INTER_NEAREST)
                            combined_mask += decoded_mask
                        except Exception as e:
                            print(f"⚠️ Failed to decode mask for anno_id {anno_id}, frame {frame_idx}: {e}")
                            continue
            
            # Clip to [0, 1]
            combined_mask = np.clip(combined_mask, 0, 1)
            mask_tensor = torch.from_numpy(combined_mask)
            
            masks.append(mask_tensor)
            
        return torch.stack(masks, dim=0)  # [T, H, W]



def create_tcam_dataloader(dataset, batch_size, shuffle=True, num_workers=2):
    """Dead simple dataloader - let PyTorch handle everything"""
    
    def collate_fn(batch):
        """Standard collate function"""
        video_frames = torch.stack([item['video_frames'] for item in batch])
        object_masks = torch.stack([item['object_masks'] for item in batch])
        expressions = [item['expression'] for item in batch]
        video_ids = [item['video_id'] for item in batch]
        expression_ids = [item['expression_id'] for item in batch]
        sample_indices = [item['sample_idx'] for item in batch]
        
        
        return {
            'video_frames': video_frames,
            'object_masks': object_masks,
            'expression': expressions,
            'video_id': video_ids,
            'expression_id': expression_ids,
            'sample_indices': sample_indices
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False  # Keep all data, even last incomplete batch
    )


if __name__ == '__main__':
    # Test the dataset
    print("🧪 Testing TCAMDataset...")
    
    dataset = TCAMDataset(
        data_root='./data',
        max_videos=10,
        min_expressions_per_video=1  # Include all videos
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test dataloader
    dataloader = create_tcam_dataloader(dataset, batch_size=4)
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Video frames: {batch['video_frames'].shape}")
        print(f"  Expressions: {batch['expression']}")
        print(f"  Video IDs: {batch['video_id']}")
        
        # Test positive mapping
        positive_mapping = dataset.get_batch_positive_mapping(batch['sample_indices'])
        print(f"  Positive mapping: {positive_mapping}")
        
        if batch_idx >= 2:  # Just test a few batches
            break
    
    print("✅ Dataset test completed!")